# SPDX-License-Identifier: Apache-2.0
"""Deterministic subprocess-boundary tests for the local Codex runtime."""

# pylint: disable=protected-access

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import main


ROOT = Path(__file__).resolve().parents[1]
FAKE_APP_SERVER = ROOT / "tests" / "fixtures" / "fake_app_server.ts"
TSCONFIG = ROOT / "tsconfig.json"


class _BoundaryChannel:
    """Small Discord-like channel used by the approval boundary test."""

    id = 500

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        member = SimpleNamespace(
            id=7,
            guild_permissions=SimpleNamespace(administrator=True),
        )
        self.guild = SimpleNamespace(
            id=500,
            get_member=lambda user_id: member if user_id == 7 else None,
        )

    async def send(self, **kwargs: Any) -> SimpleNamespace:
        self.sent.append(kwargs)
        return SimpleNamespace(id=len(self.sent))


def _node_executable() -> str | None:
    return shutil.which("node")


def _project_tool(name: str) -> Path | None:
    """Return a project-local npm executable when one has been installed."""
    suffix = ".cmd" if os.name == "nt" else ""
    candidate = ROOT / "node_modules" / ".bin" / f"{name}{suffix}"
    return candidate if candidate.is_file() else None


def _compile_fake_app_server(output_root: Path) -> Path:
    """Compile the strict TypeScript fixture into an isolated runtime directory."""
    compiler = _project_tool("tsc")
    if compiler is None:
        raise AssertionError("run npm ci before compiling the TypeScript test harness")
    result = subprocess.run(
        [
            str(compiler),
            "--project",
            str(TSCONFIG),
            "--outDir",
            str(output_root),
            "--noEmit",
            "false",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    fixture = output_root / "fake_app_server.js"
    if not fixture.is_file():
        raise AssertionError(f"TypeScript compiler did not emit {fixture}")
    return fixture


def _create_fake_cli(runtime_root: Path, node: str, fixture_source: Path) -> Path:
    """Create a platform-native launcher for the compiled Node JSONL fixture."""
    fixture = runtime_root / fixture_source.name
    shutil.copyfile(fixture_source, fixture)
    if os.name == "nt":
        launcher = runtime_root / "fake-codex.cmd"
        launcher.write_text(
            f'@echo off\r\n"{node}" "%~dp0{fixture.name}" %*\r\n',
            encoding="utf-8",
        )
        return launcher

    launcher = runtime_root / "fake-codex"
    launcher.write_text(
        f'#!/bin/sh\nexec {shlex.quote(node)} {shlex.quote(str(fixture))} "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o700)
    return launcher


class TestLocalCodexBoundary(unittest.IsolatedAsyncioTestCase):
    """Exercise the actual Python-to-Node JSONL process boundary."""

    _compiled_fixture: Path

    @classmethod
    def setUpClass(cls) -> None:
        """Compile the TypeScript responder once for this worker process."""
        if _node_executable() is None:
            raise unittest.SkipTest(
                "Node.js is required for local boundary integration tests"
            )
        if _project_tool("tsc") is None:
            raise unittest.SkipTest(
                "npm dependencies are required for local boundary integration tests"
            )
        cls._compiled_root = Path(tempfile.mkdtemp(prefix="theia-fake-server-"))
        cls.addClassCleanup(shutil.rmtree, cls._compiled_root, ignore_errors=True)
        cls._compiled_fixture = _compile_fake_app_server(cls._compiled_root)

    def setUp(self) -> None:
        self._root = Path(tempfile.mkdtemp(prefix="theia-boundary-"))
        self.addCleanup(shutil.rmtree, self._root, ignore_errors=True)
        node = _node_executable()
        if node is None:
            self.skipTest("Node.js is required for local boundary integration tests")
        workspace = self._root / "workspace"
        workspace.mkdir()
        launcher = _create_fake_cli(self._root, node, self._compiled_fixture)
        self._environment = patch.dict(
            os.environ,
            {
                "CODEX_ADAPTIVE_REASONING": "false",
                "CODEX_CWD": str(workspace),
                "CODEX_HOME": str(self._root / "global-codex"),
                "HERMES_HOME": str(self._root / "hermes"),
                "THEIA_HOME": str(self._root / "theia"),
                "THEIA_STATE": str(self._root / "sessions.json"),
                "THEIA_CODEX_CLI": str(launcher),
                "THEIA_INCLUDE_GLOBAL_MEMORY": "false",
            },
        )
        self._environment.start()
        self.addCleanup(self._environment.stop)

    async def _server(
        self,
        *,
        scenario: str = "",
        extra_environment: dict[str, str] | None = None,
    ) -> main.CodexAppServer:
        child_environment = {"FAKE_APP_SERVER_SCENARIO": scenario}
        if extra_environment:
            child_environment.update(extra_environment)
        with patch.dict(os.environ, child_environment):
            server = main.CodexAppServer()
        await server.start()
        self.addAsyncCleanup(server.close)
        return server

    async def _wait_for(self, predicate: Any) -> None:
        for _ in range(200):
            if predicate():
                return
            await asyncio.sleep(0.01)
        self.fail("timed out waiting for the fake app-server boundary")

    async def _ask(
        self,
        server: main.CodexAppServer,
        prompt: str,
        *,
        session_key: str = "boundary",
        channel: Any | None = None,
        user_id: int | None = None,
        on_event: Any | None = None,
    ) -> str:
        return await server.ask(
            prompt,
            session_key=session_key,
            channel=channel,
            user_id=user_id,
            allow_tools=True,
            on_event=on_event,
        )

    async def test_normal_responses_and_prompts_cross_the_boundary(self) -> None:
        """Verify ordinary replies preserve the prompt sent over JSONL."""
        server = await self._server(scenario="normal")

        prompt = "approval error crash should remain an ordinary prompt"
        self.assertEqual(await self._ask(server, prompt), "normal response")

        prompt_server = await self._server(scenario="prompt")
        prompt = "echo preserve this exact prompt: <opaque-value>"
        self.assertEqual(await self._ask(prompt_server, prompt), f"echo: {prompt}")

    async def test_intermediates_and_preambles_are_delivered_before_final_text(
        self,
    ) -> None:
        """Verify commentary items arrive before the final agent message."""
        server = await self._server(scenario="preamble-and-intermediate")
        events: list[tuple[str, dict[str, Any]]] = []

        async def on_event(event: str, payload: dict[str, Any]) -> None:
            events.append((event, payload))

        result = await self._ask(
            server,
            "plain request",
            on_event=on_event,
        )

        self.assertEqual(result, "streamed response")
        commentary = [
            payload.get("text")
            for event, payload in events
            if event == "item_completed"
        ]
        self.assertIn("Here is a short preamble.", commentary)
        self.assertIn("Working through the request.", commentary)
        self.assertEqual(events[-1][0], "turn_completed")

    async def test_refusal_is_a_completed_non_error_response(self) -> None:
        """Verify a model refusal remains user-visible without becoming a transport error."""
        server = await self._server(scenario="refusal")

        self.assertEqual(
            await self._ask(server, "please answer normally"),
            "I can't help with that request.",
        )

    async def test_multiple_choice_question_round_trip(self) -> None:
        """Verify structured question requests can be answered through the Discord view."""
        server = await self._server(scenario="multiple-choice")
        channel = _BoundaryChannel()
        task = asyncio.create_task(
            self._ask(
                server,
                "choose one",
                session_key="question",
                channel=channel,
                user_id=7,
            )
        )
        await self._wait_for(lambda: bool(channel.sent))

        view = channel.sent[-1]["view"]
        self.assertIsInstance(view, main._UserInputView)
        view.value = {"answers": {"color": ["Blue"]}}
        view.stop()
        self.assertEqual(await task, "choice accepted")

    async def test_invalid_api_requests_are_reported_as_protocol_errors(self) -> None:
        """Verify unknown JSON-RPC methods do not become silent timeouts."""
        server = await self._server()

        with self.assertRaisesRegex(main.CodexAppServerError, "Invalid API request"):
            await server._request("invalid/api/request", {})

    async def test_malformed_jsonrpc_requests_receive_explicit_errors(self) -> None:
        """Verify malformed JSON-RPC lines receive protocol error responses."""
        node = _node_executable()
        assert node is not None
        process = await asyncio.create_subprocess_exec(
            node,
            str(self._compiled_fixture),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        assert process.stdin is not None
        assert process.stdout is not None

        async def read_response() -> dict[str, Any]:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=3)
            self.assertTrue(line)
            return json.loads(line)

        try:
            process.stdin.write(b"not json\n")
            await process.stdin.drain()
            parse_error = await read_response()
            self.assertIsNone(parse_error["id"])
            self.assertEqual(parse_error["error"]["code"], -32600)

            process.stdin.write(b'{"id": 2, "method": "turn/start", "params": []}\n')
            await process.stdin.drain()
            params_error = await read_response()
            self.assertEqual(params_error["id"], 2)
            self.assertEqual(params_error["error"]["code"], -32602)
        finally:
            if process.returncode is None:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=3)

    async def test_rate_limits_and_exhausted_usage_are_returned(self) -> None:
        """Verify account limit snapshots cross the same request/response boundary."""
        rate_server = await self._server(scenario="rate-limit")
        credit_snapshot = await rate_server.credits()
        self.assertEqual(credit_snapshot["rateLimits"]["primary"]["usedPercent"], 100)

        usage_server = await self._server(scenario="usage-exhausted")
        usage = await usage_server.usage()
        self.assertTrue(usage["exhausted"])
        self.assertEqual(usage["usage"]["remaining"], 0)

    async def test_outage_and_authentication_failures_are_distinguishable(self) -> None:
        """Verify service outages and login-required responses retain their causes."""
        outage = await self._server(scenario="outage")
        with self.assertRaisesRegex(
            main.CodexAppServerError, "OpenAI service unavailable"
        ):
            await self._ask(outage, "service request")

        auth_failure = await self._server(scenario="auth-failure")
        with self.assertRaisesRegex(main.CodexAppServerError, "Run `/login` first"):
            await self._ask(auth_failure, "normal response")

    async def test_streamed_jsonl_events_and_malformed_line_cross_the_boundary(
        self,
    ) -> None:
        """Verify streamed notifications survive the real Node stdio hop."""
        events: list[tuple[str, dict[str, Any]]] = []

        async def on_event(event: str, payload: dict[str, Any]) -> None:
            events.append((event, payload))

        server = await self._server(scenario="malformed-stream")
        result = await self._ask(server, "ordinary request", on_event=on_event)

        self.assertEqual(result, "streamed response")
        self.assertEqual(
            [event for event, _ in events],
            [
                "item_started",
                "agent_message",
                "agent_message",
                "item_completed",
                "turn_completed",
            ],
        )
        self.assertEqual(server.status("boundary")["turn_id"], None)

    async def test_approval_round_trip_crosses_jsonl_in_both_decisions(self) -> None:
        """Verify Discord approval resolution reaches the fake app server."""
        server = await self._server(scenario="approval")
        channel = _BoundaryChannel()
        approved = asyncio.create_task(
            self._ask(
                server,
                "run safe command",
                session_key="approval-accept",
                channel=channel,
                user_id=7,
            )
        )
        await self._wait_for(lambda: bool(server._pending_approvals))

        self.assertTrue(server.resolve_approval(7, True, channel))
        self.assertEqual(await approved, "approved response")
        self.assertEqual(len(channel.sent), 1)
        self.assertIsNotNone(channel.sent[0]["view"])

        denied = asyncio.create_task(
            self._ask(
                server,
                "run another command",
                session_key="approval-deny",
                channel=channel,
                user_id=7,
            )
        )
        await self._wait_for(lambda: len(server._pending_approvals) == 1)
        self.assertTrue(server.resolve_approval(7, False, channel))
        self.assertEqual(await denied, "denied response")

    async def test_turn_error_and_interruptions_are_reported(self) -> None:
        """Verify failed and interrupted turns do not look like successful replies."""
        server = await self._server(scenario="error")

        with self.assertRaisesRegex(
            main.CodexAppServerError, "fake app-server failure"
        ):
            await self._ask(server, "return a normal result", session_key="error")

        interrupt_server = await self._server(scenario="interrupt")
        interrupted = asyncio.create_task(
            self._ask(interrupt_server, "keep working", session_key="interrupt")
        )
        await self._wait_for(
            lambda: interrupt_server.status("interrupt")["turn_id"] is not None
        )
        self.assertTrue(await interrupt_server.interrupt("interrupt"))
        with self.assertRaisesRegex(main.CodexAppServerError, "fake interruption"):
            await interrupted

    async def test_timeout_is_interrupted_and_reported_to_the_caller(self) -> None:
        """Verify a turn that never completes follows the timeout recovery path."""
        server = await self._server(
            scenario="timeout",
            extra_environment={"CODEX_TURN_TIMEOUT": "0.1"},
        )

        with self.assertRaisesRegex(
            main.CodexAppServerError, "timed out and was interrupted"
        ):
            await self._ask(server, "please wait", session_key="timeout")

    async def test_eof_and_process_crashes_release_active_turns(self) -> None:
        """Verify graceful EOF and abnormal child exit wake active callers."""
        eof_server = await self._server(scenario="eof")
        with self.assertRaisesRegex(main.CodexAppServerError, "App Server exited"):
            await self._ask(eof_server, "finish this", session_key="eof")

        crash_server = await self._server(scenario="crash")
        with self.assertRaisesRegex(main.CodexAppServerError, "App Server exited"):
            await self._ask(crash_server, "finish this", session_key="crash")

    async def test_broken_stdin_pipe_is_not_silent(self) -> None:
        """Verify writes after the child closes stdin surface as boundary failures."""
        server = await self._server()
        await server._request("close-stdin", {})
        await asyncio.sleep(0.1)

        with self.assertRaises(
            (
                main.CodexAppServerError,
                BrokenPipeError,
                ConnectionAbortedError,
                ConnectionResetError,
            )
        ):
            await server._request("after-pipe-close", {}, timeout=1)

    async def test_dead_app_server_releases_pending_protocol_requests(self) -> None:
        """Verify an exited child wakes requests waiting on its JSONL response."""
        server = await self._server()
        request = asyncio.create_task(server._request("die", {}, timeout=5))

        with self.assertRaises(main.CodexAppServerError):
            await asyncio.wait_for(request, timeout=3)

    async def test_restart_restores_state_and_reconnects_to_a_new_child(self) -> None:
        """Verify persisted sessions survive a Python restart and app-server reconnect."""
        server = await self._server()
        first = await self._ask(server, "persist this thread", session_key="restore")
        thread_id = server.status("restore")["thread_id"]
        self.assertEqual(first, "streamed response")
        self.assertIsInstance(thread_id, str)
        await server.close()

        child_code = (
            "import json, main; "
            "server = main.CodexAppServer(); "
            "print(json.dumps(server.status('restore')))"
        )
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            child_code,
            cwd=str(ROOT),
            env=os.environ.copy(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(child.communicate(), timeout=15)
        self.assertEqual(child.returncode, 0, stderr.decode())
        restored_status = json.loads(stdout.decode())
        self.assertEqual(restored_status["thread_id"], thread_id)

        reconnected = await self._server()
        self.assertEqual(reconnected.status("restore")["thread_id"], thread_id)
        self.assertEqual(
            await self._ask(
                reconnected, "continue after reconnect", session_key="restore"
            ),
            "streamed response",
        )


class TestProjectNodeBridge(unittest.TestCase):
    """Verify the locked project-local Node bridge is available to CI."""

    def test_project_codex_bridge_is_launchable(self) -> None:
        """Run the installed Node bridge without contacting Codex services."""
        node = _node_executable()
        if node is None:
            self.skipTest("Node.js is required for the project bridge smoke test")
        binary = (
            ROOT
            / "node_modules"
            / ".bin"
            / ("codex.cmd" if os.name == "nt" else "codex")
        )
        if not binary.is_file():
            self.skipTest("run npm ci before the project bridge smoke test")
        result = subprocess.run(
            [str(binary), "--version"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("codex-cli", result.stdout)
