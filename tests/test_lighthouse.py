# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable
import io

from tests.test_support import *

from theia.server.lighthouse import LighthouseView, render_lighthouse


def _snapshot(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "version": "2.0.0",
        "action": "Processing request",
        "mode": "text",
        "model": "gpt-5.6-luna",
        "reasoning": "adaptive",
        "character": {"name": "Cel", "source": "user overlay"},
        "presence": {"status": "online", "line": "reviewing the request"},
        "voice": {"providers": ("qwen",), "state": "listening"},
        "attention": {"active": "Lighthouse View", "parked": ("Memory",)},
        "mood": {
            "traits": "calm and attentive",
            "label": "engaged",
            "strength": 0.58,
        },
        "session": "Server conversation · #general",
        "workspace": {
            "entries": (
                {"category": "goal", "text": "Inspect the runtime"},
                {"category": "open_question", "text": "Should the view stay enabled?"},
            )
        },
        "runtime": {
            "codex": "connected",
            "codex_version": "0.154.0",
            "rss_bytes": 7 * 1024**3,
            "active_turns": 1,
            "workers": 0,
            "approvals": 0,
            "memory_entries": 52,
            "watchdog": "watching",
            "recovery": False,
            "update": "enabled",
            "heartbeat": {
                "state": "connected",
                "latency_ms": 14.5,
                "consecutive_failures": 0,
            },
        },
        "events": ({"timestamp": 0, "event": "turn_started"},),
    }
    value.update(overrides)
    return value


class _TTYBuffer:
    def __init__(self) -> None:
        self.value = ""

    def isatty(self) -> bool:
        return True

    def write(self, value: str) -> int:
        self.value += value
        return len(value)

    def flush(self) -> None:
        return


class LighthouseTests(unittest.IsolatedAsyncioTestCase):
    def test_render_contains_all_sections_and_redacts_unsafe_values(self) -> None:
        rendered = render_lighthouse(
            _snapshot(
                session="guild:42:channel:7:user:9 /root/private/token.txt",
                character={
                    "name": "Cel",
                    "source": "/root/.theia/personality.md",
                },
                events=({"timestamp": 0, "event": "unknown_private_event"},),
            )
        )

        for section in ("Workspace", "Runtime", "Recent events"):
            self.assertIn(section, rendered)
        for label in (
            "Status",
            "Model",
            "Character",
            "Presence",
            "Voice",
            "Attention",
            "Mood",
            "Session",
            "RSS",
            "Approvals",
            "Memory",
            "Heartbeat",
        ):
            self.assertIn(label, rendered)
        self.assertNotIn("guild:42:channel:7:user:9", rendered)
        self.assertNotIn("/root/private", rendered)
        self.assertNotIn("token.txt", rendered)
        self.assertIn("Qwen Audio Agent · listening", rendered)
        self.assertIn("7.0 GB", rendered)

    def test_empty_workspace_and_missing_subsystems_are_truthful(self) -> None:
        rendered = render_lighthouse(
            _snapshot(
                character={"name": "none", "source": "no overlay"},
                presence={"status": "unknown", "line": "none"},
                voice={"providers": (), "state": "disabled"},
                workspace={"entries": ()},
                runtime={"heartbeat": {"state": "unknown"}},
                events=(),
            )
        )
        self.assertIn("No active workspace entries", rendered)
        self.assertIn("Presence     unknown", rendered)
        self.assertIn("Voice        disabled", rendered)
        self.assertIn("No recent events", rendered)
        self.assertIn("Codex        unknown · unknown", rendered)

    async def test_noninteractive_terminal_keeps_normal_logging_and_no_heartbeat(self):
        heartbeat = AsyncMock()
        codex = SimpleNamespace(
            lighthouse_snapshot=Mock(return_value=_snapshot()),
            heartbeat=heartbeat,
        )
        view = LighthouseView(codex, output=io.StringIO())
        self.assertFalse(await view.start())
        heartbeat.assert_not_awaited()
        await view.close()

    async def test_interactive_view_updates_and_heartbeat_is_transport_only(self):
        states = [_snapshot(action="Idle"), _snapshot(action="Processing request")]
        calls: list[str] = []

        def snapshot(_key: str | None = None) -> dict[str, Any]:
            return states[min(len(calls), len(states) - 1)]

        async def heartbeat(*, timeout: float) -> dict[str, Any]:
            calls.append(f"heartbeat:{timeout}")
            return {"state": "connected"}

        output = _TTYBuffer()
        codex = SimpleNamespace(lighthouse_snapshot=snapshot, heartbeat=heartbeat)
        view = LighthouseView(
            codex,
            output=output,
            refresh_interval=0.1,
            heartbeat_interval=1,
        )
        self.assertTrue(await view.start())
        await asyncio.sleep(0.15)
        await view.close()

        self.assertTrue(calls)
        self.assertEqual(calls[0], "heartbeat:1.5")
        self.assertIn("Lighthouse View", output.value)
        self.assertIn("Runtime", output.value)

    async def test_heartbeat_uses_account_probe_without_a_turn(self) -> None:
        server = main.CodexAppServer()
        cast(Any, server)._process = SimpleNamespace(returncode=None)
        cast(Any, server)._reader_task = SimpleNamespace(done=lambda: False)
        server._request = AsyncMock(return_value={})

        result = await server.heartbeat(timeout=0.2)

        self.assertEqual(result["state"], "connected")
        server._request.assert_awaited_once_with(
            "account/read",
            {"refreshToken": False},
            timeout=0.2,
        )
        self.assertNotIn("turn/start", repr(server._request.await_args))

    async def test_heartbeat_failure_is_dashboard_only(self) -> None:
        server = main.CodexAppServer()
        cast(Any, server)._process = SimpleNamespace(returncode=None)
        cast(Any, server)._reader_task = SimpleNamespace(done=lambda: False)
        server._request = AsyncMock(side_effect=RuntimeError("transport"))

        result = await server.heartbeat()

        self.assertEqual(result["state"], "degraded")
        self.assertEqual(result["consecutive_failures"], 1)

    def test_compose_allocates_a_tty_for_lighthouse_view(self) -> None:
        compose = Path("compose.yaml").read_text(encoding="utf-8")
        self.assertIn("stdin_open: true", compose)
        self.assertIn("tty: true", compose)
