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
            "source": "session workspace",
            "entries": (
                {"category": "goal", "text": "Inspect the runtime"},
                {"category": "open_question", "text": "Should the view stay enabled?"},
            ),
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
    async def test_explicit_session_selection_updates_character_and_dashboard_state(
        self,
    ) -> None:
        server = main.CodexAppServer()
        channel = SimpleNamespace(name="general", guild=SimpleNamespace(id=1))
        user = SimpleNamespace(display_name="Alice")
        session = server.select_session(
            "guild:1:channel:2:user:7",
            channel=channel,
            user=user,
        )
        session.personality_name = "cel"
        session.personality_selected = True
        with (
            patch.object(
                server,
                "personality_selection",
                return_value={"scope": "me", "name": "cel"},
            ),
            patch.object(server, "active_personality", return_value="cel"),
            patch.object(
                server._personalities,
                "summary",
                return_value=SimpleNamespace(character_name="Cel", identifier="cel"),
            ),
        ):
            snapshot = server.lighthouse_snapshot()

        self.assertEqual(snapshot["session"]["active_count"], 1)
        self.assertEqual(
            snapshot["session"]["current"], "Server conversation · #general"
        )
        self.assertEqual(snapshot["character"]["name"], "Cel")
        self.assertTrue(
            any(event["event"] == "session_selected" for event in snapshot["events"])
        )

        await server.new_session(session.key)
        cleared = server.lighthouse_snapshot()
        self.assertEqual(cleared["session"]["active_count"], 0)
        self.assertIsNone(cleared["session"]["current"])
        self.assertIn("Session      No active session", render_lighthouse(cleared))

    async def test_failed_session_resume_is_visible_as_degraded_state(self) -> None:
        server = main.CodexAppServer()
        server._request = AsyncMock(
            side_effect=main.CodexAppServerError("transport failure")
        )
        channel = SimpleNamespace(name="general", guild=SimpleNamespace(id=1))

        with self.assertRaises(main.CodexAppServerError):
            await server.resume_session(
                "guild:1:channel:2:user:7",
                "missing-thread",
                channel=channel,
            )

        snapshot = server.lighthouse_snapshot()
        self.assertEqual(snapshot["session"]["status"], "degraded")
        self.assertEqual(snapshot["session"]["reason"], "Codex session resume failed")
        rendered = render_lighthouse(snapshot)
        self.assertIn("Session state degraded", rendered)
        self.assertIn("Codex session resume failed", rendered)

    async def test_thread_creation_keeps_the_active_discord_route(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server._request = AsyncMock(return_value={"thread": {"id": "created"}})
        channel = SimpleNamespace(name="general", guild=SimpleNamespace(id=1))
        session = server.select_session("guild:1:channel:2:user:7", channel=channel)

        await server._ensure_thread(session, allow_tools=False)

        snapshot = server.lighthouse_snapshot()
        self.assertEqual(
            snapshot["session"]["current"], "Server conversation · #general"
        )
        self.assertTrue(
            any(event["event"] == "session_created" for event in snapshot["events"])
        )

    async def test_live_session_display_tracks_active_turn_ownership(self) -> None:
        server = main.CodexAppServer()
        no_session = server.lighthouse_snapshot()
        self.assertEqual(no_session["session"]["active_count"], 0)
        self.assertIn("Session      No active session", render_lighthouse(no_session))
        restored_session = server._session("restored-session")
        restored_session.thread_id = "restored-thread"
        restored_session.last_activity_at = time.time()
        self.assertIn(
            "Session      No active session",
            render_lighthouse(server.lighthouse_snapshot()),
        )

        first_session = server._session("first-session")
        first_channel = SimpleNamespace(name="general", guild=SimpleNamespace(id=1))
        first_turn = main._TurnState(
            session=first_session,
            channel=first_channel,
            user=SimpleNamespace(display_name="Alice"),
        )
        server._turns["first-turn"] = first_turn
        first_snapshot = server.lighthouse_snapshot()
        self.assertEqual(first_snapshot["session"]["active_count"], 1)
        self.assertEqual(
            first_snapshot["session"]["current"],
            "Server conversation · #general",
        )
        self.assertIn(
            "Session      Server conversation · #general",
            render_lighthouse(first_snapshot),
        )

        second_session = server._session("second-session")
        second_channel = SimpleNamespace(name="private", guild=None)
        second_turn = main._TurnState(
            session=second_session,
            channel=second_channel,
            user=SimpleNamespace(display_name="Bob"),
        )
        server._turns["second-turn"] = second_turn
        concurrent = server.lighthouse_snapshot()
        self.assertEqual(concurrent["session"]["active_count"], 2)
        self.assertEqual(
            concurrent["session"]["current"],
            "User conversation · #private",
        )
        rendered = render_lighthouse(concurrent)
        self.assertIn("Session      2 active sessions", rendered)
        self.assertIn("Current      User conversation · #private", rendered)

        selected = server.lighthouse_snapshot("first-session")
        self.assertEqual(
            selected["session"]["current"],
            "Server conversation · #general",
        )

        first_turn.done.set_result(None)
        second_turn.done.set_result(None)
        ended = server.lighthouse_snapshot()
        self.assertEqual(ended["session"]["active_count"], 0)
        self.assertIn("Session      No active session", render_lighthouse(ended))

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

    def test_render_separates_cleanup_health_from_session_health(self) -> None:
        base_runtime = _snapshot()["runtime"]
        rendered = render_lighthouse(
            _snapshot(
                runtime={
                    **base_runtime,
                    "cleanup": {
                        "status": "degraded",
                        "reason": "expired session cleanup failed",
                    },
                }
            )
        )

        self.assertIn(
            "Cleanup      degraded · expired session cleanup failed", rendered
        )
        self.assertNotIn("Session state degraded", rendered)

    def test_render_preserves_complete_cleanup_protocol_method(self) -> None:
        rendered = render_lighthouse(
            _snapshot(
                events=(
                    {
                        "timestamp": 0,
                        "event": "log_warning",
                        "detail": (
                            "Expired Codex session cleanup failed "
                            "(method=thread/delete, code=-32600)"
                        ),
                    },
                )
            )
        )

        self.assertIn("method=thread/delete", rendered)

    def test_empty_workspace_and_missing_subsystems_are_truthful(self) -> None:
        stale_goal = "stale demo objective"
        rendered = render_lighthouse(
            _snapshot(
                character={"name": "none", "source": "no overlay"},
                presence={"status": "unknown", "line": "none"},
                voice={"providers": (), "state": "disabled"},
                workspace={
                    "source": "session workspace",
                    "entries": (),
                    "goal": stale_goal,
                },
                session_objective=None,
                runtime={"heartbeat": {"state": "unknown"}},
                events=(),
            )
        )
        self.assertIn("No active workspace entries", rendered)
        self.assertIn("Presence     unknown", rendered)
        self.assertIn("Voice        disabled", rendered)
        self.assertIn("No recent events", rendered)
        self.assertIn("Codex        unknown · unknown", rendered)
        self.assertNotIn(stale_goal, rendered)
        self.assertNotIn("Session objective", rendered)

    def test_session_objective_is_distinct_from_workspace(self) -> None:
        rendered = render_lighthouse(
            _snapshot(
                workspace={"source": "session workspace", "entries": ()},
                session_objective="Ship the current release",
            )
        )
        self.assertIn("Session objective Ship the current release", rendered)
        self.assertIn("Workspace\n  No active workspace entries", rendered)

    def test_workspace_entries_without_current_source_are_not_rendered(self) -> None:
        rendered = render_lighthouse(
            _snapshot(
                workspace={
                    "entries": ({"category": "goal", "text": "untrusted objective"},)
                },
                session_objective=None,
            )
        )
        self.assertIn("Workspace\n  No active workspace entries", rendered)
        self.assertNotIn("untrusted objective", rendered)

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
        self.assertIn("\x1b[2J", output.value)
        self.assertIn("\x1b[?1049h", output.value)
        self.assertIn("\x1b[?1049l", output.value)

    async def test_interactive_view_mutes_console_logs_and_preserves_diagnostics(self):
        output = _TTYBuffer()
        logger = logging.getLogger("theia.codex")
        handler = logging.StreamHandler(output)
        logger.addHandler(handler)
        try:
            events: list[tuple[str, str]] = []

            def record_event(event: str, detail: str) -> None:
                events.append((event, detail))

            codex = SimpleNamespace(
                lighthouse_snapshot=Mock(return_value=_snapshot()),
                heartbeat=AsyncMock(),
                _record_runtime_event=record_event,
            )
            view = LighthouseView(codex, output=output, refresh_interval=0.1)
            self.assertTrue(await view.start())
            logger.info("routine console line")
            try:
                raise RuntimeError("diagnostic traceback")
            except RuntimeError:
                logger.exception("preserved failure")
            await asyncio.sleep(0.05)
            await view.close()

            self.assertNotIn("routine console line", output.value)
            self.assertNotIn("ERROR", output.value)
            self.assertTrue(any(record.exc_info for record in view._diagnostics))
            self.assertIn(("log_error", "preserved failure"), events)
            self.assertEqual(list(handler.filters), [])
            self.assertEqual(view._diagnostic_handlers, [])
            logger.info("logging restored")
            self.assertIn("logging restored", output.value)
        finally:
            logger.removeHandler(handler)

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
