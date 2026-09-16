# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable
import io

from rich.console import Console
from rich.text import Text

from tests.test_support import *

from theia.server.lighthouse import (
    LighthouseView,
    render_lighthouse,
    render_lighthouse_diagnostics,
    render_lighthouse_rich,
)
from theia.server.lighthouse_render import render_lighthouse_diagnostics_rich
from theia.colors import THEIA_COLORS, color_value


def _snapshot(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "version": "2.0.0",
        "action": "Processing request",
        "mode": "text",
        "model": "gpt-5.6-luna",
        "reasoning": "high",
        "reasoning_mode": "adaptive",
        "character": {
            "name": "Cel",
            "source": "user override",
            "path": "~/.theia/personalities/cel.md",
        },
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
    def test_lighthouse_uses_the_shared_default_palette(self) -> None:
        self.assertEqual(
            THEIA_COLORS,
            {
                "INFO": "#A5BAFF",
                "CONNECTED": "#A5BAFF",
                "HEALTHY": "#A5BAFF",
                "ACTIVE": "#92D886",
                "WARNING": "#DFD477",
                "DEGRADED": "#DFD477",
                "ERROR": "#DD6167",
                "FATAL": "#DD6167",
                "FATAL_DARK": "#9C5256",
                "DISABLED": "#8B78BC",
            },
        )
        self.assertEqual(color_value("INFO"), 0xA5BAFF)
        self.assertEqual(color_value("DISABLED"), 0x8B78BC)

        formatter = core_module._CodexColorFormatter(use_colors=True)
        info = logging.LogRecord(
            "theia.codex", logging.INFO, "test.py", 1, "info", (), None
        )
        self.assertIn("\x1b[38;2;165;186;255m", formatter.format(info))
        fatal = logging.LogRecord(
            "theia.codex", logging.CRITICAL, "test.py", 1, "fatal", (), None
        )
        self.assertIn("\x1b[1;7m\x1b[38;2;221;97;103m", formatter.format(fatal))

        plain = render_lighthouse(_snapshot())
        styled = render_lighthouse_rich(_snapshot())
        self.assertEqual(styled.plain, plain)
        self.assertTrue(any(str(span.style) == "#A5BAFF" for span in styled.spans))

        fatal = render_lighthouse_rich(
            _snapshot(events=({"timestamp": 0, "event": "fatal"},))
        )
        self.assertTrue(
            any(
                "bold reverse" in str(span.style) and "#9C5256" in str(span.style)
                for span in fatal.spans
            )
        )

    def test_lighthouse_colors_runtime_values_by_semantics(self) -> None:
        def style_for(snapshot: dict[str, Any], value: str) -> str:
            rendered = render_lighthouse_rich(snapshot, width=120, height=40)
            offset = rendered.plain.index(value)
            return str(rendered.get_style_at_offset(Console(), offset)).casefold()

        self.assertIn("#92d886", style_for(_snapshot(), "online"))
        self.assertIn(
            "#dfd477", style_for(_snapshot(presence={"status": "idle"}), "idle")
        )
        self.assertIn(
            "#dd6167",
            style_for(_snapshot(presence={"status": "offline"}), "offline"),
        )
        self.assertIn("#a5baff", style_for(_snapshot(), "adaptive"))
        self.assertIn("#a5baff", style_for(_snapshot(), "connected"))
        self.assertIn("#a5baff", style_for(_snapshot(), "watching"))

        degraded = _snapshot(
            runtime={
                **_snapshot()["runtime"],
                "cleanup": {"status": "degraded", "reason": "timeout"},
            }
        )
        self.assertIn("#dd6167", style_for(degraded, "degraded"))
        self.assertIn("#dd6167", style_for(degraded, "timeout"))

        neutral = _snapshot(
            runtime={
                **_snapshot()["runtime"],
                "update": "not configured",
                "heartbeat": {"state": "unknown"},
            },
            attention={"active": "none"},
        )
        self.assertIn("#8b78bc", style_for(neutral, "not configured"))
        self.assertIn("#8b78bc", style_for(neutral, "none"))

    def test_diagnostic_fatal_severity_uses_the_darker_red(self) -> None:
        record = logging.LogRecord(
            "theia.process",
            logging.CRITICAL,
            "process.py",
            1,
            "process stopped",
            (),
            None,
        )
        rendered = render_lighthouse_diagnostics_rich(_snapshot(), (record,))
        offset = rendered.plain.index("CRITICAL")
        style = str(rendered.get_style_at_offset(Console(), offset)).casefold()
        self.assertIn("#9c5256", style)
        self.assertIn("bold reverse", style)

    async def test_lighthouse_uses_the_latest_adaptive_assessment(self) -> None:
        server = main.CodexAppServer()
        server.available_models = AsyncMock(
            return_value=(
                {
                    "id": "test-model",
                    "isDefault": True,
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": effort}
                        for effort in ("low", "medium", "high")
                    ],
                },
            )
        )
        server._assess_request = AsyncMock(
            return_value={"complexity": "complex", "requires_tool": True}
        )

        self.assertEqual(
            await server._select_reasoning_effort("inspect this", ()), "high"
        )
        snapshot = server.lighthouse_snapshot()

        self.assertEqual(snapshot["reasoning"], "high")
        self.assertEqual(snapshot["reasoning_mode"], "adaptive")
        rendered = render_lighthouse(snapshot)
        self.assertIn("Model        GPT-5.6 Luna · adaptive", rendered)
        self.assertIn("Reasoning    high", rendered)

    def test_lighthouse_does_not_invent_reasoning_without_an_assessment(self) -> None:
        server = main.CodexAppServer()
        snapshot = server.lighthouse_snapshot()

        self.assertIsNone(snapshot["reasoning"])
        self.assertIn("Reasoning    unknown", render_lighthouse(snapshot))

    def test_character_line_shows_profile_path_and_override_scope(self) -> None:
        rendered = render_lighthouse(_snapshot())

        self.assertIn(
            "Character    Cel · user override: ~/.theia/personalities/cel.md",
            rendered,
        )

    def test_lighthouse_character_reports_global_and_override_sources(self) -> None:
        server = main.CodexAppServer()
        profile_path = Path.home() / ".theia" / "personalities" / "cel.md"
        with (
            patch.object(
                server._personalities,
                "summary",
                return_value=SimpleNamespace(character_name="Cel", identifier="cel"),
            ),
            patch.object(
                server._personalities,
                "resolve",
                return_value=SimpleNamespace(path=profile_path),
            ),
        ):
            server._personality_scopes = {
                "everyone": {"scope": "everyone", "name": "cel"}
            }
            global_character = server._lighthouse_character(None)
            with patch.object(
                server,
                "personality_selection",
                return_value={"scope": "server", "name": "cel"},
            ):
                server_character = server._lighthouse_character(
                    SimpleNamespace(key="guild:42:channel:1:user:7")
                )
            with patch.object(
                server,
                "personality_selection",
                return_value={"scope": "me", "name": "cel"},
            ):
                user_character = server._lighthouse_character(
                    SimpleNamespace(key="guild:42:channel:1:user:7")
                )

        self.assertEqual(global_character["source"], "global")
        self.assertEqual(server_character["source"], "server override")
        self.assertEqual(user_character["source"], "user override")
        for character in (global_character, server_character, user_character):
            self.assertEqual(character["path"], "~/.theia/personalities/cel.md")

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
        self.assertIn("Workspace      2 entries", rendered)
        self.assertIn("Recent         Inspect the runtime", rendered)
        self.assertNotIn("Should the view stay enabled?", rendered)
        self.assertNotIn("Goal: Inspect the runtime", rendered)

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

    def test_recent_events_use_stable_titles_and_hide_technical_details(self) -> None:
        cleanup_timestamp = datetime(
            2026, 9, 15, 6, 51, tzinfo=timezone.utc
        ).timestamp()
        worker_timestamp = datetime(2026, 9, 15, 3, 11, tzinfo=timezone.utc).timestamp()
        completed_timestamp = datetime(
            2026, 9, 14, 19, 54, tzinfo=timezone.utc
        ).timestamp()
        snapshot = _snapshot(
            events=(
                {
                    "timestamp": cleanup_timestamp,
                    "event": "log_warning",
                    "detail": (
                        "Expired Codex session cleanup failed "
                        "(method=thread/delete, code=-32600, reason=invalidThreadId)"
                    ),
                },
                {
                    "timestamp": worker_timestamp,
                    "event": "worker_failed",
                    "detail": "Selected model is at capacity; serverOverloaded",
                },
                {
                    "timestamp": completed_timestamp,
                    "event": "turn_completed",
                    "detail": "internal protocol detail",
                },
            )
        )

        rendered = render_lighthouse(snapshot)

        self.assertIn(
            "[2026-09-15 06:51] WARNING  Cleanup failed",
            rendered,
        )
        self.assertNotIn("Worker degraded", rendered)
        self.assertIn(
            "[2026-09-14 19:54] INFO     Codex turn completed",
            rendered,
        )
        self.assertNotIn("thread/delete", rendered)
        self.assertNotIn("serverOverloaded", rendered)
        self.assertNotIn("internal protocol detail", rendered)

    def test_discord_gateway_handshake_errors_are_specific_warnings(self) -> None:
        rendered = render_lighthouse(
            _snapshot(
                events=(
                    {
                        "timestamp": 0,
                        "event": "log_error",
                        "detail": (
                            "Attempting a reconnect in 0.90s - "
                            "WSServerHandshakeError: 503, "
                            "message='Invalid response status'"
                        ),
                    },
                )
            ),
            width=120,
            height=20,
        )

        self.assertIn("Discord gateway reconnecting", rendered)
        self.assertIn("Active warning  Discord gateway reconnecting", rendered)
        self.assertNotIn("Active error  Error", rendered)

    def test_generic_internal_app_server_errors_are_diagnostic_only(self) -> None:
        snapshot = _snapshot(
            events=(
                {
                    "timestamp": 0,
                    "event": "worker_failed",
                    "detail": "serverOverloaded",
                },
            )
        )
        record = logging.LogRecord(
            "theia.codex",
            logging.INFO,
            "core.py",
            652,
            "Codex internal worker failed (status=failed, reason=serverOverloaded)",
            (),
            None,
        )
        record.module = "core"
        record.funcName = "_wait_for_turn"

        self.assertNotIn("Worker degraded", render_lighthouse(snapshot))
        diagnostics = render_lighthouse_diagnostics(snapshot, (record,))
        self.assertIn("serverOverloaded", diagnostics)
        self.assertIn("theia.codex/core._wait_for_turn", diagnostics)

    def test_diagnostic_detail_view_retains_technical_event_details(self) -> None:
        snapshot = _snapshot(
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
        record = logging.LogRecord(
            "theia.cleanup",
            logging.WARNING,
            "cleanup.py",
            10,
            "Expired Codex session cleanup failed (method=thread/delete, code=-32600)",
            (),
            None,
        )

        rendered = render_lighthouse_diagnostics(snapshot, (record,))

        self.assertIn("Theia 2.0.0 · Lighthouse Diagnostics", rendered)
        self.assertNotIn("Status       Processing request", rendered)
        self.assertNotIn("Runtime\n", rendered)
        self.assertIn("method=thread/delete", rendered)
        self.assertIn("code=-32600", rendered)
        self.assertIn("theia.cleanup", rendered)
        self.assertNotIn("No diagnostic details", rendered)

    def test_diagnostics_use_main_view_severity_and_secondary_hierarchy(self) -> None:
        console = Console()
        info = logging.LogRecord(
            "theia.runtime",
            logging.INFO,
            "runtime.py",
            1,
            "ERROR is only part of this info message; healthy",
            (),
            None,
        )
        warning = logging.LogRecord(
            "theia.worker",
            logging.WARNING,
            "worker.py",
            2,
            "Worker degraded (method=worker, duration_ms=12)",
            (),
            None,
        )
        error = logging.LogRecord(
            "theia.transport",
            logging.ERROR,
            "transport.py",
            3,
            "connection lost",
            (),
            None,
        )
        fatal = logging.LogRecord(
            "theia.process",
            logging.CRITICAL,
            "process.py",
            4,
            "process stopped",
            (),
            None,
        )

        rendered = render_lighthouse_diagnostics_rich(
            _snapshot(), (info, warning, error, fatal), width=100
        )

        info_offset = rendered.plain.index("INFO")
        warning_offset = rendered.plain.index("WARNING")
        error_offset = rendered.plain.index("ERROR")
        fatal_offset = rendered.plain.index("CRITICAL")
        self.assertIn(
            "#a5baff",
            str(rendered.get_style_at_offset(console, info_offset)).casefold(),
        )
        self.assertIn(
            "#dfd477",
            str(rendered.get_style_at_offset(console, warning_offset)).casefold(),
        )
        self.assertIn(
            "#dd6167",
            str(rendered.get_style_at_offset(console, error_offset)).casefold(),
        )
        self.assertIn(
            "bold reverse", str(rendered.get_style_at_offset(console, fatal_offset))
        )
        self.assertIn(
            "none",
            str(
                rendered.get_style_at_offset(
                    console, rendered.plain.index("theia.runtime")
                )
            ).casefold(),
        )
        self.assertIn(
            "#8b78bc",
            str(
                rendered.get_style_at_offset(
                    console, rendered.plain.index("method=worker")
                )
            ).casefold(),
        )

    def test_diagnostic_event_metadata_is_displayed_in_the_log_row(
        self,
    ) -> None:
        record = logging.LogRecord(
            "theia.worker", logging.INFO, "worker.py", 1, "accepted", (), None
        )
        record.event_name = "Worker completed"
        rendered = render_lighthouse_diagnostics_rich(_snapshot(), (record,))
        self.assertIn("Worker completed: accepted", rendered.plain)
        console = Console()
        offset = rendered.plain.index("Worker completed")
        self.assertEqual(
            "none", str(rendered.get_style_at_offset(console, offset)).casefold()
        )
        self.assertIn(
            "#a5baff",
            str(
                rendered.get_style_at_offset(console, rendered.plain.index("accepted"))
            ).casefold(),
        )

    def test_diagnostic_unknown_level_falls_back_to_info_style(self) -> None:
        record = logging.LogRecord(
            "theia.runtime", 0, "runtime.py", 1, "custom level text", (), None
        )
        record.levelname = "NOTICE"
        rendered = render_lighthouse_diagnostics_rich(_snapshot(), (record,))
        offset = rendered.plain.index("NOTICE")
        self.assertIn(
            "#a5baff", str(rendered.get_style_at_offset(Console(), offset)).casefold()
        )

    def test_responsive_layout_is_bounded_and_keeps_the_footer_visible(self) -> None:
        snapshot = _snapshot(
            action="A very long current action that must be truncated in a narrow terminal",
            events=tuple(
                {"timestamp": index, "event": "turn_completed"} for index in range(20)
            ),
        )
        for width, height in ((80, 20), (48, 12), (36, 10)):
            rendered = render_lighthouse(
                snapshot, width=width, height=height, show_keyboard_hint=True
            )
            lines = rendered.splitlines()
            self.assertEqual(len(lines), height)
            self.assertEqual(lines[-1].strip(), "F1 diagnostics")
            self.assertTrue(all(len(line) <= max(1, width - 2) for line in lines))
            self.assertIn("Status", rendered)
            self.assertIn("Model", rendered)
            self.assertIn("Character", rendered)
            self.assertIn("Session", rendered)
            self.assertIn("Codex", rendered)
            self.assertIn("Heartbeat", rendered)
            self.assertIn("Cleanup", rendered)
        self.assertIn("…", render_lighthouse(snapshot, width=48, height=12))

    def test_short_full_width_view_degrades_progressively_before_emergency(
        self,
    ) -> None:
        rendered = render_lighthouse(
            _snapshot(), width=120, height=20, show_keyboard_hint=True
        )
        lines = rendered.splitlines()

        self.assertNotIn("Latest      ", rendered)
        self.assertNotIn("", lines[:-1])
        for label in (
            "Status",
            "Model",
            "Reasoning",
            "Character",
            "Session",
            "Presence",
            "Voice",
            "Attention",
            "Mood",
            "Workspace",
            "Runtime",
            "Heartbeat",
            "Recent events",
        ):
            self.assertIn(label, rendered)

    def test_tight_compact_layout_keeps_secondary_state_and_footer_anchored(
        self,
    ) -> None:
        rendered = render_lighthouse(
            _snapshot(), width=48, height=12, show_keyboard_hint=True
        )
        lines = rendered.splitlines()

        self.assertEqual(len(lines), 12)
        self.assertIn("Presence", rendered)
        self.assertIn("Voice", rendered)
        self.assertIn("Attention", rendered)
        self.assertIn("Mood", rendered)
        self.assertIn("Workspace", rendered)
        self.assertIn("Heartbeat", rendered)
        self.assertIn("Recent events 1", rendered)
        self.assertEqual(lines[-1].strip(), "F1 diagnostics")
        self.assertTrue(lines[-2].strip())

    def test_footer_never_renders_below_a_one_row_terminal(self) -> None:
        rendered = render_lighthouse(
            _snapshot(), width=80, height=1, show_keyboard_hint=True
        )

        self.assertEqual(rendered.splitlines()[-1].strip(), "F1 diagnostics")
        self.assertEqual(len(rendered.splitlines()), 1)

    def test_live_view_rerenders_when_terminal_dimensions_change(self) -> None:
        view = LighthouseView(
            SimpleNamespace(lighthouse_snapshot=Mock(return_value=_snapshot()))
        )
        view._text_type = Text
        view._terminal_dimensions = Mock(side_effect=((120, 40), (48, 12)))

        normal = view._render_current_payload()
        compact = view._render_current_payload()

        self.assertNotEqual(normal.plain, compact.plain)
        self.assertEqual(compact.plain.splitlines()[-1].strip(), "F1 diagnostics")
        self.assertIn("Heartbeat", compact.plain)

    def test_diagnostic_separator_uses_live_width_and_escape_returns(self) -> None:
        rendered = render_lighthouse_diagnostics(
            _snapshot(version="2.0.7"), width=44, height=12
        )
        self.assertIn("Theia 2.0.7 · Lighthouse Diagnostics", rendered)
        self.assertTrue(all(len(line) <= 42 for line in rendered.splitlines()))
        self.assertEqual(len(rendered.splitlines()), 12)
        footer = rendered.splitlines()[-1].rstrip()
        self.assertTrue(footer.endswith("ESC go back"))
        self.assertIn("↑/↓ scroll", footer)

    def test_diagnostic_footer_stays_anchored_with_wrapped_details(self) -> None:
        record = logging.LogRecord(
            "theia.codex",
            logging.INFO,
            "notifications.py",
            1,
            "A diagnostic detail that is long enough to wrap across multiple terminal rows.",
            (),
            None,
        )
        rendered = render_lighthouse_diagnostics(
            _snapshot(), (record,), width=44, height=12
        )
        lines = rendered.splitlines()

        self.assertEqual(len(lines), 12)
        footer = lines[-1].rstrip()
        self.assertTrue(footer.endswith("ESC go back"))
        self.assertIn("↑/↓ scroll", footer)
        self.assertIn("detail that is long enough", rendered)

    def test_diagnostic_footer_exposes_navigation_when_history_fits(self) -> None:
        rendered = render_lighthouse_diagnostics(_snapshot(), width=80, height=12)

        footer = rendered.splitlines()[-1].rstrip()
        self.assertIn("↑/↓ scroll", footer)
        self.assertNotIn("PgUp", footer)
        self.assertNotIn("Home", footer)
        self.assertNotIn("End", footer)
        self.assertTrue(footer.endswith("ESC go back"))

    def test_diagnostic_footer_groups_white_hints_with_return_action(self) -> None:
        rendered = render_lighthouse_diagnostics_rich(_snapshot(), width=80, height=12)
        footer_start = rendered.plain.rindex("↑/↓ scroll")
        back_start = rendered.plain.rindex("ESC go back")
        console = Console()

        self.assertEqual(back_start, footer_start + len("↑/↓ scroll "))
        self.assertIn(
            "white",
            str(rendered.get_style_at_offset(console, footer_start)).casefold(),
        )
        self.assertIn(
            "white",
            str(rendered.get_style_at_offset(console, back_start)).casefold(),
        )

    def test_diagnostic_history_scrolls_without_dropping_retained_records(self) -> None:
        records = tuple(
            logging.LogRecord(
                "theia.codex",
                logging.INFO,
                "runtime.py",
                index,
                f"diagnostic record {index:02d}",
                (),
                None,
            )
            for index in range(20)
        )

        latest = render_lighthouse_diagnostics(
            _snapshot(), records, width=80, height=10, scroll_offset=0
        )
        oldest = render_lighthouse_diagnostics(
            _snapshot(), records, width=80, height=10, scroll_offset=10_000
        )

        self.assertIn("Recent diagnostics", latest)
        self.assertIn("diagnostic record 19", latest)
        self.assertNotIn("diagnostic record 00", latest)
        self.assertIn("diagnostic record 00", oldest)
        self.assertNotIn("diagnostic record 19", oldest)
        self.assertIn("↑/↓ scroll", latest)
        self.assertTrue(latest.splitlines()[-1].rstrip().endswith("ESC go back"))

    def test_empty_workspace_and_missing_subsystems_are_truthful(self) -> None:
        stale_goal = "stale demo objective"
        rendered = render_lighthouse(
            _snapshot(
                character={"name": "none", "source": "no character selected"},
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
        self.assertIn(
            "Workspace      0 entries\nRecent         No active workspace entries",
            rendered,
        )
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
        self.assertIn(
            "Workspace      0 entries\nRecent         No active workspace entries",
            rendered,
        )

    def test_workspace_entries_without_current_source_are_not_rendered(self) -> None:
        rendered = render_lighthouse(
            _snapshot(
                workspace={
                    "entries": ({"category": "goal", "text": "untrusted objective"},)
                },
                session_objective=None,
            )
        )
        self.assertIn(
            "Workspace      0 entries\nRecent         No active workspace entries",
            rendered,
        )
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
            self.assertIn("preserved failure", view.diagnostic_view())
            self.assertIn("RuntimeError: diagnostic traceback", view.diagnostic_view())
            self.assertEqual(list(handler.filters), [])
            self.assertEqual(view._diagnostic_handlers, [])
            logger.info("logging restored")
            self.assertIn("logging restored", output.value)
        finally:
            logger.removeHandler(handler)

    async def test_diagnostic_view_collects_records_from_nonterminal_loggers(self):
        output = _TTYBuffer()
        logger = logging.getLogger("theia.lighthouse.nonterminal")
        previous_propagate = logger.propagate
        previous_level = logger.level
        logger.propagate = False
        logger.setLevel(logging.INFO)
        try:
            codex = SimpleNamespace(
                lighthouse_snapshot=Mock(return_value=_snapshot()),
                heartbeat=AsyncMock(),
                _record_runtime_event=Mock(),
            )
            view = LighthouseView(codex, output=output, refresh_interval=0.1)
            self.assertTrue(await view.start())
            logger.warning("nonterminal logger detail")
            await view.close()

            details = view.diagnostic_view()
            self.assertIn("theia.lighthouse.nonterminal", details)
            self.assertIn("nonterminal logger detail", details)
        finally:
            logger.propagate = previous_propagate
            logger.setLevel(previous_level)

    async def test_f1_opens_and_esc_leaves_the_read_only_diagnostic_view(self) -> None:
        output = _TTYBuffer()
        codex = SimpleNamespace(
            lighthouse_snapshot=Mock(return_value=_snapshot()),
        )
        view = LighthouseView(codex, output=output)
        live = Mock()
        view._live = live
        view._text_type = lambda value: value

        view._handle_keyboard_text("\x1b[")
        view._handle_keyboard_text("11~")
        self.assertTrue(view._diagnostic_mode)
        self.assertTrue(live.update.called)
        self.assertIn(
            "Theia 2.0.0 · Lighthouse Diagnostics", live.update.call_args.args[0]
        )

        view._handle_keyboard_text("\x1bOP")
        self.assertTrue(view._diagnostic_mode)
        view._handle_keyboard_text("\x1b")
        await asyncio.sleep(0.1)
        self.assertFalse(view._diagnostic_mode)

        view._diagnostic_mode = True
        view._handle_keyboard_text("\x1b")
        view._handle_keyboard_text("[A")
        self.assertTrue(view._diagnostic_mode)
        self.assertEqual(view._diagnostic_scroll, 1)
        view._handle_keyboard_text("\x1b[5~")
        self.assertEqual(view._diagnostic_scroll, 9)
        view._handle_keyboard_text("\x1b[B")
        self.assertEqual(view._diagnostic_scroll, 8)
        view._handle_keyboard_text("\x1b[6~")
        self.assertEqual(view._diagnostic_scroll, 0)

        view._diagnostic_mode = True
        view._handle_keyboard_text("x")
        await asyncio.sleep(0.1)
        self.assertTrue(view._diagnostic_mode)
        view._handle_keyboard_text("\x1b")
        await asyncio.sleep(0.1)
        self.assertFalse(view._diagnostic_mode)

        view._diagnostic_mode = True
        view._diagnostic_scroll = 0
        prefix = view._handle_windows_character("\xe0", False)
        self.assertTrue(prefix)
        self.assertFalse(view._handle_windows_character("H", prefix))
        self.assertEqual(view._diagnostic_scroll, 1)

        view._diagnostic_mode = True
        prefix = view._handle_windows_character("\x00", False)
        self.assertTrue(prefix)
        self.assertFalse(view._handle_windows_character(";", prefix))
        self.assertTrue(view._diagnostic_mode)

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
        self.assertIn("TERM: xterm-256color", compose)
        self.assertIn("COLORTERM: truecolor", compose)
        self.assertIn('THEIA_CONTAINER: "1"', compose)
        self.assertIn('THEIA_UID: "${THEIA_UID:-1000}"', compose)
        self.assertIn('THEIA_GID: "${THEIA_GID:-1000}"', compose)
        self.assertEqual(compose.count("create_host_path: false"), 2)
        self.assertNotIn("\n    user:", compose)
        self.assertNotIn("userns_mode:", compose)

        dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
        self.assertIn("TERM=xterm-256color", dockerfile)
        self.assertIn("COLORTERM=truecolor", dockerfile)
        self.assertIn("ARG THEIA_UID=1000", dockerfile)
        self.assertIn("ARG THEIA_GID=1000", dockerfile)
        self.assertIn('--uid "${THEIA_UID}"', dockerfile)
        self.assertIn('--gid "${THEIA_GID}"', dockerfile)
