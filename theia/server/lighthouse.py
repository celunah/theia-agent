"""Read-only runtime state and terminal rendering for Theia's Lighthouse View."""

# Rich is optional at import time so non-interactive deployments retain normal
# Python logging when the terminal extra is unavailable.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import shutil
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..core import (
    DEFAULT_CODEX_MODEL,
    THEIA_VERSION,
    _env_bool,
    _safe_intermediate_text,
)
from .lighthouse_render import (
    LIGHTHOUSE_DIAGNOSTIC_LIMIT,
    diagnostic_scroll_limit,
    render_lighthouse,
    render_lighthouse_diagnostics,
    render_lighthouse_rich,
)

LIGHTHOUSE_ENABLED_ENV = "THEIA_LIGHTHOUSE_ENABLED"
LIGHTHOUSE_REFRESH_INTERVAL = 1.0
LIGHTHOUSE_HEARTBEAT_INTERVAL = 5.0
LIGHTHOUSE_DIAGNOSTIC_PAGE = 8
LIGHTHOUSE_ESCAPE_DELAY = 0.08

logger = logging.getLogger("theia.codex")

if TYPE_CHECKING:
    from ..core import _Session, _TurnState


class CodexLighthouseMixin:
    """Expose one bounded snapshot without creating sessions or model work."""

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def _remember_lighthouse_reasoning(self, effort: Any) -> None:
        """Keep the latest successful adaptive assessment for the live view."""
        if isinstance(effort, str) and effort.strip():
            self._lighthouse_last_adaptive_reasoning = effort.strip()[:32]

    def _lighthouse_active_sessions(
        self,
    ) -> list[tuple[_Session, _TurnState | None]]:
        """Return selected sessions and unfinished normal turns.

        The selected value is always the normal ``_Session`` object from the
        harness. Internal worker sessions and restored-but-unselected sessions
        never become dashboard sessions.
        """
        active: dict[str, tuple[int, _Session, _TurnState | None]] = {}
        for order, state in enumerate(getattr(self, "_turns", {}).values()):
            session = getattr(state, "session", None)
            if session is None:
                continue
            key = getattr(session, "key", None)
            if not isinstance(key, str) or key.startswith("__"):
                continue
            done = getattr(state, "done", None)
            done_check = getattr(done, "done", None)
            if callable(done_check) and done_check():
                continue
            try:
                canonical = self._canonical_session_key(key)
            except Exception:  # noqa: BLE001 - dashboard state is best effort
                canonical = key
            active[canonical] = (order, session, state)

        selected_key = getattr(self, "_lighthouse_active_session_key", None)
        if isinstance(selected_key, str) and selected_key:
            selected_lookup = selected_key
            try:
                selected_lookup = self._canonical_session_key(selected_key)
            except Exception:  # noqa: BLE001 - dashboard state is best effort
                selected_lookup = ""
            selected = getattr(self, "_sessions", {}).get(selected_lookup)
            if (
                selected is not None
                and not selected.key.startswith("__")
                and getattr(selected, "lighthouse_status", "active") != "inactive"
                and selected_lookup not in active
            ):
                active[selected_lookup] = (len(active), selected, None)
        return [item[1:] for item in sorted(active.values(), key=lambda item: item[0])]

    @staticmethod
    def _lighthouse_mood(session: _Session | None) -> dict[str, Any]:
        if session is None or session.mood is None:
            return {"traits": "unknown", "label": "neutral", "strength": 0.50}
        mood = session.mood
        strength = mood.strength
        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not math.isfinite(float(strength))
        ):
            strength = 0.50
        return {
            "traits": _safe_intermediate_text(mood.traits, 100) or "unknown",
            "label": _safe_intermediate_text(mood.label, 32) or "neutral",
            "strength": max(0.0, min(1.0, float(strength))),
        }

    @staticmethod
    def _lighthouse_session_label(
        session: _Session | None, turn: _TurnState | None
    ) -> str | None:
        if session is None:
            return None
        channel = getattr(turn, "channel", None) if turn is not None else None
        channel_name = _safe_intermediate_text(getattr(channel, "name", None), 60)
        if not channel_name:
            channel_name = _safe_intermediate_text(
                getattr(session, "lighthouse_channel_name", None), 60
            )
        user = getattr(turn, "user", None) if turn is not None else None
        user_name = _safe_intermediate_text(
            getattr(user, "display_name", None) or getattr(user, "name", None),
            60,
        )
        if not user_name:
            user_name = _safe_intermediate_text(
                getattr(session, "lighthouse_user_name", None), 60
            )
        guild = getattr(channel, "guild", None)
        is_guild = (
            guild is not None
            if channel is not None
            else getattr(session, "lighthouse_is_guild", None)
        )
        if is_guild is True:
            return (
                f"Server conversation · #{channel_name}"
                if channel_name
                else "Server conversation"
            )
        if is_guild is None and channel is None:
            return "Active session"
        if channel_name:
            return f"User conversation · #{channel_name}"
        return f"User conversation · @{user_name}" if user_name else "User conversation"

    def _lighthouse_character(self, session: _Session | None) -> dict[str, Any]:
        try:
            selection = (
                self.personality_selection(session.key)
                if session is not None
                else self._personality_scopes.get("everyone")
            )
            selection = dict(selection) if isinstance(selection, dict) else {}
            name = selection.get("name")
            name = name if isinstance(name, str) and name else None
            if not name:
                return {
                    "name": "none",
                    "source": "no character selected",
                    "status": "available",
                }
            summary = self._personalities.summary(name)
            profile = self._personalities.resolve(name)
            if profile is None:
                return {
                    "name": "unavailable",
                    "source": "character profile unavailable",
                    "status": "degraded",
                    "reason": "character profile unavailable",
                }
            character = _safe_intermediate_text(summary.character_name, 80) or name
            scope = str(selection.get("scope") or "everyone")
            source = {
                "me": "user override",
                "server": "server override",
            }.get(scope, "global")
            return {
                "name": character,
                "identifier": _safe_intermediate_text(summary.identifier, 80),
                "source": source,
                "path": self._lighthouse_personality_path(profile.path),
                "status": "available",
            }
        except Exception:  # noqa: BLE001 - the dashboard is best effort
            return {
                "name": "unavailable",
                "source": "overlay unavailable",
                "status": "degraded",
                "reason": "character overlay unavailable",
            }

    def _lighthouse_personality_path(self, path: Path) -> str:
        """Return a bounded logical path without exposing the host home path."""
        try:
            relative = path.resolve().relative_to(self._codex_home.resolve())
        except (OSError, ValueError):
            return "configured personality path"
        runtime_root = (
            "~/.theia"
            if self._codex_home == (Path.home() / ".theia").resolve()
            else "$THEIA_HOME"
        )
        return f"{runtime_root}/{relative.as_posix()}"

    def _lighthouse_attention(self, session: _Session | None) -> dict[str, Any]:
        if session is None or session.attention is None:
            return {"active": "none", "parked": ()}
        records = session.attention.contexts
        active = records.get(session.attention.active_context_id or "")
        parked = []
        for context_id in session.attention.parked_context_ids[:5]:
            context = records.get(context_id)
            if context is not None:
                title = _safe_intermediate_text(context.title, 100)
                if title:
                    parked.append(title)
        return {
            "active": _safe_intermediate_text(active.title, 100)
            if active is not None
            else "none",
            "active_status": active.status if active is not None else "none",
            "parked": tuple(parked),
        }

    def _lighthouse_workspace(self, session: _Session | None) -> dict[str, Any]:
        if session is None:
            return {
                "entries": (),
                "entry_count": 0,
                "revision": 0,
                "source": "session workspace",
            }
        try:
            workspace = self._workspace_snapshot(session)
        except Exception:  # noqa: BLE001 - missing workspace must not stop the view
            return {
                "entries": (),
                "entry_count": 0,
                "revision": 0,
                "source": "session workspace",
            }
        entries = []
        for item in workspace.get("entries", ()):
            if not isinstance(item, dict):
                continue
            category = _safe_intermediate_text(item.get("category"), 32)
            text = _safe_intermediate_text(item.get("text"), 180)
            if category and text:
                entries.append({"category": category, "text": text})
        try:
            revision = max(0, int(workspace.get("revision", 0)))
        except (TypeError, ValueError):
            revision = 0
        return {
            "entries": tuple(entries[:10]),
            "entry_count": len(entries),
            "revision": revision,
            "source": "session workspace",
        }

    @staticmethod
    def _lighthouse_integer(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def lighthouse_snapshot(self, session_key: str | None = None) -> dict[str, Any]:
        """Return safe live state for the operator dashboard."""
        active_sessions = self._lighthouse_active_sessions()
        requested_key = None
        if isinstance(session_key, str) and session_key:
            try:
                requested_key = self._canonical_session_key(session_key)
            except Exception:  # noqa: BLE001 - dashboard state is best effort
                requested_key = session_key
        current: tuple[_Session, _TurnState | None] | None = None
        if requested_key is not None:
            current = next(
                (
                    item
                    for item in active_sessions
                    if getattr(item[0], "key", None) == requested_key
                ),
                None,
            )
        if current is None and active_sessions:
            selected_key = getattr(self, "_lighthouse_active_session_key", None)
            if isinstance(selected_key, str):
                current = next(
                    (
                        item
                        for item in active_sessions
                        if getattr(item[0], "key", None) == selected_key
                    ),
                    None,
                )
        if current is None and active_sessions:
            current = active_sessions[-1]
        session, turn = current or (None, None)
        session_status = "inactive"
        session_reason = None
        if session is not None:
            session_status = getattr(session, "lighthouse_status", "active")
            if session_status == "inactive":
                session_status = "active"
            session_reason = (
                _safe_intermediate_text(getattr(session, "lighthouse_reason", None), 96)
                or None
            )
        session_snapshot = {
            "active_count": len(active_sessions),
            "current": self._lighthouse_session_label(session, turn),
            "status": session_status,
            "reason": session_reason,
        }
        recovery = bool(getattr(self, "_memory_recovery_active", False))
        if recovery:
            action = "Recovering Codex"
        elif session_status == "degraded":
            action = "Degraded session"
        elif turn is not None:
            action = "Processing request"
        else:
            action = "Idle"
        try:
            transport = self._codex_transport_health()
        except Exception:  # noqa: BLE001
            transport = "unknown"
        try:
            rss_bytes = self._codex_process_rss()
        except Exception:  # noqa: BLE001
            rss_bytes = None
        if not isinstance(rss_bytes, int) or rss_bytes < 0:
            rss_bytes = None
        try:
            memory = self.memory_statistics()
        except Exception:  # noqa: BLE001
            memory = {}
        internal_workers = sum(
            1
            for item in getattr(self, "_turns", {}).values()
            if item.session is not None
            and item.session.key.startswith("__")
            and not item.done.done()
        )
        active_turns = sum(
            1
            for item in getattr(self, "_turns", {}).values()
            if item.session is not None
            and not item.session.key.startswith("__")
            and not item.done.done()
        )
        process = getattr(self, "_process", None)
        connection = {
            "healthy": "connected",
            "recovering": "recovering",
            "unavailable": "disconnected",
        }.get(transport, "unknown")
        watchdog = "disabled"
        if getattr(self, "_memory_watchdog_enabled", False):
            watchdog = "recovering" if recovery else "watching"
            if time.monotonic() < getattr(self, "_memory_restart_backoff_until", 0.0):
                watchdog = "backing off"
        update = getattr(self, "_codex_updater", None)
        try:
            update_status = update.status() if update is not None else {}
        except Exception:  # noqa: BLE001
            update_status = {}
        managed_version = update_status.get("managed_version")
        codex_version = getattr(self, "_codex_version", None) or managed_version
        if not isinstance(codex_version, str) or not codex_version:
            codex_version = "unknown"
        effort = getattr(turn, "effort", None)
        if not isinstance(effort, str) or not effort.strip():
            effort = getattr(self, "_lighthouse_last_adaptive_reasoning", None)
        if not isinstance(effort, str) or not effort.strip():
            effort = None
        reasoning_mode = (
            "adaptive" if getattr(self, "_adaptive_reasoning", False) else "fixed"
        )
        snapshot = {
            "version": THEIA_VERSION,
            "action": action,
            "mode": session.mode if session is not None else "text",
            "model": getattr(turn, "model", None)
            or getattr(self, "_model", None)
            or DEFAULT_CODEX_MODEL,
            "reasoning": effort,
            "reasoning_mode": reasoning_mode,
            "character": self._lighthouse_character(session),
            "presence": {},
            "voice": {},
            "attention": self._lighthouse_attention(session),
            "mood": self._lighthouse_mood(session),
            "session": session_snapshot,
            "workspace": self._lighthouse_workspace(session),
            "runtime": {
                "codex": connection,
                "codex_version": codex_version,
                "process": "running"
                if process is not None and getattr(process, "returncode", None) is None
                else "stopped",
                "rss_bytes": rss_bytes,
                "active_turns": active_turns,
                "workers": internal_workers,
                "approvals": len(getattr(self, "_pending_approvals", {})),
                "memory_entries": self._lighthouse_integer(
                    memory.get("known_entries", 0)
                ),
                "watchdog": watchdog,
                "recovery": recovery,
                "update": "enabled" if update_status.get("enabled") else "disabled",
                "heartbeat": self.heartbeat_snapshot(),
                "cleanup": self.cleanup_snapshot()
                if callable(getattr(self, "cleanup_snapshot", None))
                else {"status": "unknown", "reason": None},
            },
            "events": self.runtime_events(limit=12),
        }
        return snapshot


class _LighthouseTerminalFilter(logging.Filter):
    """Keep all logging records out of a terminal owned by Lighthouse."""

    def filter(self, record: logging.LogRecord) -> bool:
        del record
        return False


class _LighthouseDiagnosticHandler(logging.Handler):
    """Retain diagnostic records while terminal handlers are temporarily muted."""

    def __init__(self, codex: Any, records: deque[logging.LogRecord]) -> None:
        super().__init__(level=logging.NOTSET)
        self.codex = codex
        self.records = records

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "_lighthouse_captured", False):
            return
        record._lighthouse_captured = True  # type: ignore[attr-defined]
        self.records.append(record)
        if record.levelno < logging.WARNING:
            return
        try:
            detail = _safe_intermediate_text(record.getMessage(), 120)
        except Exception:  # noqa: BLE001 - diagnostics must never affect logging
            detail = ""
        if not detail:
            return
        recorder = getattr(self.codex, "_record_runtime_event", None)
        if not callable(recorder):
            return
        event = "log_error" if record.levelno >= logging.ERROR else "log_warning"
        with contextlib.suppress(Exception):
            recorder(event, detail)


def _terminal_handler(handler: logging.Handler) -> bool:
    """Identify handlers that would write into the interactive terminal."""
    if isinstance(handler, logging.FileHandler):
        return False
    stream = getattr(handler, "stream", None) or sys.stderr
    if stream in {sys.stdout, sys.stderr}:
        return True
    checker = getattr(stream, "isatty", None)
    with contextlib.suppress(Exception):
        return bool(checker()) if callable(checker) else False
    return False


def _logging_targets() -> tuple[logging.Logger, ...]:
    """Return configured loggers and root without creating duplicate targets."""
    targets = [logging.getLogger()]
    for value in logging.Logger.manager.loggerDict.values():
        if isinstance(value, logging.Logger):
            targets.append(value)
    targets.append(logging.getLogger("theia.codex"))
    return tuple(dict.fromkeys(targets))


class LighthouseView:
    """Run an isolated Rich live view over existing Theia runtime state."""

    def __init__(
        self,
        codex: Any,
        *,
        presence: Any | None = None,
        rich_presence: Any | None = None,
        voice: Any | None = None,
        session_key: str | None = None,
        output: Any | None = None,
        input_stream: Any | None = None,
        refresh_interval: float = LIGHTHOUSE_REFRESH_INTERVAL,
        heartbeat_interval: float = LIGHTHOUSE_HEARTBEAT_INTERVAL,
        enabled: bool | None = None,
    ) -> None:
        self.codex = codex
        self.presence = presence
        self.rich_presence = rich_presence
        self.voice = voice
        self.session_key = session_key
        self.output = output if output is not None else sys.stdout
        self.input_stream = input_stream if input_stream is not None else sys.stdin
        self.refresh_interval = max(0.1, refresh_interval)
        self.heartbeat_interval = max(1.0, heartbeat_interval)
        self.enabled = (
            enabled if enabled is not None else _env_bool(LIGHTHOUSE_ENABLED_ENV, True)
        )
        self._task: asyncio.Task[None] | None = None
        self._live: Any | None = None
        self._filters: list[tuple[logging.Handler, logging.Filter]] = []
        self._diagnostic_handlers: list[tuple[logging.Logger, logging.Handler]] = []
        self._diagnostics: deque[logging.LogRecord] = deque(
            maxlen=LIGHTHOUSE_DIAGNOSTIC_LIMIT
        )
        self._diagnostic_mode = False
        self._diagnostic_scroll = 0
        self._text_type: Any | None = None
        self._keyboard_fd: int | None = None
        self._keyboard_old_attrs: Any | None = None
        self._keyboard_buffer = ""
        self._escape_handle: asyncio.TimerHandle | None = None
        self._keyboard_task: asyncio.Task[None] | None = None

    def _interactive(self) -> bool:
        if not self.enabled:
            return False
        checker = getattr(self.output, "isatty", None)
        return bool(checker()) if callable(checker) else False

    def snapshot(self) -> dict[str, Any]:
        """Build a dashboard snapshot without creating sessions or requests."""
        snapshot = self.codex.lighthouse_snapshot(self.session_key)
        snapshot = dict(snapshot) if isinstance(snapshot, dict) else {}
        if self.presence is not None:
            with contextlib.suppress(Exception):
                snapshot["presence"] = self.presence.snapshot()
        if self.rich_presence is not None:
            with contextlib.suppress(Exception):
                rich = self.rich_presence.snapshot()
                current = snapshot.setdefault("presence", {})
                if isinstance(current, dict):
                    current["line"] = rich.get("line", "none")
                    current["rich_state"] = rich.get("state", "idle")
        if self.voice is not None:
            with contextlib.suppress(Exception):
                snapshot["voice"] = self.voice.snapshot()
        return snapshot

    def diagnostic_view(self) -> str:
        """Return the explicit technical detail view without creating work."""
        return render_lighthouse_diagnostics(self.snapshot(), self._diagnostics)

    def _terminal_dimensions(self) -> tuple[int, int]:
        """Read the current terminal size without depending on a real file object."""
        try:
            fileno = self.output.fileno()
            size = os.get_terminal_size(fileno)
            return max(1, size.columns), max(1, size.lines)
        except (AttributeError, OSError, TypeError, ValueError):
            try:
                size = shutil.get_terminal_size(fallback=(80, 24))
                return max(1, size.columns), max(1, size.lines)
            except (AttributeError, OSError, TypeError, ValueError):
                return 80, 24

    def _render_current_view(self) -> str:
        width, height = self._terminal_dimensions()
        if self._diagnostic_mode:
            return render_lighthouse_diagnostics(
                self.snapshot(),
                self._diagnostics,
                width=width,
                height=height,
                scroll_offset=self._diagnostic_scroll,
            )
        return render_lighthouse(
            self.snapshot(), width=width, height=height, show_keyboard_hint=True
        )

    def _render_current_payload(self) -> Any:
        """Use styled Rich output only after Rich has been selected by ``start``."""
        text_type = self._text_type
        if getattr(text_type, "__module__", "") == "rich.text":
            width, height = self._terminal_dimensions()
            return render_lighthouse_rich(
                self.snapshot(),
                records=self._diagnostics,
                diagnostic_mode=self._diagnostic_mode,
                width=width,
                height=height,
                show_keyboard_hint=True,
                diagnostic_scroll_offset=self._diagnostic_scroll,
            )
        return (
            text_type(self._render_current_view())
            if text_type
            else self._render_current_view()
        )

    def _toggle_diagnostics(self) -> None:
        """Toggle the read-only diagnostic screen from the terminal key reader."""
        self._cancel_pending_escape()
        self._diagnostic_mode = not self._diagnostic_mode
        if self._diagnostic_mode:
            self._diagnostic_scroll = 0
        live = self._live
        text_type = self._text_type
        if live is None or text_type is None:
            return
        with contextlib.suppress(Exception):
            live.update(self._render_current_payload(), refresh=True)

    def _cancel_pending_escape(self) -> None:
        """Cancel a delayed ESC decision while an escape sequence is arriving."""
        handle, self._escape_handle = self._escape_handle, None
        if handle is not None:
            handle.cancel()

    def _complete_pending_escape(self) -> None:
        """Return to the main view after a standalone ESC keypress."""
        self._escape_handle = None
        if self._keyboard_buffer == "\x1b":
            self._keyboard_buffer = ""
            self._toggle_diagnostics()

    def _schedule_escape(self) -> None:
        """Wait briefly so an arrow-key escape sequence is not mistaken for ESC."""
        self._cancel_pending_escape()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._toggle_diagnostics()
            return
        self._escape_handle = loop.call_later(
            LIGHTHOUSE_ESCAPE_DELAY, self._complete_pending_escape
        )

    def _adjust_diagnostic_scroll(self, action: str) -> None:
        """Move through retained diagnostic rows without leaving the live view."""
        if not self._diagnostic_mode:
            return
        if action == "home":
            width, height = self._terminal_dimensions()
            self._diagnostic_scroll = diagnostic_scroll_limit(
                self.snapshot(),
                self._diagnostics,
                width=width,
                height=height,
            )
        elif action == "end":
            self._diagnostic_scroll = 0
        elif action in {"up", "page_up"}:
            self._diagnostic_scroll += (
                LIGHTHOUSE_DIAGNOSTIC_PAGE if action == "page_up" else 1
            )
        elif action in {"down", "page_down"}:
            self._diagnostic_scroll = max(
                0,
                self._diagnostic_scroll
                - (LIGHTHOUSE_DIAGNOSTIC_PAGE if action == "page_down" else 1),
            )
        else:
            return
        live = self._live
        text_type = self._text_type
        if live is None or text_type is None:
            return
        with contextlib.suppress(Exception):
            live.update(self._render_current_payload(), refresh=True)

    def _handle_keyboard_text(self, text: str) -> None:
        """Recognize view switching and diagnostic navigation key sequences."""
        if self._escape_handle is not None and text != "\x1b":
            self._cancel_pending_escape()
        if "\x1b" in text:
            self._keyboard_buffer = text[text.rfind("\x1b") :][-16:]
        elif self._keyboard_buffer.startswith("\x1b"):
            self._keyboard_buffer = (self._keyboard_buffer + text)[-16:]
        else:
            self._keyboard_buffer = ""
        function_sequences = ("\x1b[11~", "\x1bOP")
        if not self._diagnostic_mode:
            if any(
                sequence in self._keyboard_buffer for sequence in function_sequences
            ):
                self._cancel_pending_escape()
                self._keyboard_buffer = ""
                self._toggle_diagnostics()
            elif not any(
                sequence.startswith(self._keyboard_buffer)
                for sequence in function_sequences
            ):
                self._keyboard_buffer = ""
            return
        if any(sequence in self._keyboard_buffer for sequence in function_sequences):
            self._cancel_pending_escape()
            self._keyboard_buffer = ""
            return
        sequences = {
            "\x1b[A": "up",
            "\x1b[B": "down",
            "\x1b[5~": "page_up",
            "\x1b[6~": "page_down",
            "\x1b[H": "home",
            "\x1b[F": "end",
            "\x1b[1~": "home",
            "\x1b[4~": "end",
        }
        for sequence, action in sequences.items():
            if sequence in self._keyboard_buffer:
                self._cancel_pending_escape()
                self._keyboard_buffer = ""
                self._adjust_diagnostic_scroll(action)
                return
        if text == "\x1b":
            self._schedule_escape()
        elif not any(
            sequence.startswith(self._keyboard_buffer)
            for sequence in (*function_sequences, *sequences)
        ):
            self._keyboard_buffer = ""

    def _read_keyboard(self) -> None:
        """Read available POSIX terminal bytes without blocking the event loop."""
        fd = self._keyboard_fd
        if fd is None:
            return
        try:
            value = os.read(fd, 64)
        except (BlockingIOError, OSError):
            return
        if not value:
            self._stop_keyboard_input()
            return
        with contextlib.suppress(UnicodeDecodeError):
            self._handle_keyboard_text(value.decode(errors="ignore"))

    def _handle_windows_character(self, character: str, function_prefix: bool) -> bool:
        """Handle one Windows console character and return the prefix state."""
        if function_prefix:
            if character == ";" and not self._diagnostic_mode:
                self._toggle_diagnostics()
            elif self._diagnostic_mode:
                actions = {
                    "H": "up",
                    "P": "down",
                    "I": "page_up",
                    "Q": "page_down",
                    "G": "home",
                    "O": "end",
                }
                self._adjust_diagnostic_scroll(actions.get(character, ""))
            return False
        if character in {"\x00", "\xe0"}:
            return True
        self._handle_keyboard_text(character)
        return False

    async def _read_windows_keyboard(self) -> None:
        """Poll Windows console input for F1 without a blocking console read."""
        msvcrt = cast(Any, __import__("msvcrt"))

        function_prefix = False
        while True:
            while msvcrt.kbhit():
                function_prefix = self._handle_windows_character(
                    msvcrt.getwch(), function_prefix
                )
            await asyncio.sleep(0.05)

    def _start_keyboard_input(self) -> None:
        """Install a best-effort F1 reader and preserve the prior terminal mode."""
        if os.name == "nt":
            with contextlib.suppress(Exception):
                self._keyboard_task = asyncio.create_task(self._read_windows_keyboard())
            return
        stream = self.input_stream
        checker = getattr(stream, "isatty", None)
        if not callable(checker) or not checker():
            return
        fd: int | None = None
        old_attrs = None
        changed = False
        try:
            fd = int(stream.fileno())
            import termios
            import tty

            old_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            changed = True
            asyncio.get_running_loop().add_reader(fd, self._read_keyboard)
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            if changed and fd is not None and old_attrs is not None:
                with contextlib.suppress(Exception):
                    import termios

                    termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            return
        self._keyboard_fd = fd
        self._keyboard_old_attrs = old_attrs

    def _stop_keyboard_input(self) -> None:
        """Remove the F1 reader and restore the terminal's previous input mode."""
        fd, old_attrs = self._keyboard_fd, self._keyboard_old_attrs
        self._keyboard_fd = None
        self._keyboard_old_attrs = None
        self._cancel_pending_escape()
        self._keyboard_buffer = ""
        if fd is None:
            return
        with contextlib.suppress(Exception):
            asyncio.get_running_loop().remove_reader(fd)
        if old_attrs is not None:
            with contextlib.suppress(Exception):
                import termios

                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

    async def start(self) -> bool:
        """Start the live dashboard; non-TTY output keeps normal logging intact."""
        if not self._interactive():
            return False
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.text import Text
        except ImportError:
            logger.warning("Lighthouse View unavailable because Rich is not installed")
            return False
        try:
            self._text_type = Text
            handlers: list[logging.Handler] = []
            targets = _logging_targets()
            for log in targets:
                for handler in log.handlers:
                    if _terminal_handler(handler) and handler not in handlers:
                        terminal_filter = _LighthouseTerminalFilter()
                        handler.addFilter(terminal_filter)
                        self._filters.append((handler, terminal_filter))
                        handlers.append(handler)
            diagnostic_handler = _LighthouseDiagnosticHandler(
                self.codex, self._diagnostics
            )
            for log in targets:
                log.addHandler(diagnostic_handler)
                self._diagnostic_handlers.append((log, diagnostic_handler))
            console = Console(file=self.output, force_terminal=True)
            # Clear the previous terminal contents before the first dashboard
            # frame is rendered. Live(screen=True) then owns the alternate
            # screen until close() restores it.
            console.clear()
            self._live = Live(
                self._render_current_payload(),
                console=console,
                refresh_per_second=1,
                screen=True,
                transient=False,
                auto_refresh=False,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live.start(refresh=False)
            self._live.refresh()
            self._start_keyboard_input()
            self._task = asyncio.create_task(self._run())
            return True
        except Exception:
            self._restore_logging()
            self._stop_keyboard_input()
            self._text_type = None
            live, self._live = self._live, None
            if live is not None:
                with contextlib.suppress(Exception):
                    live.stop()
            logger.exception("Lighthouse View could not start")
            return False

    async def _run(self) -> None:
        next_heartbeat = 0.0
        try:
            while True:
                now = time.monotonic()
                if now >= next_heartbeat:
                    heartbeat = getattr(self.codex, "heartbeat", None)
                    if callable(heartbeat):
                        with contextlib.suppress(Exception):
                            heartbeat_call = cast(
                                Callable[..., Awaitable[Any]], heartbeat
                            )
                            await heartbeat_call(timeout=1.5)
                    next_heartbeat = now + self.heartbeat_interval
                if self._live is not None:
                    self._live.update(self._render_current_payload(), refresh=True)
                await asyncio.sleep(self.refresh_interval)
        except Exception:
            logger.exception("Lighthouse View stopped unexpectedly")
            self._restore_logging()
            self._stop_keyboard_input()
            self._text_type = None
            live, self._live = self._live, None
            if live is not None:
                with contextlib.suppress(Exception):
                    live.stop()

    def _restore_logging(self) -> None:
        diagnostic_handlers, self._diagnostic_handlers = (
            self._diagnostic_handlers,
            [],
        )
        for log, handler in diagnostic_handlers:
            with contextlib.suppress(Exception):
                log.removeHandler(handler)
        filters, self._filters = self._filters, []
        for handler, routine_filter in filters:
            with contextlib.suppress(Exception):
                handler.removeFilter(routine_filter)

    async def close(self) -> None:
        """Stop refresh and heartbeat tasks, then restore normal logging."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        keyboard_task, self._keyboard_task = self._keyboard_task, None
        if keyboard_task is not None:
            keyboard_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await keyboard_task
        self._stop_keyboard_input()
        live, self._live = self._live, None
        if live is not None:
            with contextlib.suppress(Exception):
                live.stop()
        self._restore_logging()
        self._text_type = None
