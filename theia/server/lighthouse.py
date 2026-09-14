"""Read-only runtime state and terminal rendering for Theia's Lighthouse View."""

# Rich is optional at import time so non-interactive deployments retain normal
# Python logging when the terminal extra is unavailable.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

from ..core import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    THEIA_VERSION,
    _env_bool,
    _safe_intermediate_text,
)

LIGHTHOUSE_ENABLED_ENV = "THEIA_LIGHTHOUSE_ENABLED"
LIGHTHOUSE_REFRESH_INTERVAL = 1.0
LIGHTHOUSE_HEARTBEAT_INTERVAL = 5.0
LIGHTHOUSE_EVENT_LIMIT = 12

_EVENT_LABELS = {
    "listening_state_entered": "Listening state entered",
    "speaking_state_entered": "Speaking state entered",
    "attention_changed": "Attention changed",
    "workspace_updated": "Workspace updated",
    "turn_started": "Codex turn started",
    "turn_completed": "Codex turn completed",
    "turn_timed_out": "Codex turn timed out",
    "character_loaded": "Character overlay loaded",
    "model_changed": "Model changed",
    "worker_started": "Worker started",
    "worker_completed": "Worker completed",
    "approval_requested": "Approval requested",
    "approval_resolved": "Approval resolved",
    "codex_starting": "Codex starting",
    "codex_connected": "Codex connected",
    "codex_recovered": "Codex recovered",
    "codex_restarted": "Codex restarted",
    "presence_updated": "Presence updated",
}

_MODEL_LABELS = {
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-6-astra": "GPT-6 Astra",
}

logger = logging.getLogger("theia.codex")

if TYPE_CHECKING:
    from ..core import _Session, _TurnState


class CodexLighthouseMixin:
    """Expose one bounded snapshot without creating sessions or model work."""

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def _lighthouse_session(self, session_key: str | None) -> _Session | None:
        sessions = getattr(self, "_sessions", {})
        if isinstance(session_key, str) and session_key:
            canonical = self._canonical_session_key(session_key)
            session = sessions.get(canonical)
            if session is not None and not canonical.startswith("__"):
                return session
        active = [
            state.session
            for state in getattr(self, "_turns", {}).values()
            if getattr(state, "session", None) is not None
            and not state.session.key.startswith("__")
        ]
        if active:
            return active[-1]
        normal = [
            session for session in sessions.values() if not session.key.startswith("__")
        ]
        return max(
            normal,
            key=lambda item: item.last_activity_at or 0.0,
            default=None,
        )

    def _lighthouse_turn(self, session: _Session | None) -> _TurnState | None:
        if session is None:
            return None
        for state in getattr(self, "_turns", {}).values():
            if state.session is session and not state.done.done():
                return state
        return None

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
        _session: _Session | None, turn: _TurnState | None
    ) -> str:
        channel = getattr(turn, "channel", None)
        channel_name = _safe_intermediate_text(getattr(channel, "name", None), 60)
        user = getattr(turn, "user", None)
        user_name = _safe_intermediate_text(
            getattr(user, "display_name", None) or getattr(user, "name", None),
            60,
        )
        if getattr(channel, "guild", None) is not None:
            return (
                f"Server conversation · #{channel_name}"
                if channel_name
                else "Server conversation"
            )
        return f"User conversation · {user_name}" if user_name else "User conversation"

    def _lighthouse_character(self, session: _Session | None) -> dict[str, str]:
        if session is None:
            return {"name": "none", "source": "no overlay"}
        try:
            selection = self.personality_selection(session.key)
            name = self.active_personality(session.key)
            if not name:
                return {"name": "none", "source": "no overlay"}
            summary = self._personalities.summary(name)
            character = _safe_intermediate_text(summary.character_name, 80) or name
            scope = selection.get("scope") if isinstance(selection, dict) else None
            source = {
                "me": "user overlay",
                "server": "server overlay",
                "everyone": "global overlay",
            }.get(str(scope), "personality overlay")
            return {
                "name": character,
                "identifier": _safe_intermediate_text(summary.identifier, 80),
                "source": source,
            }
        except Exception:  # noqa: BLE001 - the dashboard is best effort
            return {"name": "unavailable", "source": "overlay unavailable"}

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
            return {"entries": (), "revision": 0}
        try:
            workspace = self._workspace_snapshot(session)
        except Exception:  # noqa: BLE001 - missing workspace must not stop the view
            return {"entries": (), "revision": 0}
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
        return {"entries": tuple(entries[:10]), "revision": revision}

    @staticmethod
    def _lighthouse_integer(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def lighthouse_snapshot(self, session_key: str | None = None) -> dict[str, Any]:
        """Return safe live state for the operator dashboard."""
        session = self._lighthouse_session(session_key)
        turn = self._lighthouse_turn(session)
        recovery = bool(getattr(self, "_memory_recovery_active", False))
        if recovery:
            action = "Recovering Codex"
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
        effort = getattr(turn, "effort", None) or (
            DEFAULT_REASONING_EFFORT
            if getattr(self, "_adaptive_reasoning", False)
            else "standard"
        )
        snapshot = {
            "version": THEIA_VERSION,
            "action": action,
            "mode": session.mode if session is not None else "text",
            "model": getattr(turn, "model", None)
            or getattr(self, "_model", None)
            or DEFAULT_CODEX_MODEL,
            "reasoning": effort,
            "character": self._lighthouse_character(session),
            "presence": {},
            "voice": {},
            "attention": self._lighthouse_attention(session),
            "mood": self._lighthouse_mood(session),
            "session": self._lighthouse_session_label(session, turn),
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
            },
            "events": self.runtime_events(limit=12),
        }
        return snapshot


def _dashboard_text(value: Any, limit: int = 160) -> str:
    """Keep terminal values bounded and free of control sequences or paths."""
    text = _safe_intermediate_text(value, limit)
    text = re.sub(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text[:limit].strip()


def _dashboard_session_label(value: Any) -> str:
    """Reject opaque session labels so an accidental raw key cannot be shown."""
    text = _dashboard_text(value, 100)
    if re.search(
        r"\b(?:guild|server|channel|user|session)(?:[_ -]?id)?\s*[:=]\s*\d+",
        text,
        re.IGNORECASE,
    ):
        return "unknown"
    return text or "unknown"


def _model_label(value: Any) -> str:
    model = _dashboard_text(value, 80).casefold()
    return _MODEL_LABELS.get(model, model or "unknown")


def _format_rss(value: Any) -> str:
    if not isinstance(value, int) or value < 0:
        return "unknown"
    if value >= 1024**3:
        return f"{value / 1024**3:.1f} GB"
    if value >= 1024**2:
        return f"{value / 1024**2:.0f} MB"
    return f"{value / 1024:.0f} KB"


def _format_event_time(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "--:--:--"
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).strftime(
            "%H:%M:%S"
        )
    except (OverflowError, OSError, ValueError):
        return "--:--:--"


def _render_presence(snapshot: dict[str, Any]) -> str:
    status = _dashboard_text(snapshot.get("status"), 24) or "unknown"
    line = _dashboard_text(snapshot.get("line"), 128)
    if line and line.casefold() != "none":
        return f"{status} · {line}"
    return status


def _voice_label(snapshot: dict[str, Any]) -> str:
    providers = snapshot.get("providers")
    provider = (
        providers[0] if isinstance(providers, (list, tuple)) and providers else None
    )
    names = {
        "qwen": "Qwen Audio Agent",
        "codex-realtime": "Codex Realtime",
        "custom": "Custom backend",
    }
    if not provider:
        return "disabled"
    label = names.get(str(provider), "unavailable")
    state = _dashboard_text(snapshot.get("state"), 24) or "idle"
    if state == "disabled":
        return "disabled"
    return f"{label} · {state}"


def render_lighthouse(snapshot: dict[str, Any]) -> str:
    """Render a sanitized Lighthouse View snapshot as compact terminal text."""
    character = snapshot.get("character")
    character = character if isinstance(character, dict) else {}
    name = _dashboard_text(character.get("name"), 80) or "none"
    source = _dashboard_text(character.get("source"), 48) or "unknown"
    mood = snapshot.get("mood")
    mood = mood if isinstance(mood, dict) else {}
    mood_label = _dashboard_text(mood.get("label"), 32).title() or "Unknown"
    mood_traits = _dashboard_text(mood.get("traits"), 80)
    try:
        mood_strength = max(0.0, min(1.0, float(mood.get("strength", 0.5))))
    except (TypeError, ValueError):
        mood_strength = 0.5
    mood_line = f"{mood_label} ({mood_strength:.0%})"
    if mood_traits and mood_traits.casefold() != "unknown":
        mood_line += f" · {mood_traits}"
    attention = snapshot.get("attention")
    attention = attention if isinstance(attention, dict) else {}
    attention_line = _dashboard_text(attention.get("active"), 100) or "none"
    workspace = snapshot.get("workspace")
    workspace = workspace if isinstance(workspace, dict) else {}
    entries = workspace.get("entries")
    entries = entries if isinstance(entries, (list, tuple)) else ()
    runtime = snapshot.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    heartbeat = runtime.get("heartbeat")
    heartbeat = heartbeat if isinstance(heartbeat, dict) else {}
    heartbeat_state = _dashboard_text(heartbeat.get("state"), 24) or "unknown"
    latency = heartbeat.get("latency_ms")
    latency_text = (
        f"{float(latency):.0f} ms"
        if isinstance(latency, (int, float)) and not isinstance(latency, bool)
        else "unknown"
    )
    failures = heartbeat.get("consecutive_failures", 0)
    try:
        failures = max(0, int(failures))
    except (TypeError, ValueError):
        failures = 0
    active_turns = runtime.get("active_turns", 0)
    workers = runtime.get("workers", 0)
    approvals = runtime.get("approvals", 0)
    memory_entries = runtime.get("memory_entries", 0)
    presence = snapshot.get("presence")
    presence = presence if isinstance(presence, dict) else {}
    lines = [
        f"Theia {_dashboard_text(snapshot.get('version'), 24) or 'unknown'} · Lighthouse View",
        "────────────────────────────────────────",
        f"Status       {_dashboard_text(snapshot.get('action'), 80) or 'Unknown'}",
        f"Mode         {_dashboard_text(snapshot.get('mode'), 24).title() or 'Unknown'}",
        (
            f"Model        {_model_label(snapshot.get('model'))} · "
            f"{_dashboard_text(snapshot.get('reasoning'), 40) or 'unknown'} reasoning"
        ),
        f"Character    {name} · loaded from {source}",
        f"Presence     {_render_presence(presence)}",
        f"Voice        {_voice_label(snapshot.get('voice') if isinstance(snapshot.get('voice'), dict) else {})}",
        f"Attention    {attention_line}",
        f"Mood         {mood_line}",
        f"Session      {_dashboard_session_label(snapshot.get('session'))}",
        "────────────────────────────────────────",
        "Workspace",
    ]
    workspace_count_before = len(lines)
    if not entries:
        lines.append("  No active workspace entries")
    else:
        for item in entries[:10]:
            if not isinstance(item, dict):
                continue
            category = _dashboard_text(item.get("category"), 32)
            category = category.replace("_", " ").title()
            text = _dashboard_text(item.get("text"), 180)
            if category and text:
                lines.append(f"  • {category}: {text}")
        if len(lines) == workspace_count_before + 1:
            lines.append("  No active workspace entries")
    parked = attention.get("parked")
    if isinstance(parked, (list, tuple)):
        for title in parked[:3]:
            safe_title = _dashboard_text(title, 100)
            if safe_title:
                lines.append(f"  • Parked topic: {safe_title}")
    lines.extend(
        [
            "────────────────────────────────────────",
            "Runtime",
            (
                f"  Codex        {_dashboard_text(runtime.get('codex_version'), 32) or 'unknown'} · "
                f"{_dashboard_text(runtime.get('codex'), 24) or 'unknown'}"
            ),
            f"  RSS          {_format_rss(runtime.get('rss_bytes'))}",
            f"  Active turns {_dashboard_text(active_turns, 12) or '0'}",
            f"  Workers      {_dashboard_text(workers, 12) or '0'}",
            f"  Approvals    {_dashboard_text(approvals, 12) or '0'}",
            f"  Memory       {_dashboard_text(memory_entries, 12) or '0'} entries",
            f"  Watchdog     {_dashboard_text(runtime.get('watchdog'), 24) or 'unknown'}",
            f"  Recovery     {'active' if runtime.get('recovery') else 'inactive'}",
            f"  Heartbeat    {heartbeat_state} · {latency_text} · {failures} failures",
            f"  Codex update {_dashboard_text(runtime.get('update'), 24) or 'unknown'}",
            "────────────────────────────────────────",
            "Recent events",
        ]
    )
    events = snapshot.get("events")
    events = events if isinstance(events, (list, tuple)) else ()
    if not events:
        lines.append("  No recent events")
    else:
        for event in list(events)[-LIGHTHOUSE_EVENT_LIMIT:][::-1]:
            if not isinstance(event, dict):
                continue
            event_name = str(event.get("event") or "").casefold()
            label = _EVENT_LABELS.get(event_name, "Runtime state changed")
            lines.append(f"  [{_format_event_time(event.get('timestamp'))}] {label}")
    return "\n".join(lines)


class _RoutineLogFilter(logging.Filter):
    """Keep warnings and errors visible while Lighthouse owns the terminal."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING


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
        self.refresh_interval = max(0.1, refresh_interval)
        self.heartbeat_interval = max(1.0, heartbeat_interval)
        self.enabled = (
            enabled if enabled is not None else _env_bool(LIGHTHOUSE_ENABLED_ENV, True)
        )
        self._task: asyncio.Task[None] | None = None
        self._live: Any | None = None
        self._filters: list[tuple[logging.Handler, logging.Filter]] = []

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
            log = logging.getLogger("theia.codex")
            for handler in log.handlers:
                routine_filter = _RoutineLogFilter()
                handler.addFilter(routine_filter)
                self._filters.append((handler, routine_filter))
            console = Console(file=self.output, force_terminal=True)
            self._live = Live(
                Text(render_lighthouse(self.snapshot())),
                console=console,
                refresh_per_second=1,
                screen=False,
                transient=False,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live.start(refresh=True)
            self._task = asyncio.create_task(self._run(Text))
            return True
        except Exception:
            self._restore_logging()
            self._live = None
            logger.exception("Lighthouse View could not start")
            return False

    async def _run(self, text_type: Any) -> None:
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
                    self._live.update(text_type(render_lighthouse(self.snapshot())))
                await asyncio.sleep(self.refresh_interval)
        except Exception:
            self._restore_logging()
            logger.exception("Lighthouse View stopped unexpectedly")

    def _restore_logging(self) -> None:
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
        live, self._live = self._live, None
        if live is not None:
            with contextlib.suppress(Exception):
                live.stop()
        self._restore_logging()
