"""Read-only rendering helpers for Theia's Lighthouse terminal views."""

# Rich is optional at runtime; these imports remain local to render functions.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from ..colors import rich_style
from ..core import THEIA_VERSION, _safe_intermediate_text

LIGHTHOUSE_EVENT_LIMIT = 12
LIGHTHOUSE_DIAGNOSTIC_LIMIT = 256
DEFAULT_LIGHTHOUSE_WIDTH = 120
DEFAULT_LIGHTHOUSE_HEIGHT = 40

_MODEL_LABELS = {
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-6-astra": "GPT-6 Astra",
}

_EVENT_LABELS = {
    "listening_state_entered": "Listening state entered",
    "speaking_state_entered": "Speaking state entered",
    "attention_changed": "Attention changed",
    "workspace_updated": "Workspace updated",
    "turn_started": "Codex turn started",
    "turn_completed": "Codex turn completed",
    "turn_timed_out": "Codex turn timed out",
    "turn_cancelled": "Codex turn stopped",
    "cleanup_failed": "Cleanup failed",
    "turn_failed": "Codex turn failed",
    "log_warning": "Warning",
    "log_error": "Error",
    "character_loaded": "Character overlay loaded",
    "model_changed": "Model changed",
    "worker_started": "Worker started",
    "worker_completed": "Worker completed",
    "worker_failed": "Worker degraded",
    "approval_requested": "Approval requested",
    "approval_resolved": "Approval resolved",
    "session_created": "Session created",
    "session_resumed": "Session resumed",
    "session_selected": "Session selected",
    "session_reset": "Session reset",
    "session_degraded": "Session degraded",
    "codex_starting": "Codex starting",
    "codex_start_failed": "Codex start failed",
    "codex_stopped": "Codex stopped",
    "codex_connected": "Codex connected",
    "codex_recovered": "Codex recovered",
    "codex_restarted": "Codex restarted",
    "presence_updated": "Presence updated",
}

_EVENT_WARNING_NAMES = frozenset(
    {
        "cleanup_failed",
        "turn_failed",
        "turn_timed_out",
        "worker_failed",
        "session_degraded",
        "codex_start_failed",
    }
)


def _stable_log_event_name(detail: Any) -> str | None:
    """Map known log messages to stable titles without displaying their detail."""
    text = str(detail or "").casefold()
    if "cleanup" in text and "failed" in text:
        return "cleanup_failed"
    if "internal worker" in text and ("failed" in text or "timed out" in text):
        return "worker_failed"
    if "turn timed out" in text:
        return "turn_timed_out"
    if "turn failed" in text:
        return "turn_failed"
    return None


def _event_title(event_name: str, detail: Any = None) -> str:
    """Return a stable, operator-facing title for one runtime event."""
    if event_name in {"log_warning", "log_error"}:
        event_name = _stable_log_event_name(detail) or event_name
    return _EVENT_LABELS.get(event_name, "Runtime state changed")


def _event_severity(event_name: str, detail: Any = None) -> str:
    """Return the compact severity displayed beside a stable event title."""
    if event_name in {"fatal", "critical", "log_critical"}:
        return "FATAL"
    if event_name == "log_error":
        return "ERROR"
    if event_name == "log_warning":
        event_name = _stable_log_event_name(detail) or event_name
    if event_name in _EVENT_WARNING_NAMES:
        return "WARNING"
    return "INFO"


def _dashboard_text(value: Any, limit: int = 160) -> str:
    """Keep terminal values bounded and free of control sequences or paths."""
    safe_limit = max(1, limit)
    text = _safe_intermediate_text(value, safe_limit + 1)
    text = re.sub(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = text.strip()
    if len(text) > safe_limit:
        return text[: max(1, safe_limit - 1)].rstrip() + "…"
    return text


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


def _render_session_lines(value: Any) -> list[str]:
    """Render only the current session state supplied by the harness."""
    if isinstance(value, dict):
        try:
            active_count = max(0, int(value.get("active_count", 0)))
        except (TypeError, ValueError):
            active_count = 0
        if active_count == 0:
            return ["Session      No active session"]
        current = _dashboard_session_label(value.get("current"))
        if current == "unknown":
            current = "Active session"
        if active_count == 1:
            return [f"Session      {current}"]
        return [
            f"Session      {active_count} active sessions",
            f"Current      {current}",
        ]
    if isinstance(value, str) and value.strip():
        return [f"Session      {_dashboard_session_label(value)}"]
    return ["Session      No active session"]


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
        return "---- --:--"
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )
    except (OverflowError, OSError, ValueError):
        return "---- --:--"


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


def _dimensions(width: int | None, height: int | None) -> tuple[int, int]:
    try:
        resolved_width = (
            width
            if isinstance(width, int) and not isinstance(width, bool)
            else DEFAULT_LIGHTHOUSE_WIDTH
        )
    except (TypeError, ValueError):
        resolved_width = DEFAULT_LIGHTHOUSE_WIDTH
    try:
        resolved_height = (
            height
            if isinstance(height, int) and not isinstance(height, bool)
            else DEFAULT_LIGHTHOUSE_HEIGHT
        )
    except (TypeError, ValueError):
        resolved_height = DEFAULT_LIGHTHOUSE_HEIGHT
    return max(1, resolved_width), max(1, resolved_height)


def _usable_width(width: int) -> int:
    """Leave a cell on either side for the full-screen renderer's padding."""
    return max(1, width - 2)


def _separator(width: int) -> str:
    return "─" * _usable_width(width)


def _fit_line(line: str, width: int) -> str:
    if len(line) <= width:
        return line
    if width <= 1:
        return "…"[:width]
    return line[: width - 1].rstrip() + "…"


def _footer(width: int, hint: str) -> str:
    usable = _usable_width(width)
    hint = _fit_line(hint, usable)
    return " " * max(0, usable - len(hint)) + hint


def _workspace_lines(workspace: dict[str, Any]) -> list[str]:
    source = _dashboard_text(workspace.get("source"), 40).casefold()
    entries = workspace.get("entries")
    entries = (
        entries
        if source == "session workspace" and isinstance(entries, (list, tuple))
        else ()
    )
    try:
        count = max(0, int(workspace.get("entry_count", len(entries))))
    except (TypeError, ValueError):
        count = len(entries)
    recent = "No active workspace entries"
    if entries and isinstance(entries[0], dict):
        recent = _dashboard_text(entries[0].get("text"), 180) or recent
    label = "entry" if count == 1 else "entries"
    return [f"Workspace      {count} {label}", f"Recent         {recent}"]


def _runtime_parts(snapshot: dict[str, Any]) -> dict[str, str]:
    runtime = snapshot.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    heartbeat = runtime.get("heartbeat")
    heartbeat = heartbeat if isinstance(heartbeat, dict) else {}
    cleanup = runtime.get("cleanup")
    cleanup = cleanup if isinstance(cleanup, dict) else {}
    heartbeat_state = _dashboard_text(heartbeat.get("state"), 24) or "unknown"
    latency = heartbeat.get("latency_ms")
    latency_text = (
        f"{float(latency):.0f} ms"
        if isinstance(latency, (int, float)) and not isinstance(latency, bool)
        else "unknown"
    )
    try:
        failures = max(0, int(heartbeat.get("consecutive_failures", 0)))
    except (TypeError, ValueError):
        failures = 0
    cleanup_status = _dashboard_text(cleanup.get("status"), 24) or "unknown"
    cleanup_reason = _dashboard_text(cleanup.get("reason"), 120)
    cleanup_line = cleanup_status + (f" · {cleanup_reason}" if cleanup_reason else "")
    return {
        "codex": (
            f"{_dashboard_text(runtime.get('codex_version'), 32) or 'unknown'} · "
            f"{_dashboard_text(runtime.get('codex'), 24) or 'unknown'}"
        ),
        "rss": _format_rss(runtime.get("rss_bytes")),
        "turns": _dashboard_text(runtime.get("active_turns", 0), 12) or "0",
        "workers": _dashboard_text(runtime.get("workers", 0), 12) or "0",
        "approvals": _dashboard_text(runtime.get("approvals", 0), 12) or "0",
        "memory": f"{_dashboard_text(runtime.get('memory_entries', 0), 12) or '0'} entries",
        "watchdog": _dashboard_text(runtime.get("watchdog"), 24) or "unknown",
        "recovery": "active" if runtime.get("recovery") else "inactive",
        "heartbeat": f"{heartbeat_state} · {latency_text} · {failures} failures",
        "cleanup": cleanup_line,
        "update": _dashboard_text(runtime.get("update"), 24) or "unknown",
    }


def _event_lines(snapshot: dict[str, Any]) -> list[str]:
    events = snapshot.get("events")
    events = events if isinstance(events, (list, tuple)) else ()
    if not events:
        return ["  No recent events"]
    lines: list[str] = []
    for event in list(events)[-LIGHTHOUSE_EVENT_LIMIT:][::-1]:
        if not isinstance(event, dict):
            continue
        event_name = str(event.get("event") or "").casefold()
        label = _event_title(event_name, event.get("detail"))
        severity = _event_severity(event_name, event.get("detail"))
        lines.append(
            f"  [{_format_event_time(event.get('timestamp'))}] {severity:<8} {label}"
        )
    return lines or ["  No recent events"]


def _latest_problem(snapshot: dict[str, Any]) -> str:
    events = snapshot.get("events")
    events = events if isinstance(events, (list, tuple)) else ()
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        name = str(event.get("event") or "").casefold()
        if _event_severity(name, event.get("detail")) in {"WARNING", "ERROR", "FATAL"}:
            return _event_title(name, event.get("detail"))
    return "none"


def _base_lines(
    snapshot: dict[str, Any], *, width: int
) -> tuple[list[str], dict[str, str]]:
    character = snapshot.get("character")
    character = character if isinstance(character, dict) else {}
    name = _dashboard_text(character.get("name"), 80) or "none"
    source = _dashboard_text(character.get("source"), 48) or "unknown"
    reason = _dashboard_text(character.get("reason"), 96)
    mood = snapshot.get("mood")
    mood = mood if isinstance(mood, dict) else {}
    mood_label = _dashboard_text(mood.get("label"), 32).title() or "Unknown"
    mood_traits = _dashboard_text(mood.get("traits"), 80)
    try:
        strength = max(0.0, min(1.0, float(mood.get("strength", 0.5))))
    except (TypeError, ValueError):
        strength = 0.5
    mood_line = f"{mood_label} ({strength:.0%})"
    if mood_traits and mood_traits.casefold() != "unknown":
        mood_line += f" · {mood_traits}"
    attention = snapshot.get("attention")
    attention = attention if isinstance(attention, dict) else {}
    attention_line = _dashboard_text(attention.get("active"), 100) or "none"
    presence = snapshot.get("presence")
    presence = presence if isinstance(presence, dict) else {}
    session = snapshot.get("session")
    session = session if isinstance(session, dict) else {}
    session_reason = _dashboard_text(session.get("reason"), 96)
    objective = _dashboard_text(snapshot.get("session_objective"), 180)
    runtime = _runtime_parts(snapshot)
    lines = [
        f"Theia {_dashboard_text(snapshot.get('version'), 24) or THEIA_VERSION} · Lighthouse View",
        _separator(width),
        f"Status       {_dashboard_text(snapshot.get('action'), 80) or 'Unknown'}",
        f"Mode         {_dashboard_text(snapshot.get('mode'), 24).title() or 'Unknown'}",
        (
            f"Model        {_model_label(snapshot.get('model'))} · "
            f"{_dashboard_text(snapshot.get('reasoning_mode'), 24) or 'unknown'}"
        ),
        f"Reasoning    {_dashboard_text(snapshot.get('reasoning'), 40) or 'unknown'}",
        f"Character    {name} · loaded from {source}"
        + (f" · {reason}" if reason else ""),
        f"Presence     {_render_presence(presence)}",
        f"Voice        {_voice_label(snapshot.get('voice') if isinstance(snapshot.get('voice'), dict) else {})}",
        f"Attention    {attention_line}",
        f"Mood         {mood_line}",
    ]
    session_lines = _render_session_lines(snapshot.get("session"))
    lines.extend(session_lines)
    if session_reason:
        lines.append(
            "Session state "
            f"{_dashboard_text(session.get('status'), 24) or 'degraded'} · {session_reason}"
        )
    if objective:
        lines.append(f"Session objective {objective}")
    return lines, runtime


def _normal_lines(snapshot: dict[str, Any], *, width: int) -> list[str]:
    lines, runtime = _base_lines(snapshot, width=width)
    workspace = snapshot.get("workspace")
    workspace = workspace if isinstance(workspace, dict) else {}
    lines.extend(
        [_separator(width), *_workspace_lines(workspace), _separator(width), "Runtime"]
    )
    lines.extend(
        [
            f"  Codex        {runtime['codex']}",
            f"  RSS          {runtime['rss']}",
            f"  Active turns {runtime['turns']}",
            f"  Workers      {runtime['workers']}",
            f"  Approvals    {runtime['approvals']}",
            f"  Memory       {runtime['memory']}",
            f"  Watchdog     {runtime['watchdog']}",
            f"  Recovery     {runtime['recovery']}",
            f"  Heartbeat    {runtime['heartbeat']}",
            f"  Codex update {runtime['update']}",
            f"  Cleanup      {runtime['cleanup']}",
            _separator(width),
            "Recent events",
            *_event_lines(snapshot),
        ]
    )
    return lines


def _compact_lines(snapshot: dict[str, Any], *, width: int) -> list[str]:
    base, runtime = _base_lines(snapshot, width=width)
    # Keep the critical identity and health fields, then summarize the three
    # potentially unbounded sections instead of dropping them silently.
    status = next(
        (line for line in base if line.startswith("Status       ")),
        "Status       Unknown",
    )
    model = next(
        (line for line in base if line.startswith("Model        ")),
        "Model        unknown",
    )
    reasoning = next(
        (line for line in base if line.startswith("Reasoning    ")),
        "Reasoning    unknown",
    )
    character = next(
        (line for line in base if line.startswith("Character    ")), "Character    none"
    )
    session = [
        line for line in base if line.startswith(("Session      ", "Current      "))
    ]
    workspace = snapshot.get("workspace")
    workspace = workspace if isinstance(workspace, dict) else {}
    workspace_lines = _workspace_lines(workspace)
    events = snapshot.get("events")
    event_count = len(events) if isinstance(events, (list, tuple)) else 0
    recent_event = _event_lines(snapshot)[0].strip()
    lines = [
        base[0],
        _separator(width),
        status,
        f"{model} · {reasoning.removeprefix('Reasoning    ')}",
        character,
    ]
    lines.extend(session or ["Session      No active session"])
    lines.extend(
        [
            _separator(width),
            workspace_lines[0],
            workspace_lines[1],
            f"Runtime       Codex {runtime['codex']} · Heartbeat {runtime['heartbeat']}",
            f"Cleanup      {runtime['cleanup']}",
            f"Recent events {event_count} · {recent_event}",
        ]
    )
    problem = _latest_problem(snapshot)
    if problem != "none":
        lines.append(f"Active error  {problem}")
    lines.append(_separator(width))
    return lines


def _emergency_lines(snapshot: dict[str, Any], *, width: int) -> list[str]:
    base, runtime = _base_lines(snapshot, width=width)
    status = next(
        (line for line in base if line.startswith("Status       ")),
        "Status       Unknown",
    )
    model = next(
        (line for line in base if line.startswith("Model        ")),
        "Model        unknown",
    )
    reasoning = next(
        (line for line in base if line.startswith("Reasoning    ")),
        "Reasoning    unknown",
    )
    character = next(
        (line for line in base if line.startswith("Character    ")), "Character    none"
    )
    session = next(
        (line for line in base if line.startswith("Session      ")),
        "Session      No active session",
    )
    problem = _latest_problem(snapshot)
    return [
        base[0],
        _separator(width),
        status,
        f"{model} · {reasoning.removeprefix('Reasoning    ')}",
        character,
        session,
        f"Codex       {runtime['codex']}",
        f"Heartbeat   {runtime['heartbeat']}",
        f"Cleanup     {runtime['cleanup']}",
        f"Latest      {problem}",
        _separator(width),
    ]


def _ultra_emergency_lines(snapshot: dict[str, Any], *, width: int) -> list[str]:
    """Pack every required health field into the smallest useful screen."""
    base, runtime = _base_lines(snapshot, width=width)
    status = next(
        (line for line in base if line.startswith("Status       ")),
        "Status       Unknown",
    )
    model = next(
        (line for line in base if line.startswith("Model        ")),
        "Model        unknown",
    )
    reasoning = next(
        (line for line in base if line.startswith("Reasoning    ")),
        "Reasoning    unknown",
    )
    character = next(
        (line for line in base if line.startswith("Character    ")),
        "Character    none",
    )
    session = next(
        (line for line in base if line.startswith("Session      ")),
        "Session      No active session",
    )
    problem = _latest_problem(snapshot)
    return [
        base[0],
        _separator(width),
        status,
        f"{model} · {reasoning.removeprefix('Reasoning    ')}",
        f"{character} · {session.removeprefix('Session      ')}",
        f"Codex {runtime['codex']} · Heartbeat {runtime['heartbeat']}",
        f"Cleanup {runtime['cleanup']} · Latest {problem}",
    ]


def _fit_dashboard(lines: list[str], *, width: int, height: int, hint: str) -> str:
    usable = _usable_width(width)
    # The footer is reserved before content is clipped, so it cannot be hidden
    # by a short terminal.  Preserve the most recent state lines at the end.
    body_limit = max(1, height - (1 if hint else 0))
    if len(lines) > body_limit:
        keep = max(1, body_limit)
        important = [
            line
            for line in lines
            if line.startswith(
                (
                    "Status",
                    "Model",
                    "Reasoning",
                    "Character",
                    "Session",
                    "Current",
                    "Codex",
                    "Heartbeat",
                    "Cleanup",
                    "Latest",
                    "Active error",
                )
            )
        ]
        # Keep the title and preserve important rows in their original order.
        # This prevents clipping from moving the title below state or hiding
        # the footer behind an overlong event feed.
        lines = (lines[:1] + important)[:keep]
    rendered = [_fit_line(line, usable) for line in lines]
    if hint:
        rendered.append(_footer(width, hint))
    return "\n".join(rendered)


def render_lighthouse(
    snapshot: dict[str, Any],
    *,
    width: int | None = None,
    height: int | None = None,
    show_keyboard_hint: bool = False,
) -> str:
    """Render a bounded, dimension-aware Lighthouse View as terminal text."""
    width, height = _dimensions(width, height)
    if width < 60 or height < 14:
        lines = (
            _ultra_emergency_lines(snapshot, width=width)
            if height < 10
            else _emergency_lines(snapshot, width=width)
        )
    elif width < 100 or height < 24:
        lines = _compact_lines(snapshot, width=width)
    else:
        lines = _normal_lines(snapshot, width=width)
    hint = "F1 diagnostics" if show_keyboard_hint else ""
    return _fit_dashboard(lines, width=width, height=height, hint=hint)


def _diagnostic_message(record: Any) -> str:
    """Return a bounded diagnostic message without exposing unsafe payloads."""
    try:
        message = record.getMessage()
    except Exception:  # noqa: BLE001 - diagnostics must never affect the view
        return ""
    return _dashboard_text(message, 700)


def _diagnostic_exception(record: Any) -> str:
    """Return safe exception metadata while keeping traceback paths private."""
    exc_info = getattr(record, "exc_info", None)
    if not exc_info or len(exc_info) < 2:
        return ""
    exception = exc_info[1]
    name = _dashboard_text(getattr(exc_info[0], "__name__", "Exception"), 80)
    message = _dashboard_text(exception, 240)
    return f" · {name}: {message}" if message else f" · {name}"


def _record_severity(record: Any) -> str:
    """Classify a record from its structured numeric severity only."""
    try:
        level = int(getattr(record, "levelno", logging.INFO))
    except (TypeError, ValueError):
        level = logging.INFO
    if level >= logging.CRITICAL:
        return "FATAL"
    if level >= logging.ERROR:
        return "ERROR"
    if level >= logging.WARNING:
        return "WARNING"
    return "INFO"


def _record_level_name(record: Any) -> str:
    """Keep a safe custom level name while using INFO as its fallback style."""
    value = _dashboard_text(getattr(record, "levelname", "INFO"), 12).upper()
    return value or _record_severity(record)


def _diagnostic_event_prefix(message: str, record: Any) -> str:
    """Find an event title from structured metadata or the message prefix."""
    explicit = getattr(record, "event_name", None) or getattr(record, "event", None)
    if isinstance(explicit, str) and explicit.strip():
        return _dashboard_text(explicit, 120)
    # A plain message is not an event field. Never infer a title by matching
    # words in it, because that would make severity or hierarchy ambiguous.
    del message
    return ""


def _styled_diagnostic_line(record: Any) -> Any:
    from rich.text import Text

    message = _diagnostic_message(record)
    severity = _record_severity(record)
    level_name = _record_level_name(record)
    logger_name = _dashboard_text(getattr(record, "name", "root"), 48) or "root"
    module_name = _dashboard_text(getattr(record, "module", ""), 40)
    logger_label = (
        f"{logger_name}/{module_name}"
        if module_name and module_name.casefold() != logger_name.casefold()
        else logger_name
    )
    timestamp = _format_event_time(getattr(record, "created", None))
    exception = _diagnostic_exception(record)
    explicit_event = getattr(record, "event_name", None) or getattr(
        record, "event", None
    )
    explicit_event = (
        _dashboard_text(explicit_event, 120)
        if isinstance(explicit_event, str) and explicit_event.strip()
        else ""
    )
    line = Text()
    line.append("  [", style=rich_style("DISABLED"))
    line.append(timestamp, style=rich_style("DISABLED"))
    line.append("] ")
    line.append(
        f"{level_name:<8}", style=rich_style(severity, emphasis=severity == "FATAL")
    )
    line.append(" ")
    line.append(logger_label, style=rich_style("INFO"))
    line.append(": ")
    if explicit_event:
        line.append(explicit_event, style=f"bold {rich_style('INFO')}")
        line.append(": ")
    message_start = len(line.plain)
    line.append(message, style=rich_style(severity, emphasis=severity == "FATAL"))
    if exception:
        line.append(exception, style=rich_style(severity, emphasis=severity == "FATAL"))
    content = line.plain[message_start:]
    prefix = _diagnostic_event_prefix(message, record) if not explicit_event else ""
    if prefix and content.startswith(prefix):
        line.stylize(
            f"bold {rich_style('INFO')}",
            message_start,
            message_start + len(prefix),
        )
    for match in re.finditer(
        r"\b(?:method|status|duration(?:_ms)?|error_type|code)=[^, )]+", content
    ):
        line.stylize(
            rich_style("DISABLED"),
            message_start + match.start(),
            message_start + match.end(),
        )
    for match in re.finditer(
        r"\b(?:connected|completed|accepted|healthy)\b", content, re.IGNORECASE
    ):
        line.stylize(
            rich_style("INFO"),
            message_start + match.start(),
            message_start + match.end(),
        )
    for match in re.finditer(
        r"\b(?:disabled|inactive|none|unknown)\b", content, re.IGNORECASE
    ):
        line.stylize(
            rich_style("DISABLED"),
            message_start + match.start(),
            message_start + match.end(),
        )
    return line


def _styled_diagnostic_event(event: dict[str, Any]) -> Any:
    from rich.text import Text

    name = str(event.get("event") or "").casefold()
    severity = _event_severity(name, event.get("detail"))
    label = _event_title(name, event.get("detail"))
    line = Text("  [", style=rich_style("DISABLED"))
    line.append(
        _format_event_time(event.get("timestamp")), style=rich_style("DISABLED")
    )
    line.append("] ")
    line.append(
        f"{severity:<8}", style=rich_style(severity, emphasis=severity == "FATAL")
    )
    line.append(" ")
    line.append(label, style=f"bold {rich_style('INFO')}")
    detail = _dashboard_text(event.get("detail"), 180)
    if detail and name not in {"log_warning", "log_error"}:
        line.append(": ")
        line.append(detail, style=rich_style(severity))
    return line


def render_lighthouse_diagnostics_rich(
    snapshot: dict[str, Any],
    records: Any = (),
    *,
    width: int | None = None,
    height: int | None = None,
) -> Any:
    """Render the complete technical log with structured severity styling."""
    from rich.text import Text

    width, height = _dimensions(width, height)
    rendered = Text()
    version = _dashboard_text(snapshot.get("version"), 24) or THEIA_VERSION
    rendered.append(
        f"Theia {version} · Lighthouse Diagnostics", style=rich_style("INFO")
    )
    rendered.append("\n")
    rendered.append(_separator(width), style=rich_style("INFO"))
    events = snapshot.get("events")
    events = events if isinstance(events, (list, tuple)) else ()
    event_items = [
        event
        for event in list(events)[-LIGHTHOUSE_EVENT_LIMIT:][::-1]
        if isinstance(event, dict)
        and str(event.get("event") or "").casefold() not in {"log_warning", "log_error"}
        and _dashboard_text(event.get("detail"), 180)
    ]
    for event in event_items:
        rendered.append("\n")
        rendered.append(_styled_diagnostic_event(event))
    if event_items:
        rendered.append("\n")
        rendered.append(_separator(width), style=rich_style("INFO"))
    valid_records = [
        record
        for record in list(records)[-LIGHTHOUSE_DIAGNOSTIC_LIMIT:][::-1]
        if _diagnostic_message(record)
    ]
    if valid_records:
        for record in valid_records:
            rendered.append("\n")
            rendered.append(_styled_diagnostic_line(record))
    elif not event_items:
        rendered.append("\n  No diagnostic details", style=rich_style("DISABLED"))
    rendered.append("\n")
    rendered.append(_separator(width), style=rich_style("INFO"))
    rendered.append("\n")
    rendered.append(_footer(width, "ESC go back"), style=rich_style("INFO"))
    return rendered


def render_lighthouse_diagnostics(
    snapshot: dict[str, Any],
    records: Any = (),
    *,
    width: int | None = None,
    height: int | None = None,
) -> str:
    """Return diagnostic details as plain text for compatibility and tests."""
    return render_lighthouse_diagnostics_rich(
        snapshot, records, width=width, height=height
    ).plain


def render_lighthouse_rich(
    snapshot: dict[str, Any],
    *,
    records: Any = (),
    diagnostic_mode: bool = False,
    width: int | None = None,
    height: int | None = None,
    show_keyboard_hint: bool = False,
) -> Any:
    """Render a Lighthouse screen with shared semantic terminal colors."""
    from rich.text import Text

    if diagnostic_mode:
        return render_lighthouse_diagnostics_rich(
            snapshot, records, width=width, height=height
        )
    plain = render_lighthouse(
        snapshot,
        width=width,
        height=height,
        show_keyboard_hint=show_keyboard_hint,
    )
    rendered = Text(plain)
    offset = 0
    status_styles = {
        "INFO": rich_style("INFO"),
        "WARNING": rich_style("WARNING"),
        "ERROR": rich_style("ERROR"),
        "FATAL": rich_style("FATAL", emphasis=True),
        "CONNECTED": rich_style("CONNECTED"),
        "HEALTHY": rich_style("HEALTHY"),
        "DEGRADED": rich_style("DEGRADED"),
        "DISABLED": rich_style("DISABLED"),
    }
    for line in plain.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        if body.startswith(("Theia ", "Workspace", "Runtime", "Recent events")):
            rendered.stylize(status_styles["INFO"], offset, offset + len(body))
        if body.startswith("─"):
            rendered.stylize(status_styles["INFO"], offset, offset + len(body))
        if body.startswith("  [") and "] " in body:
            marker_end = body.find("] ") + 2
            name = body[marker_end:].split(maxsplit=1)[0].upper()
            if name in status_styles:
                rendered.stylize(
                    status_styles[name],
                    offset + marker_end,
                    offset + marker_end + len(name),
                )
        offset += len(line)
    return rendered
