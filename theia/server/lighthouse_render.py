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

_POSITIVE_STATUS_TERMS = (
    "adaptive",
    "connected",
    "enabled",
    "watching",
    "completed",
    "accepted",
    "healthy",
)
_NEUTRAL_STATUS_TERMS = (
    "not configured",
    "unknown",
    "none",
    "disabled",
    "inactive",
)
_DEGRADED_STATUS_TERMS = (
    "failed worker",
    "degraded",
    "timeout",
    "unavailable",
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


def _is_internal_worker_event(event_name: str, detail: Any = None) -> bool:
    """Keep generic internal App Server failures out of the main dashboard."""
    if event_name == "worker_failed":
        return True
    return event_name in {"log_warning", "log_error"} and (
        _stable_log_event_name(detail) == "worker_failed"
    )


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


def _severity_style(severity: str, *, emphasis: bool = False) -> str:
    """Return the Lighthouse style for a structured severity."""
    name = "FATAL_DARK" if severity == "FATAL" else severity
    return rich_style(name, emphasis=emphasis)


def _stylize_terms(
    rendered: Any,
    line_offset: int,
    value_start: int,
    value: str,
    terms: tuple[str, ...],
    style: str,
) -> None:
    """Apply a semantic style to exact terms in a known field value."""
    if not value or not terms:
        return
    pattern = (
        r"(?<![\w-])(?:"
        + "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
        + r")(?![\w-])"
    )
    for match in re.finditer(pattern, value, re.IGNORECASE):
        rendered.stylize(
            style,
            line_offset + value_start + match.start(),
            line_offset + value_start + match.end(),
        )


def _style_dashboard_value(
    rendered: Any,
    line_offset: int,
    body: str,
    prefix: str,
    *,
    presence: bool = False,
) -> None:
    """Color a dashboard field after its stable, known label."""
    if not body.startswith(prefix):
        return
    value_start = len(prefix)
    value = body[value_start:]
    if presence:
        status = value.split(" · ", 1)[0].strip().casefold()
        if status in {"online", "active"}:
            style = rich_style("ACTIVE")
        elif status in {"idle", "away"}:
            style = rich_style("WARNING")
        elif status in {"offline", "unavailable"}:
            style = rich_style("ERROR")
        else:
            style = rich_style("DISABLED")
        status_start = value_start + len(value) - len(value.lstrip())
        _stylize_terms(rendered, line_offset, status_start, status, (status,), style)
        return
    _stylize_terms(
        rendered,
        line_offset,
        value_start,
        value,
        _DEGRADED_STATUS_TERMS,
        rich_style("ERROR"),
    )
    _stylize_terms(
        rendered,
        line_offset,
        value_start,
        value,
        _NEUTRAL_STATUS_TERMS,
        rich_style("DISABLED"),
    )
    _stylize_terms(
        rendered,
        line_offset,
        value_start,
        value,
        _POSITIVE_STATUS_TERMS,
        rich_style("INFO"),
    )


def _style_dashboard_line(rendered: Any, body: str, line_offset: int) -> None:
    """Apply semantic colors to known Lighthouse fields and event rows."""
    field_prefixes = (
        ("Presence     ", True),
        ("Presence ", True),
        ("Voice        ", False),
        ("Voice ", False),
        ("Model        ", False),
        ("Reasoning    ", False),
        ("Attention    ", False),
        ("Attention ", False),
        ("Mood         ", False),
        ("Mood ", False),
        ("Session state ", False),
        ("  Codex        ", False),
        ("Codex       ", False),
        ("Codex ", False),
        ("  Watchdog     ", False),
        ("Watchdog ", False),
        ("  Recovery     ", False),
        ("Recovery ", False),
        ("  Heartbeat    ", False),
        ("Heartbeat ", False),
        ("  Codex update ", False),
        ("Codex update ", False),
        ("  Cleanup      ", False),
        ("Cleanup ", False),
        ("Runtime Codex ", False),
    )
    for prefix, is_presence in field_prefixes:
        if body.startswith(prefix):
            _style_dashboard_value(
                rendered,
                line_offset,
                body,
                prefix,
                presence=is_presence,
            )
            return
    if not (body.startswith("  [") and "] " in body):
        return
    marker_end = body.find("] ") + 2
    severity = body[marker_end:].split(maxsplit=1)[0].upper()
    if severity not in {"INFO", "WARNING", "ERROR", "FATAL"}:
        return
    severity_start = marker_end
    rendered.stylize(
        _severity_style(severity, emphasis=severity == "FATAL"),
        line_offset + severity_start,
        line_offset + severity_start + len(severity),
    )
    title_start = marker_end + 9
    title = body[title_start:]
    _stylize_terms(
        rendered,
        line_offset,
        title_start,
        title,
        _DEGRADED_STATUS_TERMS,
        rich_style("ERROR"),
    )
    _stylize_terms(
        rendered,
        line_offset,
        title_start,
        title,
        _POSITIVE_STATUS_TERMS,
        rich_style("INFO"),
    )


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
        if _is_internal_worker_event(event_name, event.get("detail")):
            continue
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
        if _is_internal_worker_event(name, event.get("detail")):
            continue
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


def _compact_parts(snapshot: dict[str, Any], *, width: int) -> dict[str, list[str]]:
    """Build compact sections in the dashboard's stated priority order."""
    base, runtime = _base_lines(snapshot, width=width)
    core = [
        next(
            (line for line in base if line.startswith("Status       ")),
            "Status       Unknown",
        ),
        next(
            (line for line in base if line.startswith("Model        ")),
            "Model        unknown",
        ),
        next(
            (line for line in base if line.startswith("Reasoning    ")),
            "Reasoning    unknown",
        ),
        next(
            (line for line in base if line.startswith("Character    ")),
            "Character    none",
        ),
    ]
    session = [
        line for line in base if line.startswith(("Session      ", "Current      "))
    ] or ["Session      No active session"]
    secondary = [
        next(
            (line for line in base if line.startswith("Presence     ")),
            "Presence     unknown",
        ),
        next(
            (line for line in base if line.startswith("Voice        ")),
            "Voice        disabled",
        ),
        next(
            (line for line in base if line.startswith("Attention    ")),
            "Attention    none",
        ),
        next(
            (line for line in base if line.startswith("Mood         ")),
            "Mood         unknown",
        ),
    ]
    workspace = snapshot.get("workspace")
    workspace = workspace if isinstance(workspace, dict) else {}
    workspace_lines = _workspace_lines(workspace)
    runtime_lines = [
        f"Codex       {runtime['codex']}",
        f"Heartbeat   {runtime['heartbeat']}",
        f"Cleanup     {runtime['cleanup']}",
    ]
    events = _event_lines(snapshot)
    problem = _latest_problem(snapshot)
    errors = [f"Active error  {problem}"] if problem != "none" else []
    return {
        "header": [base[0], _separator(width)],
        "core": core + session,
        "secondary": secondary,
        "workspace": workspace_lines,
        "runtime": runtime_lines,
        "errors": errors,
        "events": ["Recent events", *events],
        "separator": [_separator(width)],
    }


def _compact_event_label(snapshot: dict[str, Any]) -> str:
    """Return one bounded event title for the tight layout."""
    events = snapshot.get("events")
    events = events if isinstance(events, (list, tuple)) else ()
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        name = str(event.get("event") or "").casefold()
        if _is_internal_worker_event(name, event.get("detail")):
            continue
        return _event_title(name, event.get("detail"))
    return "none"


def _tight_compact_lines(snapshot: dict[str, Any], *, width: int) -> list[str]:
    """Keep every compact category visible in a short but usable terminal."""
    parts = _compact_parts(snapshot, width=width)
    secondary = parts["secondary"]
    workspace = parts["workspace"]
    runtime = parts["runtime"]
    events = snapshot.get("events")
    event_count = (
        sum(
            1
            for event in events
            if isinstance(event, dict)
            and not _is_internal_worker_event(
                str(event.get("event") or "").casefold(), event.get("detail")
            )
        )
        if isinstance(events, (list, tuple))
        else 0
    )
    event_label = _compact_event_label(snapshot)
    event_label = event_label.removeprefix("Codex ")
    event_text = f"Recent events {event_count}"
    if event_count:
        event_text += f" · {event_label}"
    if parts["errors"]:
        event_text = "Error " + _latest_problem(snapshot)
    presence_state = secondary[0].removeprefix("Presence     ").split(" · ")[0]
    voice_value = secondary[1].removeprefix("Voice        ")
    voice_name, _, voice_state = voice_value.partition(" · ")
    voice_name = {
        "Qwen Audio Agent": "Qwen",
        "Codex Realtime": "Realtime",
        "Custom backend": "Custom",
    }.get(voice_name, voice_name)
    attention_value = secondary[2].removeprefix("Attention    ")
    mood_value = secondary[3].removeprefix("Mood         ").split(" · ")[0]
    codex_state = runtime[0].rsplit(" · ", maxsplit=1)[-1]
    heartbeat_state = runtime[1].removeprefix("Heartbeat   ").split(" · ")[0]
    cleanup_state = runtime[2].removeprefix("Cleanup     ").split(" · ")[0]
    core = parts["core"]
    model = core[1]
    reasoning = core[2].removeprefix("Reasoning    ")
    return [
        *parts["header"],
        core[0],
        f"{model} · Reasoning {reasoning}",
        *core[3:],
        f"Presence {presence_state} · Voice {voice_name or 'disabled'}"
        + (f" · {voice_state}" if voice_state else ""),
        f"Attention {_dashboard_text(attention_value, 18)} · Mood {mood_value}",
        f"{workspace[0]} · {workspace[1].removeprefix('Recent         ')}",
        f"Runtime Codex {codex_state} · Heartbeat {heartbeat_state}",
        f"Cleanup {cleanup_state} · {event_text}",
    ]


def _progressive_lines(
    snapshot: dict[str, Any], *, width: int, body_budget: int
) -> list[str]:
    """Use the available height continuously before resorting to emergency."""
    parts = _compact_parts(snapshot, width=width)
    tight = _tight_compact_lines(snapshot, width=width)
    expanded = [
        *parts["header"],
        *parts["core"],
        *parts["secondary"],
        *parts["workspace"],
        "Runtime",
        *parts["runtime"],
        *parts["errors"],
        *parts["events"],
        *parts["separator"],
    ]
    if body_budget == len(expanded) - 1:
        # The final separator is lower priority than a populated row when the
        # viewport is exactly one line short of the expanded compact layout.
        return expanded[:body_budget]
    if body_budget < len(expanded):
        lines = tight[:body_budget]
        older_events = parts["events"][2:]
        lines.extend(older_events[: body_budget - len(lines)])
        return lines[:body_budget]

    lines: list[str] = []
    for line in expanded:
        if len(lines) >= body_budget:
            break
        lines.append(line)
    return lines[:body_budget]


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
        character,
        session,
        f"Codex {runtime['codex']}",
        f"Heartbeat {runtime['heartbeat']}",
        f"Cleanup {runtime['cleanup']} · Latest {problem}",
    ]


def _micro_emergency_lines(snapshot: dict[str, Any], *, width: int) -> list[str]:
    """Retain the essential fields when even the emergency view is cramped."""
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
    return [
        base[0],
        _separator(width),
        status,
        f"{model} · {reasoning.removeprefix('Reasoning    ')}",
        f"{character} · {session.removeprefix('Session      ')}",
        f"Codex {runtime['codex']} · Heartbeat {runtime['heartbeat']} · Cleanup {runtime['cleanup']}",
    ]


def _fit_dashboard(lines: list[str], *, width: int, height: int, hint: str) -> str:
    usable = _usable_width(width)
    body_limit = max(0 if hint else 1, height - (1 if hint else 0))
    lines = lines[:body_limit]
    rendered = [_fit_line(line, usable) for line in lines]
    if hint:
        rendered.extend([""] * max(0, body_limit - len(rendered)))
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
    body_budget = max(
        0 if show_keyboard_hint else 1,
        height - (1 if show_keyboard_hint else 0),
    )
    full_lines = _normal_lines(snapshot, width=width)
    tight_lines = _tight_compact_lines(snapshot, width=width)
    if width >= 64 and len(full_lines) <= body_budget:
        lines = full_lines
    elif width < 36 or body_budget < len(tight_lines):
        emergency = _ultra_emergency_lines(snapshot, width=width)
        lines = (
            _micro_emergency_lines(snapshot, width=width)
            if body_budget < len(emergency)
            else emergency
        )
    else:
        lines = _progressive_lines(snapshot, width=width, body_budget=body_budget)
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


def _diagnostic_identifier(value: Any, limit: int) -> str:
    """Return one safe logger/module/function identifier for diagnostics."""
    value = _dashboard_text(value, limit)
    return re.sub(r"[^A-Za-z0-9_.<>-]", "", value)


def _diagnostic_source_label(record: Any) -> str:
    """Show the safe logger, module, and originating Theia function."""
    logger_name = _diagnostic_identifier(getattr(record, "name", "root"), 48)
    logger_name = logger_name or "root"
    module_name = _diagnostic_identifier(getattr(record, "module", ""), 40)
    function_name = _diagnostic_identifier(getattr(record, "funcName", ""), 64)
    source = logger_name
    if module_name and module_name.casefold() != logger_name.casefold():
        source += f"/{module_name}"
    if function_name and function_name != "<module>":
        source += f".{function_name}"
    return source


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
    source_label = _diagnostic_source_label(record)
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
        f"{level_name:<8}",
        style=_severity_style(severity, emphasis=severity == "FATAL"),
    )
    line.append(" ")
    line.append(source_label, style=rich_style("INFO"))
    line.append(": ")
    if explicit_event:
        line.append(explicit_event, style=f"bold {rich_style('INFO')}")
        line.append(": ")
    message_start = len(line.plain)
    line.append(message, style=_severity_style(severity, emphasis=severity == "FATAL"))
    if exception:
        line.append(
            exception, style=_severity_style(severity, emphasis=severity == "FATAL")
        )
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
        r"(?<![\w-])(?:adaptive|connected|enabled|watching|completed|accepted|healthy)(?![\w-])",
        content,
        re.IGNORECASE,
    ):
        line.stylize(
            rich_style("INFO"),
            message_start + match.start(),
            message_start + match.end(),
        )
    for match in re.finditer(
        r"(?<![\w-])(?:failed\s+worker|degraded|timeout|unavailable)(?![\w-])",
        content,
        re.IGNORECASE,
    ):
        line.stylize(
            rich_style("ERROR"),
            message_start + match.start(),
            message_start + match.end(),
        )
    for match in re.finditer(
        r"(?<![\w-])(?:not\s+configured|disabled|inactive|none|unknown)(?![\w-])",
        content,
        re.IGNORECASE,
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
        f"{severity:<8}",
        style=_severity_style(severity, emphasis=severity == "FATAL"),
    )
    line.append(" ")
    label_start = len(line.plain)
    line.append(label, style=f"bold {rich_style('INFO')}")
    _stylize_terms(
        line,
        0,
        label_start,
        label,
        _DEGRADED_STATUS_TERMS,
        rich_style("ERROR"),
    )
    _stylize_terms(
        line,
        0,
        label_start,
        label,
        _POSITIVE_STATUS_TERMS,
        rich_style("INFO"),
    )
    detail = _dashboard_text(event.get("detail"), 180)
    if detail and name not in {"log_warning", "log_error"}:
        line.append(": ")
        detail_start = len(line.plain)
        line.append(detail, style=_severity_style(severity))
        _stylize_terms(
            line,
            0,
            detail_start,
            detail,
            _DEGRADED_STATUS_TERMS,
            rich_style("ERROR"),
        )
        _stylize_terms(
            line,
            0,
            detail_start,
            detail,
            _NEUTRAL_STATUS_TERMS,
            rich_style("DISABLED"),
        )
        _stylize_terms(
            line,
            0,
            detail_start,
            detail,
            _POSITIVE_STATUS_TERMS,
            rich_style("INFO"),
        )
    return line


def _wrap_diagnostic_lines(lines: list[Any], *, width: int) -> list[Any]:
    """Wrap styled diagnostic rows so terminal height calculations stay true."""
    from rich.console import Console

    console = Console(width=max(1, _usable_width(width)), color_system=None)
    wrapped: list[Any] = []
    for line in lines:
        wrapped.extend(
            line.wrap(
                console,
                width=max(1, _usable_width(width)),
                overflow="fold",
                no_wrap=False,
            )
            or [line]
        )
    return wrapped


def _fit_diagnostic_lines(lines: list[Any], *, width: int, height: int) -> list[Any]:
    """Keep the diagnostic header and footer visible in a short terminal."""
    from rich.text import Text

    wrapped = _wrap_diagnostic_lines(lines, width=width)
    content_budget = max(0, height - 1)
    if len(wrapped) > content_budget:
        if content_budget == 0:
            return []
        if content_budget == 1:
            return wrapped[:1]
        if content_budget == 2:
            return [wrapped[0], wrapped[-1]]
        marker = Text("  … diagnostic details truncated", style=rich_style("DISABLED"))
        tail_count = content_budget - 3
        return [wrapped[0], marker, *wrapped[-(tail_count + 1) :]]
    return [*wrapped, *([Text()] * (content_budget - len(wrapped)))]


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
    lines: list[Text] = []
    version = _dashboard_text(snapshot.get("version"), 24) or THEIA_VERSION
    lines.append(
        Text(f"Theia {version} · Lighthouse Diagnostics", style=rich_style("INFO"))
    )
    lines.append(Text(_separator(width), style=rich_style("INFO")))
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
        lines.append(_styled_diagnostic_event(event))
    if event_items:
        lines.append(Text(_separator(width), style=rich_style("INFO")))
    valid_records = [
        record
        for record in list(records)[-LIGHTHOUSE_DIAGNOSTIC_LIMIT:][::-1]
        if _diagnostic_message(record)
    ]
    if valid_records:
        lines.extend(_styled_diagnostic_line(record) for record in valid_records)
    elif not event_items:
        lines.append(Text("  No diagnostic details", style=rich_style("DISABLED")))
    lines.append(Text(_separator(width), style=rich_style("INFO")))

    rendered = Text()
    for line in _fit_diagnostic_lines(lines, width=width, height=height):
        rendered.append(line)
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
        "FATAL": _severity_style("FATAL", emphasis=True),
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
        _style_dashboard_line(rendered, body, offset)
        offset += len(line)
    return rendered
