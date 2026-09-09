"""Discord usage, diagnostics, account, and login embeds."""

# The façade keeps lazy compatibility bridges for historical patch points.
# pylint: disable=cyclic-import

from __future__ import annotations

import math
import time
from typing import Any

import discord

from .support import _current_revision, _frontend_embed, _frontend_label
from ..core import THEIA_VERSION, _truncate


def _format_count(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "Unavailable"
    return f"{value:,}"


def _format_whole_seconds(value: Any) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return "Unavailable"
    return f"{value:,.0f}"


def _format_percent(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:g}% used"
    return "Unavailable"


def _format_reset(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(value))
    return "Unavailable"


def _usage_embed(
    result: dict[str, Any],
    *,
    channel: Any | None = None,
    user: discord.abc.User | None = None,
) -> discord.Embed:
    summary = result.get("summary") if isinstance(result, dict) else None
    if not isinstance(summary, dict):
        return _frontend_embed(
            "command:usage",
            "Usage unavailable",
            "Usage data is currently unavailable. Please try again later.",
            channel=channel,
            user=user,
            color=discord.Color.orange(),
        )
    if not any(value is not None for value in summary.values()):
        return _frontend_embed(
            "command:usage",
            "Usage unavailable",
            "Usage data is currently unavailable. Please try again later.",
            channel=channel,
            user=user,
            color=discord.Color.orange(),
        )
    embed = _frontend_embed(
        "command:usage",
        "Usage",
        "Usage tracked from Theia's conversation threads.",
        channel=channel,
        user=user,
        context={
            "lifetime_tokens": _format_count(summary.get("lifetimeTokens")),
            "total_cumulative_tokens": _format_count(
                summary.get("totalCumulativeTokens", summary.get("lifetimeTokens"))
            ),
            "peak_daily_tokens": _format_count(summary.get("peakDailyTokens")),
            "current_streak": _format_count(summary.get("currentStreakDays")),
            "longest_streak": _format_count(summary.get("longestStreakDays")),
            "longest_running_turn": _format_whole_seconds(
                summary.get("longestRunningTurnSec")
            ),
        },
    )
    fields = (
        (
            "label:usage_lifetime_tokens",
            "Total cumulative tokens",
            _format_count(
                summary.get("totalCumulativeTokens", summary.get("lifetimeTokens"))
            ),
        ),
        (
            "label:usage_peak_daily_tokens",
            "Peak daily tokens",
            _format_count(summary.get("peakDailyTokens")),
        ),
        (
            "label:usage_current_streak",
            "Current streak",
            _format_count(summary.get("currentStreakDays")),
        ),
        (
            "label:usage_longest_streak",
            "Longest streak",
            _format_count(summary.get("longestStreakDays")),
        ),
        (
            "label:usage_longest_running_turn",
            "Longest running turn",
            f"{_format_whole_seconds(summary.get('longestRunningTurnSec'))} seconds",
        ),
    )
    for target, name, value in fields:
        embed.add_field(
            name=_frontend_label(target, name, channel=channel, user=user),
            value=value,
            inline=True,
        )
    return embed


def _debug_embed(
    state: dict[str, Any],
    *,
    channel: Any | None = None,
    user: discord.abc.User | None = None,
) -> discord.Embed:
    """Render bounded live diagnostics without exposing prompts or protocol data."""
    runtime = state.get("runtime") if isinstance(state, dict) else {}
    configuration = state.get("configuration") if isinstance(state, dict) else {}
    session = state.get("session") if isinstance(state, dict) else {}
    counts = state.get("counts") if isinstance(state, dict) else {}
    usage = state.get("usage") if isinstance(state, dict) else {}
    if not isinstance(runtime, dict):
        runtime = {}
    if not isinstance(configuration, dict):
        configuration = {}
    if not isinstance(session, dict):
        session = {}
    if not isinstance(counts, dict):
        counts = {}
    if not isinstance(usage, dict):
        usage = {}
    workers = counts.get("internal_workers")
    if isinstance(workers, dict):
        worker_text = ", ".join(
            f"{name}: {value}"
            for name, value in sorted(workers.items())
            if isinstance(name, str) and isinstance(value, int)
        )
    else:
        worker_text = ""
    mood = session.get("mood")
    if isinstance(mood, dict):
        raw_strength = mood.get("strength", 0.5)
        if (
            isinstance(raw_strength, (int, float))
            and not isinstance(raw_strength, bool)
            and math.isfinite(float(raw_strength))
        ):
            mood_strength = max(0.0, min(1.0, float(raw_strength)))
        else:
            mood_strength = 0.5
        mood_text = (
            f"{_truncate(mood.get('traits') or 'unknown', 120)} "
            f"({mood.get('label') or 'neutral'}, "
            f"{mood_strength * 100:.0f}%)"
        )
    else:
        mood_text = "Unavailable"
    process = str(runtime.get("process") or "unknown")
    exit_code = runtime.get("exit_code")
    if process != "running" and isinstance(exit_code, int):
        process = f"{process} (exit {exit_code})"
    state_text = (
        "recovery blocked"
        if runtime.get("state_recovery_blocked")
        else "dirty"
        if runtime.get("state_dirty")
        else "clean"
    )
    embed = _frontend_embed(
        "command:debug",
        "Theia debug state",
        "Live, sanitized runtime diagnostics. This view refreshes while it is open.",
        channel=channel,
        user=user,
        context={
            "model": str(configuration.get("model") or "Unavailable"),
            "mode": str(session.get("mode") or "Unavailable"),
            "personality": str(session.get("personality") or "None"),
            "active_turns": _format_count(counts.get("active_turns")),
            "pending_approvals": _format_count(counts.get("pending_approvals")),
        },
    )
    fields = (
        (
            "label:debug_runtime",
            "Runtime",
            "\n".join(
                (
                    f"Process: {process}",
                    "Authentication: "
                    + ("authenticated" if runtime.get("authenticated") else "required"),
                    f"State: {state_text}",
                    f"Protocol requests pending: {_format_count(counts.get('pending_protocol_requests'))}",
                )
            ),
        ),
        (
            "label:debug_configuration",
            "Configuration",
            "\n".join(
                (
                    f"Model: {configuration.get('model') or 'Unavailable'}",
                    f"Approval: {configuration.get('approval_level') or 'Unavailable'}",
                    "Adaptive reasoning: "
                    + ("on" if configuration.get("adaptive_reasoning") else "off"),
                    "Self-improvement: "
                    + ("on" if configuration.get("self_improvement") else "off"),
                )
            ),
        ),
        (
            "label:debug_session",
            "Current session",
            "\n".join(
                (
                    f"Mode: {session.get('mode') or 'Unavailable'}",
                    f"Personality: {session.get('personality') or 'None'}",
                    f"Mood: {mood_text}",
                    f"Thread: {_truncate(session.get('thread_id') or 'none', 80)}",
                    f"Turn: {_truncate(session.get('turn_id') or 'idle', 80)}",
                )
            ),
        ),
        (
            "label:debug_counts",
            "Activity",
            "\n".join(
                (
                    f"Sessions: {_format_count(counts.get('sessions'))}",
                    f"Loaded threads: {_format_count(counts.get('loaded_threads'))}",
                    f"Active turns: {_format_count(counts.get('active_turns'))}",
                    f"Pending approvals: {_format_count(counts.get('pending_approvals'))}",
                    f"Background tasks: {_format_count(counts.get('background_tasks'))}",
                    f"Internal workers: {worker_text or 'none'}",
                )
            ),
        ),
        (
            "label:debug_usage",
            "Theia usage",
            "\n".join(
                (
                    f"Cumulative tokens: {_format_count(usage.get('cumulative_tokens'))}",
                    (
                        "Longest turn: "
                        f"{_format_whole_seconds(usage.get('longest_turn_seconds'))} seconds"
                    ),
                )
            ),
        ),
    )
    for target, name, value in fields:
        embed.add_field(
            name=_frontend_label(target, name, channel=channel, user=user),
            value=_truncate(value, 1024),
            inline=False,
        )
    embed.set_footer(
        text=_frontend_label(
            "label:debug_live_footer",
            "Live updates every 2 seconds. Stop them with the button.",
            channel=channel,
            user=user,
        )
    )
    return embed


def _credits_embed(
    result: dict[str, Any],
    *,
    channel: Any | None = None,
    user: discord.abc.User | None = None,
) -> discord.Embed:
    snapshot = result.get("rateLimits") if isinstance(result, dict) else None
    if not isinstance(snapshot, dict):
        return _frontend_embed(
            "command:credits",
            "Credits unavailable",
            "Credits data is currently unavailable. Please try again later.",
            channel=channel,
            user=user,
            color=discord.Color.orange(),
        )
    credit_details = snapshot.get("credits")
    if not isinstance(credit_details, dict):
        return _frontend_embed(
            "command:credits",
            "Credits unavailable",
            "Credits data is currently unavailable. Please try again later.",
            channel=channel,
            user=user,
            color=discord.Color.orange(),
        )
    balance = credit_details.get("balance")
    balance_text = str(balance) if balance is not None else "Unavailable"
    if credit_details.get("unlimited") is True:
        balance_text = "Unlimited"
    balance_available = balance is not None or credit_details.get("unlimited") is True
    embed = _frontend_embed(
        "command:credits",
        "Credits" if balance_available else "Credits unavailable",
        "Current Codex credit information."
        if balance_available
        else "Credits data is currently unavailable. Please try again later.",
        channel=channel,
        user=user,
        context={"balance": balance_text},
        color=discord.Color.blurple() if balance_available else discord.Color.orange(),
    )
    embed.add_field(
        name=_frontend_label(
            "label:credits_balance",
            "Balance",
            channel=channel,
            user=user,
        ),
        value=balance_text,
        inline=True,
    )
    embed.add_field(
        name=_frontend_label(
            "label:credits_status",
            "Status",
            channel=channel,
            user=user,
        ),
        value=(
            "Unlimited"
            if credit_details.get("unlimited")
            else "Metered"
            if balance is not None
            else "Unavailable"
        ),
        inline=True,
    )
    primary = snapshot.get("primary") or {}
    secondary = snapshot.get("secondary") or {}
    embed.add_field(
        name=_frontend_label(
            "label:credits_five_hour_limit",
            "5-hour limit",
            channel=channel,
            user=user,
        ),
        value=(
            f"{_format_percent(primary.get('usedPercent'))}\n"
            f"Resets {_format_reset(primary.get('resetsAt'))}"
        ),
        inline=False,
    )
    embed.add_field(
        name=_frontend_label(
            "label:credits_weekly_limit",
            "Weekly limit",
            channel=channel,
            user=user,
        ),
        value=(
            f"{_format_percent(secondary.get('usedPercent'))}\n"
            f"Resets {_format_reset(secondary.get('resetsAt'))}"
        ),
        inline=False,
    )
    return embed


_PLAN_LABELS = {
    "free": "Free",
    "go": "Go",
    "plus": "Plus",
    "pro": "Pro",
    "team": "Team",
    "business": "Business",
    "enterprise": "Enterprise",
    "edu": "Edu",
}
_PLAN_PRICES = {
    "plus": "$20/mo",
    "pro": "$200/mo",
    "team": "$25/user/mo",
    "business": "$25/user/mo",
}


def _about_account(user: discord.abc.User | None) -> str:
    if user is None:
        return "Unavailable"
    name = getattr(user, "name", None) or getattr(user, "display_name", None)
    if not name:
        return "Unavailable"
    return f"@{str(name).lstrip('@')}"


def _about_plan(account: dict[str, Any] | None) -> str:
    if not isinstance(account, dict):
        return "Unavailable"
    raw_plan = account.get("planType") or account.get("plan_type")
    if not isinstance(raw_plan, str) or not raw_plan.strip():
        return "Unavailable"
    normalized = raw_plan.strip().casefold().replace("_", "-")
    name = _PLAN_LABELS.get(normalized, raw_plan.strip().replace("_", " ").title())
    price = account.get("monthlyPrice") or account.get("monthly_price")
    if isinstance(price, str) and price.strip():
        return f"{name} ({price.strip()})"
    mapped_price = _PLAN_PRICES.get(normalized)
    return f"{name} ({mapped_price})" if mapped_price else name


def _about_embed(
    *,
    account: dict[str, Any] | None,
    cli_version: str | None,
    mode: str,
    personality: str | None,
    channel: Any | None = None,
    user: discord.abc.User | None = None,
) -> discord.Embed:
    """Build the private, structured runtime-information embed."""
    embed = _frontend_embed(
        "command:about",
        "About Theia",
        "Current Theia and Codex account information.",
        channel=channel,
        user=user,
    )
    embed.add_field(
        name=_frontend_label(
            "label:about_theia_agent",
            "Theia Agent",
            channel=channel,
            user=user,
        ),
        value=f"{THEIA_VERSION} ({_current_revision()})",
        inline=False,
    )
    embed.add_field(
        name=_frontend_label(
            "label:about_codex_cli",
            "Codex CLI",
            channel=channel,
            user=user,
        ),
        value=cli_version or "Unavailable",
        inline=False,
    )
    embed.add_field(
        name=_frontend_label(
            "label:about_account",
            "Account",
            channel=channel,
            user=user,
        ),
        value=_about_account(user),
        inline=True,
    )
    embed.add_field(
        name=_frontend_label(
            "label:about_plan",
            "Plan",
            channel=channel,
            user=user,
        ),
        value=_about_plan(account),
        inline=True,
    )
    embed.add_field(
        name=_frontend_label(
            "label:about_mode",
            "Mode",
            channel=channel,
            user=user,
        ),
        value=(mode or "unknown").title(),
        inline=True,
    )
    embed.add_field(
        name=_frontend_label(
            "label:about_personality",
            "Personality",
            channel=channel,
            user=user,
        ),
        value=personality or "None",
        inline=True,
    )
    return embed


def _personality_summary_embed(
    summary: dict[str, Any] | None,
    mood: dict[str, Any],
    presence_line: str | None,
    *,
    channel: Any | None = None,
    user: discord.abc.User | None = None,
) -> discord.Embed:
    """Build the private character card for the active personality."""
    if summary is None:
        title = "No personality selected"
        description = "No character is active for this Discord session."
        personality = "none"
        character_name = ""
        character_slug = ""
        known_entries = 0
        known_users = 0
    else:
        character_name = str(summary.get("character_name") or "Character")
        character_slug = str(summary.get("identifier") or "character")
        title = f"{character_name} ({character_slug})"
        description = str(
            summary.get("description") or "No character description is available."
        )
        personality = str(summary.get("name") or character_slug)
        known_entries = max(0, int(summary.get("known_entries") or 0))
        known_users = max(0, int(summary.get("known_users") or 0))

    raw_strength = mood.get("strength")
    strength = (
        float(raw_strength)
        if isinstance(raw_strength, (int, float)) and not isinstance(raw_strength, bool)
        else 0.5
    )
    strength = max(0.0, min(1.0, strength))
    mood_label = str(mood.get("label") or "neutral").strip().title() or "Neutral"
    mood_text = f"{mood_label} ({strength * 100:.0f}%)"
    current_presence = presence_line.strip() if presence_line else "Unavailable"
    current_presence = current_presence or "Unavailable"
    context = {
        "command": "/personality",
        "personality": personality,
        "character_name": character_name,
        "character_slug": character_slug,
        "mood": mood_text,
        "presence": current_presence,
        "known_entries": known_entries,
        "known_users": known_users,
    }
    scope = ""
    set_by = ""
    embed_scope = ""
    embed_set_by = ""
    if summary is not None and summary.get("scope"):
        scope = str(summary["scope"])
        raw_set_by = summary.get("set_by")
        set_by = (
            f"Discord user ID: {raw_set_by}"
            if isinstance(raw_set_by, int) and not isinstance(raw_set_by, bool)
            else "Legacy selection"
        )
        context.update({"personality_scope": scope, "personality_set_by": set_by})
        embed_scope = _frontend_label(
            "label:personality_scope",
            "Scope",
            channel=channel,
            user=user,
            context=context,
        )
        embed_set_by = _frontend_label(
            "label:personality_set_by",
            "Set by",
            channel=channel,
            user=user,
            context=context,
        )
    embed = _frontend_embed(
        "command:personality",
        title,
        description,
        channel=channel,
        user=user,
        context=context,
    )
    embed.add_field(
        name=_frontend_label(
            "label:personality_known_entries",
            "Known Entries",
            channel=channel,
            user=user,
            context=context,
        ),
        value=str(known_entries),
        inline=False,
    )
    if embed_scope:
        embed.add_field(name=embed_scope, value=scope, inline=True)
        embed.add_field(name=embed_set_by, value=set_by, inline=True)
    embed.add_field(
        name=_frontend_label(
            "label:personality_known_users",
            "Known Users",
            channel=channel,
            user=user,
            context=context,
        ),
        value=str(known_users),
        inline=False,
    )
    embed.add_field(
        name=_frontend_label(
            "label:personality_mood",
            "Mood",
            channel=channel,
            user=user,
            context=context,
        ),
        value=mood_text,
        inline=True,
    )
    embed.add_field(
        name=_frontend_label(
            "label:personality_presence",
            "Presence",
            channel=channel,
            user=user,
            context=context,
        ),
        value=current_presence,
        inline=True,
    )
    footer = _frontend_label(
        "label:personality_footer",
        "Add or change the character with `/personality <file> <slug>`.",
        channel=channel,
        user=user,
        context=context,
    )
    embed.set_footer(text=footer)
    return embed


def _login_required_embed(
    *,
    channel: Any | None = None,
    user: discord.abc.User | None = None,
) -> discord.Embed:
    return _frontend_embed(
        "label:login_required",
        "Login required",
        "Please use `/login` before starting or controlling a Codex request.",
        channel=channel,
        user=user,
        color=discord.Color.orange(),
    )
