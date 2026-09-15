"""Discord usage, diagnostics, account, and login embeds."""

# The façade keeps lazy compatibility bridges for historical patch points.
# pylint: disable=cyclic-import

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

import discord

from ..colors import discord_color
from .support import (
    _current_revision,
    _frontend_embed,
    _frontend_label,
)
from ..core import THEIA_VERSION, _safe_intermediate_text, _truncate
from ..server.usage import PRICING_REGISTRY


def _format_count(value: Any) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        return "Unavailable"
    return f"{value:,.0f}"


def _format_whole_seconds(value: Any) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return "Unavailable"
    return f"{value:,.0f}"


def _format_duration_ms(value: Any) -> str:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return "not run"
    return f"{max(0.0, float(value)):,.1f} ms"


def _format_percent(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:g}% used"
    return "Unavailable"


def _format_reset(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(value))
    return "Unavailable"


def _format_usage_tokens(value: Any, *, estimated: bool = False) -> str:
    rendered = _format_count(value)
    return (
        f"~{rendered} (estimated)"
        if estimated and rendered != "Unavailable"
        else rendered
    )


def _format_usd(value: Any, *, available: bool = True) -> str:
    if (
        not available
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        return "Pricing unavailable"
    if value == 0:
        return "$0.00 USD"
    if value < 0.0001:
        return "<$0.0001 USD"
    if value < 0.01:
        return f"${value:.4f} USD"
    return f"${value:,.2f} USD"


def _usage_cost_value(result: dict[str, Any]) -> str:
    estimate = result.get("estimate") if isinstance(result, dict) else None
    if not isinstance(estimate, dict):
        return "Pricing unavailable"
    available = bool(estimate.get("available"))
    if "available" not in estimate:
        # Older ephemeral view snapshots used credit units. Never reinterpret
        # those values as USD when restoring a view after an upgrade.
        available = estimate.get("currency") == "USD"
    value = _format_usd(estimate.get("total"), available=available)
    return value


def _usage_cost_breakdown(result: dict[str, Any]) -> str:
    """Render routed-model costs only in the detailed view."""
    estimate = result.get("estimate") if isinstance(result, dict) else None
    if not isinstance(estimate, dict) or not estimate.get("available", False):
        return "Pricing unavailable"
    by_model = estimate.get("byModel")
    if not isinstance(by_model, dict) or len(by_model) <= 1:
        return "Unavailable"
    lines: list[str] = []
    amounts = {
        pricing.display_name: by_model.get(pricing.display_name, 0.0)
        for pricing in PRICING_REGISTRY.values()
    }
    for model, amount in amounts.items():
        rendered = _format_usd(amount)
        lines.append(
            f"{_safe_intermediate_text(model, 80)}: {rendered.removesuffix(' USD')}"
        )
    combined = _format_usd(estimate.get("total"))
    lines.append(f"Combined: {combined.removesuffix(' USD')}")
    return "\n".join(lines)


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
            color=discord_color("WARNING"),
        )
    if not any(value is not None for value in summary.values()):
        return _frontend_embed(
            "command:usage",
            "Usage unavailable",
            "Usage data is currently unavailable. Please try again later.",
            channel=channel,
            user=user,
            color=discord_color("WARNING"),
        )
    embed = _frontend_embed(
        "command:usage",
        "Usage",
        f"Usage statistics for {result.get('date') or 'the selected date'}",
        channel=channel,
        user=user,
    )
    exact = result.get("exact") if isinstance(result, dict) else {}
    exact = exact if isinstance(exact, dict) else {}
    cumulative_tokens = summary.get("totalCumulativeProcessedTokens")
    if not isinstance(cumulative_tokens, (int, float)) or isinstance(
        cumulative_tokens, bool
    ):
        cumulative_tokens = summary.get(
            "totalCumulativeTokens", summary.get("lifetimeTokens")
        )
    embed.add_field(name="Today", value="\u200b", inline=False)
    embed.add_field(
        name=_frontend_label(
            "label:usage_main_agent_tokens",
            "Main-agent tokens",
            channel=channel,
            user=user,
        ),
        value=_format_count(
            exact.get("mainAgentTokens", exact.get("totalProcessedTokens"))
        ),
        inline=True,
    )
    embed.add_field(
        name=_frontend_label(
            "label:usage_subagent_tokens",
            "Subagent tokens",
            channel=channel,
            user=user,
        ),
        value=_format_count(exact.get("subagentTokens", 0)),
        inline=True,
    )
    embed.add_field(
        name=_frontend_label(
            "label:usage_total_processed_tokens",
            "Total processed tokens",
            channel=channel,
            user=user,
        ),
        value=_format_count(
            exact.get(
                "totalProcessedTokens",
                exact.get("mainAgentTokens", exact.get("totalTokens")),
            )
        ),
        inline=True,
    )
    embed.add_field(
        name=_frontend_label(
            "label:usage_estimated_api_cost",
            "Estimated API cost",
            channel=channel,
            user=user,
        ),
        value=_usage_cost_value(result),
        inline=True,
    )
    embed.add_field(name="Activity", value="\u200b", inline=False)
    historical_fields = (
        (
            "label:usage_lifetime_tokens",
            "Total cumulative tokens",
            _format_count(cumulative_tokens),
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
            _format_whole_seconds(summary.get("longestRunningTurnSec")),
        ),
    )
    for target, name, value in historical_fields:
        if target == "label:usage_longest_running_turn" and value != "Unavailable":
            value = f"{value} seconds"
        embed.add_field(
            name=_frontend_label(target, name, channel=channel, user=user),
            value=value,
            inline=True,
        )
    embed.set_footer(
        text="Estimated using configured API pricing. This is not the user's actual subscription charge."
    )
    return embed


def _usage_detail_value(value: Any, *, estimated: bool = False) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _format_usage_tokens(value, estimated=estimated)
    return _safe_intermediate_text(str(value), 180) or "Unavailable"


def _usage_details_embed(
    result: dict[str, Any],
    *,
    channel: Any | None = None,
    user: discord.abc.User | None = None,
) -> discord.Embed:
    """Render the owner-locked detailed view for one usage snapshot."""
    detailed = result.get("detailed") if isinstance(result, dict) else None
    detailed = detailed if isinstance(detailed, dict) else {}
    categories = detailed.get("categories")
    categories = categories if isinstance(categories, dict) else {}
    fields = (
        (
            "usage_detail_system_instructions",
            "System instructions",
            "system_instructions",
        ),
        (
            "usage_detail_identity_self_model",
            "Identity / self-model",
            "identity_self_model",
        ),
        ("usage_detail_memory_data", "Memory data", "memory_data"),
        ("usage_detail_skill_data", "Skill data", "skill_data"),
        ("usage_detail_user_history", "User history", "user_history"),
        ("usage_detail_tool_definitions", "Tool definitions", "tool_definitions"),
        ("usage_detail_tool_results", "Tool results", "tool_results"),
        ("usage_detail_routing_context", "Routing context", "routing_context"),
        ("usage_detail_subagent_usage", "Subagent usage", "subagentUsage"),
    )
    embed = _frontend_embed(
        "command:usage",
        "Detailed usage",
        f"Usage statistics for {result.get('date') or 'the selected date'}",
        channel=channel,
        user=user,
    )
    for target, default_label, key in fields:
        raw = categories.get(key) if key in categories else detailed.get(key)
        if isinstance(raw, dict):
            value = raw.get("value")
            estimated = bool(raw.get("estimated"))
        else:
            value = raw
            estimated = False
        embed.add_field(
            name=_frontend_label(target, default_label, channel=channel, user=user),
            value=_usage_detail_value(value, estimated=estimated),
            inline=True,
        )
    embed.add_field(
        name=_frontend_label(
            "label:usage_detail_reasoning_tokens",
            "Reasoning tokens",
            channel=channel,
            user=user,
        ),
        value=_usage_detail_value(detailed.get("reasoningTokens")),
        inline=True,
    )
    estimate = result.get("estimate") if isinstance(result, dict) else None
    by_model = estimate.get("byModel") if isinstance(estimate, dict) else None
    if isinstance(by_model, dict) and len(by_model) > 1:
        embed.add_field(
            name=_frontend_label(
                "label:usage_detail_model_pricing",
                "Model-specific API pricing",
                channel=channel,
                user=user,
            ),
            value=_usage_cost_breakdown(result),
            inline=False,
        )
    for target, default_label, key in (
        ("usage_detail_retries", "Retries", "retries"),
        ("usage_detail_failed_turns", "Failed turns", "failedTurns"),
        ("usage_detail_api_calls", "API calls", "apiCalls"),
        ("usage_detail_subagent_turns", "Subagent turns", "subagentTurns"),
        (
            "usage_detail_long_running_turns",
            "Long-running turns",
            "longRunningTurns",
        ),
    ):
        embed.add_field(
            name=_frontend_label(
                f"label:{target}", default_label, channel=channel, user=user
            ),
            value=_format_count(detailed.get(key)),
            inline=True,
        )
    overhead = detailed.get("unattributedOverhead")
    if isinstance(overhead, dict):
        overhead_value = _usage_detail_value(
            overhead.get("value"), estimated=bool(overhead.get("estimated"))
        )
    else:
        overhead_value = "Unavailable"
    embed.add_field(
        name=_frontend_label(
            "label:usage_detail_unattributed_overhead",
            "Unattributed overhead",
            channel=channel,
            user=user,
        ),
        value=overhead_value,
        inline=True,
    )
    return embed


def _audit_timestamp(value: Any) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        return "Unavailable"
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    except (OverflowError, OSError, ValueError):
        return "Unavailable"


def _self_improvement_history_embed(
    records: list[dict[str, Any]],
    *,
    channel: Any | None = None,
    user: discord.abc.User | None = None,
) -> discord.Embed:
    """Render safe recent self-improvement audit metadata."""
    embed = _frontend_embed(
        "command:improvements",
        "Self-improvement history",
        "Recent validated changes. Content and private review data are not shown.",
        channel=channel,
        user=user,
    )
    if not records:
        embed.description = "No self-improvement changes have been recorded."
        return embed
    for record in records[:20]:
        change_id = _truncate(record.get("id") or "unknown", 64)
        status = _truncate(record.get("status") or "unknown", 20)
        category = _truncate(record.get("category") or "unknown", 32)
        target = _truncate(record.get("target") or "unknown", 100)
        reason = _truncate(record.get("reason") or "No reason recorded.", 180)
        embed.add_field(
            name=f"{change_id} · {status}",
            value=(
                f"Category: {category}\n"
                f"Target: {target}\n"
                f"When: {_audit_timestamp(record.get('timestamp'))}\n"
                f"Reason: {reason}"
            ),
            inline=False,
        )
    return embed


def _self_improvement_preview_embed(
    record: dict[str, Any],
    *,
    channel: Any | None = None,
    user: discord.abc.User | None = None,
) -> discord.Embed:
    """Render one safe self-improvement change preview."""
    embed = _frontend_embed(
        "command:improvements",
        "Self-improvement change preview",
        "Only bounded audit metadata is shown; review prompts and file contents are private.",
        channel=channel,
        user=user,
    )
    embed.add_field(
        name="Change ID", value=_truncate(record.get("id"), 64), inline=True
    )
    embed.add_field(
        name="Status", value=_truncate(record.get("status"), 20), inline=True
    )
    embed.add_field(
        name="Category", value=_truncate(record.get("category"), 32), inline=True
    )
    embed.add_field(
        name="Target", value=_truncate(record.get("target"), 100), inline=True
    )
    embed.add_field(
        name="When", value=_audit_timestamp(record.get("timestamp")), inline=True
    )
    embed.add_field(
        name="Revert available",
        value="Yes" if record.get("revertible") else "No",
        inline=True,
    )
    embed.add_field(
        name="Previous content hash",
        value=_truncate(record.get("previous_content_hash") or "none", 64),
        inline=False,
    )
    embed.add_field(
        name="New content hash",
        value=_truncate(record.get("new_content_hash") or "none", 64),
        inline=False,
    )
    embed.add_field(
        name="Reason",
        value=_truncate(record.get("reason") or "No reason recorded.", 180),
        inline=False,
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
            color=discord_color("WARNING"),
        )
    credit_details = snapshot.get("credits")
    if not isinstance(credit_details, dict):
        return _frontend_embed(
            "command:credits",
            "Credits unavailable",
            "Credits data is currently unavailable. Please try again later.",
            channel=channel,
            user=user,
            color=discord_color("WARNING"),
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
        color=discord_color("INFO" if balance_available else "DEGRADED"),
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
        color=discord_color("WARNING"),
    )
