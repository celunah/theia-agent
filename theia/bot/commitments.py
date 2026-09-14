"""Private Discord controls for a user's session-scoped open loops."""

from __future__ import annotations

from typing import Any

import discord
from discord import app_commands

from ..core import CodexAppServerError, _safe_error_reason
from .support import _frontend_embed, _require_login, session_key


def _kind_label(value: Any) -> str:
    return str(value or "open loop").replace("_", " ").title()


def _commitments_embed(
    records: list[dict[str, Any]], *, channel: Any, user: Any
) -> discord.Embed:
    if not records:
        return _frontend_embed(
            "command:commitments",
            "Open loops",
            "There are no active open loops in this session.",
            channel=channel,
            user=user,
            color=discord.Color.blurple(),
        )
    lines = [
        f"**{_kind_label(item.get('kind'))}** — {item.get('text', '')}\n"
        f"ID: `{item.get('commitment_id', '')}`"
        for item in records
    ]
    return _frontend_embed(
        "command:commitments",
        "Open loops",
        "\n\n".join(lines),
        channel=channel,
        user=user,
        color=discord.Color.blurple(),
    )


async def handle_commitments_command(
    bot: Any,
    interaction: discord.Interaction,
    action: str | app_commands.Choice[str] | None = None,
    commitment_id: str | None = None,
) -> None:
    """List or explicitly update only the invoking user's current session."""
    if not await _require_login(interaction):
        return
    selected = (
        (
            action.value
            if isinstance(action, app_commands.Choice)
            else (action or "list")
        )
        .strip()
        .casefold()
    )
    if selected not in {"list", "complete", "dismiss", "promote"}:
        await interaction.response.send_message(
            embed=_frontend_embed(
                "command:commitments",
                "Unknown action",
                "Choose list, complete, dismiss, or promote.",
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.orange(),
            ),
            ephemeral=True,
        )
        return
    if selected != "list" and not (commitment_id or "").strip():
        await interaction.response.send_message(
            embed=_frontend_embed(
                "command:commitments",
                "Open loop ID required",
                f"Provide an open loop ID to {selected} it.",
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.orange(),
            ),
            ephemeral=True,
        )
        return
    await interaction.response.defer(ephemeral=True)
    key = session_key(interaction.channel, interaction.user.id)
    try:
        if selected == "list":
            embed = _commitments_embed(
                bot.codex.session_commitments(key),
                channel=interaction.channel,
                user=interaction.user,
            )
        elif selected == "promote":
            bot.codex.promote_commitment(key, commitment_id or "")
            embed = _frontend_embed(
                "command:commitments",
                "Open loop promoted",
                "That open loop was explicitly added to your durable memory.",
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.green(),
            )
        else:
            status = "completed" if selected == "complete" else "dismissed"
            bot.codex.update_commitment(key, commitment_id or "", status)
            embed = _frontend_embed(
                "command:commitments",
                "Open loop updated",
                f"The open loop was marked {selected}.",
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.green(),
            )
    except (CodexAppServerError, OSError, ValueError) as exc:
        embed = _frontend_embed(
            "command:commitments",
            "Open loops unavailable",
            _safe_error_reason(exc),
            channel=interaction.channel,
            user=interaction.user,
            color=discord.Color.orange(),
        )
    await interaction.followup.send(embed=embed, ephemeral=True)
