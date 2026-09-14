"""Administrator controls for self-improvement audit records."""

from __future__ import annotations

from typing import Any

import discord

from ..core import _is_super_admin_user, _safe_error_reason
from ..server.core import CodexAppServerError
from .embeds import (
    _self_improvement_history_embed,
    _self_improvement_preview_embed,
)
from .support import _frontend_embed, _require_server_admin


async def handle_self_improvement_command(
    bot: Any,
    interaction: discord.Interaction,
    action: str | None,
    change_id: str | None,
) -> None:
    """Handle safe list, preview, and Super Admin-only revert operations."""
    await bot.presence.touch()
    if not await _require_server_admin(
        interaction,
        message="Only Theia administrators can inspect self-improvement changes.",
    ):
        return
    selected = (action or "list").strip().casefold()
    if selected not in {"list", "preview", "revert"}:
        await interaction.response.send_message(
            embed=_frontend_embed(
                "command:improvements",
                "Unknown action",
                "Choose list, preview, or revert.",
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.orange(),
            ),
            ephemeral=True,
        )
        return
    if selected in {"preview", "revert"} and not (change_id or "").strip():
        await interaction.response.send_message(
            embed=_frontend_embed(
                "command:improvements",
                "Change ID required",
                f"Provide a change ID to {selected} a self-improvement change.",
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.orange(),
            ),
            ephemeral=True,
        )
        return
    if selected == "revert" and not _is_super_admin_user(interaction.user.id):
        await interaction.response.send_message(
            embed=_frontend_embed(
                "command:improvements",
                "Super Admin access required",
                "Only a Theia Super Admin can revert self-improvement changes.",
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.orange(),
            ),
            ephemeral=True,
        )
        return
    try:
        if selected == "list":
            embed = _self_improvement_history_embed(
                bot.codex.self_improvement_history(),
                channel=interaction.channel,
                user=interaction.user,
            )
        elif selected == "preview":
            embed = _self_improvement_preview_embed(
                bot.codex.self_improvement_preview(change_id or ""),
                channel=interaction.channel,
                user=interaction.user,
            )
        else:
            result = await bot.codex.revert_self_improvement(
                change_id or "", super_admin=True
            )
            embed = _self_improvement_preview_embed(
                result,
                channel=interaction.channel,
                user=interaction.user,
            )
            embed.title = "Self-improvement change reverted"
    except (CodexAppServerError, OSError) as exc:
        embed = _frontend_embed(
            "command:improvements",
            "Self-improvement unavailable",
            _safe_error_reason(exc),
            channel=interaction.channel,
            user=interaction.user,
            color=discord.Color.orange(),
        )
    await interaction.response.send_message(embed=embed, ephemeral=True)
