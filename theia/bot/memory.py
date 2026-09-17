"""Discord command and owner-locked callbacks for Theia's memory explorer."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import discord
from discord import app_commands

from ..colors import discord_color
from ..core import _is_super_admin_user
from ..server.core import CodexAppServerError
from ..delivery import _MemoryView
from .support import (
    _frontend_embed,
    _is_server_admin,
    _require_login,
    _require_server_admin,
    _send_command_failure,
    _channel_id,
    _guild_id,
    session_key,
)


async def memory_scope_autocomplete(
    _interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    """Suggest the three memory scopes supported by the slash command."""
    values = ("me", "server", "everyone")
    requested = current.strip().casefold()
    return [
        app_commands.Choice(name=value, value=value)
        for value in values
        if not requested or requested in value
    ]


def _scope_value(scope: str | app_commands.Choice[str] | None) -> str:
    if isinstance(scope, app_commands.Choice):
        return scope.value
    return (scope or "me").strip().casefold() or "me"


def _memory_session_key(view: _MemoryView) -> str:
    return (
        f"guild:{view.guild_id or 0}:channel:{view.channel_id or 0}:"
        f"user:{view.owner_id or 0}"
    )


def _view_admin_state(
    interaction: discord.Interaction,
) -> tuple[bool, bool]:
    super_admin = _is_super_admin_user(getattr(interaction.user, "id", None))
    server_admin = _is_server_admin(interaction.user, interaction.channel)
    return server_admin, super_admin


def _mutation_callback(
    bot: Any,
    view: _MemoryView,
) -> Callable[[discord.Interaction, str, str, str | None], Awaitable[tuple[bool, str]]]:
    async def mutate(
        interaction: discord.Interaction,
        action: str,
        record_id: str,
        replacement: str | None,
    ) -> tuple[bool, str]:
        """Apply one owner-authorized memory edit and refresh the Discord view."""
        server_admin, super_admin = _view_admin_state(interaction)
        kwargs = {
            "actor_user_id": interaction.user.id,
            "actor_guild_id": _guild_id(interaction.channel),
            "server_admin": server_admin,
            "super_admin": super_admin,
            "confirmed": True,
        }
        try:
            if action == "forget":
                bot.codex.forget_memory(
                    _memory_session_key(view),
                    record_id,
                    view.scope,
                    **kwargs,
                )
                view.remove_record(record_id)
                message = "Memory forgotten."
            else:
                bot.codex.edit_memory(
                    _memory_session_key(view),
                    record_id,
                    replacement or "",
                    view.scope,
                    **kwargs,
                )
                view.update_record(record_id, replacement or "")
                message = "Memory updated."
            if view.message is not None:
                await view.message.edit(embed=view.embed(), view=view._view())
            await view._notify_state_change()
            return True, message
        except (CodexAppServerError, OSError) as exc:
            return False, str(exc)

    return mutate


async def handle_memory_command(
    bot: Any,
    interaction: discord.Interaction,
    scope: str | app_commands.Choice[str] | None = None,
    search: str | None = None,
    record_id: str | None = None,
) -> None:
    """Authorize and display one private memory explorer."""
    selected_scope = _scope_value(scope)
    super_admin = _is_super_admin_user(getattr(interaction.user, "id", None))
    if selected_scope == "me":
        if not await _require_login(interaction):
            return
    elif selected_scope == "server":
        if not await _require_server_admin(
            interaction,
            message="Only a server administrator can inspect current server memory.",
        ):
            return
    elif not super_admin:
        await interaction.response.send_message(
            embed=_frontend_embed(
                "label:administrator_access_required",
                "Super Admin access required",
                "Only a Theia Super Admin can inspect everyone or a targeted scope.",
                channel=interaction.channel,
                user=interaction.user,
                color=discord_color("WARNING"),
            ),
            ephemeral=True,
        )
        return
    await interaction.response.defer(ephemeral=True)
    server_admin = _is_server_admin(interaction.user, interaction.channel)
    try:
        result = bot.codex.memory_view(
            session_key(interaction.channel, interaction.user.id),
            selected_scope,
            # Keep the complete authorized set in the view so its Search
            # modal can replace or clear a slash-command search.
            search=None,
            record_id=record_id,
            actor_user_id=interaction.user.id,
            actor_guild_id=_guild_id(interaction.channel),
            server_admin=server_admin,
            super_admin=super_admin,
        )
    except CodexAppServerError as exc:
        await _send_command_failure(interaction, "Memory unavailable", exc)
        return
    records = result.get("records")
    if not isinstance(records, list):
        records = (
            result.get("entries") if isinstance(result.get("entries"), list) else []
        )
    view = _MemoryView(
        records,
        character_name=str(result.get("character_name") or "Theia"),
        character_slug=str(result.get("character_slug") or "theia"),
        scope=selected_scope,
        owner_id=interaction.user.id,
        customizer=bot.customizations,
        guild_id=_guild_id(interaction.channel),
        channel_id=_channel_id(interaction.channel),
        total_entries=result.get("total_entries"),
        search_query=search or "",
    )
    view.on_mutation = _mutation_callback(bot, view)
    view.on_view_created = bot.register_view
    message = await interaction.followup.send(
        embed=view.embed(),
        view=view,
        ephemeral=True,
        wait=True,
    )
    view.message = message
    if message is not None:
        await bot.register_view(view, message)
