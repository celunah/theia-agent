"""Discord personality profile and prompt inspection commands."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import discord
from discord import app_commands

from ..colors import discord_color
from ..core import (
    DEFAULT_PERSONALITY_SCOPE,
    PERSONALITY_SCOPES,
    CodexAppServerError,
    _safe_error_reason,
)
from .embeds import _personality_prompt_embeds, _personality_summary_embed
from .support import (
    _frontend_embed,
    _frontend_label,
    _require_server_admin,
    _user_installable_command,
    session_key,
)


class _PersonalityPromptView(discord.ui.View):
    """Owner-locked, ephemeral navigation for prompt inspection pages."""

    def __init__(
        self,
        pages: list[discord.Embed],
        *,
        owner_id: int,
        channel: Any | None,
        user: discord.abc.User,
    ) -> None:
        super().__init__(timeout=900)
        self.pages = pages
        self.owner_id = owner_id
        self.index = 0
        previous = discord.ui.Button(
            label=_frontend_label(
                "label:previous_button", "Previous", channel=channel, user=user
            )[:80],
            style=discord.ButtonStyle.secondary,
        )
        following = discord.ui.Button(
            label=_frontend_label(
                "label:next_button", "Next", channel=channel, user=user
            )[:80],
            style=discord.ButtonStyle.secondary,
        )

        async def previous_callback(interaction: discord.Interaction) -> None:
            if not await self.interaction_check(interaction):
                return
            self.index = max(0, self.index - 1)
            self._sync_buttons(previous, following)
            await interaction.response.edit_message(
                embed=self.pages[self.index],
                view=self,
                allowed_mentions=discord.AllowedMentions.none(),
            )

        async def next_callback(interaction: discord.Interaction) -> None:
            if not await self.interaction_check(interaction):
                return
            self.index = min(len(self.pages) - 1, self.index + 1)
            self._sync_buttons(previous, following)
            await interaction.response.edit_message(
                embed=self.pages[self.index],
                view=self,
                allowed_mentions=discord.AllowedMentions.none(),
            )

        previous.callback = previous_callback
        following.callback = next_callback
        self.add_item(previous)
        self.add_item(following)
        self._sync_buttons(previous, following)

    def _sync_buttons(
        self, previous: discord.ui.Button, following: discord.ui.Button
    ) -> None:
        previous.disabled = self.index == 0
        following.disabled = self.index == len(self.pages) - 1

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.owner_id:
            return True
        await interaction.response.send_message(
            "Only the user who requested this prompt can navigate it.",
            ephemeral=True,
        )
        return False


def register_personality_commands(
    bot: Any,
    personality_autocomplete: Callable[..., Any],
) -> tuple[app_commands.Group, app_commands.Command, app_commands.Command]:
    """Register the personality group and return compatibility command handles."""
    group = _user_installable_command(
        app_commands.Group(
            name="personality",
            description="View and manage character profiles",
        )
    )

    @_user_installable_command
    @group.command(name="profile", description="View or manage a character profile")
    @app_commands.describe(
        file="A Markdown or plain-text personality prompt",
        name="The profile name, or `none` to clear the active personality",
        scope="Who should use this personality: me, server, or everyone",
    )
    @app_commands.choices(
        scope=[
            app_commands.Choice(name=scope, value=scope) for scope in PERSONALITY_SCOPES
        ]
    )
    @app_commands.autocomplete(name=personality_autocomplete)
    async def profile(
        interaction: discord.Interaction,
        file: discord.Attachment | None = None,
        name: str | None = None,
        scope: app_commands.Choice[str] | None = None,
    ) -> None:
        """Show, upload, select, or clear a personality profile."""
        await bot.presence.touch()
        await interaction.response.defer(ephemeral=True)
        selected_scope = (
            scope.value
            if isinstance(scope, app_commands.Choice)
            else str(scope or DEFAULT_PERSONALITY_SCOPE)
        )
        if file is None and name is None:
            key = session_key(interaction.channel, interaction.user.id)
            try:
                summary = await bot.codex.personality_summary(key)
                mood = bot.codex.mood_state(key)
            except CodexAppServerError as exc:
                await interaction.followup.send(
                    embed=_frontend_embed(
                        "command:personality",
                        "Personality unavailable",
                        _safe_error_reason(exc),
                        channel=interaction.channel,
                        user=interaction.user,
                        color=discord_color("WARNING"),
                    ),
                    ephemeral=True,
                )
                return
            await interaction.followup.send(
                embed=_personality_summary_embed(
                    summary,
                    mood,
                    bot.rich_presence.current_line,
                    channel=interaction.channel,
                    user=interaction.user,
                ),
                ephemeral=True,
            )
            return
        if selected_scope != "me" and not await _require_server_admin(
            interaction,
            message=(
                "Only server administrators can change server or everyone "
                "personalities."
            ),
        ):
            return
        try:
            selected = await bot.codex.configure_personality(
                session_key(interaction.channel, interaction.user.id),
                name=name,
                attachment=file,
                scope=selected_scope,
                actor_user_id=interaction.user.id,
                guild_id=getattr(getattr(interaction, "guild", None), "id", None),
            )
        except CodexAppServerError as exc:
            await interaction.followup.send(
                embed=_frontend_embed(
                    "command:personality",
                    "Personality unavailable",
                    _safe_error_reason(exc),
                    channel=interaction.channel,
                    user=interaction.user,
                    color=discord_color("WARNING"),
                ),
                ephemeral=True,
            )
            return
        if selected is None:
            description = (
                f"The active Codex personality has been cleared for `{selected_scope}`."
            )
            title = "Personality cleared"
        elif file is not None:
            description = (
                f"Personality `{selected}` was uploaded and is now active for "
                f"`{selected_scope}`."
            )
            title = "Personality uploaded"
        else:
            description = (
                f"Personality `{selected}` is now active for `{selected_scope}`."
            )
            title = "Personality selected"
        await interaction.followup.send(
            embed=_frontend_embed(
                "command:personality",
                title,
                description,
                channel=interaction.channel,
                user=interaction.user,
                context={
                    "personality": selected or "none",
                    "personality_scope": selected_scope,
                },
                color=discord_color("HEALTHY"),
            ),
            ephemeral=True,
        )

    @_user_installable_command
    @group.command(name="prompt", description="Inspect the current prompt layers")
    async def prompt(interaction: discord.Interaction) -> None:
        """Show the invoking user's active presentation prompt components."""
        await bot.presence.touch()
        await interaction.response.defer(ephemeral=True)
        try:
            parts = bot.codex.personality_prompt_parts(
                session_key(interaction.channel, interaction.user.id)
            )
        except CodexAppServerError as exc:
            await interaction.followup.send(
                embed=_frontend_embed(
                    "command:personality",
                    "Prompt unavailable",
                    _safe_error_reason(exc),
                    channel=interaction.channel,
                    user=interaction.user,
                    color=discord_color("WARNING"),
                ),
                ephemeral=True,
            )
            return
        pages = _personality_prompt_embeds(
            parts,
            channel=interaction.channel,
            user=interaction.user,
        )
        view = (
            _PersonalityPromptView(
                pages,
                owner_id=interaction.user.id,
                channel=interaction.channel,
                user=interaction.user,
            )
            if len(pages) > 1
            else None
        )
        if view is None:
            await interaction.followup.send(
                embed=pages[0],
                ephemeral=True,
                allowed_mentions=discord.AllowedMentions.none(),
            )
        else:
            await interaction.followup.send(
                embed=pages[0],
                view=view,
                ephemeral=True,
                allowed_mentions=discord.AllowedMentions.none(),
            )

    bot.tree.add_command(group)
    return group, profile, prompt
