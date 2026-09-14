"""Owner-locked controls for the ephemeral usage snapshot."""

from __future__ import annotations

from typing import Any

import discord

from ..ui import _PersistentViewMixin, _check_interaction_owner
from .embeds import _usage_details_embed, _usage_embed
from .support import _frontend_label


def _safe_usage_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    """Persist only numeric usage data needed to restore the two-button view."""
    snapshot: dict[str, Any] = {"date": str(result.get("date") or "")[:32]}
    for key in ("summary", "exact", "estimate", "detailed"):
        value = result.get(key)
        if isinstance(value, dict):
            snapshot[key] = value
    return snapshot


class _UsageView(_PersistentViewMixin, discord.ui.View):
    """Toggle compact and detailed usage without creating another message."""

    def __init__(
        self,
        result: dict[str, Any],
        *,
        owner_id: int | None,
        channel: Any | None = None,
        user: discord.abc.User | None = None,
        token: str | None = None,
        recovered: bool = False,
    ) -> None:
        self._init_persistence("usage", token, recovered=recovered)
        super().__init__(timeout=None if recovered else 900)
        self.result = _safe_usage_snapshot(result)
        self.owner_id = owner_id
        self.channel = channel
        self.user = user
        self.showing_details = False
        self.toggle_button = discord.ui.Button(
            label=self._label(),
            style=discord.ButtonStyle.primary,
            custom_id=self._custom_id("details"),
        )
        self.toggle_button.callback = self._toggle
        self.add_item(self.toggle_button)

    def _label(self) -> str:
        target = (
            "label:usage_hide_details"
            if self.showing_details
            else "label:usage_show_details"
        )
        default = "Hide Details" if self.showing_details else "Show Details"
        return _frontend_label(target, default, channel=self.channel, user=self.user)

    async def _toggle(self, interaction: discord.Interaction) -> None:
        if not await self.interaction_check(interaction):
            return
        self.showing_details = not self.showing_details
        self.toggle_button.label = self._label()
        embed = (
            _usage_details_embed(self.result, channel=self.channel, user=self.user)
            if self.showing_details
            else _usage_embed(self.result, channel=self.channel, user=self.user)
        )
        await interaction.response.edit_message(embed=embed, view=self)
        await self._notify_state_change()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return await _check_interaction_owner(interaction, self.owner_id)

    def persistence_data(self) -> dict[str, Any]:
        return {
            "user_id": self.owner_id,
            "result": self.result,
            "showing_details": self.showing_details,
        }
