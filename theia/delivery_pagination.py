"""Discord pagination views and bounded response delivery."""

from __future__ import annotations

import contextlib
import io
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

import discord

from .audio import AudioOutput
from .core import _render_frontend_label
from .delivery_text import _split_pages
from .ui import _PersistentViewMixin

SendMessage = Callable[..., Awaitable[Any]]
ViewRegistrar = Callable[[discord.ui.View, Any], Awaitable[None]]


class _PaginatorView(_PersistentViewMixin, discord.ui.View):
    def __init__(
        self,
        pages: list[str],
        *,
        owner_id: int | None,
        customizer: Any | None = None,
        guild_id: int | None = None,
        timeout: float = 900,
        token: str | None = None,
        recovered: bool = False,
        index: int = 0,
    ) -> None:
        self._init_persistence("paginator", token, recovered=recovered)
        super().__init__(timeout=None if recovered else timeout)
        self.pages = pages
        self.owner_id = owner_id
        self.customizer = customizer
        self.guild_id = guild_id
        self.index = max(0, min(index, max(0, len(pages) - 1)))
        self.message: discord.Message | discord.WebhookMessage | None = None
        previous = discord.ui.Button(
            label=_render_frontend_label(
                customizer,
                guild_id,
                "label:previous_button",
                "Previous",
            ),
            style=discord.ButtonStyle.secondary,
            custom_id=self._custom_id("previous"),
        )
        following = discord.ui.Button(
            label=_render_frontend_label(
                customizer,
                guild_id,
                "label:next_button",
                "Next",
            ),
            style=discord.ButtonStyle.secondary,
            custom_id=self._custom_id("next"),
        )

        async def previous_callback(interaction: discord.Interaction) -> None:
            if not await self.interaction_check(interaction):
                return
            self.index = max(0, self.index - 1)
            await interaction.response.edit_message(
                content=self.content(), view=self._view()
            )
            await self._notify_state_change()

        async def next_callback(interaction: discord.Interaction) -> None:
            if not await self.interaction_check(interaction):
                return
            self.index = min(len(self.pages) - 1, self.index + 1)
            await interaction.response.edit_message(
                content=self.content(), view=self._view()
            )
            await self._notify_state_change()

        previous.callback = previous_callback
        following.callback = next_callback
        self.add_item(previous)
        self.add_item(following)
        self._sync_buttons()

    def _view(self) -> _PaginatorView:
        self._sync_buttons()
        return self

    def _sync_buttons(self) -> None:
        buttons = [
            child for child in self.children if isinstance(child, discord.ui.Button)
        ]
        if len(buttons) == 2:
            buttons[0].disabled = self.index == 0
            buttons[1].disabled = self.index == len(self.pages) - 1

    def content(self) -> str:
        return self.pages[self.index]

    def persistence_data(self) -> dict[str, Any]:
        return {
            "pages": self.pages,
            "index": self.index,
            "owner_id": self.owner_id,
            "guild_id": self.guild_id,
        }

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if self.owner_id is not None and interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "Only the user who requested this response can navigate it.",
                ephemeral=True,
            )
            return False
        return True

    async def handle_reaction(
        self, reaction: discord.Reaction, user: discord.abc.User
    ) -> None:
        if self.owner_id is not None and user.id != self.owner_id:
            return
        if reaction.emoji == "◀️":
            self.index = max(0, self.index - 1)
        elif reaction.emoji == "▶️":
            self.index = min(len(self.pages) - 1, self.index + 1)
        else:
            return
        if self.message is not None:
            with contextlib.suppress(discord.DiscordException):
                await self.message.edit(content=self.content())
        with contextlib.suppress(discord.DiscordException):
            await reaction.remove(user)

    async def on_timeout(self) -> None:
        if self.message is not None:
            _reaction_paginators.pop(self.message.id, None)


_reaction_paginators: dict[int, _PaginatorView] = {}


async def send_paginated(
    send: SendMessage,
    response: str,
    *,
    title: str = "Codex",
    color: discord.Color | None = None,
    owner_id: int | None = None,
    speech: Iterable[AudioOutput] = (),
    customizer: Any | None = None,
    guild_id: int | None = None,
    on_view_created: ViewRegistrar | None = None,
    **kwargs: Any,
) -> Any:
    """Send a response using components, reactions, or message splitting as fallback."""
    del title, color
    pages = _split_pages(response)
    speech_outputs = tuple(speech)
    view = (
        _PaginatorView(
            pages,
            owner_id=owner_id,
            customizer=customizer,
            guild_id=guild_id,
        )
        if len(pages) > 1
        else None
    )

    def send_kwargs(
        content: str,
        *,
        page_view: _PaginatorView | None,
        include_speech: bool,
    ) -> dict[str, Any]:
        values = dict(kwargs)
        values.pop("view", None)
        values.update(
            content=content,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        if page_view is not None:
            values["view"] = page_view
        if include_speech and speech_outputs:
            values["files"] = [
                discord.File(io.BytesIO(output.data), filename=output.filename)
                for output in speech_outputs
            ]
        return values

    try:
        message = await send(
            **send_kwargs(pages[0], page_view=view, include_speech=bool(speech_outputs))
        )
        if view is not None and on_view_created is not None:
            with contextlib.suppress(Exception):
                await on_view_created(view, message)
        if view is not None:
            view.message = message
        return message
    except (discord.DiscordException, TypeError):
        message = await send(
            **send_kwargs(pages[0], page_view=None, include_speech=bool(speech_outputs))
        )
        if view is None:
            return message
        try:
            await message.add_reaction("◀️")
            await message.add_reaction("▶️")
            if getattr(message, "id", None) is not None:
                view.message = message
                _reaction_paginators[message.id] = view
            return message
        except (discord.DiscordException, AttributeError):
            for page in pages[1:]:
                await send(**send_kwargs(page, page_view=None, include_speech=False))
            return message
