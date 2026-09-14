"""Translate Codex progress and final responses into resilient Discord messages."""

import asyncio
import contextlib
import io
import time
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

import discord

from .audio import AudioOutput
from .core import (
    _codex_logger,
    _command_embed,
    _is_tool_item,
    _render_frontend_label,
    _safe_error_reason,
    _safe_intermediate_text,
    _subtext,
    _truncate,
)
from .customization import CustomizationError
from .ui import (
    _PersistentViewMixin,
    _PromptModal,
    _check_interaction_owner,
)

SendMessage = Callable[..., Awaitable[Any]]
SpeakText = Callable[[str], Awaitable[None]]
ImagePathResolver = Callable[[dict[str, Any]], Path | None]
ImageAction = Callable[
    [discord.Interaction, str, tuple[Path, ...], Any], Awaitable[None]
]
ViewRegistrar = Callable[[discord.ui.View, Any], Awaitable[None]]
INTERMEDIATE_STATUS_LIMIT = 1990
logger = _codex_logger()


def _split_pages(text: str, limit: int = 1900) -> list[str]:
    text = text or ""
    if not text:
        return [""]
    pages: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n\n", 0, limit + 1)
        if split_at < limit // 2:
            split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at <= 0:
            split_at = limit
        pages.append(remaining[:split_at])
        remaining = remaining[split_at:]
    pages.append(remaining)
    return pages


def _format_thought_duration(seconds: float) -> str:
    elapsed = max(0, int(seconds))
    if elapsed < 60:
        unit = "second" if elapsed == 1 else "seconds"
        return f"Thought for {elapsed} {unit}"
    minutes, remainder = divmod(elapsed, 60)
    minute_unit = "minute" if minutes == 1 else "minutes"
    if remainder == 0:
        return f"Thought for {minutes} {minute_unit}"
    second_unit = "second" if remainder == 1 else "seconds"
    return f"Thought for {minutes} {minute_unit} and {remainder} {second_unit}"


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

    def _view(self) -> "_PaginatorView":
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


MemoryMutation = Callable[
    [discord.Interaction, str, str, str | None], Awaitable[tuple[bool, str]]
]


class _MemoryConfirmationView(_PersistentViewMixin, discord.ui.View):
    def __init__(
        self,
        owner_id: int | None,
        *,
        action: str,
        record_id: str,
        replacement: str | None,
        scope: str,
        channel_id: int | None,
        on_mutation: MemoryMutation | None,
        customizer: Any | None = None,
        guild_id: int | None = None,
        on_view_created: ViewRegistrar | None = None,
        timeout: float = 300,
        token: str | None = None,
        recovered: bool = False,
    ) -> None:
        self._init_persistence("memory-confirm", token, recovered=recovered)
        super().__init__(timeout=None if recovered else timeout)
        self.owner_id = owner_id
        self.action = action
        self.record_id = record_id
        self.replacement = replacement
        self.scope = scope
        self.channel_id = channel_id
        self.on_mutation = on_mutation
        self.customizer = customizer
        self.guild_id = guild_id
        self.on_view_created = on_view_created
        confirm = discord.ui.Button(
            label=_render_frontend_label(
                customizer,
                guild_id,
                "label:memory_confirm",
                "Confirm",
            ),
            style=discord.ButtonStyle.danger,
            custom_id=self._custom_id("confirm"),
        )
        cancel = discord.ui.Button(
            label=_render_frontend_label(
                customizer,
                guild_id,
                "label:memory_cancel",
                "Cancel",
            ),
            style=discord.ButtonStyle.secondary,
            custom_id=self._custom_id("cancel"),
        )

        async def confirm_callback(interaction: discord.Interaction) -> None:
            if not await self.interaction_check(interaction):
                return
            await interaction.response.defer(ephemeral=True)
            if self.on_mutation is None:
                message = "This memory action is unavailable."
            else:
                try:
                    _, message = await self.on_mutation(
                        interaction,
                        self.action,
                        self.record_id,
                        self.replacement,
                    )
                except Exception:  # noqa: BLE001 - mutation failure is user-safe
                    message = "The memory action failed safely; nothing was changed."
            await interaction.followup.send(message, ephemeral=True)
            for child in self.children:
                if isinstance(child, discord.ui.Button):
                    child.disabled = True
            await self._notify_state_change()
            self.stop()

        async def cancel_callback(interaction: discord.Interaction) -> None:
            if not await self.interaction_check(interaction):
                return
            await interaction.response.edit_message(
                content="The memory action was cancelled.",
                view=self,
            )
            self.stop()

        confirm.callback = confirm_callback
        cancel.callback = cancel_callback
        self.add_item(confirm)
        self.add_item(cancel)

    def persistence_data(self) -> dict[str, Any]:
        return {
            "user_id": self.owner_id,
            "action": self.action,
            "record_id": self.record_id,
            "replacement": self.replacement,
            "scope": self.scope,
            "channel_id": self.channel_id,
            "guild_id": self.guild_id,
        }

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return await _check_interaction_owner(interaction, self.owner_id)


class _MemorySearchModal(discord.ui.Modal):
    def __init__(self, view: "_MemoryView") -> None:
        super().__init__(
            title=_truncate(
                _render_frontend_label(
                    view.customizer,
                    view.guild_id,
                    "label:input_modal_title",
                    "Search memories",
                ),
                45,
            ),
            custom_id=f"theia:memory-search:{view.persistence_token}",
        )
        self.view = view
        self.query = discord.ui.TextInput(
            label=_render_frontend_label(
                view.customizer,
                view.guild_id,
                "label:memory_search_query",
                "Search",
            ),
            placeholder="Leave empty to show every record.",
            default=view.search_query,
            required=False,
            max_length=80,
        )
        self.add_item(self.query)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if not await self.view.interaction_check(interaction):
            return
        self.view.search_query = _safe_intermediate_text(str(self.query), 80).strip()
        self.view.index = 0
        await interaction.response.edit_message(
            embed=self.view.embed(),
            view=self.view._view(),
        )
        await self.view._notify_state_change()


class _MemoryEditModal(discord.ui.Modal):
    def __init__(self, view: "_MemoryView", record: dict[str, Any]) -> None:
        super().__init__(
            title=_truncate(
                _render_frontend_label(
                    view.customizer,
                    view.guild_id,
                    "label:input_modal_title",
                    "Edit memory",
                ),
                45,
            ),
            custom_id=f"theia:memory-edit:{view.persistence_token}",
        )
        self.view = view
        self.record_id = str(record.get("record_id") or "")
        self.value = discord.ui.TextInput(
            label=_render_frontend_label(
                view.customizer,
                view.guild_id,
                "label:memory_edit_text",
                "Memory",
            ),
            default=_safe_intermediate_text(record.get("text"), 3500),
            style=discord.TextStyle.paragraph,
            required=True,
            max_length=3500,
        )
        self.add_item(self.value)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if not await self.view.interaction_check(interaction):
            return
        await self.view.open_confirmation(
            interaction,
            action="edit",
            record_id=self.record_id,
            replacement=str(self.value),
        )


class _MemoryView(_PersistentViewMixin, discord.ui.View):
    def __init__(
        self,
        records: list[dict[str, Any]] | list[str],
        *,
        character_name: str,
        character_slug: str = "theia",
        scope: str = "me",
        owner_id: int | None,
        total_entries: int | None = None,
        customizer: Any | None = None,
        guild_id: int | None = None,
        channel_id: int | None = None,
        on_mutation: MemoryMutation | None = None,
        on_view_created: ViewRegistrar | None = None,
        timeout: float = 900,
        token: str | None = None,
        recovered: bool = False,
        index: int = 0,
        search_query: str = "",
    ) -> None:
        self._init_persistence("memory", token, recovered=recovered)
        super().__init__(timeout=None if recovered else timeout)
        self.records = self._normalize_records(records)
        self.character_name = character_name or "Theia"
        self.character_slug = character_slug or "theia"
        self.scope = scope
        self.owner_id = owner_id
        self.customizer = customizer
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.on_mutation = on_mutation
        self.on_view_created = on_view_created
        self.search_query = _safe_intermediate_text(search_query, 80).strip()
        self.index = max(0, min(index, max(0, len(self._filtered_records()) - 1)))
        self.total_entries = (
            total_entries
            if isinstance(total_entries, int) and not isinstance(total_entries, bool)
            else len(self.records)
        )
        self.message: discord.Message | discord.WebhookMessage | None = None
        self._buttons: dict[str, discord.ui.Button] = {}
        self._add_button(
            "previous",
            _render_frontend_label(
                customizer, guild_id, "label:previous_button", "Previous"
            ),
            discord.ButtonStyle.secondary,
            self._previous,
        )
        self._add_button(
            "next",
            _render_frontend_label(customizer, guild_id, "label:next_button", "Next"),
            discord.ButtonStyle.secondary,
            self._next,
        )
        self._add_button(
            "search",
            _render_frontend_label(
                customizer, guild_id, "label:memory_search", "Search"
            ),
            discord.ButtonStyle.secondary,
            self._search,
        )
        self._add_button(
            "forget",
            _render_frontend_label(
                customizer, guild_id, "label:memory_forget", "Forget"
            ),
            discord.ButtonStyle.danger,
            self._forget,
        )
        self._add_button(
            "edit",
            _render_frontend_label(customizer, guild_id, "label:memory_edit", "Edit"),
            discord.ButtonStyle.primary,
            self._edit,
        )
        self._sync_buttons()

    @staticmethod
    def _normalize_records(
        records: list[dict[str, Any]] | list[str],
    ) -> list[dict[str, Any]]:
        allowed = {
            "record_id",
            "text",
            "scope",
            "character_name",
            "character_slug",
            "source_file",
            "source_category",
            "created_at",
            "updated_at",
            "confidence",
            "origin",
            "display_metadata",
        }
        normalized: list[dict[str, Any]] = []
        for index, value in enumerate(records or []):
            if isinstance(value, dict):
                text = _safe_intermediate_text(value.get("text"), 3500)
                record_id = _safe_intermediate_text(value.get("record_id"), 64)
                if not text or not record_id:
                    continue
                item = {key: value[key] for key in allowed if key in value}
                item["text"] = text
                item["record_id"] = record_id
                for key in (
                    "scope",
                    "character_name",
                    "character_slug",
                    "source_file",
                    "source_category",
                    "created_at",
                    "updated_at",
                    "origin",
                ):
                    if key in item:
                        item[key] = _safe_intermediate_text(item[key], 160)
                if isinstance(item.get("confidence"), (int, float)):
                    item["confidence"] = max(0.0, min(1.0, float(item["confidence"])))
                else:
                    item.pop("confidence", None)
                metadata = item.get("display_metadata")
                if isinstance(metadata, dict):
                    item["display_metadata"] = {
                        key: _safe_intermediate_text(metadata[key], 80)
                        for key in ("source", "scope", "updated")
                        if key in metadata
                    }
                else:
                    item.pop("display_metadata", None)
                normalized.append(item)
            elif isinstance(value, str) and value.strip():
                normalized.append(
                    {
                        "record_id": f"legacy-{index}",
                        "text": _safe_intermediate_text(value, 3500),
                        "scope": "legacy/unscoped",
                        "source_category": "legacy memory",
                        "origin": "legacy memory",
                        "display_metadata": {
                            "source": "legacy memory",
                            "scope": "legacy/unscoped",
                            "updated": "unknown",
                        },
                    }
                )
        return normalized

    def _add_button(
        self,
        action: str,
        label: str,
        style: discord.ButtonStyle,
        callback: Callable[[discord.Interaction], Awaitable[None]],
    ) -> None:
        button = discord.ui.Button(
            label=label,
            style=style,
            custom_id=self._custom_id(action),
        )
        button.callback = callback  # type: ignore[assignment]
        self._buttons[action] = button
        self.add_item(button)

    def _filtered_records(self) -> list[dict[str, Any]]:
        query = self.search_query.casefold()
        if not query:
            return list(self.records)
        return [
            record
            for record in self.records
            if query
            in f"{record.get('text', '')} {record.get('source_category', '')}".casefold()
        ]

    def _current_record(self) -> dict[str, Any] | None:
        records = self._filtered_records()
        return records[self.index] if records and self.index < len(records) else None

    @property
    def entries(self) -> list[str]:
        return [str(record.get("text") or "") for record in self.records]

    def _view(self) -> "_MemoryView":
        self._sync_buttons()
        return self

    def _sync_buttons(self) -> None:
        filtered = self._filtered_records()
        self.index = max(0, min(self.index, max(0, len(filtered) - 1)))
        self._buttons["previous"].disabled = not filtered or self.index == 0
        self._buttons["next"].disabled = not filtered or self.index == len(filtered) - 1
        disabled = not bool(filtered)
        self._buttons["forget"].disabled = disabled
        self._buttons["edit"].disabled = disabled

    def embed(self) -> discord.Embed:
        character_name = _safe_intermediate_text(self.character_name, 120) or "Theia"
        filtered = self._filtered_records()
        record = self._current_record()
        context = {
            "character_name": character_name,
            "character_slug": _safe_intermediate_text(self.character_slug, 80)
            or "theia",
            "count": len(filtered),
            "page": self.index + 1,
            "pages": max(1, len(filtered)),
            "scope": _safe_intermediate_text(self.scope, 80),
        }
        total_label = _render_frontend_label(
            self.customizer,
            self.guild_id,
            "label:memory_total_entries",
            "Total entries",
            context=context,
        )
        if record is None:
            body = "No memories match this view."
        else:
            body = _safe_intermediate_text(record.get("text"), 3500)
            metadata = record.get("display_metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            source = _safe_intermediate_text(
                metadata.get("source")
                or record.get("source_category")
                or record.get("origin"),
                60,
            ).replace("_", " ")
            record_scope = _safe_intermediate_text(
                metadata.get("scope") or record.get("scope"), 60
            )
            updated = _safe_intermediate_text(metadata.get("updated"), 40) or "unknown"
            provenance = (
                f"Source: {source or 'memory'} | Scope: {record_scope or 'unknown'} "
                f"| Updated: {updated}"
            )
            body = f"{body or 'Memory entry unavailable.'}\n\n{provenance}"
            record_id = _safe_intermediate_text(record.get("record_id"), 24)
            if record_id:
                body += f"\nRecord: {record_id}"
        if self.search_query:
            body = f"Search: {self.search_query}\n\n{body}"
        description = f"{total_label}: {len(filtered)}\n\n{body}"
        embed = _command_embed(
            f"{character_name}'s Memory",
            description,
            target="command:memory",
            guild_id=self.guild_id,
            customizer=self.customizer,
            context=context,
        )
        if len(filtered) > 1 or self.search_query:
            embed.set_footer(
                text=_render_frontend_label(
                    self.customizer,
                    self.guild_id,
                    "label:memory_page",
                    f"Page {self.index + 1} of {max(1, len(filtered))}",
                    context=context,
                )
            )
        return embed

    async def _previous(self, interaction: discord.Interaction) -> None:
        if not await self.interaction_check(interaction):
            return
        self.index = max(0, self.index - 1)
        await interaction.response.edit_message(embed=self.embed(), view=self._view())
        await self._notify_state_change()

    async def _next(self, interaction: discord.Interaction) -> None:
        if not await self.interaction_check(interaction):
            return
        self.index += 1
        self._sync_buttons()
        await interaction.response.edit_message(embed=self.embed(), view=self._view())
        await self._notify_state_change()

    async def _search(self, interaction: discord.Interaction) -> None:
        if await self.interaction_check(interaction):
            await interaction.response.send_modal(_MemorySearchModal(self))

    async def _forget(self, interaction: discord.Interaction) -> None:
        if not await self.interaction_check(interaction):
            return
        record = self._current_record()
        if record is not None:
            await self.open_confirmation(
                interaction,
                action="forget",
                record_id=str(record.get("record_id") or ""),
            )

    async def _edit(self, interaction: discord.Interaction) -> None:
        if not await self.interaction_check(interaction):
            return
        record = self._current_record()
        if record is not None:
            await interaction.response.send_modal(_MemoryEditModal(self, record))

    async def open_confirmation(
        self,
        interaction: discord.Interaction,
        *,
        action: str,
        record_id: str,
        replacement: str | None = None,
    ) -> None:
        confirmation = _MemoryConfirmationView(
            self.owner_id,
            action=action,
            record_id=record_id,
            replacement=replacement,
            scope=self.scope,
            channel_id=self.channel_id,
            on_mutation=self.on_mutation,
            customizer=self.customizer,
            guild_id=self.guild_id,
            on_view_created=self.on_view_created,
            recovered=self.recovered,
        )
        await interaction.response.send_message(
            content=(
                "Confirm forgetting this memory?"
                if action == "forget"
                else "Confirm replacing this memory?"
            ),
            view=confirmation,
            ephemeral=True,
        )
        if self.on_view_created is not None:
            try:
                await self.on_view_created(
                    confirmation, await interaction.original_response()
                )
            except (AttributeError, discord.DiscordException):
                pass

    def remove_record(self, record_id: str) -> None:
        self.records = [
            record for record in self.records if record.get("record_id") != record_id
        ]
        self._sync_buttons()

    def update_record(self, record_id: str, replacement: str) -> None:
        for record in self.records:
            if record.get("record_id") == record_id:
                record["text"] = replacement
                break
        self._sync_buttons()

    def persistence_data(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "entries": self.entries,
            "character_name": self.character_name,
            "character_slug": self.character_slug,
            "scope": self.scope,
            "user_id": self.owner_id,
            "guild_id": self.guild_id,
            "channel_id": self.channel_id,
            "index": self.index,
            "total_entries": self.total_entries,
            "search_query": self.search_query,
        }

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return await _check_interaction_owner(interaction, self.owner_id)


_reaction_paginators: dict[int, _PaginatorView] = {}


class _ImageResultView(_PersistentViewMixin, discord.ui.View):
    """Owner-only controls for a generated image attachment."""

    def __init__(
        self,
        user_id: int | None,
        image_paths: tuple[Path, ...],
        *,
        on_action: ImageAction,
        channel: Any | None = None,
        customizer: Any | None = None,
        guild_id: int | None = None,
        timeout: float = 900,
        token: str | None = None,
        recovered: bool = False,
        message_id: int | None = None,
    ) -> None:
        self._init_persistence("image", token, recovered=recovered)
        super().__init__(timeout=None if recovered else timeout)
        self.user_id = user_id
        self.image_paths = image_paths
        self.on_action = on_action
        self.channel = channel
        self.customizer = customizer
        self.guild_id = guild_id
        self.message: Any | None = None
        self.message_id = message_id

        follow_up = discord.ui.Button(
            label=_render_frontend_label(
                customizer,
                guild_id,
                "label:image_follow_up",
                "Follow up",
            ),
            style=discord.ButtonStyle.primary,
            custom_id=self._custom_id("follow-up"),
        )

        async def follow_up_callback(interaction: discord.Interaction) -> None:
            if await self.interaction_check(interaction):
                await interaction.response.send_modal(
                    _PromptModal(
                        self.user_id,
                        on_submit=self._follow_up_submit,
                        channel=self.channel,
                        customizer=self.customizer,
                        title="Image follow-up",
                        placeholder="Describe what to do with this image next.",
                    )
                )

        follow_up.callback = follow_up_callback
        self.add_item(follow_up)

    def persistence_data(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "image_paths": [str(path) for path in self.image_paths],
            "guild_id": self.guild_id,
        }

    async def _follow_up_submit(
        self,
        interaction: discord.Interaction,
        prompt: str,
    ) -> None:
        await self.on_action(interaction, prompt, self.image_paths, self)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Allow image controls only to the user who generated the image."""
        return await _check_interaction_owner(interaction, self.user_id)


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


class _ResponseDelivery:
    def __init__(
        self,
        send: SendMessage,
        kwargs: dict[str, Any],
        *,
        owner_id: int | None = None,
        channel: Any | None = None,
        speak_text: SpeakText | None = None,
        customizer: Any | None = None,
        guild_id: int | None = None,
        context: dict[str, Any] | None = None,
        image_path_resolver: ImagePathResolver | None = None,
        on_image_action: ImageAction | None = None,
        on_view_created: ViewRegistrar | None = None,
        image_message: Any | None = None,
        image_view: _ImageResultView | None = None,
        existing_image_paths: Iterable[Path] = (),
    ) -> None:
        self.send = send
        self.kwargs = kwargs
        self.owner_id = owner_id
        self.channel = channel
        self.speak_text = speak_text
        self.customizer = customizer
        self.guild_id = guild_id
        self.context = dict(context or {})
        self.image_path_resolver = image_path_resolver
        self.on_image_action = on_image_action
        self.on_view_created = on_view_created
        self.image_message = image_message
        self.image_view = image_view
        self.existing_image_paths = tuple(
            path for path in existing_image_paths if isinstance(path, Path)
        )
        self._image_paths: list[Path] = []
        self._image_item_ids: set[str] = set()
        self.status_message: discord.Message | discord.WebhookMessage | None = None
        self.last_edit = 0.0
        self.thought_started_at: float | None = None
        self.lock = asyncio.Lock()

    @property
    def image_paths(self) -> tuple[Path, ...]:
        """Return generated images collected during this turn."""
        return tuple(self._image_paths)

    def _remember_image(self, item: dict[str, Any]) -> None:
        if self.image_path_resolver is None:
            return
        item_id = str(item.get("id") or "")
        if item_id and item_id in self._image_item_ids:
            return
        try:
            path = self.image_path_resolver(item)
        except (OSError, TypeError, ValueError) as exc:
            logger.info(
                "Could not prepare a generated image for Discord (error=%s)",
                type(exc).__name__,
            )
            return
        if path is None:
            return
        if item_id:
            self._image_item_ids.add(item_id)
        if path not in self._image_paths:
            self._image_paths.append(path)

    async def start(self) -> None:
        """Reserve the delivery lifecycle hook for future initial-status behavior."""
        return

    async def on_event(self, event: str, payload: dict[str, Any]) -> None:
        """Translate selected Codex events into compact Discord progress updates."""
        async with self.lock:
            if event == "thread_opening":
                message = _safe_intermediate_text(
                    payload.get("text"), INTERMEDIATE_STATUS_LIMIT
                )
                if message:
                    try:
                        await self.send(
                            content=_subtext(message),
                            allowed_mentions=discord.AllowedMentions.none(),
                            **self.kwargs,
                        )
                    except discord.DiscordException:
                        return
                    if self.speak_text is not None:
                        with contextlib.suppress(Exception):
                            await self.speak_text(message)
                return
            if event == "agent_message":
                # App-server agent-message events are emitted for every text
                # delta. Wait for item_completed so Discord receives one full
                # preamble/intermediate instead of a visibly streaming status.
                return
            if event in {"item_started", "tool_activity"}:
                if event == "tool_activity" or _is_tool_item(payload):
                    await self._set_status("Thinking", "Thinking")
                return
            if event == "item_completed":
                if str(payload.get("type") or "").casefold() == "imagegeneration":
                    self._remember_image(payload)
                if _is_tool_item(payload):
                    await self._set_status("Thinking", "Thinking")
                elif (
                    payload.get("type") == "agentMessage"
                    and payload.get("phase") != "final_answer"
                ):
                    message = _safe_intermediate_text(
                        payload.get("text"), INTERMEDIATE_STATUS_LIMIT
                    )
                    if message:
                        await self._set_status("Intermediate", message, force=True)
                        if self.speak_text is not None:
                            with contextlib.suppress(Exception):
                                await self.speak_text(message)
                return
            if event == "verified_change":
                for status in payload.get("statuses") or []:
                    if status in {
                        "Memory created",
                        "Memory updated",
                        "Skill created",
                        "Skill updated",
                    }:
                        await self._set_status(status, "Thinking", force=True)
                return

    def _status_text(self, title: str, description: str) -> str:
        targets = {
            "Thinking": "label:thinking",
            "Intermediate": "label:intermediate",
            "Memory created": "label:memory_created",
            "Memory updated": "label:memory_updated",
            "Skill created": "label:skill_created",
            "Skill updated": "label:skill_updated",
        }
        target = targets.get(title)
        if self.customizer is None or target is None:
            return description if title == "Intermediate" else title
        context = dict(self.context)
        context.update({"status": title, "text": description})
        try:
            if title == "Intermediate":
                value = self.customizer.render(
                    self.guild_id,
                    target,
                    "content",
                    description,
                    context=context,
                )
            else:
                label = getattr(self.customizer, "label", None)
                value = (
                    label(  # pylint: disable=not-callable
                        self.guild_id,
                        target,
                        title,
                        context=context,
                    )
                    if callable(label)
                    else self.customizer.render(
                        self.guild_id,
                        target,
                        "label",
                        title,
                        context=context,
                    )
                )
            return str(value or (description if title == "Intermediate" else title))
        except CustomizationError:
            return description if title == "Intermediate" else title

    async def _set_status(
        self, title: str, description: str, *, force: bool = False
    ) -> None:
        description = description or title
        now = time.monotonic()
        status = self._status_text(title, description)
        if title == "Thinking" and self.thought_started_at is None:
            self.thought_started_at = now
        if not force and now - self.last_edit < 0.8:
            return
        content = _subtext(status)
        try:
            if self.status_message is None:
                self.status_message = await self.send(
                    content=content,
                    allowed_mentions=discord.AllowedMentions.none(),
                    **self.kwargs,
                )
            else:
                await self.status_message.edit(content=content)
            self.last_edit = now
        except discord.DiscordException:
            return

    async def finalize(
        self,
        response: str,
        *,
        failed: bool = False,
        error_reason: str | None = None,
        speech: Iterable[AudioOutput] = (),
        image_paths: Iterable[Path] = (),
        on_image_action: ImageAction | None = None,
    ) -> None:
        """Replace progress status with a final response or image-view update."""
        async with self.lock:
            if self.status_message is not None and self.thought_started_at is not None:
                thought = _format_thought_duration(
                    time.monotonic() - self.thought_started_at
                )
                if self.customizer is not None:
                    try:
                        label = getattr(self.customizer, "label", None)
                        thought = (
                            label(
                                self.guild_id,
                                "label:thought_duration",
                                thought,
                                context={
                                    **self.context,
                                    "duration": thought.removeprefix("Thought for "),
                                    "status": "Thought duration",
                                    "text": thought,
                                },
                            )
                            if callable(label)
                            else self.customizer.render(
                                self.guild_id,
                                "label:thought_duration",
                                "label",
                                thought,
                                context={
                                    **self.context,
                                    "duration": thought.removeprefix("Thought for "),
                                    "status": "Thought duration",
                                    "text": thought,
                                },
                            )
                        )
                    except CustomizationError as exc:
                        logger.debug(
                            "Could not render thought status with frontend preferences "
                            "(error=%s)",
                            type(exc).__name__,
                        )
                with contextlib.suppress(discord.DiscordException, AttributeError):
                    await self.status_message.edit(content=_subtext(thought))
        paths = tuple(
            dict.fromkeys(
                path
                for path in (image_paths or self.image_paths)
                if isinstance(path, Path)
            )
        )
        action = on_image_action or self.on_image_action
        if failed:
            reason = _safe_error_reason(error_reason)
            embed = _command_embed(
                "Request failed",
                f"Codex could not complete this request.\n\nReason: {reason}",
                color=discord.Color.red(),
                target="label:request_failed",
                guild_id=self.guild_id,
                customizer=self.customizer,
                context={**self.context, "reason": reason, "status": "failed"},
            )
            if self.image_message is not None and await self._edit_image_message(
                "",
                self.existing_image_paths,
                on_image_action=action,
                speech=(),
                embed=embed,
            ):
                return
            await self.send(
                embed=embed,
                allowed_mentions=discord.AllowedMentions.none(),
                **self.kwargs,
            )
            return
        final_response = (
            response or "Codex completed the request without a text response."
        )
        if self.image_message is not None and await self._edit_image_message(
            final_response,
            paths or self.existing_image_paths,
            on_image_action=action,
            speech=speech,
        ):
            return
        if paths:
            await self._send_response_with_images(
                final_response,
                paths,
                speech=speech,
                on_image_action=action,
            )
            return
        await send_paginated(
            self.send,
            final_response,
            title="Codex",
            owner_id=self.owner_id,
            speech=speech,
            customizer=self.customizer,
            guild_id=self.guild_id,
            on_view_created=self.on_view_created,
            **self.kwargs,
        )

    def _new_image_view(
        self,
        image_paths: tuple[Path, ...],
        on_image_action: ImageAction | None,
    ) -> _ImageResultView | None:
        if on_image_action is None and self.image_view is not None:
            on_image_action = self.image_view.on_action
        if on_image_action is None:
            return None
        if self.image_view is not None:
            self.image_view.image_paths = image_paths
            self.image_view.on_action = on_image_action
            return self.image_view
        return _ImageResultView(
            self.owner_id,
            image_paths,
            on_action=on_image_action,
            channel=self.channel,
            customizer=self.customizer,
            guild_id=self.guild_id,
        )

    @staticmethod
    def _image_file(path: Path, index: int) -> discord.File:
        suffix = path.suffix.casefold()
        if not suffix or len(suffix) > 12 or not suffix[1:].isalnum():
            suffix = ".png"
        return discord.File(str(path), filename=f"theia-image-{index}{suffix}")

    @staticmethod
    def _speech_files(speech: Iterable[AudioOutput]) -> list[discord.File]:
        return [
            discord.File(io.BytesIO(output.data), filename=output.filename)
            for output in speech
        ]

    async def _send_response_with_images(
        self,
        response: str,
        image_paths: tuple[Path, ...],
        *,
        speech: Iterable[AudioOutput],
        on_image_action: ImageAction | None,
    ) -> None:
        """Send final text, generated files, and controls as one message."""
        pages = _split_pages(response)
        speech_outputs = tuple(speech)
        view = self._new_image_view(image_paths, on_image_action)
        files: list[discord.File] = [
            self._image_file(path, index)
            for index, path in enumerate(image_paths[:10], start=1)
        ]
        files.extend(self._speech_files(speech_outputs))
        values = dict(self.kwargs)
        values.pop("files", None)
        values.pop("view", None)
        values.update(
            content=pages[0],
            files=files,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        if view is not None:
            values["view"] = view
        try:
            message = await self.send(**values)
        except (discord.DiscordException, OSError, TypeError) as exc:
            logger.info(
                "Could not deliver a generated image with its response (error=%s)",
                type(exc).__name__,
            )
            await send_paginated(
                self.send,
                response,
                owner_id=self.owner_id,
                customizer=self.customizer,
                guild_id=self.guild_id,
                speech=speech_outputs,
                on_view_created=self.on_view_created,
                **self.kwargs,
            )
            return
        finally:
            for file in files:
                file.close()

        if view is not None:
            view.message = message
            view.message_id = getattr(message, "id", None)
            if self.on_view_created is not None:
                with contextlib.suppress(Exception):
                    await self.on_view_created(view, message)
        for page in pages[1:]:
            await self.send(
                content=page,
                allowed_mentions=discord.AllowedMentions.none(),
                **self.kwargs,
            )

    async def _edit_image_message(
        self,
        response: str,
        image_paths: tuple[Path, ...],
        *,
        on_image_action: ImageAction | None,
        speech: Iterable[AudioOutput],
        embed: discord.Embed | None = None,
    ) -> bool:
        """Edit the original image message after a follow-up completes."""
        if self.image_message is None:
            return False
        pages = _split_pages(response)
        view = self._new_image_view(image_paths, on_image_action)
        speech_outputs = tuple(speech)
        files: list[discord.File] = []
        new_images = bool(self.image_paths)
        if new_images:
            files.extend(
                self._image_file(path, index)
                for index, path in enumerate(image_paths[:10], start=1)
            )
        files.extend(self._speech_files(speech_outputs))
        values = dict(self.kwargs)
        values.pop("files", None)
        values.pop("view", None)
        values.update(
            content=pages[0] or None,
            embed=embed,
            view=view,
        )
        if files:
            attachments = (
                []
                if new_images
                else list(getattr(self.image_message, "attachments", ()))
            )
            values["attachments"] = [*attachments, *files]
        try:
            await self.image_message.edit(**values)
        except (discord.DiscordException, OSError, TypeError) as exc:
            logger.info(
                "Could not edit the original generated image response (error=%s)",
                type(exc).__name__,
            )
            return False
        finally:
            for file in files:
                file.close()

        if view is not None:
            view.message = self.image_message
            view.message_id = getattr(self.image_message, "id", None)
            if self.on_view_created is not None:
                with contextlib.suppress(Exception):
                    await self.on_view_created(view, self.image_message)
        for page in pages[1:]:
            await self.send(
                content=page,
                allowed_mentions=discord.AllowedMentions.none(),
                **self.kwargs,
            )
        return True


async def send_response(send: SendMessage, response: str, **kwargs: Any) -> None:
    """Send a response through the standard Discord pagination path."""
    await send_paginated(send, response, **kwargs)
