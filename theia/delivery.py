"""Translate Codex progress and final responses into resilient Discord messages."""

import asyncio
import contextlib
import io
import time
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any, cast

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
ImageAction = Callable[[discord.Interaction, str, tuple[Path, ...]], Awaitable[None]]
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
    """Render a short thought duration using seconds or minutes and seconds."""
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
            style=discord.ButtonStyle.primary,
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
        """Return the page currently selected by the paginator."""
        return self.pages[self.index]

    def persistence_data(self) -> dict[str, Any]:
        return {
            "pages": self.pages,
            "index": self.index,
            "owner_id": self.owner_id,
            "guild_id": self.guild_id,
        }

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Allow pagination controls only for the response owner."""
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
        """Move the paginator and edit its message for an authorized reaction."""
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
        """Remove the expired paginator from the reaction dispatch table."""
        if self.message is not None:
            _reaction_paginators.pop(self.message.id, None)


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
    ) -> None:
        self._init_persistence("image", token, recovered=recovered)
        super().__init__(timeout=None if recovered else timeout)
        self.user_id = user_id
        self.image_paths = image_paths
        self.on_action = on_action
        self.channel = channel
        self.customizer = customizer
        self.guild_id = guild_id

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
        await self.on_action(interaction, prompt, self.image_paths)

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
        """Replace progress status with a safe error or paginated final response."""
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
        if failed:
            reason = _safe_error_reason(error_reason)
            await self.send(
                embed=_command_embed(
                    "Request failed",
                    f"Codex could not complete this request.\n\nReason: {reason}",
                    color=discord.Color.red(),
                    target="label:request_failed",
                    guild_id=self.guild_id,
                    customizer=self.customizer,
                    context={**self.context, "reason": reason, "status": "failed"},
                ),
                allowed_mentions=discord.AllowedMentions.none(),
                **self.kwargs,
            )
            return
        await send_paginated(
            self.send,
            response or "Codex completed the request without a text response.",
            title="Codex",
            owner_id=self.owner_id,
            speech=speech,
            customizer=self.customizer,
            guild_id=self.guild_id,
            on_view_created=self.on_view_created,
            **self.kwargs,
        )
        paths = tuple(
            dict.fromkeys(
                path
                for path in (image_paths or self.image_paths)
                if isinstance(path, Path)
            )
        )
        if paths:
            await self._send_images(
                paths,
                on_image_action=on_image_action or self.on_image_action,
                on_view_created=self.on_view_created,
            )

    async def _send_images(
        self,
        image_paths: tuple[Path, ...],
        *,
        on_image_action: ImageAction | None,
        on_view_created: ViewRegistrar | None,
    ) -> None:
        """Send generated image files and attach their follow-up controls."""
        files: list[discord.File] = []
        try:
            for index, path in enumerate(image_paths[:10], start=1):
                suffix = path.suffix.casefold()
                if not suffix or len(suffix) > 12 or not suffix[1:].isalnum():
                    suffix = ".png"
                files.append(
                    discord.File(
                        str(path),
                        filename=f"theia-image-{index}{suffix}",
                    )
                )
            message = await self.send(
                files=files,
                allowed_mentions=discord.AllowedMentions.none(),
                **self.kwargs,
            )
        except (discord.DiscordException, OSError, TypeError) as exc:
            logger.info(
                "Could not deliver a generated image to Discord (error=%s)",
                type(exc).__name__,
            )
            return
        finally:
            for file in files:
                file.close()

        if on_image_action is None:
            return
        view = _ImageResultView(
            self.owner_id,
            image_paths,
            on_action=on_image_action,
            channel=self.channel,
            customizer=self.customizer,
            guild_id=self.guild_id,
        )
        edit = getattr(message, "edit", None)
        if not callable(edit):
            return
        try:
            edit_async = cast(Callable[..., Awaitable[Any]], edit)
            await edit_async(view=view)
            if on_view_created is not None:
                with contextlib.suppress(Exception):
                    await on_view_created(view, message)
        except (discord.DiscordException, TypeError) as exc:
            logger.info(
                "Could not attach generated image controls (error=%s)",
                type(exc).__name__,
            )


async def send_response(send: SendMessage, response: str, **kwargs: Any) -> None:
    """Send a response through the standard Discord pagination path."""
    await send_paginated(send, response, **kwargs)
