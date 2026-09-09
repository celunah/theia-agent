"""Discord bot lifecycle and background service ownership."""

# Lazy callbacks bridge event handlers that remain publicly exposed by the façade.
# pylint: disable=cyclic-import,import-outside-toplevel,missing-kwoa

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, cast

import discord
from discord.ext import commands

from ..server.core import CodexAppServer
from .support import (
    DEBUG_REFRESH_INTERVAL,
    PERSISTENT_VIEW_FILE,
    _PersistentViewStore,
)
from .embeds import _debug_embed
from ..customization import FrontendCustomizationStore
from ..delivery import _ImageResultView, _PaginatorView
from ..presence import PresenceManager, RichPresenceManager
from ..recaps import NightlyRecapManager
from ..ui import _DecisionView, _DebugView, _FormView, _UserInputView
from ..voice import VoiceModeManager
from ..core import _codex_logger

logger = _codex_logger()

intents = discord.Intents.default()
intents.message_content = True


def _stale_interaction_fallback_delay() -> float:
    from . import core as bot_module

    return float(getattr(bot_module, "STALE_INTERACTION_FALLBACK_DELAY", 2.0))


async def _run_image_follow_up_callback(*args: Any, **kwargs: Any) -> Any:
    from . import core as bot_module

    return await bot_module._run_image_follow_up(*args, **kwargs)


async def _on_message_callback(message: discord.Message) -> None:
    from . import core as bot_module

    await bot_module.on_message(message)


class TheiaBot(commands.Bot):
    """Discord bot lifecycle owner for Theia's Codex, voice, and presence services."""

    def __init__(self) -> None:
        super().__init__(command_prefix=(), intents=intents, help_command=None)
        self.customizations = FrontendCustomizationStore()
        self.codex = CodexAppServer()
        self._persistent_views = _PersistentViewStore(
            self.codex.runtime_home() / PERSISTENT_VIEW_FILE
        )
        self._interaction_recovery_tasks: set[asyncio.Task[Any]] = set()
        self.codex.set_frontend_customizer(self.customizations)
        self.codex.set_view_registrar(self.register_view)
        self._participating_threads: set[int] = set()
        self._known_channels: dict[int, Any] = {}
        self._request_tasks: set[asyncio.Task[Any]] = set()
        self._debug_tasks: set[asyncio.Task[Any]] = set()
        self._restart_task: asyncio.Task[None] | None = None
        self._retention_task: asyncio.Task[None] | None = None
        self._nightly_recap_task: asyncio.Task[None] | None = None
        self._gateway_presence_lock = asyncio.Lock()
        self.presence = PresenceManager(self._change_presence_when_ready)
        self.rich_presence = RichPresenceManager(
            self._change_rich_presence,
            self.codex.generate_presence,
        )
        self.recaps = NightlyRecapManager(self.codex.runtime_home())
        self.voice = VoiceModeManager(
            transcribe=self.codex.transcribe_audio,
            synthesize=self.codex.synthesize_response,
            realtime_available=lambda: self.codex.voice_provider == "codex-realtime",
            realtime_start=self.codex.start_realtime_voice,
            realtime_audio=self.codex.append_realtime_audio,
            realtime_speech=self.codex.append_realtime_speech,
            realtime_stop=self.codex.stop_realtime_voice,
            realtime_authorized=self._voice_session_allows_tools,
        )

    def _voice_session_allows_tools(self, session: Any) -> bool:
        """Recheck voice tool access through the current Discord member."""
        from .support import _voice_session_allows_tools

        return _voice_session_allows_tools(session)

    def schedule_request(self, coroutine: Coroutine[Any, Any, None]) -> None:
        """Run one agentic request independently of its Discord event callback."""
        task = asyncio.create_task(coroutine)
        self._request_tasks.add(task)
        task.add_done_callback(self._request_task_done)

    async def register_view(self, view: Any, message: Any) -> None:
        """Persist a component view after Discord has assigned its message ID."""
        self._persistent_views.register(view, message)

    def _restore_persistent_view(self, record: dict[str, Any]) -> Any | None:
        """Rebuild one persisted view without restoring an old process callback."""
        state = record.get("state")
        kind = record.get("kind")
        token = record.get("token")
        if not isinstance(state, dict) or not isinstance(kind, str):
            return None
        if not isinstance(token, str) or not token:
            return None

        user_id = state.get("user_id")
        user_id = user_id if isinstance(user_id, int) and user_id > 0 else None
        guild_id = state.get("guild_id")
        guild_id = guild_id if isinstance(guild_id, int) and guild_id > 0 else None
        customizer = self.customizations
        if kind == "paginator":
            pages = state.get("pages")
            if (
                not isinstance(pages, list)
                or not pages
                or not all(isinstance(page, str) for page in pages)
            ):
                return None
            index = state.get("index", 0)
            index = index if isinstance(index, int) else 0
            return _PaginatorView(
                pages,
                owner_id=user_id,
                customizer=customizer,
                guild_id=guild_id,
                token=token,
                recovered=True,
                index=index,
            )
        if kind == "image":
            paths: list[Path] = []
            for value in state.get("image_paths", []):
                if not isinstance(value, str):
                    continue
                path = self.codex.image_artifact_path(
                    {"type": "imageGeneration", "savedPath": value}
                )
                if path is not None and path not in paths:
                    paths.append(path)
            if not paths:
                return None

            async def follow_up(
                interaction: discord.Interaction,
                prompt: str,
                image_paths: tuple[Path, ...],
                view: _ImageResultView,
            ) -> None:
                await _run_image_follow_up_callback(
                    interaction,
                    prompt,
                    image_paths,
                    channel=interaction.channel,
                    image_view=view,
                    image_message=view.message,
                )

            return _ImageResultView(
                user_id,
                tuple(paths),
                on_action=follow_up,
                customizer=customizer,
                guild_id=guild_id,
                token=token,
                recovered=True,
                message_id=record.get("message_id"),
            )
        if kind == "decision":
            raw_choices = state.get("choices")
            if not isinstance(raw_choices, list):
                return None
            choices: list[tuple[str, str, discord.ButtonStyle]] = []
            for choice in raw_choices:
                if not isinstance(choice, dict):
                    continue
                label = choice.get("label")
                value = choice.get("value")
                style = choice.get("style")
                if (
                    not isinstance(label, str)
                    or not isinstance(value, str)
                    or not isinstance(style, int)
                ):
                    continue
                try:
                    button_style = discord.ButtonStyle(style)
                except (TypeError, ValueError):
                    continue
                choices.append((label, value, button_style))
            if not choices:
                return None
            return _DecisionView(
                user_id,
                choices,
                token=token,
                recovered=True,
            )
        if kind == "debug":
            return _DebugView(
                user_id,
                customizer=customizer,
                guild_id=guild_id,
                token=token,
                recovered=True,
            )
        if kind == "form":
            prompt = state.get("prompt")
            if not isinstance(prompt, str):
                return None
            return _FormView(
                user_id,
                prompt=prompt,
                customizer=customizer,
                guild_id=guild_id,
                token=token,
                recovered=True,
            )
        if kind == "user-input":
            questions = state.get("questions")
            if not isinstance(questions, list) or not all(
                isinstance(question, dict) for question in questions
            ):
                return None
            question_index = state.get("question_index", 0)
            question_index = question_index if isinstance(question_index, int) else 0
            answers = state.get("answers")
            answers = answers if isinstance(answers, dict) else None
            return _UserInputView(
                user_id,
                questions,
                customizer=customizer,
                guild_id=guild_id,
                token=token,
                recovered=True,
                question_index=question_index,
                answers=answers,
            )
        return None

    def _request_task_done(self, task: asyncio.Task[Any]) -> None:
        self._request_tasks.discard(task)
        if task.cancelled():
            return
        try:
            error = task.exception()
        except asyncio.InvalidStateError:
            return
        if error is not None:
            logger.error(
                "Background Theia request failed (error=%s)",
                type(error).__name__,
            )

    async def _cancel_request_tasks(self) -> None:
        """Stop agentic requests before shared Codex and Discord resources close."""
        tasks = tuple(self._request_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._request_tasks.clear()

    async def _cancel_interaction_recovery_tasks(self) -> None:
        """Cancel delayed stale-interaction acknowledgements during shutdown."""
        tasks = tuple(self._interaction_recovery_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._interaction_recovery_tasks.clear()

    async def _acknowledge_unhandled_interaction(
        self, interaction: discord.Interaction
    ) -> None:
        await asyncio.sleep(_stale_interaction_fallback_delay())
        if interaction.response.is_done():
            return
        with contextlib.suppress(discord.DiscordException):
            await interaction.response.send_message(
                "This control expired or was interrupted by a restart. "
                "Please start a new request.",
                ephemeral=True,
            )

    def _schedule_interaction_recovery(self, interaction: discord.Interaction) -> None:
        task = asyncio.create_task(self._acknowledge_unhandled_interaction(interaction))
        self._interaction_recovery_tasks.add(task)
        task.add_done_callback(self._interaction_recovery_tasks.discard)

    async def on_interaction(self, interaction: discord.Interaction) -> None:
        """Acknowledge component or modal interactions unknown after a restart."""
        if interaction.type in {
            discord.InteractionType.component,
            discord.InteractionType.modal_submit,
        }:
            self._schedule_interaction_recovery(interaction)

    def schedule_debug_refresh(
        self,
        message: Any,
        view: _DebugView,
        *,
        session_key_value: str,
        channel: Any | None,
        user: discord.abc.User | None,
    ) -> None:
        """Refresh one diagnostic message independently of agent request tasks."""
        task = asyncio.create_task(
            self._refresh_debug_message(
                message,
                view,
                session_key_value=session_key_value,
                channel=channel,
                user=user,
            )
        )
        self._debug_tasks.add(task)
        task.add_done_callback(self._debug_task_done)

    def _debug_task_done(self, task: asyncio.Task[Any]) -> None:
        self._debug_tasks.discard(task)
        if task.cancelled():
            return
        with contextlib.suppress(asyncio.InvalidStateError):
            error = task.exception()
            if error is not None:
                logger.debug("Live debug view stopped (error=%s)", type(error).__name__)

    async def _refresh_debug_message(
        self,
        message: Any,
        view: _DebugView,
        *,
        session_key_value: str,
        channel: Any | None,
        user: discord.abc.User | None,
    ) -> None:
        while not view.is_finished():
            await asyncio.sleep(DEBUG_REFRESH_INTERVAL)
            if view.is_finished():
                return
            try:
                await message.edit(
                    embed=_debug_embed(
                        self.codex.debug_state(session_key_value),
                        channel=channel,
                        user=user,
                    ),
                    view=view,
                )
            except (discord.DiscordException, AttributeError) as exc:
                logger.debug(
                    "Live debug view could not be refreshed (error=%s)",
                    type(exc).__name__,
                )
                view.stop()
                return

    async def _cancel_debug_tasks(self) -> None:
        """Stop live diagnostic refreshes before shared resources close."""
        tasks = tuple(self._debug_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._debug_tasks.clear()

    async def _change_presence_when_ready(self, **kwargs: Any) -> None:
        """Defer presence changes until Discord has established the gateway."""
        if not self.is_ready():
            return
        if "status" in kwargs and "activity" not in kwargs:
            kwargs["activity"] = self.rich_presence.current_activity
        async with self._gateway_presence_lock:
            await self.change_presence(**kwargs)

    async def _change_rich_presence(
        self, *, activity: discord.BaseActivity | None
    ) -> None:
        """Change activity while preserving the independent online status."""
        status = self.presence.current_status
        if status is None:
            return
        await self._change_presence_when_ready(
            status=status,
            activity=activity,
        )

    async def setup_hook(self) -> None:
        """Start Codex, synchronize slash commands, and begin background services."""
        await super().setup_hook()
        self._persistent_views.restore(self, self._restore_persistent_view)
        await self.codex.start()
        await self.tree.sync()
        await self.presence.start()
        await self.rich_presence.start()
        self._retention_task = asyncio.create_task(self._retention_loop())
        if self.recaps.enabled:
            self._nightly_recap_task = asyncio.create_task(self._nightly_recap_loop())

    async def close(self) -> None:
        """Stop background services and close Discord and Codex resources in order."""
        await self._cancel_interaction_recovery_tasks()
        await self._cancel_debug_tasks()
        await self._cancel_request_tasks()
        if self._retention_task is not None:
            self._retention_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._retention_task
            self._retention_task = None
        if self._nightly_recap_task is not None:
            self._nightly_recap_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._nightly_recap_task
            self._nightly_recap_task = None
        await self.rich_presence.close()
        await self.presence.close()
        await self.voice.close()
        await self.codex.close()
        await super().close()

    async def on_ready(self) -> None:
        """Refresh the presence after Discord establishes or restores the gateway."""
        await self.presence.on_ready()

    async def _retention_loop(self) -> None:
        while True:
            try:
                await self.codex.enforce_retention()
            except Exception as exc:  # noqa: BLE001 - janitor must stay alive
                logger.warning(
                    "Codex session retention check failed (error=%s)",
                    type(exc).__name__,
                )
            await asyncio.sleep(60 * 60)

    async def _generate_nightly_recap(
        self, prompt: str, source_session_key: str | None
    ) -> str | None:
        """Run one private recap generation turn through the Codex boundary."""
        return await self.codex.generate_nightly_recap(
            prompt,
            session_key=source_session_key,
        )

    async def _nightly_recap_loop(self) -> None:
        """Generate pending recaps at local midnight and after missed wakeups."""
        while True:
            try:
                await self.recaps.process_due(self._generate_nightly_recap)
            except Exception as exc:  # noqa: BLE001 - scheduler must stay alive
                logger.warning(
                    "Nightly recap pass failed (error=%s)", type(exc).__name__
                )
            await asyncio.sleep(self.recaps.seconds_until_midnight())

    async def backfill_after_resume(self) -> None:
        """Replay bounded messages missed while the Discord gateway was disconnected."""
        limit_text = os.getenv("THEIA_BACKFILL_LIMIT") or os.getenv(
            "DISCORD_BACKFILL_LIMIT", "20"
        )
        try:
            limit = max(0, min(100, int(limit_text)))
        except ValueError:
            limit = 20
        if limit == 0:
            return
        for channel_id in self.codex.channel_checkpoints():
            channel = self.get_channel(channel_id)
            if channel is not None:
                self._known_channels[channel_id] = channel
        for channel_id, channel in tuple(self._known_channels.items()):
            checkpoint = self.codex.channel_checkpoint(channel_id)
            if checkpoint is None:
                continue
            history = getattr(channel, "history", None)
            if not callable(history):
                continue
            history_call = cast(Callable[..., Any], history)
            try:
                missed = [
                    item
                    async for item in history_call(
                        limit=limit, after=discord.Object(id=checkpoint)
                    )
                ]
            except discord.DiscordException as exc:
                logger.info(
                    "Could not backfill a Discord channel after reconnect "
                    "(channel_id=%s, error=%s)",
                    channel_id,
                    type(exc).__name__,
                )
                continue
            for item in reversed(missed):
                await _on_message_callback(item)
