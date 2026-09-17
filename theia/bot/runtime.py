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

from ..server.core import CodexAppServer, CodexAppServerError
from .support import (
    PERSISTENT_VIEW_FILE,
    _PersistentViewStore,
)
from .usage import _UsageView
from ..customization import FrontendCustomizationStore
from ..delivery import (
    _ImageResultView,
    _MemoryConfirmationView,
    _MemoryView,
    _PaginatorView,
)
from ..presence import PresenceManager, RichPresenceManager
from ..recaps import NightlyRecapManager
from ..ui import _DecisionView, _FormView, _UserInputView
from ..voice import VoiceModeManager
from ..server.lighthouse import LighthouseView
from ..core import _codex_logger
from .memory import _mutation_callback

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
            audio_provider=self.codex.audio_provider,
            provider_name=lambda: self.codex.voice_provider,
            realtime_available=lambda: self.codex.voice_provider == "codex-realtime",
            realtime_start=self.codex.start_realtime_voice,
            realtime_audio=self.codex.append_realtime_audio,
            realtime_speech=self.codex.append_realtime_speech,
            realtime_stop=self.codex.stop_realtime_voice,
            realtime_authorized=self._voice_session_allows_tools,
        )
        self.lighthouse = LighthouseView(
            self.codex,
            presence=self.presence,
            rich_presence=self.rich_presence,
            voice=self.voice,
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
        if kind == "usage":
            result = state.get("result")
            if user_id is None or not isinstance(result, dict):
                return None
            view = _UsageView(
                result,
                owner_id=user_id,
                token=token,
                recovered=True,
            )
            view.showing_details = bool(state.get("showing_details"))
            view.toggle_button.label = view._label()
            return view
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
        if kind == "memory":
            if user_id is None:
                return None
            records = state.get("records")
            entries = state.get("entries")
            if not isinstance(records, list):
                records = entries
            if not isinstance(records, list) or not all(
                isinstance(entry, (str, dict)) for entry in records
            ):
                return None
            character_name = state.get("character_name", "Theia")
            character_slug = state.get("character_slug", "theia")
            scope = state.get("scope", "me")
            if not all(
                isinstance(value, str)
                for value in (character_name, character_slug, scope)
            ):
                return None
            index = state.get("index", 0)
            index = index if isinstance(index, int) else 0
            total_entries = state.get("total_entries")
            total_entries = total_entries if isinstance(total_entries, int) else None
            channel_id = state.get("channel_id")
            channel_id = channel_id if isinstance(channel_id, int) else None
            view = _MemoryView(
                records,
                character_name=character_name,
                character_slug=character_slug,
                scope=scope,
                owner_id=user_id,
                total_entries=total_entries,
                customizer=customizer,
                guild_id=guild_id,
                channel_id=channel_id,
                token=token,
                recovered=True,
                index=index,
                search_query=(
                    state.get("search_query")
                    if isinstance(state.get("search_query"), str)
                    else ""
                ),
            )
            view.on_mutation = _mutation_callback(self, view)
            view.on_view_created = self.register_view
            return view
        if kind == "memory-confirm":
            if user_id is None:
                return None
            action = state.get("action")
            record_id = state.get("record_id")
            scope = state.get("scope")
            channel_id = state.get("channel_id")
            replacement = state.get("replacement")
            if (
                action not in {"forget", "edit"}
                or not isinstance(record_id, str)
                or not isinstance(scope, str)
                or not isinstance(channel_id, int)
                or (replacement is not None and not isinstance(replacement, str))
            ):
                return None

            async def mutate(
                interaction: discord.Interaction,
                requested_action: str,
                requested_id: str,
                requested_replacement: str | None,
            ) -> tuple[bool, str]:
                from .support import _guild_id, _is_server_admin
                from ..core import _is_super_admin_user

                super_admin = _is_super_admin_user(interaction.user.id)
                try:
                    result = (
                        self.codex.forget_memory(
                            f"guild:{guild_id or 0}:channel:{channel_id}:"
                            f"user:{user_id or 0}",
                            requested_id,
                            scope,
                            actor_user_id=interaction.user.id,
                            actor_guild_id=_guild_id(interaction.channel),
                            server_admin=_is_server_admin(
                                interaction.user, interaction.channel
                            ),
                            super_admin=super_admin,
                            confirmed=True,
                        )
                        if requested_action == "forget"
                        else self.codex.edit_memory(
                            f"guild:{guild_id or 0}:channel:{channel_id}:"
                            f"user:{user_id or 0}",
                            requested_id,
                            requested_replacement or "",
                            scope,
                            actor_user_id=interaction.user.id,
                            actor_guild_id=_guild_id(interaction.channel),
                            server_admin=_is_server_admin(
                                interaction.user, interaction.channel
                            ),
                            super_admin=super_admin,
                            confirmed=True,
                        )
                    )
                except (CodexAppServerError, OSError) as exc:
                    return False, str(exc)
                return True, (
                    "Memory forgotten."
                    if result.get("action") == "forget"
                    else "Memory updated."
                )

            return _MemoryConfirmationView(
                user_id,
                action=action,
                record_id=record_id,
                replacement=replacement,
                scope=scope,
                channel_id=channel_id,
                on_mutation=mutate,
                customizer=customizer,
                guild_id=guild_id,
                token=token,
                recovered=True,
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
        try:
            await super().setup_hook()
        except Exception as exc:  # noqa: BLE001 - preserve the operator dashboard
            self.codex.record_startup_failure(
                "Theia's Discord runtime could not initialize.", block=False
            )
            logger.critical(
                "FATAL: Theia startup degraded (component=Discord runtime, error=%s)",
                type(exc).__name__,
            )
        try:
            self._persistent_views.restore(self, self._restore_persistent_view)
        except Exception as exc:  # noqa: BLE001 - preserve the operator dashboard
            self.codex.record_startup_failure(
                "Theia persistent views could not be restored.", block=False
            )
            logger.critical(
                "FATAL: Theia startup degraded (component=persistent views, error=%s)",
                type(exc).__name__,
            )
        try:
            await self.lighthouse.start()
        except Exception as exc:  # noqa: BLE001 - preserve startup diagnostics
            self.codex.record_startup_failure(
                "Theia Lighthouse could not start.", block=False
            )
            logger.critical(
                "FATAL: Theia startup degraded (component=Lighthouse, error=%s)",
                type(exc).__name__,
            )
        for operation, reason in (
            (self.codex.start, "Theia could not start the Codex App Server."),
            (self.tree.sync, "Theia Discord commands could not be synchronized."),
            (self.presence.start, "Theia presence could not start."),
            (self.rich_presence.start, "Theia rich presence could not start."),
        ):
            try:
                await operation()
            except Exception as exc:  # noqa: BLE001 - keep Lighthouse available
                self.codex.record_startup_failure(reason, block=False)
                logger.critical(
                    "FATAL: Theia startup degraded (error=%s)", type(exc).__name__
                )
        try:
            retention_coroutine = self._retention_loop()
            try:
                self._retention_task = asyncio.create_task(retention_coroutine)
            except Exception:
                retention_coroutine.close()
                raise
            if self.recaps.enabled:
                recap_coroutine = self._nightly_recap_loop()
                try:
                    self._nightly_recap_task = asyncio.create_task(recap_coroutine)
                except Exception:
                    recap_coroutine.close()
                    raise
        except Exception as exc:  # noqa: BLE001 - keep Lighthouse available
            self.codex.record_startup_failure(
                "Theia background services could not start.", block=False
            )
            logger.critical(
                "FATAL: Theia startup degraded (component=background services, error=%s)",
                type(exc).__name__,
            )

    async def close(self) -> None:
        """Stop background services and close Discord and Codex resources in order."""
        await self._cancel_interaction_recovery_tasks()
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
        await self.lighthouse.close()
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
