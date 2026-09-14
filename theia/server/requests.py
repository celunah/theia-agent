"""Request preparation and conversational turn orchestration."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import discord

from .policy import (
    MAX_CODEX_MEMORY_TURN_RETRIES,
    SESSION_ARCHIVE_AFTER,
    SESSION_DELETE_AFTER,
    _MEMORY_RETRIEVAL_HINT_RE,
)
from ..core import (
    DEFAULT_MODE,
    CodexAppServerError,
    CodexTransientRestartError,
    _Session,
    _TurnDiagnostics,
    _TurnState,
    _codex_logger,
)

logger = _codex_logger()


class CodexRequestMixin:
    if TYPE_CHECKING:
        _model: str | None
        _approval_level: str
        _adaptive_reasoning: bool

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    async def _run_turn_with_recovery(
        self,
        session_key: str,
        session: _Session,
        *,
        turn_prompt: str,
        attachment_list: tuple[Any, ...],
        prepared_attachments: list[dict[str, Any]],
        effort: str,
        channel: discord.abc.Messageable | None,
        user_id: int | None,
        user: Any | None,
        allow_tools: bool,
        thread_source: discord.Message | None,
        user_prompt: str | None,
        on_channel_change: Callable[[discord.abc.Messageable], None] | None,
        on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None,
        interaction_sender: Callable[..., Awaitable[Any]] | None,
        allow_discord_tools: bool,
        mood_input: str,
        recent_context: str | None,
        summary_injected: bool,
        diagnostics: _TurnDiagnostics | None = None,
    ) -> str:
        """Run a turn, retrying only interruptions caused by memory recovery."""
        diagnostics = diagnostics or session.turn_diagnostics
        prompted_user_text = user_prompt or turn_prompt
        schedule_mood = True
        for attempt in range(MAX_CODEX_MEMORY_TURN_RETRIES + 1):
            await self._ensure_running()
            await self._ensure_thread(
                session,
                allow_tools=allow_tools,
                include_dynamic_tools=allow_discord_tools,
            )
            turn_params: dict[str, Any] = {
                "threadId": session.thread_id,
                "input": self._user_input(
                    turn_prompt, attachment_list, prepared_attachments
                ),
                "effort": effort,
            }
            if self._model is not None:
                turn_params["model"] = self._model
            result = await self._request("turn/start", turn_params)
            turn = result.get("turn", {})
            turn_id = turn.get("id")
            if not turn_id:
                raise CodexAppServerError("Codex did not return a turn id.")
            if summary_injected:
                session.pending_self_improvement_summary = None
                self._persist_state()

            state = self._turns.setdefault(
                str(turn_id),
                _TurnState(
                    thread_id=session.thread_id,
                    session=session,
                    channel=channel,
                    user_id=user_id,
                    user=user,
                    allow_tools=allow_tools,
                    thread_source=thread_source,
                    user_prompt=prompted_user_text,
                    on_channel_change=on_channel_change,
                    on_event=on_event,
                    interaction_sender=interaction_sender,
                    allow_discord_tools=allow_discord_tools,
                    diagnostics=diagnostics,
                ),
            )
            state.thread_id = session.thread_id
            state.session = session
            state.channel = channel
            state.user_id = user_id
            state.user = user
            state.allow_tools = allow_tools
            state.thread_source = thread_source
            state.user_prompt = prompted_user_text
            state.on_channel_change = on_channel_change
            state.on_event = on_event
            state.interaction_sender = interaction_sender
            state.allow_discord_tools = allow_discord_tools
            state.diagnostics = diagnostics
            session.turn_id = str(turn_id)
            if schedule_mood:
                self._schedule_mood_appraisal(
                    session,
                    mood_input,
                    recent_context=recent_context,
                    diagnostics=diagnostics,
                )
                schedule_mood = False
            try:
                return await self._wait_for_turn(
                    session_key, session, state, str(turn_id)
                )
            except CodexTransientRestartError:
                if attempt >= MAX_CODEX_MEMORY_TURN_RETRIES:
                    raise
                delay = min(2.0 * (2**attempt), 10.0)
                logger.warning(
                    "Retrying a turn interrupted by Codex memory recovery "
                    "(attempt=%d, delay_seconds=%.1f)",
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)
        raise CodexTransientRestartError(
            "Codex could not complete the turn after memory recovery."
        )

    async def _prepare_session_for_activity(
        self, session: _Session, *, now: float | None = None
    ) -> None:
        activity_at = time.time() if now is None else now
        thread_id = session.thread_id
        if thread_id and session.last_activity_at is not None:
            inactive_for = max(0.0, activity_at - session.last_activity_at)
            if inactive_for >= SESSION_DELETE_AFTER:
                try:
                    await self.delete_thread(thread_id)
                except CodexAppServerError as exc:
                    message = str(exc).casefold()
                    if (
                        "not found" not in message
                        and "unknown thread" not in message
                        and "no rollout found" not in message
                    ):
                        raise
                    self._forget_thread(thread_id)
            elif session.archived:
                await self.unarchive_thread(thread_id)

        session.last_activity_at = activity_at
        self._persist_state()

    async def enforce_retention(self, *, now: float | None = None) -> dict[str, int]:
        """Archive or delete inactive mapped sessions according to policy."""
        await self._ensure_running()
        checked_at = time.time() if now is None else now
        archived = 0
        deleted = 0
        pruned_sessions = 0
        for session in tuple(self._sessions.values()):
            if session.lock is None:
                session.lock = asyncio.Lock()
            async with session.lock:
                if not session.thread_id:
                    has_session_metadata = bool(
                        session.mode != DEFAULT_MODE
                        or session.personality_name
                        or session.personality_selected
                        or session.pending_self_improvement_summary
                        or session.tool_policy is not None
                        or session.attention is not None
                        or (
                            session.workspace is not None
                            and bool(session.workspace.entries)
                        )
                        or bool(session.commitments)
                    )
                    if not has_session_metadata and (
                        session.last_activity_at is None
                        or checked_at - session.last_activity_at >= SESSION_DELETE_AFTER
                    ):
                        self._forget_session(session.key)
                        pruned_sessions += 1
                    continue
                if session.turn_id or session.last_activity_at is None:
                    continue
                inactive_for = max(0.0, checked_at - session.last_activity_at)
                if inactive_for >= SESSION_DELETE_AFTER:
                    thread_id = session.thread_id
                    try:
                        await self.delete_thread(thread_id)
                    except CodexAppServerError as exc:
                        message = str(exc).casefold()
                        if (
                            "not found" not in message
                            and "unknown thread" not in message
                            and "no rollout found" not in message
                        ):
                            logger.warning(
                                "Could not delete an expired Codex session (error=%s)",
                                type(exc).__name__,
                            )
                            continue
                        self._forget_thread(thread_id)
                    deleted += 1
                elif inactive_for >= SESSION_ARCHIVE_AFTER and not session.archived:
                    try:
                        await self._request(
                            "thread/archive", {"threadId": session.thread_id}
                        )
                    except CodexAppServerError as exc:
                        logger.warning(
                            "Could not archive an inactive Codex session (error=%s)",
                            type(exc).__name__,
                        )
                        continue
                    self._set_thread_archived(session.thread_id, True)
                    self._set_thread_loaded(session.thread_id, False)
                    self._persist_state()
                    archived += 1
        if archived or deleted or pruned_sessions:
            logger.info(
                "Applied Codex session retention (archived=%d, deleted=%d, "
                "pruned_sessions=%d)",
                archived,
                deleted,
                pruned_sessions,
            )
        if pruned_sessions:
            self._persist_state()
        self._prune_attachment_cache()
        return {"archived": archived, "deleted": deleted}

    def _prune_attachment_cache(self, *, required_bytes: int = 0) -> int:
        """Remove expired cache entries before the private cache exceeds its quota."""
        try:
            entries = tuple(self._attachment_root.iterdir())
        except FileNotFoundError:
            return 0
        except OSError as exc:
            logger.warning(
                "Could not inspect the attachment cache (error=%s)",
                type(exc).__name__,
            )
            return 0

        files: list[tuple[float, int, Path]] = []
        total = 0
        for path in entries:
            if path.is_symlink() or not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            total += stat.st_size
            files.append((stat.st_mtime, stat.st_size, path))
        if total + required_bytes <= self._attachment_cache_limit:
            return 0

        cutoff = time.time() - self._attachment_cache_max_age
        removed = 0
        for modified_at, size, path in sorted(files):
            if modified_at > cutoff:
                continue
            try:
                path.unlink()
            except OSError as exc:
                logger.warning(
                    "Could not remove an expired attachment cache entry (error=%s)",
                    type(exc).__name__,
                )
                continue
            total -= size
            removed += 1
            if total + required_bytes <= self._attachment_cache_limit:
                break
        if total + required_bytes > self._attachment_cache_limit:
            logger.warning(
                "Attachment cache quota reached; refusing additional cache data "
                "(bytes=%d, limit=%d)",
                total,
                self._attachment_cache_limit,
            )
            if required_bytes:
                raise CodexAppServerError(
                    "The private attachment cache is full; try again later."
                )
        return removed

    async def ask(
        self,
        prompt: str,
        *,
        session_key: str,
        channel: discord.abc.Messageable | None,
        user_id: int | None,
        user: Any | None = None,
        attachments: Iterable[Any] = (),
        allow_tools: bool = True,
        thread_source: discord.Message | None = None,
        user_prompt: str | None = None,
        historical_context: str | None = None,
        on_channel_change: Callable[[discord.abc.Messageable], None] | None = None,
        on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
        interaction_sender: Callable[..., Awaitable[Any]] | None = None,
        allow_discord_tools: bool = True,
    ) -> str:
        """Run a user request in a session and return its completed response text."""
        await self._ensure_running()
        await self.refresh_account()
        if self.account is None and self.requires_openai_auth:
            raise CodexAppServerError("Run `/login` first.")

        session = self._session(session_key)
        assert session.lock is not None
        attachment_list = tuple(attachments)
        logger.info(
            "Codex request accepted (prompt_characters=%d, attachments=%d, "
            "tools_allowed=%s)",
            len(prompt),
            len(attachment_list),
            allow_tools,
        )
        async with session.lock:
            await self._ensure_running()
            await self._prepare_session_for_activity(session)
            diagnostics = _TurnDiagnostics()
            session.turn_diagnostics = diagnostics
            attachment_preparation = await self._prepare_attachment_manifest(
                attachment_list
            )
            prepared_attachments = list(attachment_preparation.inputs)
            attachment_manifest = attachment_preparation.manifest
            mood_input = user_prompt or prompt
            attention_transition = await self._prepare_attention_for_turn(
                session,
                mood_input,
                recent_global_context=prompt if user_prompt else None,
                historical_context=historical_context,
                diagnostics=diagnostics,
            )
            effort = await self._select_reasoning_effort(
                prompt, attachment_list, diagnostics=diagnostics
            )
            logger.info(
                "Starting Codex turn (adaptive_reasoning=%s, effort=%s, attachments=%d)",
                self._adaptive_reasoning,
                effort,
                len(attachment_list),
            )
            previous_thread_id = session.thread_id
            await self._ensure_thread(
                session,
                allow_tools=allow_tools,
                include_dynamic_tools=allow_discord_tools,
            )
            if (
                isinstance(channel, discord.Thread)
                and session.thread_id
                and session.thread_id != previous_thread_id
            ):
                discord_thread_name = str(getattr(channel, "name", "")).strip()
                if discord_thread_name:
                    try:
                        await self.set_thread_name(
                            session.thread_id,
                            discord_thread_name,
                        )
                    except CodexAppServerError as exc:
                        logger.debug(
                            "Could not name Codex thread from Discord thread "
                            "(error=%s)",
                            type(exc).__name__,
                        )
            workspace = self._workspace_snapshot(session)
            memory_context = None
            memory_retrieval_used = bool(
                allow_tools and _MEMORY_RETRIEVAL_HINT_RE.search(user_prompt or prompt)
            )
            if memory_retrieval_used:
                memory_context = await self.generate_memory_retrieval(
                    prompt,
                    session_key=session_key,
                    allow_tools=allow_tools,
                )
            self_model = self._safe_self_model_snapshot(
                session,
                allow_tools=allow_tools,
                allow_discord_tools=allow_discord_tools,
                phase="starting",
                memory_retrieval_used=memory_retrieval_used,
                attachment_manifest=attachment_manifest,
            )
            turn_prompt, summary_injected = self._turn_prompt_with_summary(
                session,
                prompt,
                memory_context=memory_context,
                attention_transition=attention_transition,
                self_model=self_model or {},
                workspace=workspace,
            )
            response = await self._run_turn_with_recovery(
                session_key,
                session,
                turn_prompt=turn_prompt,
                attachment_list=attachment_list,
                prepared_attachments=prepared_attachments,
                effort=effort,
                channel=channel,
                user_id=user_id,
                user=user,
                allow_tools=allow_tools,
                thread_source=thread_source,
                user_prompt=user_prompt,
                on_channel_change=on_channel_change,
                on_event=on_event,
                interaction_sender=interaction_sender,
                allow_discord_tools=allow_discord_tools,
                mood_input=mood_input,
                recent_context=prompt if user_prompt else None,
                summary_injected=summary_injected,
                diagnostics=diagnostics,
            )
            self._record_attention_response(session, response)
            completed_self_model = self._safe_self_model_snapshot(
                session,
                allow_tools=allow_tools,
                allow_discord_tools=allow_discord_tools,
                phase="completed",
                memory_retrieval_used=memory_retrieval_used,
                attachment_manifest=attachment_manifest,
            )
            self._schedule_self_improvement_review(
                session,
                user_prompt or prompt,
                response,
                channel=channel,
                user_id=user_id,
                user=user,
                allow_tools=allow_tools,
                diagnostics=diagnostics,
            )
            self._schedule_workspace_review(
                session,
                user_prompt or prompt,
                response,
                recent_context=prompt if user_prompt else None,
                self_model=completed_self_model or {},
                diagnostics=diagnostics,
            )
            return response
