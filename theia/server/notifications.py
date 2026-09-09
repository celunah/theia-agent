"""Codex notification handling for the App Server protocol."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import discord

from ..core import (
    _TurnState,
    _codex_logger,
    _is_tool_item,
    _safe_log_label,
    _verified_change_status,
)

logger = _codex_logger()


class CodexNotificationMixin:
    if TYPE_CHECKING:
        _memory_roots: Any
        _skill_roots: Any
        _skills_refresh_task: asyncio.Task[Any] | None
        _turns: Any

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def _handle_notification(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        params = message.get("params") or {}
        logger.debug(
            "Handling Codex notification (method=%s)",
            _safe_log_label(method),
        )
        if method == "account/updated":
            auth_mode = params.get("authMode")
            self.account = {"type": auth_mode} if auth_mode else None
            if not auth_mode:
                self.clear_authenticated_users()
            self._models = ()
            self._models_loaded_at = 0.0
            self._provider_capabilities = None
            self._provider_capabilities_key = None
            logger.info(
                "Codex account state changed (authenticated=%s)",
                bool(auth_mode),
            )
            return
        if method == "account/rateLimits/updated":
            rate_limits = params.get("rateLimits")
            self._rate_limits = rate_limits if isinstance(rate_limits, dict) else None
            logger.debug(
                "Codex rate limits updated (available=%s)",
                self._rate_limits is not None,
            )
            return
        if method == "account/login/completed":
            login_id = params.get("loginId")
            if self._login_channel is not None and login_id == self._login_id:
                channel = self._login_channel
                user_id = self._login_user_id
                guild_id = self._login_guild_id
                login_sender = self._login_sender
                self._login_channel = None
                self._login_id = None
                self._login_user_id = None
                self._login_guild_id = None
                self._login_sender = None
                if params.get("success"):
                    self.account = {"type": "chatgpt"}
                    if user_id is not None:
                        self.mark_authenticated(user_id, guild_id=guild_id)
                    access_message = (
                        "Codex is ready. Everyone in this server can now use "
                        "`/btw` or `/skill`."
                        if guild_id is not None
                        else "Codex is ready. You can now use `/btw` or `/skill`."
                    )
                    embed = self._frontend_embed(
                        channel,
                        "command:login",
                        "Authentication completed",
                        access_message,
                        color=discord.Color.green(),
                    )
                    if login_sender is not None:
                        self._background_send_callback(login_sender, embed)
                    else:
                        self._background_send(channel, embed)
                    logger.info("Codex login completed successfully")
                else:
                    embed = self._frontend_embed(
                        channel,
                        "command:login",
                        "Login failed",
                        "Codex login did not complete. Please try `/login` again.",
                        color=discord.Color.red(),
                    )
                    if login_sender is not None:
                        self._background_send_callback(login_sender, embed)
                    else:
                        self._background_send(channel, embed)
                    logger.warning("Codex login completed unsuccessfully")
            return

        if method == "thread/tokenUsage/updated":
            self._record_token_usage(params)
            return

        state = self._find_turn(params)
        thread = params.get("thread") or {}
        thread_id = str(
            params.get("threadId")
            or (thread.get("id") if isinstance(thread, dict) else "")
            or ""
        )
        realtime_state = self._realtime_for_thread(thread_id)
        if (
            realtime_state is not None
            and isinstance(method, str)
            and method.startswith("thread/realtime/")
        ):
            self._handle_realtime_notification(realtime_state, method, params)
            return
        if method == "thread/started":
            if thread_id:
                self._set_thread_loaded(thread_id, True)
            return
        if method == "thread/status/changed":
            status = params.get("status") or {}
            status_type = status.get("type") if isinstance(status, dict) else status
            if thread_id:
                self._set_thread_loaded(
                    thread_id,
                    str(status_type or "").casefold() != "notloaded",
                )
            return
        if method == "thread/closed":
            if thread_id:
                self._set_thread_loaded(thread_id, False)
            return
        if method == "thread/deleted":
            if thread_id:
                self._forget_thread(thread_id)
            return
        if method in {"thread/archived", "thread/unarchived"}:
            if thread_id:
                self._set_thread_loaded(thread_id, False)
            return
        if method == "skills/changed":
            logger.info("Codex skill catalog changed; refreshing it")
            self._skills_cache = ()
            self._skills_loaded_at = 0.0
            if self._skills_refresh_task is None or self._skills_refresh_task.done():
                self._skills_refresh_task = asyncio.create_task(
                    self._refresh_skills_after_change()
                )
            return
        if method == "item/agentMessage/delta":
            delta = params.get("delta")
            if state is not None and isinstance(delta, str):
                logger.debug(
                    "Received Codex agent message delta (characters=%d)",
                    len(delta),
                )
                item_id = str(params.get("itemId") or "")
                item = state.agent_messages.setdefault(
                    item_id, {"text": "", "phase": None}
                )
                item["text"] = str(item.get("text") or "") + delta
                state.last_agent_message_id = item_id or state.last_agent_message_id
                self._emit(
                    state,
                    "agent_message",
                    {
                        "text": item["text"],
                        "delta": delta,
                        "item_id": item_id,
                        "phase": item.get("phase"),
                    },
                )
            return
        if method == "item/commandExecution/outputDelta":
            if state is not None:
                logger.debug("Received Codex tool output delta")
                self._emit(state, "tool_activity", {})
            return
        if method == "item/started":
            item = params.get("item") or {}
            if state is not None:
                logger.debug(
                    "Codex item started (type=%s)",
                    _safe_log_label(item.get("type")),
                )
                if _is_tool_item(item):
                    logger.info(
                        "Codex tool started (type=%s)",
                        _safe_log_label(item.get("type")),
                    )
                if item.get("type") == "agentMessage":
                    item_id = str(item.get("id") or "")
                    state.agent_messages[item_id] = {
                        "text": str(item.get("text") or ""),
                        "phase": item.get("phase"),
                    }
                    state.last_agent_message_id = item_id or state.last_agent_message_id
                self._emit(state, "item_started", item)
            return
        if method == "item/completed":
            item = params.get("item") or {}
            if state is not None:
                logger.debug(
                    "Codex item completed (type=%s)",
                    _safe_log_label(item.get("type")),
                )
                if _is_tool_item(item):
                    logger.info(
                        "Codex tool completed (type=%s)",
                        _safe_log_label(item.get("type")),
                    )
                state.items.append(item)
                if item.get("type") == "agentMessage":
                    item_id = str(item.get("id") or "")
                    state.agent_messages[item_id] = {
                        "text": str(item.get("text") or ""),
                        "phase": item.get("phase"),
                    }
                    state.last_agent_message_id = item_id or state.last_agent_message_id
                    if item.get("phase") != "commentary" and isinstance(
                        item.get("text"), str
                    ):
                        state.final_text = item["text"]
                self._emit(state, "item_completed", item)
                verified = _verified_change_status(
                    item, self._memory_roots, self._skill_roots
                )
                if verified:
                    logger.info(
                        "Verified Codex file changes (statuses=%d)",
                        len(verified),
                    )
                    self._emit(state, "verified_change", {"statuses": verified})
            return
        if method == "turn/completed":
            turn = params.get("turn") or {}
            turn_id = turn.get("id") or params.get("turnId")
            if turn_id:
                state = self._turns.setdefault(str(turn_id), _TurnState())
                state.completed = turn
                state.thread_id = state.thread_id or params.get("threadId")
                for item in turn.get("items", []):
                    if (
                        isinstance(item, dict)
                        and str(item.get("type") or "").casefold() == "imagegeneration"
                    ):
                        self._emit(state, "item_completed", item)
                    if (
                        item.get("type") == "agentMessage"
                        and isinstance(item.get("text"), str)
                        and item.get("phase") != "commentary"
                    ):
                        state.final_text = item["text"]
                if state.thread_id:
                    self._clear_pending_for_turn(state.thread_id, str(turn_id))
                self._emit(state, "turn_completed", turn)
                logger.debug(
                    "Codex turn notification completed (status=%s, items=%d)",
                    turn.get("status") or "unknown",
                    len(turn.get("items", []))
                    if isinstance(turn.get("items"), list)
                    else 0,
                )
                if not state.done.done():
                    state.done.set_result(None)
            return
        if method in {"context/compacted", "thread/compacted"} and state is not None:
            self._emit(state, "compacted", params)
            return
        if method == "error" and state is not None:
            state.completed = {
                "status": "failed",
                "error": params.get("error") or params,
            }
            if not state.done.done():
                state.done.set_result(None)
            logger.warning("Codex turn error notification received")
