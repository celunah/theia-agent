"""JSONL transport and protocol handlers for the Codex App Server."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from typing import TYPE_CHECKING, Any, cast

import discord

from ..core import (
    CodexAppServerError,
    _PendingApproval,
    _TurnState,
    _codex_logger,
    _env_float,
    _error_message,
    _path_from_value,
    _path_is_under,
    _safe_approval_reason,
    _safe_intermediate_text,
    _safe_log_label,
    _subtext,
    _truncate,
)
from .policy import (
    MAX_ATTACHMENT_BYTES,
    _APPROVAL_PATH_RE,
    _APPROVAL_RISK_DANGEROUS,
    _APPROVAL_RISK_SAFE,
    _APPROVAL_RISK_VERY_DANGEROUS,
    _APPROVAL_SAFE_COMMAND_RE,
    _APPROVAL_VERY_DANGEROUS_RE,
)
from ..ui import _DecisionView, _FormView, _UserInputView

logger = _codex_logger()


class CodexTransportMixin:
    if TYPE_CHECKING:
        _next_request_id: int
        _skills_refresh_task: asyncio.Task[Any] | None

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    async def _request(
        self,
        method: str,
        params: dict[str, Any] | None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        request_id = self._next_request_id
        self._next_request_id += 1
        future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[request_id] = future
        method_label = _safe_log_label(method)
        started_at = time.monotonic()
        logger.debug(
            "Codex protocol request started (method=%s, pending=%d)",
            method_label,
            len(self._pending),
        )
        try:
            await self._send({"method": method, "id": request_id, "params": params})
            wait_for = self._request_timeout if timeout is None else timeout
            response = (
                await asyncio.wait_for(future, wait_for) if wait_for else await future
            )
        except asyncio.TimeoutError as exc:
            self._pending.pop(request_id, None)
            logger.warning(
                "Codex protocol request timed out (method=%s, duration_ms=%.1f)",
                method_label,
                (time.monotonic() - started_at) * 1000,
            )
            raise CodexAppServerError(f"Codex {method} timed out.") from exc
        except BaseException as exc:
            self._pending.pop(request_id, None)
            logger.debug(
                "Codex protocol request aborted (method=%s, error=%s)",
                method_label,
                type(exc).__name__,
            )
            raise

        if "error" in response:
            error = response["error"]
            message = _error_message(error) or "unknown error"
            logger.warning(
                "Codex protocol request failed (method=%s, duration_ms=%.1f)",
                method_label,
                (time.monotonic() - started_at) * 1000,
            )
            raise CodexAppServerError(f"Codex {method} failed: {message}")
        result = response.get("result", {})
        if not isinstance(result, dict):
            logger.debug(
                "Codex protocol request completed with non-object result "
                "(method=%s, duration_ms=%.1f)",
                method_label,
                (time.monotonic() - started_at) * 1000,
            )
            return {}
        logger.debug(
            "Codex protocol request completed (method=%s, result_keys=%d, duration_ms=%.1f)",
            method_label,
            len(result),
            (time.monotonic() - started_at) * 1000,
        )
        return result

    async def _send(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise CodexAppServerError("The Codex App Server is not running.")
        async with self._write_lock:
            logger.debug(
                "Writing Codex protocol message (method=%s, has_request_id=%s)",
                _safe_log_label(message.get("method")),
                "id" in message,
            )
            process.stdin.write(
                (json.dumps(message, separators=(",", ":")) + "\n").encode()
            )
            await process.stdin.drain()

    async def _read_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        failure: BaseException | None = None
        cancelled = False
        try:
            async for raw_line in process.stdout:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Ignored malformed Codex App Server output")
                    continue
                if "method" in message:
                    logger.debug(
                        "Received Codex protocol request (method=%s, expects_response=%s)",
                        _safe_log_label(message.get("method")),
                        "id" in message,
                    )
                    if "id" in message:
                        task = asyncio.create_task(self._handle_server_request(message))
                        self._server_tasks.add(task)
                        task.add_done_callback(self._server_task_done)
                    else:
                        self._handle_notification(message)
                elif "id" in message:
                    logger.debug(
                        "Received Codex protocol response (has_error=%s, pending=%s)",
                        "error" in message,
                        message.get("id") in self._pending,
                    )
                    request_id = message["id"]
                    future = self._pending.pop(request_id, None)
                    if future is not None and not future.done():
                        future.set_result(message)
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:  # noqa: BLE001 - fail all pending protocol waits
            failure = exc
            logger.error(
                "Codex App Server output loop failed (error=%s)",
                type(exc).__name__,
            )
        finally:
            failure = failure or CodexAppServerError("The Codex App Server exited.")
            if not cancelled:
                logger.warning("Codex App Server output loop stopped")
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(failure)
            self._pending.clear()
            for state in self._turns.values():
                if not state.done.done():
                    state.done.set_exception(failure)

    def _server_task_done(self, task: asyncio.Task[Any]) -> None:
        self._server_tasks.discard(task)
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        except asyncio.InvalidStateError:
            return
        if error is not None:
            logger.error(
                "Codex server request handler failed (error=%s)",
                type(error).__name__,
            )

    async def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        async for raw_line in process.stderr:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                self._stderr_tail = (self._stderr_tail + [line])[-20:]
                logger.debug(
                    "Codex App Server emitted stderr output (characters=%d)",
                    len(line),
                )

    async def _handle_server_request(self, message: dict[str, Any]) -> None:
        method = str(message.get("method") or "")
        params = message.get("params") or {}
        logger.debug(
            "Handling Codex server request (method=%s, parameter_keys=%d)",
            _safe_log_label(method),
            len(params) if isinstance(params, dict) else 0,
        )
        try:
            result = await self._server_request_result(
                method, params, request_id=message.get("id")
            )
        except CodexAppServerError as exc:
            logger.warning(
                "Codex server request rejected (method=%s, error=%s)",
                _safe_log_label(method),
                type(exc).__name__,
            )
            await self._send(
                {
                    "id": message["id"],
                    "error": {"code": -32000, "message": str(exc)},
                }
            )
            return
        await self._send({"id": message["id"], "result": result})
        logger.debug(
            "Codex server request completed (method=%s, result_keys=%d)",
            _safe_log_label(method),
            len(result),
        )

    async def _server_request_result(
        self,
        method: str,
        params: dict[str, Any],
        *,
        request_id: Any | None = None,
    ) -> dict[str, Any]:
        state = self._find_turn(params)
        channel = state.channel if state else None
        user_id = state.user_id if state else None
        if method in {"item/commandExecution/requestApproval", "execCommandApproval"}:
            return await self._approval_request(
                channel,
                user_id,
                state,
                params,
                kind="command",
                request_id=request_id,
            )
        if method in {"item/fileChange/requestApproval", "applyPatchApproval"}:
            return await self._approval_request(
                channel,
                user_id,
                state,
                params,
                kind="file_change",
                request_id=request_id,
            )
        if method == "item/permissions/requestApproval":
            return await self._approval_request(
                channel,
                user_id,
                state,
                params,
                kind="permissions",
                request_id=request_id,
            )
        if method == "item/tool/requestUserInput":
            return await self._request_user_input(channel, user_id, params, state=state)
        if method == "mcpServer/elicitation/request":
            return await self._mcp_elicitation(channel, user_id, params, state=state)
        if method == "item/tool/call":
            return await self._dynamic_tool_call(state, params)
        raise CodexAppServerError(f"Unsupported server request: {method}")

    async def _send_turn_message(
        self,
        state: _TurnState | None,
        *,
        channel: discord.abc.Messageable | None = None,
        **kwargs: Any,
    ) -> Any:
        """Send through an interaction webhook when a turn has no bot channel."""
        if state is not None and state.interaction_sender is not None:
            message = await state.interaction_sender(**kwargs)
        else:
            target = channel or (state.channel if state is not None else None)
            if target is None:
                raise discord.DiscordException(
                    "No Discord message destination is available."
                )
            message = await target.send(**kwargs)
        view = kwargs.get("view")
        if view is not None and self._view_registrar is not None:
            with contextlib.suppress(Exception):
                await self._view_registrar(view, message)
        return message

    async def _approval_request(
        self,
        channel: discord.abc.Messageable | None,
        user_id: int | None,
        state: _TurnState | None,
        params: dict[str, Any],
        *,
        kind: str,
        request_id: Any | None = None,
    ) -> dict[str, Any]:
        if state is None:
            logger.warning(
                "Could not surface Codex approval request because no active turn "
                "matched it"
            )
            return self._approval_result(kind, params, approved=False)
        thread_id = str(params.get("threadId") or (state.thread_id if state else ""))
        turn_id = str(params.get("turnId") or self._turn_id_for_state(state))
        approval_id = params.get("approvalId")
        approval_id = str(approval_id) if approval_id else None
        item_id = str(
            params.get("itemId")
            or approval_id
            or f"{kind}-approval-{request_id or turn_id or thread_id}"
        )
        if not channel or user_id is None or not thread_id or not turn_id:
            logger.warning(
                "Could not surface Codex approval request because its Discord "
                "routing information is unavailable"
            )
            return self._approval_result(kind, params, approved=False)

        if not state.allow_tools:
            logger.info("Codex approval request is unavailable for this turn")
            await self._announce_unavailable_approval(
                state,
                channel,
                kind,
                params,
                "tool access is disabled for this turn",
            )
            return self._approval_result(kind, params, approved=False)
        if not self._has_turn_server_admin_access(channel, user_id, state.user):
            state.allow_tools = False
            logger.info(
                "Codex approval request is unavailable after administrator access "
                "changed"
            )
            await self._announce_unavailable_approval(
                state,
                channel,
                kind,
                params,
                "administrator access is no longer available",
            )
            return self._approval_result(kind, params, approved=False)

        risk = self._approval_risk(kind, params)
        if self._should_auto_approve(risk):
            logger.info(
                "Auto-approved Codex request (kind=%s, level=%s, risk=%s)",
                _safe_log_label(kind),
                self._approval_level,
                risk,
            )
            return self._approval_result(kind, params, approved=True)

        logger.info("Codex requested user approval (kind=%s)", _safe_log_label(kind))

        key = ":".join((str(user_id), thread_id, turn_id, item_id, approval_id or ""))
        pending = _PendingApproval(
            key=key,
            user_id=user_id,
            channel_id=getattr(channel, "id", None),
            thread_id=thread_id,
            turn_id=turn_id,
            item_id=item_id,
            approval_id=approval_id,
            kind=kind,
            params=dict(params),
            future=asyncio.get_running_loop().create_future(),
        )
        self._pending_approvals[key] = pending
        view: _DecisionView | None = None

        async def resolve_from_button(
            decision: str, current_user: discord.abc.User
        ) -> None:
            if pending.future.done():
                return
            if not self._has_current_server_admin_access(
                channel, user_id, current_user=current_user
            ):
                state.allow_tools = False
                logger.info(
                    "Rejected approval button after administrator access changed"
                )
                approved = False
            else:
                approved = decision == "accept"
            pending.future.set_result(
                self._approval_result(kind, params, approved=approved)
            )
            self._pending_approvals.pop(key, None)

        try:
            view = _DecisionView(
                user_id,
                [
                    (
                        self._frontend_label(
                            channel, "label:approve_button", "Approve"
                        ),
                        "accept",
                        discord.ButtonStyle.success,
                    ),
                    (
                        self._frontend_label(channel, "label:deny_button", "Deny"),
                        "decline",
                        discord.ButtonStyle.danger,
                    ),
                ],
                on_decision=resolve_from_button,
            )
            summary = self._approval_summary(kind, params)
            reason = _safe_approval_reason(params.get("reason"))
            description = reason or summary
            embed = self._frontend_embed(
                channel,
                "label:approval_needed",
                "Approval needed",
                description,
                context={"reason": reason, "status": "approval"},
                color=discord.Color.orange(),
            )
            embed.set_footer(text="You can also use /approve or /deny.")
            await self._send_turn_message(
                state,
                channel=channel,
                embed=embed,
                view=view,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            return await asyncio.wait_for(
                asyncio.shield(pending.future),
                _env_float("CODEX_APPROVAL_TIMEOUT", 300),
            )
        except (asyncio.TimeoutError, discord.DiscordException):
            logger.warning(
                "Codex approval request ended without approval (kind=%s)",
                _safe_log_label(kind),
            )
            return self._approval_result(kind, params, approved=False)
        finally:
            if view is not None:
                view.stop()
            if self._pending_approvals.get(key) is pending:
                self._pending_approvals.pop(key, None)

    @staticmethod
    def _approval_strings(value: Any, *, depth: int = 0) -> Iterable[str]:
        """Yield bounded string values from JSON approval parameters."""
        if depth > 5:
            return
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for nested in value.values():
                yield from CodexTransportMixin._approval_strings(
                    nested, depth=depth + 1
                )
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield from CodexTransportMixin._approval_strings(
                    nested, depth=depth + 1
                )

    @classmethod
    def _approval_command_strings(cls, params: dict[str, Any]) -> tuple[str, ...]:
        values: list[str] = []
        for key in (
            "command",
            "cmd",
            "commandLine",
            "command_line",
            "input",
            "argv",
        ):
            values.extend(cls._approval_strings(params.get(key)))
        return tuple(values)

    def _approval_risk(self, kind: str, params: dict[str, Any]) -> str:
        """Classify an emitted approval request for the configured tier."""
        if kind in {"file_change", "permissions"}:
            return _APPROVAL_RISK_VERY_DANGEROUS
        if str(params.get("kind") or "").casefold() == "writestdin":
            return _APPROVAL_RISK_VERY_DANGEROUS
        if params.get("networkApprovalContext"):
            return _APPROVAL_RISK_VERY_DANGEROUS

        actions = params.get("commandActions")
        if isinstance(actions, list) and any(
            isinstance(action, dict)
            and str(action.get("type") or "").casefold()
            in {"write", "delete", "move", "applypatch"}
            for action in actions
        ):
            return _APPROVAL_RISK_VERY_DANGEROUS

        all_strings = tuple(self._approval_strings(params))
        for text in all_strings:
            if _APPROVAL_VERY_DANGEROUS_RE.search(text):
                return _APPROVAL_RISK_VERY_DANGEROUS
            for match in _APPROVAL_PATH_RE.finditer(text):
                candidate = match.group(0).rstrip(".,:!?)]}")
                path = _path_from_value(candidate)
                if path is None:
                    # A path in a foreign format is safer to treat as outside
                    # the configured workspace than to auto-approve it.
                    if ":\\" in candidate or candidate.startswith("\\\\"):
                        return _APPROVAL_RISK_VERY_DANGEROUS
                    continue
                if not _path_is_under(path, self._shared_workspace_roots):
                    return _APPROVAL_RISK_VERY_DANGEROUS

        commands = self._approval_command_strings(params)
        if any(_APPROVAL_SAFE_COMMAND_RE.match(command) for command in commands):
            return _APPROVAL_RISK_SAFE
        return _APPROVAL_RISK_DANGEROUS

    def _should_auto_approve(self, risk: str) -> bool:
        """Return whether an emitted approval can be resolved without Discord."""
        if self._approval_level == "high":
            return False
        if self._approval_level == "medium":
            return risk == _APPROVAL_RISK_SAFE
        return risk != _APPROVAL_RISK_VERY_DANGEROUS

    async def _announce_unavailable_approval(
        self,
        state: _TurnState,
        channel: discord.abc.Messageable,
        kind: str,
        params: dict[str, Any],
        reason: str,
    ) -> None:
        """Tell Discord when policy prevents an approval from being actionable."""
        description = (
            f"Codex requested approval to {self._approval_summary(kind, params)}, "
            "but this request cannot be approved in the current Discord session "
            f"because {reason}."
        )
        try:
            await self._send_turn_message(
                state,
                channel=channel,
                embed=self._frontend_embed(
                    channel,
                    "label:approval_needed",
                    "Approval needed",
                    description,
                    context={"reason": reason, "status": "unavailable"},
                    color=discord.Color.orange(),
                ),
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except discord.DiscordException:
            logger.warning(
                "Could not surface unavailable Codex approval request (kind=%s)",
                _safe_log_label(kind),
            )

    @staticmethod
    def _approval_summary(kind: str, params: dict[str, Any]) -> str:
        if kind == "file_change":
            return "apply changes to workspace files"
        if kind == "permissions":
            return "grant additional tool permissions for this turn"
        if str(params.get("kind") or "").casefold() == "writestdin":
            return "provide input to an existing command"
        if params.get("networkApprovalContext"):
            return "access an external network resource"
        return "run a command using the configured Codex tools"

    @staticmethod
    def _approval_result(
        kind: str, params: dict[str, Any], *, approved: bool
    ) -> dict[str, Any]:
        if kind == "permissions":
            return {
                "permissions": params.get("permissions") if approved else {},
                "scope": "turn",
            }
        return {"decision": "accept" if approved else "decline"}

    async def _decision(
        self,
        state: _TurnState | None,
        channel: discord.abc.Messageable | None,
        user_id: int | None,
        content: str,
        choices: list[tuple[str, str, discord.ButtonStyle]],
    ) -> str:
        if channel is None:
            logger.warning("Codex choice request has no Discord channel")
            return "decline"
        view = _DecisionView(
            user_id,
            [
                (
                    self._frontend_label(
                        channel,
                        (
                            "label:approve_button"
                            if value == "accept"
                            else "label:deny_button"
                            if value == "decline"
                            else "label:answer_button"
                        ),
                        label,
                    ),
                    value,
                    style,
                )
                for label, value, style in choices
            ],
        )
        try:
            await self._send_turn_message(
                state,
                channel=channel,
                content=_subtext(
                    "Confirmation needed. "
                    + (
                        _safe_intermediate_text(content, 1800)
                        or "Codex needs your confirmation."
                    )
                ),
                view=view,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            await view.wait()
        except discord.DiscordException:
            logger.warning("Codex choice request could not be delivered")
            return "decline"
        logger.info(
            "Codex choice request resolved (decision=%s)", view.value or "decline"
        )
        return view.value or "decline"

    async def _request_user_input(
        self,
        channel: Any | None,
        user_id: int | None,
        params: dict[str, Any],
        *,
        state: _TurnState | None = None,
    ) -> dict[str, Any]:
        questions = [
            item for item in params.get("questions", []) if isinstance(item, dict)
        ]
        if channel is None or not questions:
            logger.warning("Codex user-input request has no usable questions")
            return {"answers": {}}
        logger.info(
            "Codex requested user input (questions=%d, multiple_choice=%s)",
            len(questions),
            bool(questions[0].get("options")),
        )
        view = _UserInputView(
            user_id,
            questions,
            channel=channel,
            customizer=self._frontend_customizer,
        )
        try:
            message = view.message_kwargs()
            message["allowed_mentions"] = discord.AllowedMentions.none()
            await self._send_turn_message(state, channel=channel, **message)
            await view.wait()
        except discord.DiscordException:
            logger.warning("Codex user-input request could not be delivered")
            return {"answers": {}}
        logger.info("Codex user-input request resolved")
        return view.value or {"answers": {}}

    async def _mcp_elicitation(
        self,
        channel: discord.abc.Messageable | None,
        user_id: int | None,
        params: dict[str, Any],
        *,
        state: _TurnState | None = None,
    ) -> dict[str, Any]:
        if channel is None:
            logger.warning("Codex elicitation request has no Discord channel")
            return {"action": "decline"}
        logger.info(
            "Codex elicitation request received (mode=%s)",
            _safe_log_label(params.get("mode")),
        )
        message = _truncate(
            params.get("message") or "Codex needs input from an MCP server.", 1700
        )
        if params.get("mode") == "url":
            decision = await self._decision(
                state,
                channel,
                user_id,
                f"{message}\n{params.get('url', '')}",
                [
                    ("Open / allow", "accept", discord.ButtonStyle.success),
                    ("Decline", "decline", discord.ButtonStyle.danger),
                ],
            )
            return {"action": decision if decision == "accept" else "decline"}

        schema = params.get("requestedSchema") or {}
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        names = ", ".join(str(name) for name in properties) or "the requested fields"
        view = _FormView(
            user_id,
            prompt=f"JSON object with these fields: {names}",
            channel=channel,
            customizer=self._frontend_customizer,
        )
        try:
            await self._send_turn_message(
                state,
                channel=channel,
                content=_subtext(
                    f"{_safe_intermediate_text(message) or 'Codex needs your input.'}\n"
                    f"Reply with a JSON object containing: {names}"
                ),
                view=view,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            await view.wait()
        except discord.DiscordException:
            logger.warning("Codex elicitation request could not be delivered")
            return {"action": "decline"}
        if not isinstance(view.value, dict):
            return {"action": "decline"}
        return {"action": "accept", "content": view.value}

    async def _dynamic_create_thread(
        self,
        state: _TurnState,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        def result(text: str, *, success: bool) -> dict[str, Any]:
            return {
                "contentItems": [{"type": "inputText", "text": text}],
                "success": success,
            }

        arguments = params.get("arguments")
        if isinstance(arguments, str):
            with contextlib.suppress(ValueError):
                arguments = json.loads(arguments)
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return result(
                "create_thread accepts an optional name and opening_message.",
                success=False,
            )
        raw_name = arguments.get("name")
        if raw_name:
            name = str(raw_name)
        else:
            prompt = re.sub(r"\s+", " ", state.user_prompt or "").strip()
            name = f"Codex: {prompt}" if prompt else "Codex request"
        name = re.sub(r"\s+", " ", name).strip()
        name = _truncate(name, 100).strip(" -:;,.()[]{}") or "Codex request"

        channel = state.channel
        if state.discord_thread is not None:
            await self._emit_thread_opening(state, arguments.get("opening_message"))
            return result(
                "Thread setup is complete. Continue with the user's request now; "
                "do not mention thread setup or call create_thread again.",
                success=True,
            )
        if isinstance(channel, discord.Thread):
            # A user may ask for a thread while already inside one. Discord does
            # not support nesting threads, so apply the requested name to the
            # current thread and let the turn continue there.
            if raw_name:
                await self._apply_discord_thread_name(channel, name)
                if state.thread_id:
                    with contextlib.suppress(Exception):
                        await self.set_thread_name(state.thread_id, name)
            await self._emit_thread_opening(state, arguments.get("opening_message"))
            return result(
                "Thread setup is complete. Continue with the user's request now; "
                "do not mention thread setup or call create_thread again.",
                success=True,
            )
        if channel is None or getattr(channel, "guild", None) is None:
            logger.info("Rejected Discord thread creation outside a server")
            return result(
                "Discord threads are only available in server channels.",
                success=False,
            )

        create_thread = getattr(state.thread_source, "create_thread", None)
        if not callable(create_thread):
            create_thread = getattr(channel, "create_thread", None)
        if not callable(create_thread):
            logger.info("Discord thread creation is unavailable in this channel")
            return result(
                "Discord cannot create a thread in the current channel.",
                success=False,
            )
        try:
            create_thread_async = cast(Callable[..., Awaitable[Any]], create_thread)
            response_channel = await create_thread_async(
                name=name,
                auto_archive_duration=1440,
            )
        except (discord.DiscordException, TypeError, RuntimeError) as exc:
            logger.info(
                "Could not create a Discord thread from the Codex tool (error=%s)",
                type(exc).__name__,
            )
            return result(
                "Discord could not create the requested thread; continue in the current channel.",
                success=False,
            )
        if response_channel is None or not callable(
            getattr(response_channel, "send", None)
        ):
            logger.info("Discord thread creation returned no usable channel")
            return result(
                "Discord did not return a usable thread; continue in the current channel.",
                success=False,
            )

        await self._apply_discord_thread_name(response_channel, name)
        if state.thread_id:
            try:
                await self.set_thread_name(state.thread_id, name)
            except (CodexAppServerError, OSError) as exc:
                logger.info(
                    "Could not assign the Codex session name for the Discord thread "
                    "(error=%s)",
                    type(exc).__name__,
                )
        state.discord_thread = response_channel
        state.channel = response_channel
        if state.on_channel_change is not None:
            try:
                state.on_channel_change(response_channel)
            except Exception as exc:  # noqa: BLE001 - routing is supplementary
                logger.warning(
                    "Discord response routing could not switch to the new thread "
                    "(error=%s)",
                    type(exc).__name__,
                )
        await self._emit_thread_opening(state, arguments.get("opening_message"))
        logger.info("Codex created a Discord response thread")
        return result(
            "Discord thread created. Continue the response in the new thread "
            "without repeating the opening response.",
            success=True,
        )

    async def _emit_thread_opening(
        self,
        state: _TurnState,
        value: Any,
    ) -> None:
        """Deliver a Codex-provided opening message through the intermediate path."""
        if state.discord_thread_opening_sent:
            return
        opening_message = _safe_intermediate_text(value, 1900)
        if not opening_message:
            return
        payload = {
            "type": "agentMessage",
            "phase": "commentary",
            "text": opening_message,
        }
        try:
            if state.on_event is not None:
                await state.on_event("thread_opening", payload)
            elif state.interaction_sender is not None:
                await state.interaction_sender(
                    content=_subtext(opening_message),
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            elif state.channel is not None:
                # This fallback is used only by direct callers without a
                # Discord delivery callback; normal turns use the callback.
                await state.channel.send(
                    content=_subtext(opening_message),
                    allowed_mentions=discord.AllowedMentions.none(),
                )
        except Exception as exc:  # noqa: BLE001 - opening text must not block the turn
            logger.info(
                "Could not deliver the Discord thread opening response (error=%s)",
                type(exc).__name__,
            )
            return
        state.discord_thread_opening_sent = True

    async def _apply_discord_thread_name(
        self,
        channel: discord.abc.Messageable,
        name: str,
    ) -> None:
        edit = getattr(channel, "edit", None)
        if not callable(edit):
            return
        try:
            edit_async = cast(Callable[..., Awaitable[Any]], edit)
            await edit_async(name=name)
        except (discord.DiscordException, TypeError, RuntimeError) as exc:
            logger.info(
                "Could not assign the Discord thread name (error=%s)",
                type(exc).__name__,
            )

    async def _dynamic_tool_call(
        self,
        state: _TurnState | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if state is None or state.channel is None:
            logger.warning("Rejected Codex Discord tool request without a channel")
            return {
                "contentItems": [
                    {"type": "inputText", "text": "No Discord channel is available."}
                ],
                "success": False,
            }
        if not state.allow_discord_tools:
            logger.info("Rejected Discord tool request for an account-installed turn")
            return {
                "contentItems": [
                    {
                        "type": "inputText",
                        "text": "Discord tools are unavailable for this installation.",
                    }
                ],
                "success": False,
            }
        if not state.allow_tools or not self._has_turn_server_admin_access(
            state.channel, state.user_id, state.user
        ):
            state.allow_tools = False
            logger.info("Rejected Codex Discord tool request for a restricted user")
            return {
                "contentItems": [
                    {
                        "type": "inputText",
                        "text": "Discord tools are unavailable to this requester.",
                    }
                ],
                "success": False,
            }
        tool = str(params.get("tool") or "")
        logger.info(
            "Codex requested Discord tool (tool=%s)",
            _safe_log_label(tool),
        )
        if tool not in {"send_message", "sendMessage", "create_thread"} or params.get(
            "namespace"
        ) not in {None, "discord"}:
            return {
                "contentItems": [
                    {"type": "inputText", "text": f"Unknown Discord tool: {tool}"}
                ],
                "success": False,
            }
        if tool == "create_thread":
            return await self._dynamic_create_thread(state, params)
        arguments = params.get("arguments")
        if isinstance(arguments, str):
            with contextlib.suppress(ValueError):
                arguments = json.loads(arguments)
        if not isinstance(arguments, dict) or not arguments.get("content"):
            return {
                "contentItems": [
                    {"type": "inputText", "text": "send_message requires content."}
                ],
                "success": False,
            }
        files: list[discord.File] = []
        raw_files = arguments.get("files") or arguments.get("attachments") or []
        if isinstance(raw_files, (str, dict)):
            raw_files = [raw_files]
        if isinstance(raw_files, list):
            for item in raw_files[:10]:
                raw_path = item.get("path") if isinstance(item, dict) else item
                path = _path_from_value(raw_path)
                if path is None or not _path_is_under(
                    path, self._shared_workspace_roots
                ):
                    continue
                try:
                    if path.is_symlink():
                        continue
                    resolved = path.resolve(strict=True)
                    if (
                        not _path_is_under(resolved, self._shared_workspace_roots)
                        or not resolved.is_file()
                    ):
                        continue
                    if resolved.stat().st_size > MAX_ATTACHMENT_BYTES:
                        continue
                    files.append(discord.File(str(resolved), filename=resolved.name))
                except OSError:
                    continue
        logger.debug("Codex Discord tool prepared (files=%d)", len(files))
        try:
            await state.channel.send(
                _truncate(arguments["content"], 2000),
                files=files or None,
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except (discord.DiscordException, OSError) as exc:
            logger.warning(
                "Codex Discord tool could not send a message (error=%s)",
                type(exc).__name__,
            )
            return {
                "contentItems": [
                    {
                        "type": "inputText",
                        "text": "Discord could not send the message.",
                    }
                ],
                "success": False,
            }
        finally:
            for file in files:
                file.close()
        return {
            "contentItems": [{"type": "inputText", "text": "Message sent."}],
            "success": True,
        }

    def _find_turn(self, params: dict[str, Any]) -> _TurnState | None:
        turn_id = params.get("turnId")
        thread_id = params.get("threadId")
        if turn_id and str(turn_id) in self._turns:
            state = self._turns[str(turn_id)]
            if thread_id is None or state.thread_id == str(thread_id):
                return state
        if thread_id:
            normalized_thread_id = str(thread_id)
            for state in reversed(tuple(self._turns.values())):
                if state.thread_id == normalized_thread_id:
                    return state
        return None

    def _turn_id_for_state(self, target: _TurnState) -> str:
        """Return the local turn key for an app-server state object."""
        for turn_id, state in reversed(tuple(self._turns.items())):
            if state is target:
                return turn_id
        return ""

    def _emit(self, state: _TurnState, event: str, payload: dict[str, Any]) -> None:
        if state.on_event is None:
            return
        logger.debug(
            "Dispatching Codex event to Discord delivery (event=%s)",
            _safe_log_label(event),
        )
        task = asyncio.create_task(
            cast(Coroutine[Any, Any, None], state.on_event(event, payload))
        )
        state.event_tasks.append(task)

    def _background_send(
        self, channel: discord.abc.Messageable, content: discord.Embed
    ) -> None:
        self._background_send_callback(channel.send, content)

    def _background_send_callback(
        self, send: Callable[..., Awaitable[Any]], content: discord.Embed
    ) -> None:
        task = asyncio.create_task(cast(Coroutine[Any, Any, Any], send(embed=content)))
        self._server_tasks.add(task)
        task.add_done_callback(self._server_task_done)
