"""Ephemeral synthesis, recap, presence, and attachment workers."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import time
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import discord

from .policy import (
    AUDIO_ATTACHMENT_SUFFIXES,
    IMAGE_SUFFIXES,
    MAX_ATTACHMENT_BATCH_BYTES,
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENT_TEXT_BYTES,
    MAX_ATTACHMENTS_PER_REQUEST,
    TEXT_ATTACHMENT_SUFFIXES,
)
from ..audio import AudioOutput, AudioProtocolError
from ..core import (
    AGENT_NAME,
    BASE_PRIORS,
    DEFAULT_REASONING_EFFORT,
    CodexAppServerError,
    _Session,
    _TurnState,
    _codex_logger,
    _path_from_value,
    _path_is_under,
    _truncate,
)
from .prompts import (
    _ASSESSMENT_COMPLEXITIES,
    _ASSESSMENT_DEVELOPER_INSTRUCTIONS,
    _ASSESSMENT_OUTPUT_SCHEMA,
    _NIGHTLY_RECAP_DEVELOPER_INSTRUCTIONS,
    _NIGHTLY_RECAP_OUTPUT_SCHEMA,
    _PRESENCE_ACTIVITY_TYPES,
    _PRESENCE_DEVELOPER_INSTRUCTIONS,
    _PRESENCE_OUTPUT_SCHEMA,
)

logger = _codex_logger()


class CodexWorkerMixin:
    if TYPE_CHECKING:
        _model: str | None
        _approval_level: str
        _adaptive_reasoning: bool
        _self_improvement_enabled: bool

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    async def synthesize_response(self, text: str) -> tuple[AudioOutput, ...]:
        """Create optional Discord audio without making TTS failure a chat failure."""
        if not self._audio.tts.enabled:
            return ()
        try:
            return await self._audio.synthesize_many(text)
        except AudioProtocolError as exc:
            logger.warning(
                "Optional TTS response failed (error=%s)", type(exc).__name__
            )
            return ()

    async def generate_nightly_recap(
        self,
        prompt: str,
        *,
        session_key: str | None = None,
        timeout: float | None = None,
    ) -> str | None:
        """Generate one private, no-tool recap without extending a user thread."""
        await self._ensure_running()
        session_id = f"__nightly_recap__:{time.monotonic_ns()}"
        session = _Session(
            key=session_id,
            personality_name=(
                self.active_personality(session_key)
                if session_key is not None
                else None
            ),
        )
        self._sessions[session_id] = session
        state: _TurnState | None = None
        thread_id: str | None = None
        turn_id: str | None = None
        wait_timeout = self._nightly_recap_timeout if timeout is None else timeout
        request_timeout = max(1.0, min(wait_timeout, self._request_timeout))
        try:
            thread_result = await self._request(
                "thread/start",
                {
                    "cwd": str(self._attachment_root),
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                    "ephemeral": True,
                    "runtimeWorkspaceRoots": [],
                    "baseInstructions": self._system_instructions(
                        session, allow_tools=False
                    ),
                    "developerInstructions": _NIGHTLY_RECAP_DEVELOPER_INSTRUCTIONS,
                    **({"model": self._model} if self._model is not None else {}),
                },
                timeout=request_timeout,
            )
            thread_id = str((thread_result.get("thread") or {}).get("id") or "")
            if not thread_id:
                return None
            turn_result = await self._request(
                "turn/start",
                {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": prompt}],
                    "effort": "low",
                    "outputSchema": _NIGHTLY_RECAP_OUTPUT_SCHEMA,
                    **({"model": self._model} if self._model is not None else {}),
                },
                timeout=request_timeout,
            )
            turn_id = str((turn_result.get("turn") or {}).get("id") or "")
            if not turn_id:
                return None
            session.thread_id = thread_id
            session.turn_id = turn_id
            state = _TurnState(
                thread_id=thread_id,
                session=session,
                allow_tools=False,
            )
            self._turns[turn_id] = state
            response = await self._wait_for_turn(
                session_id,
                session,
                state,
                turn_id,
                timeout=wait_timeout,
            )
            return self._parse_nightly_recap(response)
        except asyncio.CancelledError:
            if thread_id and turn_id:
                with contextlib.suppress(Exception):
                    await self._request(
                        "turn/interrupt",
                        {"threadId": thread_id, "turnId": turn_id},
                        timeout=2.0,
                    )
            raise
        finally:
            if turn_id:
                self._turns.pop(turn_id, None)
            self._sessions.pop(session_id, None)

    @staticmethod
    def _parse_nightly_recap(text: str) -> str | None:
        """Parse the bounded recap field returned by the ephemeral turn."""
        candidates = [text.strip()]
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))
        for candidate in candidates:
            candidate = candidate.removeprefix("```json").removesuffix("```").strip()
            try:
                value = json.loads(candidate)
            except (TypeError, ValueError):
                continue
            if not isinstance(value, dict) or not isinstance(value.get("recap"), str):
                continue
            recap = re.sub(r"\s+", " ", value["recap"]).strip()
            if recap:
                return _truncate(recap, 8000)
        return None

    async def generate_presence(
        self,
        prompt: str,
        *,
        session_key: str | None = None,
        timeout: float = 8.0,
    ) -> dict[str, str] | None:
        """Generate one short activity line in a disposable, no-tool turn."""
        await self._ensure_running()
        session_id = f"__presence__:{time.monotonic_ns()}"
        session = _Session(
            key=session_id,
            personality_name=(
                self.active_personality(session_key)
                if session_key is not None
                else None
            ),
        )
        self._sessions[session_id] = session
        state: _TurnState | None = None
        thread_id: str | None = None
        turn_id: str | None = None
        request_timeout = max(1.0, min(timeout, self._request_timeout))
        try:
            thread_result = await self._request(
                "thread/start",
                {
                    "cwd": str(self._attachment_root),
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                    "ephemeral": True,
                    "runtimeWorkspaceRoots": [],
                    "baseInstructions": self._system_instructions(
                        session, allow_tools=False
                    ),
                    "developerInstructions": _PRESENCE_DEVELOPER_INSTRUCTIONS,
                    **({"model": self._model} if self._model is not None else {}),
                },
                timeout=request_timeout,
            )
            thread_id = str((thread_result.get("thread") or {}).get("id") or "")
            if not thread_id:
                return None
            turn_result = await self._request(
                "turn/start",
                {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": prompt}],
                    "effort": "low",
                    "outputSchema": _PRESENCE_OUTPUT_SCHEMA,
                    **({"model": self._model} if self._model is not None else {}),
                },
                timeout=request_timeout,
            )
            turn_id = str((turn_result.get("turn") or {}).get("id") or "")
            if not turn_id:
                return None
            session.thread_id = thread_id
            session.turn_id = turn_id
            state = _TurnState(
                thread_id=thread_id,
                session=session,
                allow_tools=False,
            )
            self._turns[turn_id] = state
            response = await self._wait_for_turn(
                session_id,
                session,
                state,
                turn_id,
                timeout=timeout,
            )
            return self._parse_presence(response)
        except asyncio.CancelledError:
            if thread_id and turn_id:
                with contextlib.suppress(Exception):
                    await self._request(
                        "turn/interrupt",
                        {"threadId": thread_id, "turnId": turn_id},
                        timeout=2.0,
                    )
            raise
        finally:
            if turn_id:
                self._turns.pop(turn_id, None)
            self._sessions.pop(session_id, None)

    @staticmethod
    def _parse_presence(text: str) -> dict[str, str] | None:
        """Parse a bounded activity object without retaining the turn."""
        candidates = [text.strip()]
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))
        for candidate in candidates:
            candidate = candidate.removeprefix("```json").removesuffix("```").strip()
            try:
                value = json.loads(candidate)
            except (TypeError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            activity_type = str(value.get("activity_type") or "").casefold()
            activity_text = value.get("text")
            if activity_type not in _PRESENCE_ACTIVITY_TYPES or not isinstance(
                activity_text, str
            ):
                continue
            activity_text = re.sub(r"\s+", " ", activity_text).strip()
            activity_text = activity_text[:128].rstrip()
            if activity_text:
                return {
                    "activity_type": activity_type,
                    "text": activity_text,
                }
        return None

    async def _select_reasoning_effort(
        self, prompt: str, attachments: Iterable[discord.Attachment]
    ) -> str:
        if not self._adaptive_reasoning:
            logger.debug("Adaptive reasoning disabled; using medium")
            return DEFAULT_REASONING_EFFORT

        try:
            models = await self.available_models()
        except (CodexAppServerError, OSError) as exc:
            models = ()
            logger.warning(
                "Could not load Codex model capabilities for reasoning selection "
                "(error=%s)",
                type(exc).__name__,
            )

        assessment_effort = self._supported_effort("low", models)
        assessment = await self._assess_request(
            prompt, attachments, effort=assessment_effort
        )
        if assessment is None:
            logger.warning("Codex reasoning pre-assessment unavailable; using medium")
            return DEFAULT_REASONING_EFFORT
        if not assessment["requires_tool"]:
            selected = self._supported_effort("low", models)
            logger.debug("Pre-assessment selected %s for a no-tool request", selected)
            return selected

        requested = {
            "simple": "medium",
            "moderate": "medium",
            "complex": "high",
            "very_complex": "max",
        }[assessment["complexity"]]
        selected = self._supported_effort(requested, models)
        logger.debug(
            "Pre-assessment selected %s for a %s tool-backed request",
            selected,
            assessment["complexity"],
        )
        return selected

    async def _assess_request(
        self,
        prompt: str,
        attachments: Iterable[discord.Attachment],
        *,
        effort: str,
    ) -> dict[str, Any] | None:
        """Run a hidden, ephemeral planning turn before the user turn."""
        logger.debug("Starting hidden Codex reasoning pre-assessment")
        key = f"__assessment__:{time.monotonic_ns()}"
        session = _Session(key=key)
        state: _TurnState | None = None
        try:
            params: dict[str, Any] = {
                "cwd": self._cwd,
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "ephemeral": True,
                "baseInstructions": BASE_PRIORS,
                "developerInstructions": _ASSESSMENT_DEVELOPER_INSTRUCTIONS,
            }
            if self._model is not None:
                params["model"] = self._model
            result = await self._request("thread/start", params)
            thread_id = str((result.get("thread") or {}).get("id") or "")
            if not thread_id:
                raise CodexAppServerError(
                    "Codex did not return an assessment thread id."
                )

            turn_params: dict[str, Any] = {
                "threadId": thread_id,
                "input": [
                    {
                        "type": "text",
                        "text": self._assessment_prompt(prompt, attachments),
                    }
                ],
                "effort": effort,
                "outputSchema": _ASSESSMENT_OUTPUT_SCHEMA,
            }
            if self._model is not None:
                turn_params["model"] = self._model
            turn_result = await self._request("turn/start", turn_params)
            turn_id = str((turn_result.get("turn") or {}).get("id") or "")
            if not turn_id:
                raise CodexAppServerError("Codex did not return an assessment turn id.")

            session.thread_id = thread_id
            session.turn_id = turn_id
            state = _TurnState(thread_id=thread_id, session=session)
            self._turns[turn_id] = state
            text = await self._wait_for_turn(
                key,
                session,
                state,
                turn_id,
                timeout=self._assessment_timeout,
            )
            return self._parse_assessment(text)
        except (CodexAppServerError, OSError) as exc:
            logger.debug(
                "Hidden Codex reasoning pre-assessment failed (error=%s)",
                type(exc).__name__,
            )
            return None
        finally:
            if state is not None and state.event_tasks:
                await asyncio.gather(*state.event_tasks, return_exceptions=True)
            self._sessions.pop(key, None)

    @staticmethod
    def _assessment_prompt(
        prompt: str, attachments: Iterable[discord.Attachment]
    ) -> str:
        filenames = [
            str(getattr(item, "filename", "attachment")) for item in attachments
        ]
        attachment_text = ", ".join(filenames) if filenames else "none"
        return (
            "Classify this request without answering it. A tool is required when "
            "completing the request would need file, web, code, account, or other "
            "external state access. Return JSON only with complexity set to one of "
            "simple, moderate, complex, very_complex and requires_tool set to true "
            "or false.\n\n"
            f"<task>{prompt}</task>\n"
            f"<attachments>{attachment_text}</attachments>"
        )

    @staticmethod
    def _parse_assessment(text: str) -> dict[str, Any] | None:
        candidates = [text.strip()]
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))
        for candidate in candidates:
            candidate = candidate.removeprefix("```json").removesuffix("```").strip()
            try:
                value = json.loads(candidate)
            except (TypeError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            complexity = str(value.get("complexity") or "").casefold()
            requires_tool = value.get("requires_tool")
            if complexity in _ASSESSMENT_COMPLEXITIES and isinstance(
                requires_tool, bool
            ):
                return {
                    "complexity": complexity,
                    "requires_tool": requires_tool,
                }
        return None

    def _supported_effort(
        self, requested: str, models: Iterable[dict[str, Any]]
    ) -> str:
        model = self._selected_model_metadata(models)
        if model is None:
            return DEFAULT_REASONING_EFFORT
        advertised = model.get("supportedReasoningEfforts")
        supported: dict[str, str] = {}
        if isinstance(advertised, list):
            for option in advertised:
                value = (
                    option.get("reasoningEffort")
                    if isinstance(option, dict)
                    else option
                )
                if isinstance(value, str) and value.strip():
                    supported.setdefault(value.casefold(), value)
        if not supported:
            return DEFAULT_REASONING_EFFORT

        requested = requested.casefold()
        if requested == "low":
            candidates = ("low", "light", "minimal", "medium")
        elif requested == "max":
            candidates = ("max", "xhigh", "high", "medium", "low")
        elif requested == "xhigh":
            candidates = ("xhigh", "max", "high", "medium", "low")
        elif requested == "high":
            candidates = ("high", "xhigh", "max", "medium", "low")
        else:
            candidates = ("medium", "high", "xhigh", "max", "low")
        for candidate in candidates:
            if candidate in supported:
                return supported[candidate]
        return next(iter(supported.values()))

    def _selected_model_metadata(
        self, models: Iterable[dict[str, Any]]
    ) -> dict[str, Any] | None:
        values = tuple(models)
        if self._model is not None:
            for model in values:
                if model.get("id") == self._model:
                    return model
        return next((model for model in values if model.get("isDefault")), None) or (
            values[0] if values else None
        )

    async def _prepare_attachments(
        self, attachments: Iterable[Any]
    ) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        total_bytes = 0
        for index, attachment in enumerate(attachments, start=1):
            if index > MAX_ATTACHMENTS_PER_REQUEST:
                raise CodexAppServerError(
                    f"A request may include at most {MAX_ATTACHMENTS_PER_REQUEST} attachments."
                )
            filename = str(
                getattr(attachment, "filename", "attachment") or "attachment"
            )
            size = getattr(attachment, "size", None)
            if isinstance(size, int) and size > MAX_ATTACHMENT_BYTES:
                raise CodexAppServerError("An attachment is too large to process.")
            if (
                isinstance(size, int)
                and size >= 0
                and total_bytes + size > MAX_ATTACHMENT_BATCH_BYTES
            ):
                raise CodexAppServerError(
                    "The attachments are too large to process together."
                )
            read = getattr(attachment, "read", None)
            if not callable(read):
                prepared.append(
                    {
                        "type": "text",
                        "text": f"Attachment `{filename}` is available at {getattr(attachment, 'url', '')}.",
                    }
                )
                continue
            try:
                read_async = cast(Callable[[], Awaitable[Any]], read)
                raw = await read_async()
            except Exception as exc:
                raise CodexAppServerError(
                    "An attachment could not be downloaded."
                ) from exc
            if not isinstance(raw, bytes) or len(raw) > MAX_ATTACHMENT_BYTES:
                raise CodexAppServerError("An attachment is too large or invalid.")
            total_bytes += len(raw)
            if total_bytes > MAX_ATTACHMENT_BATCH_BYTES:
                raise CodexAppServerError(
                    "The attachments are too large to process together."
                )
            path = self._store_attachment(filename, raw)
            content_type = str(getattr(attachment, "content_type", "") or "")
            suffix = Path(filename).suffix.casefold()
            if content_type.casefold().startswith("image/"):
                prepared.append({"type": "localImage", "path": str(path)})
            elif (
                content_type.casefold().startswith("audio/")
                or suffix in AUDIO_ATTACHMENT_SUFFIXES
            ):
                prepared.append({"type": "localAudio", "path": str(path)})
                if self._audio.transcription.enabled:
                    try:
                        transcript = await self._audio.transcribe(
                            filename,
                            raw,
                            content_type,
                        )
                    except AudioProtocolError as exc:
                        raise CodexAppServerError(
                            f"Audio transcription failed: {exc}"
                        ) from exc
                    if transcript:
                        prepared.append(
                            {
                                "type": "text",
                                "text": (
                                    f"Transcript of attached audio `{filename}`:\n"
                                    f"{transcript}"
                                ),
                            }
                        )
            elif (
                content_type.casefold().startswith("text/")
                or suffix in TEXT_ATTACHMENT_SUFFIXES
            ):
                text = raw[:MAX_ATTACHMENT_TEXT_BYTES].decode("utf-8", errors="replace")
                prepared.append(
                    {
                        "type": "text",
                        "text": (
                            f"Attachment `{filename}` is available at {path}.\n"
                            f"Its text content follows:\n{text}"
                        ),
                    }
                )
            else:
                prepared.append(
                    {
                        "type": "text",
                        "text": f"Attachment `{filename}` is available at {path}.",
                    }
                )
        return prepared

    def _store_attachment(self, filename: str, raw: bytes) -> Path:
        suffix = Path(filename).suffix.casefold()
        if not suffix or len(suffix) > 16 or not re.fullmatch(r"\.[a-z0-9]+", suffix):
            suffix = ".bin"
        digest = hashlib.sha256(filename.encode("utf-8") + b"\0" + raw).hexdigest()
        path = self._attachment_root / f"{digest}{suffix}"
        temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
        try:
            self._attachment_root.mkdir(parents=True, exist_ok=True)
            self._attachment_root.chmod(0o700)
            if path.is_symlink():
                path.unlink()
            elif path.exists():
                if not path.is_file() or path.read_bytes() != raw:
                    path.unlink()
                else:
                    return path
            self._prune_attachment_cache(required_bytes=len(raw))
            temporary.write_bytes(raw)
            temporary.chmod(0o600)
            temporary.replace(path)
        except OSError as exc:
            raise CodexAppServerError(
                "The attachment could not be cached in the private runtime."
            ) from exc
        finally:
            with contextlib.suppress(OSError):
                temporary.unlink()
        return path

    def image_artifact_path(self, item: dict[str, Any]) -> Path | None:
        """Return a safe generated-image path from one Codex image item."""
        if str(item.get("type") or "").casefold() != "imagegeneration":
            return None
        raw_path = item.get("savedPath") or item.get("saved_path")
        path = _path_from_value(raw_path)
        if path is None:
            return None
        try:
            if path.is_symlink():
                return None
            resolved = path.resolve(strict=True)
            if (
                not resolved.is_file()
                or resolved.suffix.casefold() not in IMAGE_SUFFIXES
                or not _path_is_under(resolved, self._image_artifact_roots)
                or resolved.stat().st_size > MAX_ATTACHMENT_BYTES
            ):
                return None
        except OSError:
            return None
        return resolved

    def _user_input(
        self,
        prompt: str,
        attachments: Iterable[discord.Attachment],
        prepared: Iterable[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if prepared is not None:
            result.extend(prepared)
            return result
        for attachment in attachments:
            content_type = (attachment.content_type or "").casefold()
            if content_type.startswith("image/"):
                result.append({"type": "image", "url": attachment.url})
            elif content_type.startswith("audio/"):
                result.append({"type": "audio", "url": attachment.url})
            else:
                result.append(
                    {
                        "type": "text",
                        "text": f"Attachment `{attachment.filename}`: {attachment.url}",
                    }
                )
        return result

    async def _ensure_thread(
        self,
        session: _Session,
        *,
        allow_tools: bool = True,
        include_dynamic_tools: bool = True,
    ) -> None:
        await self._ensure_running()
        instruction_fingerprint = self._instruction_fingerprint(
            session,
            allow_tools,
            include_dynamic_tools=include_dynamic_tools,
        )
        if session.thread_id is not None and (
            session.instruction_fingerprint != instruction_fingerprint
            or session.tool_policy != allow_tools
        ):
            self._reset_session_thread(session)
            self._persist_state()

        if session.thread_id and not session.loaded:
            self._claim_usage_thread(session.thread_id)
            params: dict[str, Any] = {
                "threadId": session.thread_id,
                "runtimeWorkspaceRoots": [
                    str(path) for path in self._workspace_roots(allow_tools)
                ],
                "approvalPolicy": self._approval_policy(allow_tools),
                "sandbox": self._sandbox(allow_tools),
            }
            params.update(
                self._thread_instruction_params(
                    session,
                    allow_tools,
                    include_dynamic_tools=False,
                )
            )
            if self._model is not None:
                params["model"] = self._model
            try:
                await self._request("thread/resume", params)
            except CodexAppServerError as exc:
                message = str(exc).casefold()
                if (
                    "not found" not in message
                    and "unknown thread" not in message
                    and "no rollout found" not in message
                ):
                    raise
                session.thread_id = None
            else:
                session.instruction_fingerprint = instruction_fingerprint
                session.tool_policy = allow_tools
                session.loaded = True
                self._set_thread_loaded(session.thread_id, True)
                self._persist_state()
                logger.debug("Resumed Codex thread with current instructions")

        if session.thread_id is None:
            params: dict[str, Any] = {
                "cwd": self._thread_cwd(allow_tools),
                "approvalPolicy": self._approval_policy(allow_tools),
                "sandbox": self._sandbox(allow_tools),
                "threadSource": AGENT_NAME.casefold(),
                "runtimeWorkspaceRoots": [
                    str(path) for path in self._workspace_roots(allow_tools)
                ],
            }
            params.update(
                self._thread_instruction_params(
                    session,
                    allow_tools,
                    include_dynamic_tools=include_dynamic_tools,
                )
            )
            if self._model is not None:
                params["model"] = self._model
            result = await self._request("thread/start", params)
            thread = result.get("thread") or {}
            session.thread_id = thread.get("id")
            if not session.thread_id:
                raise CodexAppServerError("Codex did not return a thread id.")
            self._claim_usage_thread(session.thread_id)
            session.loaded = True
            session.instruction_fingerprint = instruction_fingerprint
            session.tool_policy = allow_tools
            self._set_thread_loaded(session.thread_id, True)
            self._persist_state()
            logger.info(
                "Created Codex thread (tools_allowed=%s, workspace_roots=%d)",
                allow_tools,
                len(self._workspace_roots(allow_tools)),
            )
        else:
            session.loaded = True
            if session.thread_id:
                self._set_thread_loaded(session.thread_id, True)
            logger.debug("Reused loaded Codex thread")
