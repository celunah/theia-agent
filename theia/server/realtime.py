"""Codex Realtime voice-session protocol support."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ..core import CodexAppServerError, _codex_logger, _safe_log_label

logger = _codex_logger()

CODEX_REALTIME_FEATURE = "realtime_conversation"
CODEX_REALTIME_PROVIDER = "codex-realtime"
_REALTIME_AUDIO_MAX_BYTES = 256 * 1024
_REALTIME_AUDIO_REQUEST_TIMEOUT = 10.0


@dataclass
class _RealtimeState:
    """Track one thread-scoped Codex Realtime session."""

    session_key: str
    thread_id: str
    allow_tools: bool
    on_event: Callable[[str, dict[str, Any]], Awaitable[None]]
    started: asyncio.Future[None]
    closed: asyncio.Future[None]
    realtime_session_id: str | None = None
    failed: bool = False


class CodexRealtimeMixin:
    """Expose the bounded Realtime protocol over the existing JSONL transport."""

    if TYPE_CHECKING:
        _realtime_sessions: dict[str, _RealtimeState]
        _realtime_feature_enabled: bool
        _realtime_model: str
        _realtime_voice: str
        _request_timeout: float

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    @property
    def realtime_voice_available(self) -> bool:
        """Whether the selected Codex App Server exposes Realtime voice."""
        return self._realtime_feature_enabled

    @property
    def voice_provider(self) -> str | None:
        """Return the active voice provider, preferring complete custom audio."""
        if self.custom_audio_configured:
            if self._audio.transcription.enabled and self._audio.tts.enabled:
                return "custom"
            return None
        return CODEX_REALTIME_PROVIDER if self.realtime_voice_available else None

    @property
    def custom_audio_configured(self) -> bool:
        """Whether either custom audio endpoint was configured."""
        return bool(self._audio.transcription.base_url or self._audio.tts.base_url)

    async def start_realtime_voice(
        self,
        session_key: str,
        allow_tools: bool,
        on_event: Callable[[str, dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Start a Codex Realtime audio session on the session's thread."""
        if not self.realtime_voice_available:
            raise CodexAppServerError(
                "Codex Realtime voice is unavailable in this installation."
            )
        if self.voice_provider != CODEX_REALTIME_PROVIDER:
            raise CodexAppServerError(
                "Codex Realtime is not the active voice provider."
            )
        await self._ensure_running()
        await self.refresh_account()
        if self.account is None and self.requires_openai_auth:
            raise CodexAppServerError("Run `/login` first.")

        session = self._session(session_key)
        canonical_key = session.key
        existing = self._realtime_sessions.get(canonical_key)
        if existing is not None and not existing.closed.done():
            existing.on_event = on_event
            return
        assert session.lock is not None
        async with session.lock:
            await self._prepare_session_for_activity(session)
            await self._ensure_thread(session, allow_tools=allow_tools)
            thread_id = session.thread_id
            if not thread_id:
                raise CodexAppServerError("Theia could not create a Codex thread.")
            loop = asyncio.get_running_loop()
            state = _RealtimeState(
                session_key=canonical_key,
                thread_id=thread_id,
                allow_tools=allow_tools,
                on_event=on_event,
                started=loop.create_future(),
                closed=loop.create_future(),
            )
            self._realtime_sessions[canonical_key] = state
            params: dict[str, Any] = {
                "threadId": thread_id,
                "outputModality": "audio",
                "transport": {"type": "websocket"},
            }
            if self._realtime_voice:
                params["voice"] = self._realtime_voice
            if self._realtime_model:
                params["model"] = self._realtime_model
            try:
                result = await self._request("thread/realtime/start", params)
                session_id = result.get("realtimeSessionId")
                if isinstance(session_id, str):
                    state.realtime_session_id = session_id
                await asyncio.wait_for(
                    asyncio.shield(state.started), self._request_timeout
                )
            except BaseException:
                self._realtime_sessions.pop(canonical_key, None)
                raise
        logger.info("Codex Realtime voice started")

    async def append_realtime_audio(
        self,
        session_key: str,
        pcm: bytes,
        sample_rate: int,
        num_channels: int,
    ) -> None:
        """Append one bounded PCM chunk to an active Realtime session."""
        if not pcm:
            return
        if len(pcm) > _REALTIME_AUDIO_MAX_BYTES:
            raise CodexAppServerError("Realtime audio chunk is too large.")
        if sample_rate <= 0 or num_channels <= 0:
            raise CodexAppServerError("Realtime audio metadata is invalid.")
        state = self._realtime_sessions.get(self._canonical_session_key(session_key))
        if state is None or state.closed.done():
            raise CodexAppServerError("The Codex Realtime voice session is closed.")
        await self._request(
            "thread/realtime/appendAudio",
            {
                "threadId": state.thread_id,
                "audio": {
                    "data": base64.b64encode(pcm).decode("ascii"),
                    "sampleRate": sample_rate,
                    "numChannels": num_channels,
                },
            },
            timeout=_REALTIME_AUDIO_REQUEST_TIMEOUT,
        )

    async def append_realtime_speech(self, session_key: str, text: str) -> None:
        """Ask an active Realtime session to speak text supplied by Theia."""
        value = text.strip()
        if not value:
            return
        state = self._realtime_sessions.get(self._canonical_session_key(session_key))
        if state is None or state.closed.done():
            raise CodexAppServerError("The Codex Realtime voice session is closed.")
        await self._request(
            "thread/realtime/appendSpeech",
            {"threadId": state.thread_id, "text": value[:4096]},
            timeout=_REALTIME_AUDIO_REQUEST_TIMEOUT,
        )

    async def stop_realtime_voice(self, session_key: str) -> bool:
        """Stop a Realtime transport and forget its callback state."""
        canonical_key = self._canonical_session_key(session_key)
        state = self._realtime_sessions.pop(canonical_key, None)
        if state is None:
            return False
        try:
            if not state.closed.done():
                await self._request(
                    "thread/realtime/stop",
                    {"threadId": state.thread_id},
                    timeout=_REALTIME_AUDIO_REQUEST_TIMEOUT,
                )
        except CodexAppServerError as exc:
            if not any(
                phrase in str(exc).casefold()
                for phrase in ("closed", "not found", "no active")
            ):
                raise
        finally:
            if not state.closed.done():
                state.closed.set_result(None)
        logger.info("Codex Realtime voice stopped")
        return True

    async def list_realtime_voices(self) -> dict[str, Any]:
        """Return voices supported by the installed Codex Realtime server."""
        if not self.realtime_voice_available:
            raise CodexAppServerError("Codex Realtime voice is unavailable.")
        await self._ensure_running()
        return await self._request("thread/realtime/listVoices", {})

    async def _refresh_realtime_capability(self) -> None:
        """Read the Realtime feature gate from the selected App Server."""
        self._realtime_feature_enabled = False
        try:
            result = await self._request("experimentalFeature/list", {})
        except (CodexAppServerError, OSError):
            logger.info("Codex Realtime capability is unavailable")
            return
        features = result.get("data")
        if not isinstance(features, list):
            return
        for feature in features:
            if not isinstance(feature, dict):
                continue
            if feature.get("name") != CODEX_REALTIME_FEATURE:
                continue
            self._realtime_feature_enabled = bool(feature.get("enabled"))
            logger.info(
                "Codex Realtime capability detected (enabled=%s)",
                self._realtime_feature_enabled,
            )
            return

    async def _close_realtime_sessions(self) -> None:
        """Resolve and discard Realtime callbacks before the child process closes."""
        for state in self._realtime_sessions.values():
            if not state.closed.done():
                state.closed.set_result(None)
        self._realtime_sessions.clear()

    def _realtime_for_thread(self, thread_id: str) -> _RealtimeState | None:
        if not thread_id:
            return None
        for state in self._realtime_sessions.values():
            if state.thread_id == thread_id:
                return state
        return None

    @staticmethod
    def _decode_realtime_audio(value: Any) -> bytes | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            decoded = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error):
            return None
        if not decoded or len(decoded) > _REALTIME_AUDIO_MAX_BYTES:
            return None
        return decoded

    @classmethod
    def _realtime_audio_payload(cls, audio: Any) -> dict[str, Any] | None:
        if not isinstance(audio, dict):
            return None
        data = cls._decode_realtime_audio(audio.get("data"))
        sample_rate = audio.get("sampleRate")
        num_channels = audio.get("numChannels")
        if (
            data is None
            or isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
            or isinstance(num_channels, bool)
            or not isinstance(num_channels, int)
            or num_channels <= 0
        ):
            return None
        samples = audio.get("samplesPerChannel")
        return {
            "data": data,
            "sample_rate": sample_rate,
            "num_channels": num_channels,
            "samples_per_channel": samples
            if isinstance(samples, int) and not isinstance(samples, bool)
            else None,
        }

    def _emit_realtime(
        self, state: _RealtimeState, event: str, payload: dict[str, Any]
    ) -> None:
        task = asyncio.create_task(
            cast(Coroutine[Any, Any, None], state.on_event(event, payload))
        )
        task.add_done_callback(lambda done: self._realtime_event_done(done, event))

    @staticmethod
    def _realtime_event_done(task: asyncio.Task[Any], event: str) -> None:
        with contextlib.suppress(Exception):
            error = task.exception()
            if error is not None:
                logger.debug(
                    "Codex Realtime event callback failed (event=%s, error=%s)",
                    _safe_log_label(event),
                    type(error).__name__,
                )

    def _handle_realtime_notification(
        self,
        state: _RealtimeState,
        method: str,
        params: dict[str, Any],
    ) -> None:
        if method == "thread/realtime/started":
            session_id = params.get("realtimeSessionId")
            if isinstance(session_id, str):
                state.realtime_session_id = session_id
            if not state.started.done():
                state.started.set_result(None)
            self._emit_realtime(state, "started", {})
            return
        if method == "thread/realtime/itemAdded":
            item = params.get("item")
            if isinstance(item, dict):
                self._emit_realtime(state, "item_added", item)
            return
        if method == "thread/realtime/item/started":
            item = params.get("item")
            if isinstance(item, dict):
                self._emit_realtime(state, "item_started", item)
            return
        if method == "thread/realtime/item/transcript/delta":
            delta = params.get("delta")
            item_id = params.get("itemId")
            if isinstance(delta, str) and isinstance(item_id, str):
                self._emit_realtime(
                    state,
                    "item_transcript_delta",
                    {"item_id": item_id, "delta": delta},
                )
            return
        if method == "thread/realtime/item/completed":
            item = params.get("item")
            if isinstance(item, dict):
                self._emit_realtime(state, "item_completed", item)
            return
        if method == "thread/realtime/transcript/delta":
            delta = params.get("delta")
            role = params.get("role")
            if isinstance(delta, str) and isinstance(role, str):
                self._emit_realtime(
                    state,
                    "transcript_delta",
                    {"role": role, "delta": delta},
                )
            return
        if method == "thread/realtime/transcript/done":
            text = params.get("text")
            role = params.get("role")
            if isinstance(text, str) and isinstance(role, str):
                self._emit_realtime(
                    state,
                    "transcript_done",
                    {"role": role, "text": text},
                )
            return
        if method == "thread/realtime/outputAudio/delta":
            payload = self._realtime_audio_payload(params.get("audio"))
            if payload is not None:
                self._emit_realtime(state, "output_audio", payload)
            else:
                logger.debug("Ignored invalid Codex Realtime audio chunk")
            return
        if method == "thread/realtime/sdp":
            logger.debug("Ignored unexpected Codex Realtime SDP notification")
            return
        if method == "thread/realtime/error":
            state.failed = True
            message = params.get("message")
            self._emit_realtime(
                state,
                "error",
                {"message": message if isinstance(message, str) else "unknown"},
            )
            return
        if method == "thread/realtime/closed":
            if not state.closed.done():
                state.closed.set_result(None)
            self._emit_realtime(state, "closed", {})
