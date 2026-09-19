"""Provider-neutral full-duplex audio middleware contracts."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .core import _safe_error_reason, _truncate

try:
    import aiohttp
except ImportError:  # pragma: no cover - optional protocol dependency
    aiohttp = None  # type: ignore[assignment]


AUDIO_PROVIDER_ENV = "THEIA_AUDIO_PROVIDER"
QWEN_AUDIO_URL_ENV = "THEIA_QWEN_AUDIO_URL"
QWEN_AUDIO_TOKEN_ENV = "THEIA_QWEN_AUDIO_TOKEN"
QWEN_AUDIO_SEMANTIC_ENV = "THEIA_QWEN_AUDIO_SEMANTIC_AUDIO"
AUDIO_PROVIDER_AUTO = "auto"
AUDIO_PROVIDER_QWEN = "qwen"
AUDIO_PROVIDER_CODEX_REALTIME = "codex-realtime"
AUDIO_PROVIDER_CUSTOM = "custom"
AUDIO_PROVIDER_UNAVAILABLE = "unavailable"

AUDIO_PROVIDER_EVENTS = frozenset(
    {
        "speech_started",
        "speech_stopped",
        "transcript_partial",
        "transcript_final",
        "audio_output",
        "output_interrupted",
        "provider_error",
        "provider_ready",
    }
)
AUDIO_PROVIDER_MAX_TEXT = 4096
AUDIO_PROVIDER_MAX_AUDIO_BYTES = 256 * 1024


@dataclass(frozen=True)
class AudioProviderEvent:
    """A bounded event emitted by an audio middleware provider."""

    type: str
    text: str = ""
    audio: bytes = b""
    sample_rate: int = 0
    num_channels: int = 0
    role: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.type not in AUDIO_PROVIDER_EVENTS:
            raise ValueError("Unsupported audio provider event.")
        object.__setattr__(self, "text", self.text[:AUDIO_PROVIDER_MAX_TEXT])
        object.__setattr__(self, "reason", _truncate(self.reason, 240))
        if len(self.audio) > AUDIO_PROVIDER_MAX_AUDIO_BYTES:
            raise ValueError("Audio provider output is too large.")
        if self.sample_rate < 0 or self.num_channels < 0:
            raise ValueError("Audio provider metadata is invalid.")


@dataclass(frozen=True)
class AudioProviderCapabilities:
    """Capabilities advertised by a provider without exposing its backend."""

    full_duplex: bool = True
    speech_events: bool = True
    partial_transcription: bool = True
    final_transcription: bool = True
    audio_output: bool = True
    interruption: bool = True
    semantic_audio_understanding: bool = False


AudioProviderEventCallback = Callable[[AudioProviderEvent], Awaitable[None]]


class AudioProvider(Protocol):
    """Middleware API used by the voice session manager."""

    name: str
    capabilities: AudioProviderCapabilities

    @property
    def available(self) -> bool:
        """Whether the provider has a usable configured transport."""

    async def start(
        self, session_key: str, on_event: AudioProviderEventCallback
    ) -> None:
        """Open one isolated provider session."""

    async def send_audio(
        self, session_key: str, pcm: bytes, sample_rate: int, num_channels: int
    ) -> None:
        """Send one bounded input audio chunk."""

    async def speak_text(self, session_key: str, text: str) -> None:
        """Send Theia's final response for provider-side audio output."""

    async def interrupt(self, session_key: str) -> None:
        """Interrupt provider output for a barge-in."""

    async def stop(self, session_key: str) -> None:
        """Close one provider session."""


class AudioProviderError(RuntimeError):
    """A provider transport failed without exposing its raw payload."""


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def _safe_wire_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:AUDIO_PROVIDER_MAX_TEXT]


def _decode_audio(value: Any) -> bytes | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        return None
    if not data or len(data) > AUDIO_PROVIDER_MAX_AUDIO_BYTES:
        return None
    return data


@dataclass
class _QwenSession:
    client: Any
    websocket: Any
    task: asyncio.Task[None] | None = None


class QwenAudioProvider:
    """Adapt a configured Qwen Audio Agent endpoint into the middleware API.

    The endpoint is a JSON WebSocket protocol. It performs audio processing and
    backend model delegation; it receives no personality, memory, tool, or
    Discord session state from Theia.
    """

    name = AUDIO_PROVIDER_QWEN

    def __init__(
        self,
        endpoint: str,
        *,
        token: str = "",
        semantic_audio_understanding: bool = False,
    ) -> None:
        self.endpoint = endpoint.strip()
        self._token = token.strip()
        self.capabilities = AudioProviderCapabilities(
            semantic_audio_understanding=semantic_audio_understanding
        )
        self._sessions: dict[str, _QwenSession] = {}

    @classmethod
    def from_environment(
        cls, credentials: Mapping[str, str] | None = None
    ) -> QwenAudioProvider | None:
        """Build the adapter only when an external Qwen endpoint is configured."""
        endpoint = os.getenv(QWEN_AUDIO_URL_ENV, "").strip()
        if not endpoint:
            return None
        token = (
            credentials.get(QWEN_AUDIO_TOKEN_ENV, "")
            if credentials is not None
            else os.getenv(QWEN_AUDIO_TOKEN_ENV, "")
        )
        return cls(
            endpoint,
            token=token,
            semantic_audio_understanding=_env_bool(QWEN_AUDIO_SEMANTIC_ENV),
        )

    @property
    def available(self) -> bool:
        """Return whether an endpoint and optional WebSocket dependency exist."""
        return self.endpoint.startswith(("ws://", "wss://")) and aiohttp is not None

    async def start(
        self, session_key: str, on_event: AudioProviderEventCallback
    ) -> None:
        """Open a provider connection without sending Theia's session identity."""
        if not self.available:
            raise AudioProviderError("Qwen audio middleware is unavailable.")
        if session_key in self._sessions:
            return
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        assert aiohttp is not None
        client = aiohttp.ClientSession(headers=headers)
        state: _QwenSession | None = None
        try:
            websocket = await client.ws_connect(self.endpoint, heartbeat=30)
            state = _QwenSession(client, websocket)
            self._sessions[session_key] = state
            state.task = asyncio.create_task(
                self._read_events(session_key, state, on_event)
            )
            await websocket.send_json(
                {
                    "type": "session_start",
                    "session_id": uuid.uuid4().hex,
                    "audio": {"sample_rate": 48000, "num_channels": 2},
                }
            )
            await on_event(AudioProviderEvent("provider_ready"))
        except BaseException:  # pylint: disable=try-except-raise
            self._sessions.pop(session_key, None)
            if (
                state is not None
                and state.task is not None
                and state.task is not asyncio.current_task()
            ):
                state.task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await state.task
            with contextlib.suppress(Exception):
                await client.close()
            raise

    async def send_audio(
        self, session_key: str, pcm: bytes, sample_rate: int, num_channels: int
    ) -> None:
        """Send raw PCM to the configured Qwen middleware endpoint."""
        if not pcm:
            return
        if len(pcm) > AUDIO_PROVIDER_MAX_AUDIO_BYTES:
            raise AudioProviderError("Qwen audio input is too large.")
        if sample_rate <= 0 or num_channels <= 0:
            raise AudioProviderError("Qwen audio metadata is invalid.")
        state = self._sessions.get(session_key)
        if state is None:
            raise AudioProviderError("Qwen audio middleware is disconnected.")
        await state.websocket.send_json(
            {
                "type": "audio_input",
                "audio": {
                    "data": base64.b64encode(pcm).decode("ascii"),
                    "sample_rate": sample_rate,
                    "num_channels": num_channels,
                },
            }
        )

    async def speak_text(self, session_key: str, text: str) -> None:
        """Send a completed Theia response for Qwen-side speech synthesis."""
        value = _safe_wire_text(text)
        if not value:
            return
        state = self._sessions.get(session_key)
        if state is None:
            raise AudioProviderError("Qwen audio middleware is disconnected.")
        await state.websocket.send_json({"type": "text_output", "text": value})

    async def interrupt(self, session_key: str) -> None:
        """Request provider-side output cancellation for a barge-in."""
        state = self._sessions.get(session_key)
        if state is not None:
            await state.websocket.send_json({"type": "output_interrupt"})

    async def stop(self, session_key: str) -> None:
        """Close one provider connection and cancel its event reader."""
        state = self._sessions.pop(session_key, None)
        if state is None:
            return
        current = asyncio.current_task()
        if state.task is not None and state.task is not current:
            state.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await state.task
        with contextlib.suppress(Exception):
            await state.websocket.send_json({"type": "session_stop"})
        with contextlib.suppress(Exception):
            await state.websocket.close()
        with contextlib.suppress(Exception):
            await state.client.close()

    async def close(self) -> None:
        """Close every active provider connection during shutdown."""
        for session_key in tuple(self._sessions):
            await self.stop(session_key)

    async def _read_events(
        self,
        session_key: str,
        state: _QwenSession,
        on_event: AudioProviderEventCallback,
    ) -> None:
        try:
            async for message in state.websocket:
                message_type = getattr(getattr(message, "type", None), "name", "")
                if message_type != "TEXT":
                    continue
                try:
                    value = json.loads(message.data)
                except (TypeError, ValueError):
                    continue
                event = self._event_from_wire(value)
                if event is not None:
                    await on_event(event)
        except Exception as exc:  # noqa: BLE001 - convert transport failures
            if session_key in self._sessions:
                await on_event(
                    AudioProviderEvent(
                        "provider_error",
                        reason=f"provider disconnected: {_safe_error_reason(exc)}",
                    )
                )

    @classmethod
    def _event_from_wire(cls, value: Any) -> AudioProviderEvent | None:
        if not isinstance(value, dict):
            return None
        event_type = value.get("type") or value.get("event")
        if not isinstance(event_type, str) or event_type not in AUDIO_PROVIDER_EVENTS:
            return None
        text = _safe_wire_text(value.get("text"))
        reason = _safe_wire_text(value.get("reason") or value.get("message"))
        role = _safe_wire_text(value.get("role"))[:32]
        if event_type == "audio_output":
            audio = value.get("audio")
            if not isinstance(audio, dict):
                audio = value
            data = _decode_audio(audio.get("data"))
            sample_rate = audio.get("sample_rate") or audio.get("sampleRate")
            num_channels = audio.get("num_channels") or audio.get("numChannels")
            if (
                data is None
                or not isinstance(sample_rate, int)
                or isinstance(sample_rate, bool)
                or not isinstance(num_channels, int)
                or isinstance(num_channels, bool)
            ):
                return None
            return AudioProviderEvent(
                event_type,
                audio=data,
                sample_rate=sample_rate,
                num_channels=num_channels,
            )
        return AudioProviderEvent(event_type, text=text, role=role, reason=reason)
