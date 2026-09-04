"""Optional OpenAI-compatible transcription and text-to-speech services."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any

from .core import _codex_logger, _env_float, _truncate

logger = _codex_logger()

_SUPPORTED_PROTOCOLS = frozenset({"openai", "openai-compatible", "openai_compatible"})
_SUPPORTED_TTS_FORMATS = frozenset({"mp3", "opus", "aac", "flac", "wav", "pcm"})
_MAX_TRANSCRIPTION_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_TTS_RESPONSE_BYTES = 32 * 1024 * 1024
_MAX_TTS_INPUT_CHARACTERS = 4096
_MAX_TTS_DISCORD_ATTACHMENTS = 10
_AUDIO_ERROR_LIMIT = 400


class AudioProtocolError(RuntimeError):
    """An OpenAI-compatible audio endpoint returned an unusable response."""


@dataclass(frozen=True)
class AudioServiceConfig:
    protocol: str
    base_url: str
    api_key: str
    model: str
    timeout: float

    @property
    def enabled(self) -> bool:
        return bool(self.base_url) and self.protocol in _SUPPORTED_PROTOCOLS


@dataclass(frozen=True)
class TTSConfig(AudioServiceConfig):
    voice: str
    response_format: str


@dataclass(frozen=True)
class AudioOutput:
    """Audio bytes ready to be attached to a Discord response."""

    data: bytes
    filename: str
    content_type: str


def _env_text(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value.strip()
    return default


def _protocol(name: str) -> str:
    value = name.casefold().replace(" ", "-")
    if value not in _SUPPORTED_PROTOCOLS:
        logger.warning(
            "Unsupported audio protocol configured; service disabled (protocol=%s)",
            re.sub(r"[^a-z0-9_-]", "", value)[:40] or "unknown",
        )
        return "unsupported"
    return value


def _safe_filename(filename: str) -> str:
    value = os.path.basename(filename).replace("\r", "").replace("\n", "")
    return value or "attachment"


def _safe_server_detail(raw: bytes) -> str:
    """Extract a short error detail without returning a response blob."""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    try:
        value: Any = json.loads(text)
    except (TypeError, ValueError):
        value = text
    if isinstance(value, dict):
        error = value.get("error")
        if isinstance(error, dict):
            value = error.get("message") or error.get("detail") or error
        else:
            value = value.get("message") or value.get("detail") or value
    if isinstance(value, dict):
        value = "audio server error"
    detail = re.sub(r"\s+", " ", str(value)).strip()
    detail = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [redacted]", detail)
    return _truncate(detail, _AUDIO_ERROR_LIMIT)


def _error_from_http(error: urllib.error.HTTPError, body: bytes) -> AudioProtocolError:
    detail = _safe_server_detail(body)
    suffix = f": {detail}" if detail else ""
    return AudioProtocolError(
        f"Audio server returned HTTP {error.code} {error.reason}{suffix}"
    )


def _multipart_body(
    *,
    fields: dict[str, str],
    file_name: str,
    file_content_type: str,
    file_data: bytes,
) -> tuple[bytes, str]:
    boundary = f"----TheiaAudio{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        safe_name = name.replace("\r", "").replace("\n", "")
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{safe_name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            )
        )
    safe_filename = _safe_filename(file_name).replace('"', "'")
    safe_content_type = (
        re.sub(r"[\r\n]", "", file_content_type) or "application/octet-stream"
    )
    chunks.extend(
        (
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="file"; '
                f'filename="{safe_filename}"\r\n'
            ).encode(),
            f"Content-Type: {safe_content_type}\r\n\r\n".encode(),
            file_data,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        )
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _response_json_or_text(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return body.decode("utf-8", errors="replace")


class OpenAICompatibleAudio:
    """Use separate OpenAI-compatible servers for transcription and TTS."""

    def __init__(
        self,
        transcription: AudioServiceConfig,
        tts: TTSConfig,
    ) -> None:
        self.transcription = transcription
        self.tts = tts

    @classmethod
    def from_environment(cls) -> OpenAICompatibleAudio:
        transcription = AudioServiceConfig(
            protocol=_protocol(
                _env_text(
                    "STT_PROTOCOL",
                    "THEIA_TRANSCRIPTION_PROTOCOL",
                    default="openai-compatible",
                )
            ),
            base_url=_env_text("STT_BASE_URL", "THEIA_TRANSCRIPTION_BASE_URL"),
            api_key=_env_text("STT_TOKEN", "THEIA_TRANSCRIPTION_API_KEY"),
            model=_env_text(
                "STT_MODEL", "THEIA_TRANSCRIPTION_MODEL", default="whisper-1"
            ),
            timeout=max(1.0, _env_float("THEIA_TRANSCRIPTION_TIMEOUT", 120)),
        )
        tts_format = _env_text(
            "TTS_FORMAT", "THEIA_TTS_FORMAT", default="mp3"
        ).casefold()
        if tts_format not in _SUPPORTED_TTS_FORMATS:
            logger.warning(
                "Unsupported TTS response format; using mp3 (format=%s)",
                re.sub(r"[^a-z0-9]", "", tts_format)[:20] or "unknown",
            )
            tts_format = "mp3"
        tts = TTSConfig(
            protocol=_protocol(
                _env_text(
                    "TTS_PROTOCOL",
                    "THEIA_TTS_PROTOCOL",
                    default="openai-compatible",
                )
            ),
            base_url=_env_text("TTS_BASE_URL", "THEIA_TTS_BASE_URL"),
            api_key=_env_text("TTS_TOKEN", "THEIA_TTS_API_KEY"),
            model=_env_text("TTS_MODEL", "THEIA_TTS_MODEL", default="tts-1"),
            timeout=max(1.0, _env_float("THEIA_TTS_TIMEOUT", 120)),
            voice=_env_text("TTS_VOICE", "THEIA_TTS_VOICE", default="alloy"),
            response_format=tts_format,
        )
        return cls(transcription, tts)

    @staticmethod
    def _url(config: AudioServiceConfig, endpoint: str) -> str:
        # The configured value is the API base, e.g. https://host/v1. Do not
        # append /v1 automatically because compatible self-hosted servers vary.
        return f"{config.base_url.rstrip('/')}/{endpoint}"

    @staticmethod
    def _headers(config: AudioServiceConfig, *, accept: str) -> dict[str, str]:
        headers = {"Accept": accept, "User-Agent": "Theia-Agent/1.0"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        return headers

    @staticmethod
    def _request(
        request: urllib.request.Request, *, timeout: float, limit: int
    ) -> bytes:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read()
            except OSError:
                body = b""
            raise _error_from_http(exc, body) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AudioProtocolError(
                f"Could not reach audio server: {type(exc).__name__}"
            ) from exc
        if len(body) > limit:
            raise AudioProtocolError("Audio server response was too large.")
        return body

    def _transcribe_sync(self, filename: str, raw: bytes, content_type: str) -> str:
        body, multipart_type = _multipart_body(
            fields={"model": self.transcription.model, "response_format": "json"},
            file_name=filename,
            file_content_type=content_type,
            file_data=raw,
        )
        request = urllib.request.Request(
            self._url(self.transcription, "audio/transcriptions"),
            data=body,
            headers={
                **self._headers(self.transcription, accept="application/json"),
                "Content-Type": multipart_type,
            },
            method="POST",
        )
        response = _response_json_or_text(
            self._request(
                request,
                timeout=self.transcription.timeout,
                limit=_MAX_TRANSCRIPTION_RESPONSE_BYTES,
            )
        )
        value = response.get("text") if isinstance(response, dict) else response
        text = str(value or "").strip()
        if not text:
            raise AudioProtocolError("Transcription server returned no text.")
        return text

    async def transcribe(
        self, filename: str, raw: bytes, content_type: str = ""
    ) -> str | None:
        if not self.transcription.enabled:
            return None
        logger.info("Sending audio attachment to configured transcription service")
        return await asyncio.to_thread(
            self._transcribe_sync, filename, raw, content_type
        )

    def _synthesize_sync(self, text: str) -> bytes:
        body = json.dumps(
            {
                "model": self.tts.model,
                "input": text,
                "voice": self.tts.voice,
                "response_format": self.tts.response_format,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url(self.tts, "audio/speech"),
            data=body,
            headers={
                **self._headers(self.tts, accept="audio/*"),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        response = self._request(
            request,
            timeout=self.tts.timeout,
            limit=_MAX_TTS_RESPONSE_BYTES,
        )
        # A few compatible servers return base64 JSON despite the normal raw
        # audio response. Accept that useful extension without changing the
        # standard request/response contract.
        if response.lstrip().startswith(b"{"):
            value = _response_json_or_text(response)
            if isinstance(value, dict):
                encoded = value.get("audio") or value.get("data")
                if isinstance(encoded, str):
                    try:
                        response = base64.b64decode(encoded, validate=True)
                    except (ValueError, binascii.Error):
                        raise AudioProtocolError(
                            "TTS server returned invalid base64 audio."
                        ) from None
        if not response:
            raise AudioProtocolError("TTS server returned no audio.")
        return response

    async def synthesize(self, text: str) -> AudioOutput | None:
        outputs = await self.synthesize_many(text)
        return outputs[0] if outputs else None

    async def synthesize_many(self, text: str) -> tuple[AudioOutput, ...]:
        if not self.tts.enabled or not (text or "").strip():
            return ()
        value = text.strip()
        chunks = [
            value[index : index + _MAX_TTS_INPUT_CHARACTERS]
            for index in range(0, len(value), _MAX_TTS_INPUT_CHARACTERS)
        ]
        if len(chunks) > _MAX_TTS_DISCORD_ATTACHMENTS:
            logger.info(
                "Skipping TTS for a response that exceeds Discord's attachment limit"
            )
            return ()
        outputs: list[AudioOutput] = []
        for index, chunk in enumerate(chunks, start=1):
            logger.info("Sending response text to configured TTS service")
            raw = await asyncio.to_thread(self._synthesize_sync, chunk)
            suffix = self.tts.response_format
            extension = (
                f"theia-response-{index}.{suffix}"
                if len(chunks) > 1
                else f"theia-response.{suffix}"
            )
            outputs.append(
                AudioOutput(
                    data=raw,
                    filename=extension,
                    content_type=f"audio/{suffix}",
                )
            )
        return tuple(outputs)
