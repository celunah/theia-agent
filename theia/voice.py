"""Discord voice receive, speaker-aware STT, and TTS playback support."""

from __future__ import annotations

import asyncio
import contextlib
import io
import math
import sys
import time
import wave
from array import array
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, cast

import discord

from .audio import AudioOutput, AudioProtocolError
from .core import _codex_logger, _env_float, _safe_error_reason, _subtext

try:
    from discord.ext import voice_recv
except ImportError:  # pragma: no cover - exercised when optional extras are absent
    voice_recv = None  # type: ignore[assignment]

logger = _codex_logger()

VOICE_SAMPLE_RATE = 48_000
VOICE_CHANNELS = 2
VOICE_SAMPLE_WIDTH = 2
VOICE_PACKET_SECONDS = 0.02
VOICE_SILENCE_AFTER_ENV = "THEIA_VOICE_SILENCE_AFTER"
VOICE_RMS_THRESHOLD_ENV = "THEIA_VOICE_RMS_THRESHOLD"
VOICE_MAX_UTTERANCE_ENV = "THEIA_VOICE_MAX_UTTERANCE"
DEFAULT_VOICE_SILENCE_AFTER = 0.8
DEFAULT_VOICE_RMS_THRESHOLD = 500
DEFAULT_VOICE_MAX_UTTERANCE = 30.0
VOICE_MIN_UTTERANCE = 0.15

TranscribeAudio = Callable[[str, bytes, str], Awaitable[str]]
SynthesizeText = Callable[[str], Awaitable[tuple[AudioOutput, ...]]]
VoiceTranscriptCallback = Callable[["VoiceSession", str, str], Awaitable[None]]


class VoiceModeError(RuntimeError):
    """Voice mode could not be started or used."""


@dataclass(frozen=True)
class VoiceSegment:
    """A completed speaker utterance converted to a WAV payload for STT."""

    guild_id: int
    channel_id: int
    speaker_id: int | None
    speaker_name: str
    wav_data: bytes


@dataclass
class _SpeakerBuffer:
    speaker_id: int | None
    speaker_name: str
    frames: bytearray
    started_at: float | None = None
    silence_seconds: float = 0.0


def _pcm_duration(data: bytes) -> float:
    bytes_per_second = VOICE_SAMPLE_RATE * VOICE_CHANNELS * VOICE_SAMPLE_WIDTH
    return len(data) / bytes_per_second


def _pcm_rms(data: bytes) -> float:
    usable = data[: len(data) - (len(data) % VOICE_SAMPLE_WIDTH)]
    if not usable:
        return 0.0
    samples = array("h")
    samples.frombytes(usable)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _to_wav(pcm: bytes) -> bytes:
    output = io.BytesIO()
    # typeshed currently infers Wave_read for wave.open even in write mode.
    # pylint: disable=no-member
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(VOICE_CHANNELS)
        wav_file.setsampwidth(VOICE_SAMPLE_WIDTH)
        wav_file.setframerate(VOICE_SAMPLE_RATE)
        wav_file.writeframes(pcm)
    # pylint: enable=no-member
    return output.getvalue()


if voice_recv is not None:

    class VoiceConversationSink(voice_recv.AudioSink):  # type: ignore[union-attr]
        """Split decoded Discord PCM into short per-speaker WAV utterances."""

        def __init__(
            self,
            *,
            guild_id: int,
            channel_id: int,
            loop: asyncio.AbstractEventLoop,
            on_segment: Callable[[VoiceSegment], Awaitable[None]],
            on_speech_start: Callable[[int], None],
        ) -> None:
            super().__init__()
            self.guild_id = guild_id
            self.channel_id = channel_id
            self.loop = loop
            self.on_segment = on_segment
            self.on_speech_start = on_speech_start
            self.silence_after = max(
                0.1,
                _env_float(VOICE_SILENCE_AFTER_ENV, DEFAULT_VOICE_SILENCE_AFTER),
            )
            self.rms_threshold = max(
                1.0,
                _env_float(VOICE_RMS_THRESHOLD_ENV, DEFAULT_VOICE_RMS_THRESHOLD),
            )
            self.max_utterance = max(
                1.0,
                _env_float(VOICE_MAX_UTTERANCE_ENV, DEFAULT_VOICE_MAX_UTTERANCE),
            )
            self._buffers: dict[int, _SpeakerBuffer] = {}
            self._closed = False

        def wants_opus(self) -> bool:
            """Request decoded PCM because the sink performs its own VAD."""
            return False

        def write(self, user: Any, data: Any) -> None:
            """Accumulate one decoded packet and finish utterances on silence or length."""
            if self._closed:
                return
            pcm = getattr(data, "pcm", b"")
            if not isinstance(pcm, bytes) or not pcm:
                return
            speaker_id = getattr(user, "id", None)
            if not isinstance(speaker_id, int):
                speaker_id = None
            speaker_name = str(
                getattr(user, "display_name", None)
                or getattr(user, "name", None)
                or "Unknown speaker"
            )
            key = speaker_id if speaker_id is not None else 0
            buffer = self._buffers.get(key)
            if buffer is None:
                buffer = _SpeakerBuffer(speaker_id, speaker_name, bytearray())
                self._buffers[key] = buffer
            now = time.monotonic()
            duration = _pcm_duration(pcm) or VOICE_PACKET_SECONDS
            speaking = _pcm_rms(pcm) >= self.rms_threshold
            if speaking:
                if buffer.started_at is None:
                    buffer.started_at = now
                    try:
                        # This callback only schedules the loop-side playback
                        # interruption and is intentionally synchronous.
                        self.on_speech_start(self.guild_id)
                    except Exception as exc:  # noqa: BLE001 - keep receiving audio
                        logger.debug(
                            "Voice speech-start callback failed (error=%s)",
                            type(exc).__name__,
                        )
                buffer.frames.extend(pcm)
                buffer.silence_seconds = 0.0
            elif buffer.started_at is not None:
                buffer.frames.extend(pcm)
                buffer.silence_seconds += duration

            if buffer.started_at is None:
                return
            utterance_duration = _pcm_duration(bytes(buffer.frames))
            if (
                buffer.silence_seconds >= self.silence_after
                or utterance_duration >= self.max_utterance
            ):
                self._finish(key)

        def cleanup(self) -> None:
            """Flush active speaker buffers and stop accepting new audio packets."""
            if self._closed:
                return
            self._closed = True
            for key in tuple(self._buffers):
                self._finish(key)

        def _finish(self, key: int) -> None:
            buffer = self._buffers.pop(key, None)
            if buffer is None or buffer.started_at is None:
                return
            pcm = bytes(buffer.frames)
            if _pcm_duration(pcm) < VOICE_MIN_UTTERANCE:
                return
            self._submit(
                self.on_segment(
                    VoiceSegment(
                        guild_id=self.guild_id,
                        channel_id=self.channel_id,
                        speaker_id=buffer.speaker_id,
                        speaker_name=buffer.speaker_name,
                        wav_data=_to_wav(pcm),
                    )
                )
            )

        def _submit(self, awaitable: Awaitable[Any]) -> None:
            if self.loop.is_closed():
                return
            try:
                future = asyncio.run_coroutine_threadsafe(
                    cast(Coroutine[Any, Any, Any], awaitable), self.loop
                )
            except RuntimeError:
                return

            def completed(done: Any) -> None:
                with contextlib.suppress(Exception):
                    error = done.exception()
                    if error is not None:
                        logger.debug(
                            "Voice callback failed (error=%s)",
                            type(error).__name__,
                        )

            future.add_done_callback(completed)

else:

    class VoiceConversationSink:  # pragma: no cover - dependency-gated fallback
        """Placeholder that reports the missing optional Discord voice dependency."""

        def __init__(self, **_kwargs: Any) -> None:
            raise VoiceModeError(
                "Voice receive support is unavailable; install the voice extras."
            )


class VoiceSession:
    """Bind one Discord session to a guild voice channel and transcript callback."""

    def __init__(
        self,
        *,
        session_key: str,
        user_id: int,
        guild_id: int,
        voice_channel_id: int,
        text_channel: discord.abc.Messageable,
        allow_tools: bool,
        on_transcript: VoiceTranscriptCallback,
    ) -> None:
        self.session_key = session_key
        self.user_id = user_id
        self.guild_id = guild_id
        self.voice_channel_id = voice_channel_id
        self.text_channel = text_channel
        self.allow_tools = allow_tools
        self.on_transcript = on_transcript


class VoiceModeManager:
    """Maintain voice receive/playback connections and route speakers safely."""

    def __init__(
        self,
        *,
        transcribe: TranscribeAudio,
        synthesize: SynthesizeText,
    ) -> None:
        self._transcribe = transcribe
        self._synthesize = synthesize
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sessions: dict[str, VoiceSession] = {}
        self._clients: dict[int, discord.VoiceClient] = {}
        self._sinks: dict[int, VoiceConversationSink] = {}
        self._play_locks: dict[int, asyncio.Lock] = {}
        self._playback_generation: dict[int, int] = {}

    @property
    def available(self) -> bool:
        """Return whether the optional Discord voice-receive package is installed."""
        return voice_recv is not None

    async def start(
        self,
        *,
        session_key: str,
        user_id: int,
        voice_channel: discord.abc.Connectable,
        text_channel: Any,
        allow_tools: bool,
        on_transcript: VoiceTranscriptCallback,
    ) -> VoiceSession:
        """Join a voice channel or reuse its connection for another session."""
        if voice_recv is None:
            raise VoiceModeError(
                "Voice receive support is unavailable; install the voice extras."
            )
        guild = getattr(voice_channel, "guild", None)
        guild_id = getattr(guild, "id", None)
        channel_id = getattr(voice_channel, "id", None)
        if not isinstance(guild_id, int) or not isinstance(channel_id, int):
            raise VoiceModeError("Voice mode requires a guild voice channel.")
        self._loop = asyncio.get_running_loop()
        existing = self._sessions.get(session_key)
        if existing is not None:
            if existing.voice_channel_id != channel_id:
                await self.stop(session_key)
            else:
                existing.text_channel = text_channel
                existing.allow_tools = allow_tools
                existing.on_transcript = on_transcript
                return existing

        client = self._clients.get(guild_id)
        if client is None:
            guild_client = getattr(guild, "voice_client", None)
            if guild_client is not None:
                client = guild_client
            else:
                self._ensure_opus()
                try:
                    client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
                except (discord.DiscordException, TypeError, RuntimeError) as exc:
                    raise VoiceModeError(
                        f"Could not join the voice channel: {_safe_error_reason(exc)}"
                    ) from exc
            if not hasattr(client, "listen"):
                raise VoiceModeError(
                    "The existing voice connection cannot receive audio."
                )
            self._clients[guild_id] = client
        current_channel = getattr(getattr(client, "channel", None), "id", None)
        if current_channel != channel_id:
            raise VoiceModeError(
                "The bot is already listening in another voice channel in this server."
            )

        if guild_id not in self._sinks:
            sink = VoiceConversationSink(
                guild_id=guild_id,
                channel_id=channel_id,
                loop=self._loop,
                on_segment=self._on_segment,
                on_speech_start=self._on_speech_start,
            )
            try:
                cast(Any, client).listen(sink)
            except (discord.DiscordException, TypeError, RuntimeError) as exc:
                raise VoiceModeError(
                    f"Could not start voice receive: {_safe_error_reason(exc)}"
                ) from exc
            self._sinks[guild_id] = sink

        session = VoiceSession(
            session_key=session_key,
            user_id=user_id,
            guild_id=guild_id,
            voice_channel_id=channel_id,
            text_channel=text_channel,
            allow_tools=allow_tools,
            on_transcript=on_transcript,
        )
        self._sessions[session_key] = session
        logger.info(
            "Voice mode started (sessions_in_guild=%d)",
            self._guild_session_count(guild_id),
        )
        return session

    async def stop(self, session_key: str) -> bool:
        """Stop one voice session and disconnect a guild when it has no listeners."""
        session = self._sessions.pop(session_key, None)
        if session is None:
            return False
        guild_id = session.guild_id
        if not any(item.guild_id == guild_id for item in self._sessions.values()):
            client = self._clients.pop(guild_id, None)
            self._sinks.pop(guild_id, None)
            self._play_locks.pop(guild_id, None)
            self._playback_generation.pop(guild_id, None)
            if client is not None:
                with contextlib.suppress(Exception):
                    stop_listening = getattr(client, "stop_listening", None)
                    if callable(stop_listening):
                        stop_listening()
                with contextlib.suppress(discord.DiscordException):
                    await client.disconnect()
        logger.info(
            "Voice mode stopped (sessions_in_guild=%d)",
            self._guild_session_count(guild_id),
        )
        return True

    async def close(self) -> None:
        """Stop all voice sessions and disconnect every managed guild client."""
        for session_key in tuple(self._sessions):
            await self.stop(session_key)
        for guild_id, client in tuple(self._clients.items()):
            self._clients.pop(guild_id, None)
            with contextlib.suppress(discord.DiscordException):
                await client.disconnect()

    def has_session(self, session_key: str) -> bool:
        """Return whether a Discord session currently has voice mode enabled."""
        return session_key in self._sessions

    async def speak_text(self, session_key: str, text: str) -> None:
        """Synthesize and play one text response for a voice session."""
        outputs = await self._synthesize(text)
        await self.speak_outputs(session_key, outputs)

    async def speak_outputs(
        self, session_key: str, outputs: tuple[AudioOutput, ...]
    ) -> None:
        """Play synthesized outputs serially while honoring playback cancellation."""
        session = self._sessions.get(session_key)
        if session is None or not outputs:
            return
        client = self._clients.get(session.guild_id)
        if client is None:
            return
        lock = self._play_locks.setdefault(session.guild_id, asyncio.Lock())
        async with lock:
            generation = self._playback_generation.get(session.guild_id, 0)
            for output in outputs:
                if generation != self._playback_generation.get(session.guild_id, 0):
                    return
                await self._play_output(client, output)

    async def stop_playback(self, guild_id: int) -> None:
        """Cancel queued playback for a guild and stop its current audio source."""
        self._playback_generation[guild_id] = (
            self._playback_generation.get(guild_id, 0) + 1
        )
        client = self._clients.get(guild_id)
        if client is None:
            return
        stop_playing = getattr(client, "stop_playing", None)
        if callable(stop_playing):
            with contextlib.suppress(Exception):
                stop_playing()

    async def _on_segment(self, segment: VoiceSegment) -> None:
        sessions = [
            session
            for session in self._sessions.values()
            if session.guild_id == segment.guild_id
            and session.voice_channel_id == segment.channel_id
        ]
        if segment.speaker_id is not None:
            matching = [
                session for session in sessions if session.user_id == segment.speaker_id
            ]
            # With one active voice session, accept other Discord speakers too
            # and preserve their display name in the prompt. If several users
            # have independent sessions in one channel, route only to the
            # matching owner to keep their Codex contexts isolated.
            sessions = matching or (sessions if len(sessions) == 1 else [])
        if len(sessions) != 1:
            return
        session = sessions[0]
        try:
            text = await self._transcribe(
                f"voice-{segment.speaker_id or 'unknown'}.wav",
                segment.wav_data,
                "audio/wav",
            )
        except (AudioProtocolError, RuntimeError) as exc:
            logger.warning("Voice transcription failed (error=%s)", type(exc).__name__)
            with contextlib.suppress(discord.DiscordException):
                await session.text_channel.send(
                    content=_subtext(
                        "Voice transcription failed: " + _safe_error_reason(exc)
                    ),
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            return
        if text.strip():
            await session.on_transcript(session, segment.speaker_name, text.strip())

    def _on_speech_start(self, guild_id: int) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.stop_playback(guild_id), self._loop
            )
        except RuntimeError:
            return

        def completed(done: Any) -> None:
            with contextlib.suppress(Exception):
                error = done.exception()
                if error is not None:
                    logger.debug(
                        "Voice playback interruption failed (error=%s)",
                        type(error).__name__,
                    )

        future.add_done_callback(completed)

    async def _play_output(
        self, client: discord.VoiceClient, output: AudioOutput
    ) -> None:
        if not self._is_connected(client):
            return
        loop = asyncio.get_running_loop()
        format_name = output.filename.rsplit(".", 1)[-1]
        source: discord.AudioSource | None = None
        finished = loop.create_future()

        def after(error: BaseException | None) -> None:
            loop.call_soon_threadsafe(self._finish_playback, finished, error)

        try:
            source = discord.FFmpegPCMAudio(
                io.BytesIO(output.data),
                pipe=True,
                before_options=f"-f {format_name}",
            )
            client.play(source, after=after)
            error = await finished
            if error is not None:
                raise VoiceModeError("Discord voice playback failed.") from error
        except (OSError, discord.DiscordException, VoiceModeError) as exc:
            logger.warning("Voice playback failed (error=%s)", type(exc).__name__)
        finally:
            if source is not None:
                with contextlib.suppress(Exception):
                    source.cleanup()

    @staticmethod
    def _finish_playback(
        future: asyncio.Future[Any], error: BaseException | None
    ) -> None:
        if not future.done():
            future.set_result(error)

    @staticmethod
    def _is_connected(client: discord.VoiceClient) -> bool:
        checker = getattr(client, "is_connected", None)
        return bool(checker()) if callable(checker) else True

    @staticmethod
    def _ensure_opus() -> None:
        if discord.opus.is_loaded():
            return
        try:
            discord.opus._load_default()
        except Exception as exc:
            raise VoiceModeError(
                "The Discord Opus library is unavailable for voice receive."
            ) from exc
        if not discord.opus.is_loaded():
            raise VoiceModeError(
                "The Discord Opus library is unavailable for voice receive."
            )

    def _guild_session_count(self, guild_id: int) -> int:
        return sum(session.guild_id == guild_id for session in self._sessions.values())
