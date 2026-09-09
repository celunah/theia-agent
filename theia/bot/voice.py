"""Restart and voice-request helpers for the Discord façade."""

# Resolve façade patch points lazily to avoid an import cycle at module load time.
# pylint: disable=cyclic-import,import-outside-toplevel

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from collections.abc import Awaitable, Callable
from typing import Any

import discord

from ..server.core import CodexAppServerError
from ..core import VOICE_MODE, _codex_logger, _safe_error_reason, _subtext
from ..voice import VoiceSession

logger = _codex_logger()


def _bot_module() -> Any:
    from . import core as bot_module

    return bot_module


async def _restart_in_place(*, delay: float = 0.5) -> None:
    """Gracefully close the bot and replace this process with the same command."""
    bot_module = _bot_module()
    await asyncio.sleep(delay)
    logger.info("Restarting Theia process in place")
    try:
        await bot_module.bot.close()
    except Exception:
        logger.exception("Theia shutdown raised during in-place restart")

    if "__compiled__" in globals() and sys.argv and sys.argv[0]:
        # Nuitka onefile runs the bundled modules from a temporary extraction
        # directory.  sys.executable can point into that directory after the
        # child runtime has closed; sys.argv[0] remains the user's binary.
        executable = os.path.abspath(sys.argv[0])
        argv = [executable, *sys.argv[1:]]
    else:
        executable = sys.executable or "python"
        argv = [executable, *sys.argv]
    try:
        os.execv(executable, argv)
    except Exception:
        logger.exception("Theia in-place process replacement failed")


def _voice_speak_callback(
    session_key_value: str,
) -> Callable[[str], Awaitable[None]] | None:
    bot_module = _bot_module()
    if bot_module.bot.codex.mode(
        session_key_value
    ) == VOICE_MODE and bot_module.bot.voice.has_session(session_key_value):

        async def speak(text: str) -> None:
            await bot_module.bot.voice.speak_text(session_key_value, text)

        return speak
    return None


async def _handle_voice_transcript(
    session: VoiceSession, speaker: str, transcript: str
) -> None:
    bot_module = _bot_module()
    with contextlib.suppress(discord.DiscordException):
        await session.text_channel.send(
            content=_subtext(f"{speaker}: {transcript}"),
            allowed_mentions=discord.AllowedMentions.none(),
        )
    await bot_module.bot.presence.touch()
    prompt = f"[Voice input from {speaker}]\n{transcript}"
    allow_tools = bot_module._voice_session_allows_tools(session)
    active_turn = bot_module.bot.codex.status(session.session_key).get("turn_id")
    if active_turn and session.allow_tools and not allow_tools:
        with contextlib.suppress(CodexAppServerError):
            await bot_module.bot.codex.interrupt(session.session_key)
        active_turn = None
    if active_turn:
        try:
            await bot_module.bot.codex.steer(session.session_key, prompt)
            return
        except CodexAppServerError as exc:
            if "no active codex turn" not in str(exc).casefold():
                with contextlib.suppress(discord.DiscordException):
                    await session.text_channel.send(
                        content=_subtext(
                            "I could not steer the active request: "
                            + _safe_error_reason(exc)
                        ),
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                return
    bot_module.bot.schedule_request(
        _run_voice_request(
            session,
            prompt,
            allow_tools=allow_tools,
        )
    )


async def _run_voice_request(
    session: VoiceSession,
    prompt: str,
    *,
    allow_tools: bool,
) -> None:
    """Run voice input outside the receive callback so transcription stays responsive."""
    bot_module = _bot_module()
    context = await bot_module._channel_context(session.text_channel)
    await bot_module.handle_request(
        session.text_channel.send,
        prompt,
        channel=session.text_channel,
        user_id=session.user_id,
        allow_tools=allow_tools,
        context=context,
        speak_text=_voice_speak_callback(session.session_key),
    )
