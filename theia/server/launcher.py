"""Secure Theia launcher coordination and startup-failure visibility."""

from __future__ import annotations

import asyncio
import contextlib
import signal
from typing import Any

from discord.errors import LoginFailure

from ..core import _codex_logger
from .vault import VaultError

logger = _codex_logger()


def _startup_failure_reason(error: Exception) -> str:
    """Map startup exceptions to bounded operator-facing messages."""
    if isinstance(error, LoginFailure):
        return "Discord authentication failed. Check the bot token."
    if isinstance(error, VaultError):
        return "Credential vault startup failed."
    return "Discord startup failed."


async def _wait_for_shutdown() -> None:
    """Keep the operator dashboard available until the process is stopped."""
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stopped.set)
        except (NotImplementedError, RuntimeError, ValueError):
            continue
        installed.append(signum)
    try:
        await stopped.wait()
    finally:
        for signum in installed:
            with contextlib.suppress(RuntimeError, ValueError):
                loop.remove_signal_handler(signum)


async def run_secure_launcher(bot: Any) -> None:
    """Start secure credentials and Discord without hiding startup failures."""
    bot.codex.enable_secure_credentials()
    lighthouse_started = False
    runtime_closed = False
    try:
        lighthouse_started = await bot.lighthouse.start()
        await bot.lighthouse.pause_secret_input()
        try:
            await bot.codex.unlock_secure_credentials()
        finally:
            bot.lighthouse.resume_secret_input()
        token = (
            bot.codex.secure_credential("TOKEN")
            or bot.codex.secure_credential("DISCORD_TOKEN")
            or bot.codex.secure_credential("THEIA_DISCORD_TOKEN")
        )
        if not token:
            raise VaultError("The credential vault does not contain a Discord token.")
        try:
            await bot.start(token)
        finally:
            del token
    except Exception as exc:  # noqa: BLE001 - retain only a safe failure summary
        reason = _startup_failure_reason(exc)
        bot.codex.record_startup_failure(reason, block=False)
        logger.critical("FATAL: %s", reason)
        if lighthouse_started:
            await bot.close(close_lighthouse=False)
            runtime_closed = True
            await _wait_for_shutdown()
    finally:
        if runtime_closed:
            await bot.lighthouse.close()
        else:
            await bot.close()
