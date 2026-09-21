"""Secure launcher tests for pre-Discord Lighthouse failures."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from discord.errors import LoginFailure

from theia.server.launcher import run_secure_launcher


class _FakeLighthouse:
    def __init__(self, started: bool = True) -> None:
        self.started = started
        self.closed = False

    async def start(self) -> bool:
        return self.started

    async def pause_secret_input(self) -> None:
        return None

    def resume_secret_input(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class _FakeCodex:
    def __init__(self) -> None:
        self.record_startup_failure = Mock()
        self._token = "token-value"

    def enable_secure_credentials(self) -> None:
        return None

    async def unlock_secure_credentials(self) -> None:
        return None

    def secure_credential(self, name: str) -> str:
        return self._token if name == "TOKEN" else ""


class _FakeBot:
    def __init__(self, *, lighthouse_started: bool = True) -> None:
        self.codex = _FakeCodex()
        self.lighthouse = _FakeLighthouse(lighthouse_started)
        self.close_calls: list[bool] = []

    async def start(self, _token: str) -> None:
        raise LoginFailure("private token detail")

    async def close(self, *, close_lighthouse: bool = True) -> None:
        self.close_calls.append(close_lighthouse)
        if close_lighthouse:
            await self.lighthouse.close()


class SecureLauncherTests(unittest.IsolatedAsyncioTestCase):
    async def test_discord_auth_failure_stays_on_lighthouse_until_shutdown(
        self,
    ) -> None:
        bot = _FakeBot()
        with patch(
            "theia.server.launcher._wait_for_shutdown",
            new=AsyncMock(),
        ) as wait_for_shutdown:
            await run_secure_launcher(bot)

        bot.codex.record_startup_failure.assert_called_once_with(
            "Discord authentication failed. Check the bot token.", block=False
        )
        self.assertEqual(bot.close_calls, [False])
        self.assertTrue(bot.lighthouse.closed)
        wait_for_shutdown.assert_awaited_once_with()

    async def test_noninteractive_auth_failure_closes_normally(self) -> None:
        bot = _FakeBot(lighthouse_started=False)
        with patch("theia.server.launcher._wait_for_shutdown", new=AsyncMock()) as wait:
            await run_secure_launcher(bot)

        self.assertEqual(bot.close_calls, [True])
        self.assertTrue(bot.lighthouse.closed)
        wait.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
