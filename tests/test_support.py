# The split test modules intentionally share this compatibility fixture surface.
# pylint: disable=cyclic-import,duplicate-code
import asyncio
import json
import logging
import os
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import discord
from typing_extensions import Self

import main
from scripts import build_nuitka
from scripts.configure import (
    VOICE_MODE,
    TEXT_MODE,
    ConfigurationError,
    collect_configuration,
    save_configuration,
    validate_configuration,
)
from theia import core as core_module
from theia.bot.core import _PersistentViewStore, _handle_voice_transcript, on_message
from theia.core import _path_is_under


class _Channel:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.id: Any = None
        self.guild: Any = None
        self.create_thread: Any = None
        self.edit: Any = None

    async def send(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(id=len(self.sent))


class _FailingSendChannel(_Channel):
    async def send(self, *_args, **_kwargs):
        raise discord.DiscordException("permission revoked")


def _admin_guild(user_id: int = 7, administrator: bool = True) -> Any:
    member = SimpleNamespace(
        id=user_id,
        guild_permissions=SimpleNamespace(administrator=administrator),
    )
    return SimpleNamespace(
        id=1,
        get_member=lambda candidate: member if candidate == user_id else None,
    )


MOOD_TEST_NAMESPACE = f"mood-test-{time.time_ns()}"


def _mood_test_key(name: str) -> str:
    return f"{MOOD_TEST_NAMESPACE}:{name}"


class _TypingContext:
    def __init__(self, channel: "_TypingChannel") -> None:
        self.channel = channel

    async def __aenter__(self):
        self.channel.typing_started = True

    async def __aexit__(self, *_args):
        self.channel.typing_started = False


class _TypingChannel(_Channel):
    def __init__(self) -> None:
        super().__init__()
        self.typing_started = False

    def typing(self) -> _TypingContext:
        return _TypingContext(self)


class _Message:
    id = 100

    def __init__(self) -> None:
        self.edits: list[dict] = []
        self.reactions: list[str] = []
        self.deleted = False

    async def edit(self, **kwargs):
        self.edits.append(kwargs)

    async def add_reaction(self, emoji: str) -> None:
        self.reactions.append(emoji)

    async def delete(self) -> None:
        self.deleted = True


class _ImageMessage(_Message):
    def __init__(self) -> None:
        super().__init__()
        self.attachments = (SimpleNamespace(url="https://cdn.example/image.png"),)


class _HistoryChannel(_Channel):
    def __init__(self, messages: list[SimpleNamespace]) -> None:
        super().__init__()
        self.messages = messages
        self.history_calls: list[dict] = []

    def history(self, *, limit: int, before=None):
        self.history_calls.append({"limit": limit, "before": before})
        eligible = [
            item for item in self.messages if before is None or item.id < before.id
        ]

        async def iterator():
            for item in reversed(eligible[-limit:]):
                yield item

        return iterator()


class _ForbiddenHistoryIterator:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise discord.Forbidden(
            SimpleNamespace(status=403, reason="Forbidden"), "forbidden"
        )


class _ForbiddenHistoryChannel(_Channel):
    def __init__(self, channel_id: int) -> None:
        super().__init__()
        self.id = channel_id

    def history(self, *, limit: int, after=None):  # pylint: disable=unused-argument
        return _ForbiddenHistoryIterator()


class AsyncBehaviorTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        """Keep default-policy tests independent of a developer's .env file."""
        self._approval_environment = patch.dict(
            os.environ, {"THEIA_APPROVAL_LEVEL": "high"}
        )
        self._approval_environment.start()
        self.addCleanup(self._approval_environment.stop)


__all__ = [
    "MOOD_TEST_NAMESPACE",
    "TEXT_MODE",
    "VOICE_MODE",
    "Any",
    "AsyncBehaviorTestBase",
    "AsyncMock",
    "ConfigurationError",
    "Mock",
    "Path",
    "Self",
    "SimpleNamespace",
    "_Channel",
    "_FailingSendChannel",
    "_ForbiddenHistoryChannel",
    "_ForbiddenHistoryIterator",
    "_HistoryChannel",
    "_ImageMessage",
    "_Message",
    "_PersistentViewStore",
    "_TypingChannel",
    "_TypingContext",
    "_admin_guild",
    "_handle_voice_transcript",
    "_mood_test_key",
    "_path_is_under",
    "asyncio",
    "build_nuitka",
    "cast",
    "collect_configuration",
    "core_module",
    "datetime",
    "discord",
    "json",
    "logging",
    "main",
    "on_message",
    "os",
    "patch",
    "save_configuration",
    "tempfile",
    "time",
    "timezone",
    "unittest",
    "validate_configuration",
]
