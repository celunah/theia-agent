"""Dynamic Discord presence for recent activity and long-running tool turns."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import discord

from .core import _codex_logger, _env_float, _is_tool_item, _safe_log_label

logger = _codex_logger()

PRESENCE_IDLE_AFTER_ENV = "THEIA_PRESENCE_IDLE_AFTER"
PRESENCE_LONG_TASK_AFTER_ENV = "THEIA_PRESENCE_LONG_TASK_AFTER"
PRESENCE_UPDATE_INTERVAL_ENV = "THEIA_PRESENCE_UPDATE_INTERVAL"
DEFAULT_PRESENCE_IDLE_AFTER = 15 * 60
DEFAULT_PRESENCE_LONG_TASK_AFTER = 60
DEFAULT_PRESENCE_UPDATE_INTERVAL = 15


@dataclass
class _ActiveRequest:
    started_at: float
    tool_started: bool = False


class PresenceManager:
    """Keep Discord presence aligned with meaningful bot activity.

    A request does not become DND merely because it is active. It must first
    produce a real Codex tool event and remain active past the long-task
    threshold. Context compaction is deliberately ignored.
    """

    def __init__(
        self,
        change_presence: Callable[..., Awaitable[None]],
        *,
        idle_after: float | None = None,
        long_task_after: float | None = None,
        update_interval: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._change_presence = change_presence
        self.idle_after = max(
            1.0,
            idle_after
            if idle_after is not None
            else _env_float(PRESENCE_IDLE_AFTER_ENV, DEFAULT_PRESENCE_IDLE_AFTER),
        )
        self.long_task_after = max(
            1.0,
            long_task_after
            if long_task_after is not None
            else _env_float(
                PRESENCE_LONG_TASK_AFTER_ENV, DEFAULT_PRESENCE_LONG_TASK_AFTER
            ),
        )
        self.update_interval = max(
            1.0,
            update_interval
            if update_interval is not None
            else _env_float(
                PRESENCE_UPDATE_INTERVAL_ENV, DEFAULT_PRESENCE_UPDATE_INTERVAL
            ),
        )
        self._clock = clock
        self._last_interaction: float | None = None
        self._active_requests: dict[str, _ActiveRequest] = {}
        self._current_status: discord.Status | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._closed = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._loop_task is None or self._loop_task.done():
            self._closed = False
            self._loop_task = asyncio.create_task(self._run())
        await self.refresh()

    async def on_ready(self) -> None:
        """Re-send the state after the gateway reconnects."""
        self._current_status = None
        await self.refresh()

    async def close(self) -> None:
        self._closed = True
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None

    async def touch(self) -> None:
        """Record a user interaction and immediately prefer online presence."""
        self._last_interaction = self._clock()
        await self.refresh()

    async def begin_request(self, request_id: str) -> None:
        self._active_requests[request_id] = _ActiveRequest(self._clock())
        await self.refresh()

    async def observe_event(
        self, request_id: str, event: str, payload: dict[str, Any]
    ) -> None:
        """Record only real tool activity; compaction never qualifies as a task."""
        if event == "compacted":
            return
        active = self._active_requests.get(request_id)
        if active is None:
            return
        if event == "tool_activity" or (
            event == "item_started" and _is_tool_item(payload)
        ):
            active.tool_started = True
            await self.refresh()

    async def finish_request(self, request_id: str) -> None:
        self._active_requests.pop(request_id, None)
        await self.refresh()

    async def refresh(self) -> None:
        async with self._lock:
            now = self._clock()
            long_task = any(
                request.tool_started
                and now - request.started_at >= self.long_task_after
                for request in self._active_requests.values()
            )
            if long_task:
                status = discord.Status.dnd
            elif (
                self._last_interaction is not None
                and now - self._last_interaction < self.idle_after
            ):
                status = discord.Status.online
            else:
                status = discord.Status.idle
            if status == self._current_status:
                return
            try:
                await self._change_presence(status=status)
            except Exception as exc:  # noqa: BLE001 - presence must not affect chat
                logger.debug(
                    "Could not update Discord presence (status=%s, error=%s)",
                    _safe_log_label(status.name),
                    type(exc).__name__,
                )
                return
            self._current_status = status
            logger.debug("Discord presence updated (status=%s)", status.name)

    async def _run(self) -> None:
        while not self._closed:
            await asyncio.sleep(self.update_interval)
            await self.refresh()
