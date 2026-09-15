"""Deferred generic thinking status for Discord response delivery."""

import asyncio
from typing import Any

GENERIC_THINKING_DELAY = 5.0


class GenericThinkingStatus:
    def __init__(self, delivery: Any) -> None:
        self.delivery = delivery
        self.delay = GENERIC_THINKING_DELAY
        self.task: asyncio.Task[None] | None = None

    def cancel(self) -> None:
        task = self.task
        self.task = None
        if task is not None and not task.done():
            task.cancel()

    def schedule(self) -> None:
        if (
            self.delivery.status_message is not None
            or self.delivery.thinking_summary is not None
        ):
            return
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._show())

    async def _show(self) -> None:
        try:
            await asyncio.sleep(self.delay)
            async with self.delivery.lock:
                if self.task is not asyncio.current_task():
                    return
                self.task = None
                if (
                    self.delivery.status_message is None
                    and self.delivery.thinking_summary is None
                ):
                    await self.delivery._set_status("Thinking", "Thinking", force=True)
        except asyncio.CancelledError:
            return
