# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable,duplicate-code
from tests.test_support import *


class RichPresenceManagerTests(unittest.IsolatedAsyncioTestCase):
    async def _wait_for(self, predicate: Any) -> None:
        for _ in range(100):
            if predicate():
                return
            await asyncio.sleep(0.01)
        self.fail("timed out waiting for Rich Presence update")

    async def test_active_presence_maps_type_and_suppresses_duplicate_phases(
        self,
    ) -> None:
        changes: list[dict[str, Any]] = []
        generated: list[str] = []

        async def change_presence(**kwargs: Any) -> None:
            changes.append(kwargs)

        async def generate(prompt: str, **_kwargs: Any) -> dict[str, str]:
            generated.append(prompt)
            return {"activity_type": "listening", "text": "reviewing"}

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            active_debounce=0,
            timeout=1,
        )
        try:
            await manager.begin_task(
                "task",
                session_key="guild:1:channel:2:user:3",
                guild_id=1,
                prompt="Review the request.",
                channel_context="Recent exchange.",
            )
            await self._wait_for(lambda: len(changes) == 1)
            self.assertEqual(
                changes[-1]["activity"].type, discord.ActivityType.listening
            )
            self.assertEqual(changes[-1]["activity"].name, "reviewing")

            await manager.observe_event(
                "task", "item_started", {"type": "commandExecution"}
            )
            await self._wait_for(lambda: len(generated) == 2)
            await manager.observe_event(
                "task", "item_completed", {"type": "commandExecution"}
            )
            await asyncio.sleep(0.02)
            self.assertEqual(len(generated), 2)
            await manager.observe_event(
                "task", "item_started", {"type": "commandExecution"}
            )
            await asyncio.sleep(0.02)
            self.assertEqual(len(generated), 2)
        finally:
            await manager.close()

    async def test_active_activity_overrides_idle_and_finishing_clears_it(self) -> None:
        changes: list[dict[str, Any]] = []

        async def change_presence(**kwargs: Any) -> None:
            changes.append(kwargs)

        async def generate(prompt: str, **_kwargs: Any) -> dict[str, str]:
            activity_type = "none" if "idle" in prompt else "playing"
            return {"activity_type": activity_type, "text": "available"}

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            active_debounce=0,
            timeout=1,
        )
        try:
            await manager.refresh_idle()
            await self._wait_for(lambda: bool(changes))
            self.assertIsNotNone(changes[-1]["activity"])

            await manager.begin_task(
                "task",
                session_key="guild:1:channel:2:user:3",
                guild_id=1,
                prompt="Do work.",
                channel_context=None,
            )
            self.assertIsInstance(changes[-1]["activity"], discord.CustomActivity)
            self.assertEqual(changes[-1]["activity"].name, "available")
            await self._wait_for(
                lambda: (
                    changes[-1].get("activity") is not None
                    and changes[-1]["activity"].type == discord.ActivityType.playing
                )
            )
            await manager.finish_task("task", response="Done.")
            self.assertIsInstance(changes[-1]["activity"], discord.CustomActivity)
            self.assertEqual(changes[-1]["activity"].name, "available")
        finally:
            await manager.close()

    async def test_start_generates_idle_presence_immediately(self) -> None:
        generated = asyncio.Event()

        async def change_presence(**_kwargs: Any) -> None:
            return

        async def generate(prompt: str, **_kwargs: Any) -> dict[str, str]:
            self.assertIn("idle", prompt.casefold())
            generated.set()
            return {"activity_type": "none", "text": "available"}

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            idle_interval=900,
            recent_idle_interval=600,
            timeout=1,
        )
        try:
            await manager.start()
            await asyncio.wait_for(generated.wait(), timeout=1)
        finally:
            await manager.close()

    async def test_idle_presence_uses_one_session_context_without_cross_guild_mixing(
        self,
    ) -> None:
        prompts: list[tuple[str, str | None]] = []

        async def change_presence(**_kwargs: Any) -> None:
            return

        async def generate(prompt: str, **kwargs: Any) -> dict[str, str]:
            prompts.append((prompt, kwargs.get("session_key")))
            return {"activity_type": "none", "text": "ready"}

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            active_debounce=0,
            timeout=1,
            clock=lambda: 1.0,
        )
        try:
            for request_id, session_key, guild_id, prompt in (
                ("first", "guild:1:channel:2:user:3", 1, "guild one task"),
                ("second", "guild:2:channel:4:user:5", 2, "guild two task"),
            ):
                await manager.begin_task(
                    request_id,
                    session_key=session_key,
                    guild_id=guild_id,
                    prompt=prompt,
                    channel_context=f"context for {prompt}",
                )
                await manager.finish_task(request_id, response=f"result for {prompt}")
            prompts.clear()
            await manager.refresh_idle()
            await self._wait_for(lambda: bool(prompts))
            self.assertEqual(prompts[0][1], "guild:2:channel:4:user:5")
            self.assertIn("guild two task", prompts[0][0])
            self.assertNotIn("guild one task", prompts[0][0])
        finally:
            await manager.close()

    async def test_finishing_a_task_cancels_its_pending_presence_request(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()
        never = asyncio.Event()

        async def change_presence(**_kwargs: Any) -> None:
            return

        async def generate(_prompt: str, **_kwargs: Any) -> dict[str, str]:
            started.set()
            try:
                await never.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return {"activity_type": "none", "text": "unreachable"}

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            active_debounce=0,
            timeout=10,
        )
        try:
            await manager.begin_task(
                "task",
                session_key="guild:1:channel:2:user:3",
                guild_id=1,
                prompt="Do work.",
                channel_context=None,
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            await manager.finish_task("task")
            self.assertTrue(cancelled.is_set())
        finally:
            await manager.close()

    async def test_recent_idle_context_uses_the_longer_refresh_interval(self) -> None:
        now = [0.0]

        async def change_presence(**_kwargs: Any) -> None:
            return

        async def generate(_prompt: str, **_kwargs: Any) -> dict[str, str] | None:
            return None

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            idle_interval=900,
            recent_idle_interval=600,
            context_max_age=1800,
            active_debounce=10,
            timeout=1,
            clock=lambda: now[0],
        )
        try:
            self.assertEqual(await manager._idle_delay(), 900)
            await manager.begin_task(
                "task",
                session_key="guild:1:channel:2:user:3",
                guild_id=1,
                prompt="Do work.",
                channel_context=None,
            )
            await manager.finish_task("task")
            self.assertEqual(await manager._idle_delay(), 600)
            now[0] = 1800
            self.assertEqual(await manager._idle_delay(), 900)
        finally:
            await manager.close()


class PresenceManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_bot_presence_waits_for_gateway_readiness(self) -> None:
        test_bot = main.TheiaBot()
        change_presence = AsyncMock()
        test_bot.change_presence = change_presence
        with patch.object(test_bot, "is_ready", return_value=False):
            await test_bot._change_presence_when_ready(status=discord.Status.idle)
        change_presence.assert_not_awaited()

    async def test_status_updates_retain_rich_activity(self) -> None:
        test_bot = main.TheiaBot()
        activity = discord.CustomActivity("reviewing")
        test_bot.rich_presence._current_activity = activity
        change_presence = AsyncMock()
        test_bot.change_presence = change_presence
        with patch.object(test_bot, "is_ready", return_value=True):
            await test_bot._change_presence_when_ready(status=discord.Status.dnd)

        await_args = change_presence.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertEqual(
            await_args.kwargs,
            {"status": discord.Status.dnd, "activity": activity},
        )

    async def test_rich_presence_preserves_the_current_status(self) -> None:
        test_bot = main.TheiaBot()
        test_bot.presence._current_status = discord.Status.idle
        activity = discord.CustomActivity("reviewing")
        change_presence = AsyncMock()
        test_bot.change_presence = change_presence
        with patch.object(test_bot, "is_ready", return_value=True):
            await test_bot._change_rich_presence(activity=activity)

        await_args = change_presence.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertEqual(
            await_args.kwargs,
            {"status": discord.Status.idle, "activity": activity},
        )

    async def _manager(self, now: list[float]) -> tuple[main.PresenceManager, list]:
        changes: list = []

        async def change_presence(**kwargs):
            changes.append(kwargs["status"])

        manager = main.PresenceManager(
            change_presence,
            idle_after=10,
            long_task_after=5,
            update_interval=60,
            clock=lambda: now[0],
        )
        await manager.start()
        return manager, changes

    async def test_recent_interaction_is_online_then_becomes_idle(self) -> None:
        now = [0.0]
        manager, changes = await self._manager(now)
        try:
            await manager.touch()
            self.assertEqual(changes[-1], discord.Status.online)
            now[0] = 10
            await manager.refresh()
            self.assertEqual(changes[-1], discord.Status.idle)
        finally:
            await manager.close()

    async def test_only_long_running_tool_turn_becomes_dnd(self) -> None:
        now = [0.0]
        manager, changes = await self._manager(now)
        try:
            await manager.touch()
            await manager.begin_request("tool-turn")
            await manager.observe_event(
                "tool-turn", "item_started", {"type": "commandExecution"}
            )
            now[0] = 5
            await manager.refresh()
            self.assertEqual(changes[-1], discord.Status.dnd)
            await manager.finish_request("tool-turn")
            self.assertEqual(changes[-1], discord.Status.online)
        finally:
            await manager.close()

    async def test_basic_turn_and_context_compaction_never_become_dnd(self) -> None:
        now = [0.0]
        manager, changes = await self._manager(now)
        try:
            await manager.touch()
            await manager.begin_request("basic-turn")
            now[0] = 6
            await manager.observe_event(
                "basic-turn", "compacted", {"reason": "context"}
            )
            await manager.refresh()
            now[0] = 11
            await manager.refresh()
            self.assertNotIn(discord.Status.dnd, changes)
        finally:
            await manager.close()


if __name__ == "__main__":
    unittest.main()
