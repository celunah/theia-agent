# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable,duplicate-code
from tests.test_support import *


class MemoryWatchdogTests(AsyncBehaviorTestBase):
    def test_rss_includes_the_codex_launcher_process_tree(self) -> None:
        server = main.CodexAppServer()
        server._process = cast(Any, SimpleNamespace(pid=10, returncode=None))
        root = Mock()
        child = Mock()
        root.children.return_value = [child]
        root.memory_info.return_value = SimpleNamespace(rss=100)
        child.memory_info.return_value = SimpleNamespace(rss=900)

        with patch("theia.server.lifecycle.psutil.Process", return_value=root):
            self.assertEqual(server._codex_process_rss(), 1000)

    def test_memory_watchdog_can_be_disabled(self) -> None:
        with patch.dict(os.environ, {"THEIA_CODEX_MEMORY_WATCHDOG": "false"}):
            server = main.CodexAppServer()

        server._start_memory_watchdog()
        self.assertIsNone(server._memory_watchdog_task)

    async def test_sustained_rss_breach_schedules_recovery(self) -> None:
        server = main.CodexAppServer()
        server._memory_watchdog_limit = 100
        server._memory_watchdog_interval = 0
        server._memory_breach_samples = 2
        server._codex_process_rss = Mock(return_value=200)
        recovery = AsyncMock()
        server._recover_memory_pressure = recovery

        await server._memory_watchdog_loop()
        task = server._memory_recovery_task
        self.assertIsNotNone(task)
        await task
        recovery.assert_awaited_once_with(200)

    async def test_memory_recovery_interrupts_active_turns_before_failing_them(
        self,
    ) -> None:
        server = main.CodexAppServer()
        server._memory_restart_grace = 0.01
        session = server._session("memory-test")
        session.thread_id = "thread"
        session.turn_id = "turn"
        state = main._TurnState(thread_id="thread", session=session)
        server._turns["turn"] = state
        server._request = AsyncMock(return_value={})

        await server._interrupt_active_turns()

        server._request.assert_awaited_once_with(
            "turn/interrupt",
            {"threadId": "thread", "turnId": "turn"},
            timeout=0.5,
        )
        with self.assertRaisesRegex(main.CodexAppServerError, "memory usage exceeded"):
            await state.done

    async def test_memory_recovery_restarts_codex_after_interrupting_turns(
        self,
    ) -> None:
        server = main.CodexAppServer()
        server._process = cast(Any, SimpleNamespace(returncode=None))
        server._interrupt_active_turns = AsyncMock()
        server._close_locked = AsyncMock()
        server._start_locked = AsyncMock()
        server._start_memory_watchdog = Mock()

        await server._recover_memory_pressure(700 * 1024 * 1024)

        server._interrupt_active_turns.assert_awaited_once_with()
        server._close_locked.assert_awaited_once_with()
        server._start_locked.assert_awaited_once_with()
        server._start_memory_watchdog.assert_called_once_with()
        self.assertFalse(server._memory_recovery_active)
