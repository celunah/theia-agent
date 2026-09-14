# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable
from tests.test_support import *

from theia.core import _TurnDiagnostics
from theia.server.worker_diagnostics import record_internal_request, run_worker


class WorkerObservabilityTests(AsyncBehaviorTestBase):
    async def test_low_signal_turns_skip_mood_and_attention_workers(self) -> None:
        server = main.CodexAppServer()
        server.classify_attention = AsyncMock()
        server.classify_mood = AsyncMock()
        session = server._session("low-signal")
        diagnostics = _TurnDiagnostics()

        await server._prepare_attention_for_turn(
            session,
            "ok",
            diagnostics=diagnostics,
        )
        server._schedule_mood_appraisal(
            session,
            "ok",
            diagnostics=diagnostics,
        )
        await asyncio.sleep(0)

        server.classify_attention.assert_not_awaited()
        server.classify_mood.assert_not_awaited()
        self.assertIsNone(diagnostics.attention_classifier_duration_ms)
        self.assertIsNone(diagnostics.mood_classifier_duration_ms)

    async def test_worker_diagnostics_are_bounded_and_classify_outcomes(self) -> None:
        diagnostics = _TurnDiagnostics()

        async def successful_worker() -> None:
            record_internal_request()
            await asyncio.sleep(0.001)

        await run_worker(diagnostics, "attention", successful_worker())
        with self.assertRaises(asyncio.TimeoutError):
            await run_worker(
                diagnostics,
                "mood",
                self._raise(asyncio.TimeoutError()),
            )
        with self.assertRaises(RuntimeError):
            await run_worker(
                diagnostics,
                "self_improvement",
                self._raise(RuntimeError("worker failed")),
            )

        cancellation = asyncio.create_task(
            run_worker(
                diagnostics,
                "workspace_review",
                asyncio.Event().wait(),
            )
        )
        await asyncio.sleep(0)
        cancellation.cancel()
        await asyncio.gather(cancellation, return_exceptions=True)

        self.assertGreaterEqual(diagnostics.attention_classifier_duration_ms or 0, 0)
        self.assertGreaterEqual(diagnostics.mood_classifier_duration_ms or 0, 0)
        self.assertGreaterEqual(
            diagnostics.self_improvement_duration_ms or 0,
            0,
        )
        self.assertGreaterEqual(
            diagnostics.workspace_review_duration_ms or 0,
            0,
        )
        self.assertEqual(diagnostics.approximate_internal_request_count, 1)
        self.assertEqual(diagnostics.timeout_count, 1)
        self.assertEqual(diagnostics.cancellation_count, 1)
        self.assertEqual(diagnostics.failed_worker_count, 1)

    @staticmethod
    async def _raise(error: BaseException) -> None:
        raise error

    async def test_normal_turn_diagnostics_are_safe_and_session_scoped(self) -> None:
        server = main.CodexAppServer()
        first = server._session("diagnostics-first")
        second = server._session("diagnostics-second")
        first_diagnostics = _TurnDiagnostics()
        second_diagnostics = _TurnDiagnostics()
        first.turn_diagnostics = first_diagnostics
        second.turn_diagnostics = second_diagnostics
        state = main._TurnState(
            session=first,
            diagnostics=first_diagnostics,
        )
        state.completed = {"status": "completed"}
        state.done.set_result(None)

        await server._wait_for_turn(first.key, first, state, "normal-turn")

        self.assertIsNotNone(first_diagnostics.normal_turn_duration_ms)
        self.assertIsNone(second_diagnostics.normal_turn_duration_ms)
        debug = server.debug_state(first.key)
        self.assertIn("diagnostics", debug)
        self.assertNotIn("normal-turn", repr(debug["diagnostics"]))
        self.assertNotIn(first.key, repr(debug["diagnostics"]))
        rendered = main.render_lighthouse(server.lighthouse_snapshot(first.key))
        self.assertIn("Runtime", rendered)
        self.assertIn("Workers", rendered)

    async def test_workspace_review_is_non_blocking_and_only_latest_is_pending(self):
        server = main.CodexAppServer()
        session = server._session("workspace-scheduling")
        session.lock = asyncio.Lock()
        gate = asyncio.Event()

        async def blocked_review(*_args: Any, **_kwargs: Any) -> None:
            await gate.wait()

        server._run_workspace_review = blocked_review
        diagnostics = _TurnDiagnostics()
        started = time.monotonic()
        server._schedule_workspace_review(
            session,
            "A substantial request.",
            "A completed response.",
            recent_context=None,
            self_model={},
            diagnostics=diagnostics,
        )
        self.assertLess(time.monotonic() - started, 0.1)
        first_task = session.workspace_review_task
        self.assertIsNotNone(first_task)
        await asyncio.sleep(0)

        server._schedule_workspace_review(
            session,
            "A newer request.",
            "A newer response.",
            recent_context=None,
            self_model={},
            diagnostics=diagnostics,
        )
        second_task = session.workspace_review_task
        self.assertIsNotNone(second_task)
        self.assertIsNot(first_task, second_task)
        assert first_task is not None
        await asyncio.sleep(0)
        self.assertTrue(first_task.cancelled() or first_task.done())
        self.assertEqual(session.workspace_review_generation, 2)

        assert second_task is not None
        second_task.cancel()
        await asyncio.gather(first_task, second_task, return_exceptions=True)

    async def test_workspace_worker_failure_does_not_escape_main_turn(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server._request = AsyncMock(side_effect=RuntimeError("failed worker"))
        session = server._session("workspace-failure")
        session.lock = asyncio.Lock()
        diagnostics = _TurnDiagnostics()

        server._schedule_workspace_review(
            session,
            "A request.",
            "A response.",
            recent_context=None,
            self_model={},
            diagnostics=diagnostics,
        )
        task = session.workspace_review_task
        self.assertIsNotNone(task)
        assert task is not None
        await task

        self.assertEqual(diagnostics.failed_worker_count, 1)
        self.assertEqual(server.session_workspace(session.key)["entries"], [])

    async def test_shutdown_awaits_all_registered_worker_cancellations(self) -> None:
        server = main.CodexAppServer()
        task = asyncio.create_task(asyncio.Event().wait())
        server._server_tasks.add(task)

        await server.close()

        self.assertTrue(task.cancelled())
        self.assertEqual(server._server_tasks, set())
