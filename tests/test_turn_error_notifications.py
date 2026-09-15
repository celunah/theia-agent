import asyncio
from unittest.mock import patch
from unittest.mock import AsyncMock

import main
from tests.test_support import AsyncBehaviorTestBase


class TurnErrorNotificationTests(AsyncBehaviorTestBase):
    async def test_repeated_skill_changes_share_one_refresh_log_and_task(self) -> None:
        server = main.CodexAppServer()
        refresh_started = asyncio.Event()
        release_refresh = asyncio.Event()
        refresh_calls = 0

        async def refresh() -> None:
            nonlocal refresh_calls
            refresh_calls += 1
            refresh_started.set()
            await release_refresh.wait()

        with (
            patch.object(server, "_refresh_skills_after_change", new=refresh),
            patch("theia.server.notifications.logger.info") as info,
        ):
            notification = {"method": "skills/changed", "params": {}}
            server._handle_notification(notification)
            await refresh_started.wait()
            server._handle_notification(notification)
            server._handle_notification(notification)

            info.assert_called_once_with("Codex skill catalog changed; refreshing it")
            self.assertEqual(refresh_calls, 1)

            release_refresh.set()
            task = server._skills_refresh_task
            assert task is not None
            await task
            server._handle_notification(notification)
            server._handle_notification(notification)
            info.assert_called_once_with("Codex skill catalog changed; refreshing it")
            self.assertEqual(refresh_calls, 1)

    async def test_empty_protocol_error_uses_safe_no_details_reason(self) -> None:
        server = main.CodexAppServer()
        server._send = AsyncMock()
        request = asyncio.create_task(server._request("turn/start", {}))
        await asyncio.sleep(0)
        request_id = next(iter(server._pending))
        server._pending[request_id].set_result(
            {"id": request_id, "error": {"data": {"opaque": "hidden"}}}
        )

        with self.assertRaisesRegex(
            main.CodexAppServerError,
            "Codex returned an error without details",
        ) as caught:
            await request

        self.assertNotIn(
            "The request failed for an unspecified reason",
            str(caught.exception),
        )

    async def test_internal_error_notification_is_diagnostic_only(self) -> None:
        server = main.CodexAppServer()
        session = server._session("__presence__:session")
        state = main._TurnState(thread_id="thread", session=session)
        server._turns["turn-1"] = state

        with (
            patch("theia.server.notifications.logger.warning") as warning,
            patch("theia.server.notifications.logger.debug") as debug,
        ):
            server._handle_notification(
                {
                    "method": "error",
                    "params": {
                        "threadId": "thread",
                        "turnId": "turn-1",
                        "error": {"message": "worker unavailable"},
                    },
                }
            )

        warning.assert_not_called()
        debug.assert_any_call(
            "Codex error notification observed before terminal turn state "
            "(has_reason=%s)",
            True,
        )
        self.assertEqual(state.notification_error_reason, "worker unavailable")
        self.assertFalse(state.done.done())

    async def test_thread_only_error_cannot_fail_active_turn(self) -> None:
        server = main.CodexAppServer()
        session = server._session("active-session")
        state = main._TurnState(thread_id="thread", session=session)
        server._turns["turn-1"] = state

        server._handle_notification(
            {
                "method": "error",
                "params": {
                    "threadId": "thread",
                    "error": {"message": "a stale server-level error"},
                },
            }
        )

        self.assertIsNone(state.completed)
        self.assertFalse(state.done.done())

    async def test_live_error_waits_for_terminal_turn(self) -> None:
        server = main.CodexAppServer()
        session = server._session("active-session")
        state = main._TurnState(thread_id="thread", session=session)
        server._turns["turn-1"] = state

        server._handle_notification(
            {
                "method": "error",
                "params": {
                    "threadId": "thread",
                    "turnId": "turn-1",
                    "error": {"message": "real turn failure"},
                },
            }
        )
        self.assertIsNone(state.completed)
        self.assertFalse(state.done.done())
        self.assertEqual(state.notification_error_reason, "real turn failure")

        server._handle_notification(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread",
                    "turnId": "turn-1",
                    "turn": {"id": "turn-1", "status": "failed"},
                },
            }
        )

        with self.assertRaisesRegex(main.CodexAppServerError, "real turn failure"):
            await server._wait_for_turn("active-session", session, state, "turn-1")

    async def test_error_notification_does_not_fail_successful_turn(self) -> None:
        server = main.CodexAppServer()
        session = server._session("active-session")
        state = main._TurnState(thread_id="thread", session=session)
        server._turns["turn-1"] = state

        server._handle_notification(
            {
                "method": "error",
                "params": {
                    "threadId": "thread",
                    "turnId": "turn-1",
                    "error": {"message": "temporary server notice"},
                },
            }
        )
        server._handle_notification(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread",
                    "turnId": "turn-1",
                    "turn": {
                        "id": "turn-1",
                        "status": "completed",
                        "items": [
                            {
                                "type": "agentMessage",
                                "phase": "final",
                                "text": "completed normally",
                            }
                        ],
                    },
                },
            }
        )

        self.assertEqual(
            await server._wait_for_turn("active-session", session, state, "turn-1"),
            "completed normally",
        )

    async def test_failed_turn_without_reason_uses_honest_fallback(self) -> None:
        server = main.CodexAppServer()
        session = server._session("active-session")
        state = main._TurnState(thread_id="thread", session=session)
        state.completed = {"status": "failed"}
        state.done.set_result(None)

        self.assertEqual(
            main._safe_error_reason("The request failed for an unspecified reason."),
            "Codex returned an error without details.",
        )
        with self.assertRaisesRegex(
            main.CodexAppServerError,
            "Codex reported a failed turn without a reason",
        ):
            await server._wait_for_turn("active-session", session, state, "turn-1")

    async def test_failed_turn_log_keeps_readable_error_reason(self) -> None:
        server = main.CodexAppServer()
        session = server._session("__presence__:session")
        state = main._TurnState(thread_id="thread", session=session)
        state.completed = {
            "status": "failed",
            "error": {
                "message": "",
                "codexErrorInfo": "other",
                "additionalDetails": "response stream ended",
            },
        }
        state.done.set_result(None)

        with (
            patch("theia.server.core.logger.info") as info,
            self.assertRaises(main.CodexAppServerError),
        ):
            await server._wait_for_turn(
                "__presence__:session", session, state, "turn-1"
            )

        reason = info.call_args.args[2]
        self.assertEqual(
            reason,
            "response stream ended; Codex classification: Other",
        )

    async def test_failed_turn_log_includes_safe_protocol_context(self) -> None:
        server = main.CodexAppServer()
        session = server._session("__presence__:session")
        state = main._TurnState(thread_id="thread", session=session)
        state.completed = {
            "status": "failed",
            "error": {
                "message": "Codex returned an error without details.",
                "code": -32000,
                "data": {
                    "statusCode": 503,
                    "statusText": "Service Unavailable",
                },
            },
        }
        state.done.set_result(None)

        with (
            patch("theia.server.core.logger.info") as info,
            self.assertRaises(main.CodexAppServerError),
        ):
            await server._wait_for_turn(
                "__presence__:session", session, state, "turn-1"
            )

        reason = info.call_args.args[2]
        self.assertIn("code=-32000", reason)
        self.assertIn("data=503 Service Unavailable", reason)

    async def test_specific_error_notification_beats_generic_completion_error(
        self,
    ) -> None:
        server = main.CodexAppServer()
        session = server._session("__presence__:session")
        state = main._TurnState(thread_id="thread", session=session)
        state.notification_error_reason = "response stream disconnected: 503"
        state.completed = {
            "status": "failed",
            "error": {"message": "Codex returned an error without details."},
        }
        state.done.set_result(None)

        with (
            patch("theia.server.core.logger.info") as info,
            self.assertRaises(main.CodexAppServerError),
        ):
            await server._wait_for_turn(
                "__presence__:session", session, state, "turn-1"
            )

        self.assertEqual(info.call_args.args[2], "response stream disconnected: 503")

    async def test_error_from_another_thread_cannot_fail_live_turn(self) -> None:
        server = main.CodexAppServer()
        session = server._session("active-session")
        state = main._TurnState(thread_id="thread", session=session)
        server._turns["turn-1"] = state

        server._handle_notification(
            {
                "method": "error",
                "params": {
                    "threadId": "another-thread",
                    "turnId": "turn-1",
                    "error": {"message": "unrelated failure"},
                },
            }
        )

        self.assertIsNone(state.completed)
        self.assertFalse(state.done.done())
