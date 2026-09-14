# pylint: disable=wildcard-import,unused-wildcard-import,protected-access
"""Focused tests for bounded session open loops."""

from tests.test_support import *
from theia.server.commitments import expire_commitments


COMMITMENT_TEST_NAMESPACE = f"commitment-test-{time.time_ns()}"


def _key(name: str, *, user: int = 7, guild: int = 1, channel: int = 2) -> str:
    return f"guild:{guild}:channel:{channel}:user:{user}:{COMMITMENT_TEST_NAMESPACE}:{name}"


def _session_key(*, user: int = 7, guild: int = 1, channel: int = 2) -> str:
    return f"guild:{guild}:channel:{channel}:user:{user}"


def _operation(key: str, category: str, text: str) -> dict[str, str]:
    return {"op": "upsert", "key": key, "category": category, "text": text}


def _proposal(
    key: str,
    text: str = "Resolve the Qwen audio bridge",
    *,
    kind: str = "deferred_question",
    source: str = "user_request",
) -> dict[str, Any]:
    return {
        "workspace_key": key,
        "kind": kind,
        "text": text,
        "source": source,
        "explicit": True,
    }


class CommitmentTests(AsyncBehaviorTestBase):
    def _with_workspace(self, server: Any, session: Any, key: str) -> float:
        event_at = time.time()
        self.assertTrue(
            server._apply_workspace_delta(
                session,
                [_operation(key, "open_question", "Resolve the Qwen audio bridge")],
                base_generation=1,
                base_revision=0,
                now=event_at,
            )
        )
        return event_at

    def test_explicit_commitment_is_created_and_linked(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_key("explicit"))
        event_at = self._with_workspace(server, session, "qwen_bridge")

        self.assertTrue(
            server._apply_commitment_proposals(
                session,
                [_proposal("qwen_bridge")],
                user_prompt="Please remind me to revisit the Qwen audio bridge later.",
                response="We can leave that question open for now.",
                now=event_at + 1,
            )
        )
        records = server.session_commitments(session.key)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["workspace_key"], "qwen_bridge")
        self.assertEqual(records[0]["status"], "active")

    def test_speculative_commitment_is_rejected(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_key("speculative"))
        event_at = self._with_workspace(server, session, "ordinary_note")

        self.assertFalse(
            server._apply_commitment_proposals(
                session,
                [_proposal("ordinary_note")],
                user_prompt="The audio bridge is interesting.",
                response="It has several implementation details.",
                now=event_at + 1,
            )
        )
        self.assertEqual(server.session_commitments(session.key), [])

    def test_reviewer_parser_requires_a_workspace_link(self) -> None:
        server = main.CodexAppServer()
        parsed = server._parse_workspace_review(
            '{"operations":[{"op":"upsert","key":"bridge",'
            '"category":"open_question","text":"Resolve the bridge"}],'
            '"commitments":[{"workspace_key":"bridge",'
            '"kind":"deferred_question","text":"Resolve the bridge",'
            '"source":"user_request","explicit":true}]}'
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(len(parsed[0]), 1)
        self.assertEqual(len(parsed[1]), 1)

        session = server._session(_key("review-parser"))
        self.assertTrue(
            server._apply_workspace_review(
                session,
                parsed[0],
                parsed[1],
                base_generation=1,
                base_revision=0,
                user_prompt="Please revisit the bridge later.",
                response="I will follow up.",
            )
        )
        self.assertEqual(len(server.session_commitments(session.key)), 1)

    def test_malformed_commitment_proposals_are_ignored(self) -> None:
        server = main.CodexAppServer()
        parsed = server._parse_workspace_review(
            '{"operations":[],"commitments":[{"workspace_key":[],'
            '"kind":[],"source":[],"text":[],"explicit":true}]}'
        )
        self.assertEqual(parsed, ([], []))

    def test_return_cue_is_relevant_once_and_does_not_reset_thread(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_key("return"))
        session.thread_id = "existing-thread"
        event_at = self._with_workspace(server, session, "qwen_bridge")
        server._apply_commitment_proposals(
            session,
            [_proposal("qwen_bridge")],
            user_prompt="Let's revisit the Qwen audio bridge later.",
            response="I will follow up on it.",
            now=event_at + 1,
        )
        event = {"relation": "RETURN", "new_topic": "Qwen audio bridge"}

        prompt = server._render_commitment_prompt(session, event, "Back to the bridge")
        self.assertIn("one active open loop", prompt)
        self.assertEqual(
            server._render_commitment_prompt(session, event, "Back to the bridge"), ""
        )
        self.assertEqual(session.thread_id, "existing-thread")

    def test_completion_dismissal_and_expiration_are_bounded(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_key("statuses"))
        event_at = self._with_workspace(server, session, "first_loop")
        server._apply_workspace_delta(
            session,
            [_operation("second_loop", "goal", "Finish the deployment")],
            base_generation=1,
            base_revision=1,
            now=event_at,
        )
        server._apply_commitment_proposals(
            session,
            [
                _proposal("first_loop", "Revisit the Qwen audio bridge"),
                _proposal(
                    "second_loop",
                    "Finish the deployment",
                    kind="unfinished_task",
                    source="assistant_promise",
                ),
            ],
            user_prompt="Please revisit the Qwen audio bridge.",
            response="I'll follow up and finish the deployment.",
            now=event_at + 1,
        )
        records = server.session_commitments(session.key)
        first_id = next(
            item["commitment_id"]
            for item in records
            if item["workspace_key"] == "first_loop"
        )
        second_id = next(
            item["commitment_id"]
            for item in records
            if item["workspace_key"] == "second_loop"
        )
        server.update_commitment(session.key, first_id, "completed")
        server.update_commitment(session.key, second_id, "dismissed")
        self.assertEqual(server.session_commitments(session.key), [])
        self.assertEqual(
            server.session_commitments(session.key, include_closed=True)[0]["status"],
            "dismissed",
        )
        session.commitments[first_id].status = "active"
        session.commitments[first_id].expires_at = 10.0
        self.assertTrue(expire_commitments(session, now=11.0))
        self.assertEqual(session.commitments[first_id].status, "stale")

    def test_commitments_are_isolated_by_session(self) -> None:
        server = main.CodexAppServer()
        first = server._session(_key("first"))
        second = server._session(_key("second", user=8))
        event_at = self._with_workspace(server, first, "private_loop")
        server._apply_commitment_proposals(
            first,
            [_proposal("private_loop")],
            user_prompt="Remind me to revisit the Qwen audio bridge.",
            response="I will follow up.",
            now=event_at + 1,
        )
        self.assertEqual(len(server.session_commitments(first.key)), 1)
        self.assertEqual(server.session_commitments(second.key), [])

    def test_promotion_is_explicit_and_uses_private_user_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(root / "state.json"),
                },
            ):
                server = main.CodexAppServer()
                session = server._session(_session_key())
                event_at = self._with_workspace(server, session, "promote_loop")
                server._apply_commitment_proposals(
                    session,
                    [_proposal("promote_loop")],
                    user_prompt="Please remind me to revisit the Qwen audio bridge.",
                    response="I will follow up.",
                    now=event_at + 1,
                )
                commitment_id = server.session_commitments(session.key)[0][
                    "commitment_id"
                ]
                server.promote_commitment(session.key, commitment_id)
                memory_path = root / "theia" / "memories" / "users" / "7" / "USER.md"
                self.assertIn("Resolve the Qwen audio bridge", memory_path.read_text())
                self.assertEqual(server.session_commitments(session.key), [])

    def test_commitments_persist_and_restore_with_the_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = _session_key(user=71, guild=81, channel=91)
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(root / "state.json"),
                },
            ):
                server = main.CodexAppServer()
                session = server._session(key)
                event_at = self._with_workspace(server, session, "restore_loop")
                server._apply_commitment_proposals(
                    session,
                    [_proposal("restore_loop")],
                    user_prompt="Please remind me to revisit the Qwen audio bridge.",
                    response="I will follow up.",
                    now=event_at + 1,
                )
                server._persist_state()
                restored = main.CodexAppServer()
                records = restored.session_commitments(key)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["workspace_key"], "restore_loop")
