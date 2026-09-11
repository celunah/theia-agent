# pylint: disable=wildcard-import,unused-wildcard-import,protected-access
"""Focused tests for semantic conversational attention state."""

from tests.test_support import *


ATTENTION_TEST_NAMESPACE = f"attention-test-{time.time_ns()}"


def _attention_key(name: str) -> str:
    return f"{ATTENTION_TEST_NAMESPACE}:{name}"


def _attention(session: Any) -> Any:
    assert session.attention is not None
    return session.attention


def _classification(
    relation: str,
    *,
    title: str = "Rust memory allocation",
    summary: str = "A discussion about memory allocation behavior.",
    preserve: bool = False,
    target: str | None = None,
    acknowledge: bool = True,
    confidence: float = 0.9,
    loops: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "relation": relation,
        "confidence": confidence,
        "acknowledge": acknowledge,
        "preserve_context": preserve,
        "target_context_id": target,
        "topic_title": title,
        "topic_summary": summary,
        "open_loops": loops or [],
        "reason": "The user introduced a meaningful conversational change.",
    }


class AttentionStateTests(unittest.TestCase):
    def test_initial_context_and_continuation_stay_in_one_context(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_attention_key("user"))

        self.assertIsNone(
            server._apply_attention_result(
                session,
                "How does Rust allocate memory?",
                _classification("CONTINUE", title="Rust memory"),
                now=10.0,
            )
        )
        active_id = _attention(session).active_context_id
        self.assertIsNotNone(active_id)
        self.assertIsNone(
            server._apply_attention_result(
                session,
                "What about its allocator?",
                _classification("CLARIFICATION", title="Rust memory"),
                now=20.0,
            )
        )
        self.assertEqual(_attention(session).active_context_id, active_id)
        self.assertEqual(len(_attention(session).contexts), 1)

    def test_related_extension_updates_the_active_context_without_a_bridge(
        self,
    ) -> None:
        server = main.CodexAppServer()
        session = server._session(_attention_key("extension"))
        server._apply_attention_result(
            session,
            "Watchdog design",
            _classification("CONTINUE", title="Watchdog"),
            now=10.0,
        )
        self.assertIsNone(
            server._apply_attention_result(
                session,
                "How should pressure appear visually?",
                _classification(
                    "RELATED_EXTENSION",
                    title="Watchdog presentation",
                    summary="How watchdog pressure could be presented.",
                    loops=["Choose a pressure indicator."],
                ),
                now=20.0,
            )
        )
        attention = _attention(session)
        self.assertEqual(len(attention.contexts), 1)
        active = attention.contexts[attention.active_context_id]
        self.assertEqual(active.title, "Watchdog presentation")
        self.assertEqual(active.open_loops, ["Choose a pressure indicator."])

    def test_end_closes_topic_and_low_confidence_suppresses_acknowledgement(
        self,
    ) -> None:
        server = main.CodexAppServer()
        session = server._session(_attention_key("end"))
        server._apply_attention_result(
            session,
            "Watchdog design",
            _classification("CONTINUE", title="Watchdog"),
            now=10.0,
        )
        event = server._apply_attention_result(
            session,
            "That's all for this topic.",
            _classification("END", title="Watchdog"),
            now=20.0,
        )
        self.assertIsNone(_attention(session).active_context_id)
        closed = next(iter(_attention(session).contexts.values()))
        self.assertEqual(closed.status, "closed")
        assert event is not None
        self.assertFalse(event["acknowledge"])

        session = server._session(_attention_key("uncertain"))
        server._apply_attention_result(
            session,
            "Initial topic",
            _classification("CONTINUE", title="Initial"),
            now=10.0,
        )
        event = server._apply_attention_result(
            session,
            "Maybe another subject.",
            _classification("OFF_TOPIC", title="Other", confidence=0.2),
            now=20.0,
        )
        assert event is not None
        self.assertFalse(event["acknowledge"])

    def test_context_retention_remains_bounded(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_attention_key("bounded"))
        server._apply_attention_result(
            session,
            "Initial topic",
            _classification("CONTINUE", title="Topic 0"),
            now=0.0,
        )
        for index in range(20):
            server._apply_attention_result(
                session,
                f"Topic {index + 1}",
                _classification("TOPIC_SHIFT", title=f"Topic {index + 1}"),
                now=float(index + 1),
            )
        self.assertLessEqual(len(_attention(session).contexts), 12)

    def test_topic_shift_parks_previous_context_and_off_topic_threshold_is_bounded(
        self,
    ) -> None:
        server = main.CodexAppServer()
        session = server._session(_attention_key("threshold"))
        server._apply_attention_result(
            session,
            "How should the watchdog work?",
            _classification("CONTINUE", title="Memory watchdog"),
            now=10.0,
        )
        old_id = _attention(session).active_context_id

        temporary = server._apply_attention_result(
            session,
            "Why does Rust retain memory?",
            _classification("OFF_TOPIC", preserve=True),
            now=20.0,
        )
        self.assertEqual(len(_attention(session).contexts), 1)
        self.assertEqual(_attention(session).active_context_id, old_id)
        assert temporary is not None
        self.assertTrue(temporary["acknowledge"])
        self.assertIsNone(temporary["new_context_id"])

        long_question = "Rust memory behavior matters here. " * 20
        durable = server._apply_attention_result(
            session,
            long_question,
            _classification("OFF_TOPIC", preserve=True),
            now=30.0,
        )
        self.assertEqual(len(_attention(session).contexts), 2)
        self.assertEqual(_attention(session).contexts[old_id].status, "parked")
        assert durable is not None
        self.assertEqual(durable["previous_context_id"], old_id)
        self.assertTrue(durable["return_available"])

    def test_nested_return_restores_requested_context(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_attention_key("nested"))
        server._apply_attention_result(
            session,
            "Watchdog design",
            _classification("CONTINUE", title="Watchdog"),
            now=10.0,
        )
        watchdog_id = _attention(session).active_context_id
        server._apply_attention_result(
            session,
            "Now personality behavior",
            _classification("TOPIC_SHIFT", title="Personality"),
            now=20.0,
        )
        personality_id = _attention(session).active_context_id
        server._apply_attention_result(
            session,
            "A related visual question with an unresolved design detail.",
            _classification(
                "SIDETRACK",
                title="Visual status",
                preserve=True,
                loops=["Decide how pressure should appear."],
            ),
            now=30.0,
        )
        visual_id = _attention(session).active_context_id

        event = server._apply_attention_result(
            session,
            "Back to personality behavior.",
            _classification(
                "NESTED_RETURN", target=personality_id, title="Personality"
            ),
            now=40.0,
        )
        self.assertEqual(_attention(session).active_context_id, personality_id)
        self.assertEqual(_attention(session).contexts[visual_id].status, "parked")
        self.assertEqual(_attention(session).contexts[watchdog_id].status, "parked")
        assert event is not None
        self.assertEqual(event["new_context_id"], personality_id)

    def test_repeated_transition_is_not_acknowledged_again(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_attention_key("repeat"))
        server._apply_attention_result(
            session,
            "Watchdog design",
            _classification("CONTINUE", title="Watchdog"),
            now=10.0,
        )
        first = server._apply_attention_result(
            session,
            "A short unrelated question.",
            _classification("OFF_TOPIC", title="Rust", preserve=False),
            now=20.0,
        )
        second = server._apply_attention_result(
            session,
            "Another short unrelated question.",
            _classification("OFF_TOPIC", title="Rust", preserve=False),
            now=21.0,
        )
        assert first is not None
        assert second is not None
        self.assertTrue(first["acknowledge"])
        self.assertFalse(second["acknowledge"])

    def test_explicit_remember_preserves_even_a_short_topic(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_attention_key("remember"))
        server._apply_attention_result(
            session,
            "Current topic",
            _classification("CONTINUE", title="Current"),
            now=10.0,
        )
        server._apply_attention_result(
            session,
            "Remember this thread: Rust.",
            _classification("OFF_TOPIC", title="Rust", preserve=False),
            now=20.0,
        )
        self.assertEqual(len(_attention(session).contexts), 2)

    def test_contexts_are_isolated_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                "THEIA_HOME": str(Path(directory) / "theia"),
                "THEIA_STATE": str(Path(directory) / "state.json"),
            }
            with patch.dict(os.environ, environment):
                server = main.CodexAppServer()
                first = server._session("user-a")
                second = server._session("user-b")
                server._apply_attention_result(
                    first,
                    "First user's topic",
                    _classification("CONTINUE", title="User A"),
                    now=10.0,
                )
                server._apply_attention_result(
                    second,
                    "Second user's topic",
                    _classification("CONTINUE", title="User B"),
                    now=10.0,
                )
                self.assertNotEqual(
                    _attention(first).active_context_id,
                    _attention(second).active_context_id,
                )

            with patch.dict(os.environ, environment):
                restored = main.CodexAppServer()
                snapshot = restored.conversation_attention("user-a")
                self.assertEqual(
                    next(iter(snapshot["contexts"].values()))["title"], "User A"
                )


class AttentionPromptTests(unittest.TestCase):
    def test_attention_prompt_is_after_mood_and_before_user_input(self) -> None:
        server = main.CodexAppServer()
        event = {
            "type": "conversation_transition",
            "relation": "TOPIC_SHIFT",
            "previous_topic": "Memory watchdog",
            "new_topic": "Conversational attention",
            "acknowledge": True,
            "return_available": True,
        }
        prompt, _ = server._turn_prompt_with_summary(
            server._session("attention-prompt"),
            "the current user request",
            attention_transition=event,
        )
        self.assertLess(
            prompt.index("## Current mood"), prompt.index("[Conversational attention]")
        )
        self.assertLess(
            prompt.index("[Conversational attention]"),
            prompt.index("the current user request"),
        )
        self.assertNotIn("TOPIC_SHIFT", prompt)
        self.assertIn("then answer the new subject directly", prompt)


class AttentionWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_classifier_is_ephemeral_low_effort_and_receives_all_windows(
        self,
    ) -> None:
        server = main.CodexAppServer()
        requests: list[tuple[str, dict[str, Any]]] = []

        async def request(method: str, params: dict[str, Any], **_kwargs: Any) -> dict:
            requests.append((method, params))
            if method == "thread/start":
                return {"thread": {"id": "attention-thread"}}
            return {"turn": {"id": "attention-turn"}}

        server._ensure_running = AsyncMock()
        server._request = AsyncMock(side_effect=request)
        server._wait_for_turn = AsyncMock(
            return_value=(
                '{"relation":"TOPIC_SHIFT","confidence":0.9,'
                '"acknowledge":true,"preserve_context":true,'
                '"target_context_id":null,"topic_title":"New topic",'
                '"topic_summary":"A new subject.","open_loops":[],'
                '"reason":"The subject changed."}'
            )
        )
        result = await server.classify_attention(
            "new user message",
            session_key="user-a",
            attention={
                "active_context_id": "context-a",
                "contexts": {
                    "context-a": {
                        "context_id": "context-a",
                        "title": "Active topic",
                        "summary": "Active summary",
                        "recent_exchanges": ["User: Earlier exchange"],
                        "open_loops": ["An open question"],
                    },
                    "context-b": {
                        "context_id": "context-b",
                        "title": "Parked topic",
                        "summary": "Parked summary",
                        "open_loops": [],
                    },
                },
                "parked_context_ids": ["context-b"],
            },
            recent_global_context="Global one\nGlobal two",
        )
        assert result is not None
        self.assertEqual(result["relation"], "TOPIC_SHIFT")
        self.assertEqual(
            [method for method, _ in requests], ["thread/start", "turn/start"]
        )
        self.assertEqual(requests[0][1]["approvalPolicy"], "never")
        self.assertEqual(requests[0][1]["sandbox"], "read-only")
        self.assertTrue(requests[0][1]["ephemeral"])
        self.assertEqual(requests[0][1]["baseInstructions"], main.BASE_PRIORS)
        self.assertNotIn("dynamicTools", requests[0][1])
        self.assertEqual(requests[1][1]["effort"], "low")
        worker_prompt = requests[1][1]["input"][0]["text"]
        self.assertIn("Active summary", worker_prompt)
        self.assertIn("Parked summary", worker_prompt)
        self.assertIn("Global two", worker_prompt)
        self.assertNotIn("attention-thread", server._sessions)
