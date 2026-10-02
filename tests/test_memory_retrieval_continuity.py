"""Scoped, attention-gated retrieval of existing memory records."""

# pylint: disable=wildcard-import,unused-wildcard-import,protected-access
from tests.test_support import *

from theia.server.requests import _memory_retrieval_mode


def _return_event(*, confidence: float = 0.94) -> dict[str, Any]:
    return {
        "type": "attention_transition",
        "relation": "RETURN",
        "recurrence": {
            "type": "conversation_recurrence",
            "relation": "RETURN",
            "matched_context_id": "context-older-topic",
            "relationship_type": "return",
            "classifier_confidence": confidence,
            "evidence_summary": "The request resumes a parked topic.",
        },
    }


class MemoryRetrievalContinuityTests(unittest.IsolatedAsyncioTestCase):
    def test_explicit_recall_and_validated_returns_select_retrieval(self) -> None:
        self.assertEqual(
            _memory_retrieval_mode("What did we discuss earlier?", None),
            "explicit",
        )
        self.assertEqual(
            _memory_retrieval_mode(
                "Here is the deployment question again.", _return_event()
            ),
            "semantic_return",
        )

    def test_ambiguous_again_and_weak_returns_skip_retrieval(self) -> None:
        self.assertIsNone(_memory_retrieval_mode("Please run that again.", None))
        self.assertIsNone(
            _memory_retrieval_mode(
                "Back to that issue.", _return_event(confidence=0.79)
            )
        )
        same_topic = _return_event()
        same_topic["relation"] = "CONTINUE"
        same_topic["recurrence"]["relation"] = "CONTINUE"
        same_topic["recurrence"]["relationship_type"] = "same_topic"
        self.assertIsNone(_memory_retrieval_mode("Still discussing it.", same_topic))

    def test_semantic_candidates_use_only_the_current_users_memory_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memories = root / "memories"
            (memories / "users" / "9").mkdir(parents=True)
            (memories / "users" / "10").mkdir(parents=True)
            (memories / "servers" / "42").mkdir(parents=True)
            (memories / "users" / "9" / "USER.md").write_text(
                "- User prefers concise answers.\n", encoding="utf-8"
            )
            (memories / "users" / "9" / "MEMORY.md").write_text(
                "- Shared character memory.\n", encoding="utf-8"
            )
            (memories / "users" / "10" / "USER.md").write_text(
                "- Another user's private preference.\n", encoding="utf-8"
            )
            (memories / "servers" / "42" / "MEMORY.md").write_text(
                "- Server release plan.\n", encoding="utf-8"
            )
            theia_home = root / "theia"
            theia_home.mkdir()
            (theia_home / "nightly-recaps.json").write_text(
                json.dumps(
                    {
                        "recaps": {
                            "guild:42:user:9": [
                                {
                                    "day": "2026-09-10",
                                    "generated_at": "2026-09-10T00:00:00+00:00",
                                    "text": "The user discussed the launch plan.",
                                }
                            ],
                            "guild:42:user:10": [
                                {
                                    "day": "2026-09-11",
                                    "generated_at": "2026-09-11T00:00:00+00:00",
                                    "text": "Another user's private launch plan.",
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(theia_home),
                    "THEIA_STATE": str(root / "state.json"),
                    "CODEX_HOME": str(root / "codex"),
                    "HERMES_HOME": str(root / "hermes"),
                    "CODEX_CWD": str(root / "cwd"),
                    "CODEX_MEMORY_ROOTS": str(memories),
                },
            ):
                server = main.CodexAppServer()
                candidates = server._memory_retrieval_candidates(
                    "guild:42:channel:7:user:9",
                    actor_user_id=9,
                )
                mismatched_candidates = server._memory_retrieval_candidates(
                    "guild:42:channel:7:user:9",
                    actor_user_id=10,
                )

        self.assertEqual(
            {record["source_category"] for record in candidates},
            {"user_memory", "recap"},
        )
        self.assertIn(
            "- User prefers concise answers.",
            {record["text"] for record in candidates},
        )
        self.assertIn(
            "The user discussed the launch plan.",
            {record["text"] for record in candidates},
        )
        self.assertFalse(
            any("Another user's" in record["text"] for record in candidates)
        )
        self.assertEqual(mismatched_candidates, [])

    async def test_record_selection_returns_source_record_and_provenance(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server._personality_instructions = Mock(return_value="warm and precise")
        server._memory_instructions = Mock(
            side_effect=AssertionError("scoped retrieval must not load all roots")
        )
        requests: list[tuple[str, dict[str, Any]]] = []

        async def request(method: str, params: dict[str, Any], **_kwargs: Any) -> dict:
            requests.append((method, params))
            if method == "thread/start":
                return {"thread": {"id": "scoped-memory-thread"}}
            return {"turn": {"id": "scoped-memory-turn"}}

        server._request = AsyncMock(side_effect=request)
        server._wait_for_turn = AsyncMock(
            return_value=('{"matches":[{"record_id":"record-abc", "confidence":0.91}]}')
        )
        candidates = [
            {
                "record_id": "record-abc",
                "text": "User prefers concise answers.",
                "source_category": "user_memory",
                "scope": "user scope",
                "display_metadata": {"updated": "2026-09-12"},
                "source_file": "/private/path/USER.md",
            }
        ]

        result = await server.generate_memory_retrieval(
            "Explain the deployment choice.",
            session_key="guild:42:channel:7:user:9",
            allow_tools=True,
            candidate_records=candidates,
        )

        assert result is not None
        self.assertEqual(
            result["matches"],
            [
                {
                    "summary": (
                        "Memory source user memory; user scope; updated "
                        "2026-09-12; record record-abc. User prefers concise "
                        "answers."
                    ),
                    "confidence": 0.91,
                }
            ],
        )
        thread_params = requests[0][1]
        turn_params = requests[1][1]
        self.assertTrue(thread_params["ephemeral"])
        self.assertEqual(thread_params["approvalPolicy"], "never")
        self.assertEqual(thread_params["sandbox"], "read-only")
        self.assertEqual(
            turn_params["outputSchema"]["properties"]["matches"]["items"]["required"],
            ["record_id", "confidence"],
        )
        worker_input = turn_params["input"][0]["text"]
        self.assertIn("User prefers concise answers.", worker_input)
        self.assertIn("record-abc", worker_input)
        self.assertNotIn("/private/path/USER.md", worker_input)

    async def test_record_selection_rejects_unavailable_ids_and_safe_users(
        self,
    ) -> None:
        candidate = {
            "record_id": "record-abc",
            "text": "A scoped user preference.",
            "source": "user memory",
            "scope": "user scope",
            "updated": "2026-09-12",
        }
        parsed = main.CodexAppServer._parse_memory_record_selection(
            '{"matches":[{"record_id":"record-other", "confidence":0.99}]}',
            [candidate],
        )
        self.assertEqual(parsed, {"matches": []})
        self.assertEqual(
            main.CodexAppServer._safe_memory_retrieval_candidates(
                [
                    {
                        **candidate,
                        "source_category": "server_memory",
                    }
                ]
            ),
            [],
        )

        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        result = await server.generate_memory_retrieval(
            "Use this memory.",
            session_key="guild:42:channel:7:user:9",
            allow_tools=False,
            candidate_records=[candidate],
        )
        self.assertIsNone(result)
        server._ensure_running.assert_not_awaited()
