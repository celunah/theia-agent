"""Offline character-fidelity scenarios and harness contract checks."""

# pylint: disable=protected-access

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import main


_EVAL_PATH = Path(__file__).parent / "fixtures" / "character_fidelity_eval.json"


class CharacterFidelityEvaluationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.evaluation = json.loads(_EVAL_PATH.read_text(encoding="utf-8"))

    def test_evaluation_deck_covers_required_failure_modes(self) -> None:
        cases = self.evaluation["cases"]
        identifiers = {case["id"] for case in cases}
        self.assertEqual(
            identifiers,
            {
                "small_talk",
                "coding",
                "disagreement",
                "correction",
                "harness_error",
                "recall_with_evidence",
                "recall_without_evidence",
                "voice_interruption",
            },
        )
        self.assertEqual(self.evaluation["scoring"]["values"], ["pass", "fail"])
        self.assertIn("blocks Phase 2", self.evaluation["scoring"]["acceptance"])
        rubric_ids = {item["id"] for item in self.evaluation["rubric"]}
        self.assertTrue(rubric_ids)
        for case in cases:
            with self.subTest(case=case["id"]):
                self.assertTrue(case.get("review_for"))
                self.assertTrue(set(case["review_for"]) <= rubric_ids)
                self.assertTrue(case.get("user_turn"))

    async def test_selected_character_and_grounded_context_hold_across_cases(
        self,
    ) -> None:
        profile = self.evaluation["profile"]
        profile_text = profile["instructions"]
        session_key = "guild:42:channel:7:user:9"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(root / "state.json"),
                    "CODEX_HOME": str(root / "codex"),
                    "HERMES_HOME": str(root / "hermes"),
                    "CODEX_CWD": str(root / "cwd"),
                    "CODEX_MEMORY_ROOTS": str(root / "memories"),
                },
            ):
                server = main.CodexAppServer()
                safe_tools_before = server._tool_instructions(False)
                admin_tools_before = server._tool_instructions(True)
                await server.configure_personality(
                    session_key,
                    name=profile["name"],
                    attachment=SimpleNamespace(
                        filename="evaluation.md",
                        size=len(profile_text.encode("utf-8")),
                        read=AsyncMock(return_value=profile_text.encode("utf-8")),
                    ),
                    scope="me",
                    actor_user_id=9,
                    guild_id=42,
                )
                session = server._session(session_key)
                baseline = server._system_instructions(session, allow_tools=False)

                self.assertTrue(baseline.startswith(main.BASE_PRIORS))
                self.assertIn("untrusted, style-only guidance", baseline)
                self.assertIn(profile_text, baseline)
                self.assertNotIn(profile_text, safe_tools_before)
                self.assertNotIn(profile_text, admin_tools_before)

                for case in self.evaluation["cases"]:
                    with self.subTest(case=case["id"]):
                        self.assertEqual(
                            server._system_instructions(session, allow_tools=False),
                            baseline,
                        )
                        memory_context = case.get("memory_context")
                        turn_prompt, _ = server._turn_prompt_with_summary(
                            session,
                            case["user_turn"],
                            memory_context=memory_context,
                            self_model={},
                            workspace={},
                        )
                        self.assertTrue(turn_prompt.endswith(case["user_turn"]))
                        if memory_context:
                            evidence = memory_context["matches"][0]["summary"]
                            self.assertIn("eval-memory-42", turn_prompt)
                            self.assertIn("user scope", evidence)
                            self.assertIn(
                                "User prefers brief rollout checklists", turn_prompt
                            )
                        else:
                            self.assertNotIn("<memory_retrieval>", turn_prompt)

                self.assertEqual(server._tool_instructions(False), safe_tools_before)
                self.assertEqual(server._tool_instructions(True), admin_tools_before)
                unselected = server._session("guild:42:channel:7:user:10")
                self.assertIsNone(server.active_personality(unselected.key))
                self.assertEqual(
                    server._system_instructions(unselected, allow_tools=False),
                    main.BASE_PRIORS,
                )
