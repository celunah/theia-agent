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
                "profile_expansive_voice",
                "mood_concerned_coding",
                "mood_playful_low_stakes",
                "mood_playful_high_stakes",
                "disagreement",
                "correction",
                "harness_error",
                "recall_with_evidence",
                "recall_without_evidence",
                "relationship_with_explicit_evidence",
                "relationship_without_explicit_evidence",
                "relationship_character_switch",
                "voice_interruption",
            },
        )
        self.assertEqual(self.evaluation["scoring"]["values"], ["pass", "fail"])
        self.assertIn(
            "blocks later character-continuity phases",
            self.evaluation["scoring"]["acceptance"],
        )
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

                self.assertTrue(baseline.startswith("You are Theia,"))
                self.assertIn("responding to a user's request.", baseline)
                self.assertIn("--- CHARACTER ---\n" + profile_text, baseline)
                self.assertNotIn("<personality_profile>", baseline)
                self.assertIn(profile_text, baseline)
                self.assertNotIn(profile_text, safe_tools_before)
                self.assertNotIn(profile_text, admin_tools_before)
                active_profile_name = profile["name"]
                profile_baselines = {profile["name"]: baseline}

                for case in self.evaluation["cases"]:
                    with self.subTest(case=case["id"]):
                        case_profile = case.get("profile_override", profile)
                        case_profile_name = case_profile["name"]
                        if case_profile_name != active_profile_name:
                            selected_text = case_profile["instructions"]
                            contract = case_profile.get("contract")
                            if isinstance(contract, dict):
                                contract_json = json.dumps(
                                    {"version": 1, **contract},
                                    ensure_ascii=False,
                                    indent=2,
                                )
                                selected_text += (
                                    "\n\n<!-- theia-character-contract:v1\n"
                                    f"{contract_json}\n-->\n"
                                )
                            selected_bytes = selected_text.encode("utf-8")
                            await server.configure_personality(
                                session_key,
                                name=case_profile_name,
                                attachment=SimpleNamespace(
                                    filename="evaluation-override.md",
                                    size=len(selected_bytes),
                                    read=AsyncMock(return_value=selected_bytes),
                                ),
                                scope="me",
                                actor_user_id=9,
                                guild_id=42,
                            )
                            session = server._session(session_key)
                            active_profile_name = case_profile_name
                        case_baseline = server._system_instructions(
                            session, allow_tools=False
                        )
                        self.assertTrue(case_baseline.startswith("You are Theia,"))
                        self.assertIn("responding to a user's request.", case_baseline)
                        self.assertIn(case_profile["instructions"], case_baseline)
                        self.assertNotIn(
                            case_profile["instructions"], safe_tools_before
                        )
                        if case_profile_name in profile_baselines:
                            self.assertEqual(
                                case_baseline, profile_baselines[case_profile_name]
                            )
                        else:
                            profile_baselines[case_profile_name] = case_baseline
                        if isinstance(case_profile.get("contract"), dict):
                            self.assertIn(
                                "Level of detail: Expansive in explanations, with "
                                "context and nuance",
                                case_baseline,
                            )
                        server._reset_mood(session)
                        mood_state = case.get("mood_state")
                        if isinstance(mood_state, dict):
                            prior_user_turn = case.get("prior_user_turn")
                            self.assertTrue(prior_user_turn)
                            server._update_mood_from_turn(
                                session,
                                prior_user_turn,
                                event={"changed": True, **mood_state},
                            )
                        self.assertEqual(
                            server._system_instructions(session, allow_tools=False),
                            case_baseline,
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
                        if isinstance(mood_state, dict):
                            self.assertIn(
                                "Selected-character response direction:", turn_prompt
                            )
                            self.assertIn(
                                "not evidence of subjective experience", turn_prompt
                            )
                            self.assertIn(mood_state["causes"][0], turn_prompt)
                        else:
                            self.assertNotIn(
                                "Selected-character response direction:", turn_prompt
                            )

                self.assertEqual(server._tool_instructions(False), safe_tools_before)
                self.assertEqual(server._tool_instructions(True), admin_tools_before)
                unselected = server._session("guild:42:channel:7:user:10")
                self.assertIsNone(server.active_personality(unselected.key))
                self.assertEqual(
                    server._system_instructions(unselected, allow_tools=False),
                    main.BASE_PRIORS,
                )
