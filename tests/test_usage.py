# pylint: disable=wildcard-import,unused-wildcard-import
from tests.test_support import *

from theia.server.policy import _TOKEN_USAGE_KEYS
from theia.server.usage import (
    PRICING_REGISTRY,
    estimate_api_cost,
    estimate_api_value,
)


class UsageAccountingTests(unittest.TestCase):
    def _server(self) -> main.CodexAppServer:
        server = main.CodexAppServer()
        server._persist_state = lambda: None
        server._usage_threads.clear()
        server._usage_daily.clear()
        server._usage_daily_breakdown.clear()
        server._usage_turns.clear()
        server._usage_internal_turns.clear()
        server._usage_failed_daily.clear()
        server._usage_retries_daily.clear()
        server._usage_api_calls_daily.clear()
        server._usage_subagent_turns_daily.clear()
        server._usage_long_running_turns_daily.clear()
        server._usage_api_calls = 0
        server._usage_subagent_turns = 0
        return server

    def test_cache_miss_hit_output_and_model_effort_are_preserved(self) -> None:
        server = self._server()
        session = server._session("usage-user")
        session.thread_id = "thread-luna"
        server._claim_usage_thread("thread-luna")
        server._turns["turn-luna"] = cast(
            Any,
            SimpleNamespace(
                thread_id="thread-luna",
                session=session,
                model="gpt-5.6-luna",
                effort="high",
                prompt_attribution={"system_instructions": 12, "memory_data": 4},
            ),
        )
        server._handle_notification(
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "threadId": "thread-luna",
                    "turnId": "turn-luna",
                    "tokenUsage": {
                        "last": {
                            "inputTokens": 20,
                            "cachedInputTokens": 5,
                            "outputTokens": 8,
                            "reasoningOutputTokens": 2,
                            "totalTokens": 35,
                        },
                        "total": {
                            "inputTokens": 20,
                            "cachedInputTokens": 5,
                            "outputTokens": 8,
                            "reasoningOutputTokens": 2,
                            "totalTokens": 35,
                        },
                    },
                },
            }
        )

        usage = server.theia_usage()
        self.assertEqual(usage["exact"]["inputTokens"], 20)
        self.assertEqual(usage["exact"]["cachedInputTokens"], 5)
        self.assertEqual(usage["exact"]["outputTokens"], 8)
        self.assertEqual(usage["exact"]["totalTokens"], 35)
        self.assertEqual(usage["exact"]["totalProcessedTokens"], 33)
        self.assertEqual(
            usage["detailed"]["categories"]["system_instructions"],
            {"value": 12, "estimated": True},
        )
        self.assertTrue(usage["estimate"]["available"])
        expected = estimate_api_value(
            [
                {
                    "model": "gpt-5.6-luna",
                    "effort": "high",
                    "tokens": {
                        "inputTokens": 20,
                        "cachedInputTokens": 5,
                        "outputTokens": 8,
                        "reasoningOutputTokens": 2,
                    },
                }
            ]
        )
        self.assertEqual(usage["estimate"], expected)
        self.assertEqual(usage["estimate"]["currency"], "USD")

    def test_cost_is_unavailable_when_a_model_has_no_configured_pricing(self) -> None:
        estimate = estimate_api_cost(
            [{"model": "future-model", "tokens": {"outputTokens": 100}}]
        )
        self.assertFalse(estimate["available"])
        self.assertIsNone(estimate["total"])
        self.assertEqual(estimate["unavailableModels"], ["Future Model"])

    def test_cost_is_unavailable_when_only_provider_total_is_reported(self) -> None:
        estimate = estimate_api_cost(
            [{"model": "gpt-5.6-luna", "tokens": {"totalTokens": 100}}]
        )
        self.assertFalse(estimate["available"])
        self.assertIsNone(estimate["total"])
        self.assertEqual(estimate["incompleteRecords"], ["GPT-5.6 Luna"])

    def test_pricing_registry_has_canonical_model_specific_rates(self) -> None:
        self.assertEqual(
            set(PRICING_REGISTRY),
            {"gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra"},
        )
        self.assertEqual(
            (
                PRICING_REGISTRY["gpt-5.6-luna"].input_miss,
                PRICING_REGISTRY["gpt-5.6-luna"].input_hit,
                PRICING_REGISTRY["gpt-5.6-luna"].output,
            ),
            (0.20, 0.02, 1.20),
        )
        self.assertEqual(
            (
                PRICING_REGISTRY["gpt-6-astra"].input_miss,
                PRICING_REGISTRY["gpt-6-astra"].input_hit,
                PRICING_REGISTRY["gpt-6-astra"].output,
            ),
            (10.00, 1.00, 50.00),
        )
        self.assertEqual(
            PRICING_REGISTRY["gpt-5.6-sol"].reasoning_token_pricing, "output_rate"
        )

    def test_fast_mode_uses_the_centralized_multiplier(self) -> None:
        normal = estimate_api_cost(
            [{"model": "gpt-5.6-luna", "tokens": {"outputTokens": 100}}]
        )
        fast = estimate_api_cost(
            [
                {
                    "model": "gpt-5.6-luna",
                    "fastMode": True,
                    "tokens": {"outputTokens": 100},
                }
            ]
        )
        self.assertAlmostEqual(fast["total"], normal["total"] * 1.5)

    def test_main_and_subagent_tokens_and_cost_are_combined(self) -> None:
        server = self._server()
        day = time.strftime("%Y-%m-%d", time.gmtime())
        server._usage_threads["main-thread"] = {
            "inputTokens": 1000,
            "cachedInputTokens": 0,
            "outputTokens": 5,
            "totalTokens": 1005,
        }
        server._usage_thread_fields["main-thread"] = {
            "inputTokens",
            "cachedInputTokens",
            "outputTokens",
            "totalTokens",
        }
        server._usage_daily[day] = 1005
        server._usage_daily_breakdown[day] = dict(server._usage_threads["main-thread"])
        server._usage_turns["main-thread:turn"] = {
            "day": day,
            "recorded_at": time.time(),
            "model": "gpt-5.6-luna",
            "effort": "medium",
            "tokens": {
                "inputTokens": 1000,
                "cachedInputTokens": 0,
                "outputTokens": 5,
                "totalTokens": 1005,
            },
        }
        server._usage_internal_turns["worker:turn"] = {
            "day": day,
            "recorded_at": time.time(),
            "model": "gpt-5.6-luna",
            "effort": "medium",
            "tokens": {
                "inputTokens": 10,
                "cachedInputTokens": 2,
                "outputTokens": 3,
                "totalTokens": 15,
            },
        }
        usage = server.theia_usage()
        self.assertEqual(usage["exact"]["mainAgentTokens"], 1005)
        self.assertEqual(usage["exact"]["subagentTokens"], 15)
        self.assertEqual(usage["exact"]["totalProcessedTokens"], 1020)
        self.assertAlmostEqual(usage["estimate"]["total"], 0.00021164)

    def test_internal_turn_metadata_drives_subagent_usage_and_pricing(self) -> None:
        server = self._server()
        session = main._Session(key="__attention__:test")
        server._turns["worker-turn"] = cast(
            Any,
            SimpleNamespace(
                session=session,
                model="gpt-5.6-terra",
                effort="low",
            ),
        )
        server._record_usage_api_call(session)
        server._handle_notification(
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "threadId": "worker-thread",
                    "turnId": "worker-turn",
                    "tokenUsage": {
                        "last": {
                            "inputTokens": 10,
                            "cachedInputTokens": 2,
                            "outputTokens": 3,
                            "totalTokens": 15,
                        }
                    },
                },
            }
        )
        usage = server.theia_usage()
        self.assertEqual(usage["detailed"]["subagentTurns"], 1)
        self.assertEqual(usage["exact"]["subagentTokens"], 15)
        self.assertIn("GPT-5.6 Terra", usage["estimate"]["byModel"])
        self.assertEqual(usage["estimate"]["unavailableModels"], [])

    def test_activity_metrics_are_separate_from_token_counts(self) -> None:
        server = self._server()
        session = server._session("activity-user")
        session.thread_id = "activity-thread"
        server._record_usage_api_call(session)
        server._record_usage_retry()
        server._record_failed_usage_turn(SimpleNamespace(session=session), {})
        server._record_usage_turn_duration(61.0, session, "missing-turn")
        usage = server.theia_usage()
        detailed = usage["detailed"]
        self.assertEqual(detailed["apiCalls"], 1)
        self.assertEqual(detailed["retries"], 1)
        self.assertEqual(detailed["failedTurns"], 1)
        self.assertEqual(detailed["longRunningTurns"], 1)
        self.assertNotIn("retries", usage["exact"])

    def test_multiple_models_and_daily_cumulative_statistics(self) -> None:
        server = self._server()
        for index, model in enumerate(("gpt-5.6-luna", "gpt-5.6-terra")):
            thread_id = f"thread-{index}"
            turn_id = f"turn-{index}"
            session = server._session(f"user-{index}")
            session.thread_id = thread_id
            server._claim_usage_thread(thread_id)
            server._turns[turn_id] = cast(
                Any,
                SimpleNamespace(
                    thread_id=thread_id,
                    session=session,
                    model=model,
                    effort="low" if index == 0 else "max",
                ),
            )
            total = 10 + index
            server._handle_notification(
                {
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "threadId": thread_id,
                        "turnId": turn_id,
                        "tokenUsage": {
                            "last": {"inputTokens": total, "totalTokens": total},
                            "total": {"inputTokens": total, "totalTokens": total},
                        },
                    },
                }
            )
        usage = server.theia_usage()
        self.assertEqual(usage["summary"]["totalCumulativeTokens"], 21)
        self.assertEqual(usage["summary"]["peakDailyTokens"], 21)
        self.assertEqual(
            set(usage["estimate"]["byModel"]), {"GPT-5.6 Luna", "GPT-5.6 Terra"}
        )

    def test_older_partial_records_remain_readable_without_false_zero_values(
        self,
    ) -> None:
        server = self._server()
        day = time.strftime("%Y-%m-%d", time.gmtime())
        server._usage_threads["old-thread"] = {key: 0 for key in _TOKEN_USAGE_KEYS}
        server._usage_threads["old-thread"]["totalTokens"] = 17
        server._usage_daily[day] = 17
        usage = server.theia_usage()
        self.assertIsNone(usage["exact"]["inputTokens"])
        self.assertIsNone(usage["exact"]["outputTokens"])
        self.assertEqual(usage["exact"]["totalTokens"], 17)

    def test_streaks_honor_date_boundaries(self) -> None:
        server = self._server()
        now = datetime(2026, 9, 14, tzinfo=timezone.utc).timestamp()
        server._usage_daily.update(
            {
                "2026-09-12": 2,
                "2026-09-13": 3,
                "2026-09-14": 4,
                "2026-09-10": 1,
            }
        )
        usage = server.theia_usage(now=now)
        self.assertEqual(usage["summary"]["currentStreakDays"], 3)
        self.assertEqual(usage["summary"]["longestStreakDays"], 3)

    def test_prompt_categories_are_recorded_without_changing_prompt_text(self) -> None:
        server = self._server()
        session = server._session("prompt-user")
        attribution: dict[str, int] = {}
        prompt, _ = server._turn_prompt_with_summary(
            session,
            "the actual request",
            prompt_attribution=attribution,
        )
        self.assertTrue(prompt.endswith("the actual request"))
        self.assertGreater(attribution["system_instructions"], 0)
        self.assertGreater(attribution["user_history"], 0)
        self.assertEqual(attribution.get("tool_results"), None)

    def test_state_restores_new_usage_records(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(Path(directory) / "theia"),
                    "THEIA_STATE": str(Path(directory) / "state.json"),
                },
            ),
        ):
            server = main.CodexAppServer()
            session = server._session("restore-user")
            session.thread_id = "restore-thread"
            server._claim_usage_thread("restore-thread")
            server._turns["restore-turn"] = cast(
                Any,
                SimpleNamespace(
                    thread_id="restore-thread",
                    session=session,
                    model="gpt-5.6-sol",
                    effort="medium",
                ),
            )
            server._handle_notification(
                {
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "threadId": "restore-thread",
                        "turnId": "restore-turn",
                        "tokenUsage": {
                            "last": {"outputTokens": 4, "totalTokens": 4},
                            "total": {"outputTokens": 4, "totalTokens": 4},
                        },
                    },
                }
            )
            restored = main.CodexAppServer()
            usage = restored.theia_usage()
        self.assertEqual(usage["summary"]["totalCumulativeTokens"], 4)
        self.assertEqual(usage["exact"]["outputTokens"], 4)


class UsageViewTests(unittest.TestCase):
    def test_details_embed_has_all_requested_categories(self) -> None:
        result = {
            "date": "2026-09-14",
            "summary": {"lifetimeTokens": 30},
            "exact": {"totalTokens": 30},
            "estimate": {"total": 0.1},
            "detailed": {
                "categories": {"system_instructions": {"value": 3, "estimated": True}},
                "reasoningTokens": 2,
                "retries": 1,
                "failedTurns": 1,
                "unattributedOverhead": {"value": 4, "estimated": True},
            },
        }
        embed = main._usage_details_embed(result)
        self.assertEqual(
            {field.name for field in embed.fields},
            {
                "System instructions",
                "Identity / self-model",
                "Memory data",
                "Skill data",
                "User history",
                "Tool definitions",
                "Tool results",
                "Routing context",
                "Subagent usage",
                "Reasoning tokens",
                "Retries",
                "Failed turns",
                "API calls",
                "Subagent turns",
                "Long-running turns",
                "Unattributed overhead",
            },
        )
        self.assertIn("~3", str(embed.fields[0].value))

    def test_usage_embed_uses_fields_and_processed_token_semantics(self) -> None:
        embed = main._usage_embed(
            {
                "date": "2026-09-14",
                "summary": {
                    "totalCumulativeTokens": 500,
                    "peakDailyTokens": 100,
                    "currentStreakDays": 1,
                    "longestStreakDays": 2,
                    "longestRunningTurnSec": 12.75,
                },
                "exact": {
                    "inputTokens": 20,
                    "cachedInputTokens": 5,
                    "outputTokens": 8,
                    "totalProcessedTokens": 33,
                },
                "estimate": {"available": True, "total": 0.0034, "byModel": {}},
            }
        )
        self.assertEqual(embed.description, "Usage statistics for 2026-09-14")
        fields = {field.name: field.value for field in embed.fields}
        self.assertEqual(fields["Main-agent tokens"], "33")
        self.assertEqual(fields["Subagent tokens"], "0")
        self.assertEqual(fields["Total processed tokens"], "33")
        self.assertEqual(fields["Estimated API cost"], "$0.0034 USD")
        self.assertEqual(fields["Longest running turn"], "13 seconds")
        self.assertNotIn("credits", str(embed.to_dict()).casefold())

    def test_usage_embed_falls_back_to_provider_cumulative_total(self) -> None:
        embed = main._usage_embed(
            {
                "summary": {
                    "totalCumulativeTokens": 42,
                    "totalCumulativeProcessedTokens": None,
                    "longestRunningTurnSec": 1,
                }
            }
        )
        fields = {field.name: field.value for field in embed.fields}
        self.assertEqual(fields["Total cumulative tokens"], "42")

    def test_details_show_model_specific_pricing_and_combined_cost(self) -> None:
        embed = main._usage_details_embed(
            {
                "date": "2026-09-14",
                "estimate": {
                    "available": True,
                    "total": 0.30,
                    "byModel": {
                        "GPT-5.6 Luna": 0.10,
                        "GPT-5.6 Terra": 0.20,
                    },
                },
                "detailed": {"categories": {}},
            }
        )
        pricing_field = next(
            field
            for field in embed.fields
            if field.name == "Model-specific API pricing"
        )
        pricing_text = (
            pricing_field.value if isinstance(pricing_field.value, str) else ""
        )
        self.assertIn("GPT-5.6 Luna: $0.10", pricing_text)
        self.assertIn("GPT-5.6 Terra: $0.20", pricing_text)
        self.assertIn("GPT-6 Astra: $0.00", pricing_text)
        self.assertIn("Combined: $0.30", pricing_text)

    def test_state_restores_usage_activity_counters(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(Path(directory) / "theia"),
                    "THEIA_STATE": str(Path(directory) / "state.json"),
                },
            ),
        ):
            server = main.CodexAppServer()
            session = server._session("restore-activity")
            server._record_usage_api_call(session)
            restored = main.CodexAppServer()
            day = time.strftime("%Y-%m-%d", time.gmtime())
            self.assertEqual(restored._usage_api_calls, 1)
            self.assertEqual(restored._usage_api_calls_daily[day], 1)

    def test_usage_view_edits_one_ephemeral_message_and_locks_owner(self) -> None:
        result = {
            "date": "2026-09-14",
            "summary": {"lifetimeTokens": 30},
            "exact": {"totalTokens": 30},
            "estimate": {"total": 0.1},
            "detailed": {"categories": {}},
        }
        view = main._UsageView(result, owner_id=7)
        self.assertEqual(view.toggle_button.label, "Show Details")
        denied_response = SimpleNamespace(send_message=AsyncMock())
        denied = SimpleNamespace(user=SimpleNamespace(id=8), response=denied_response)
        self.assertFalse(
            asyncio.run(view.interaction_check(cast(discord.Interaction, denied)))
        )
        denied_response.send_message.assert_awaited_once()

        response = SimpleNamespace(edit_message=AsyncMock())
        allowed = SimpleNamespace(user=SimpleNamespace(id=7), response=response)
        asyncio.run(view._toggle(cast(discord.Interaction, allowed)))
        response.edit_message.assert_awaited_once()
        self.assertEqual(view.toggle_button.label, "Hide Details")
