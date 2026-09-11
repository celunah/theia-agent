# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable,duplicate-code
from tests.test_support import *


def _apply_mood_event(
    server: Any,
    session: Any,
    text: str,
    *,
    now: float,
    label: str = "concerned",
    traits: str = "careful and concerned",
    strength: float = 0.72,
    causes: list[str] | None = None,
) -> bool:
    return server._update_mood_from_turn(
        session,
        text,
        event={
            "changed": True,
            "label": label,
            "traits": traits,
            "strength": strength,
            "causes": causes or ["The user described a problem."],
        },
        now=now,
    )


class AsyncBehaviorTests(AsyncBehaviorTestBase):
    async def test_mood_is_after_personality_and_before_user_input(self) -> None:
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
                await server.configure_personality(
                    "session",
                    name="warm",
                    attachment=SimpleNamespace(
                        filename="warm.md",
                        size=14,
                        read=AsyncMock(return_value=b"Be warm."),
                    ),
                )
                session = server._session("session")
                turn_prompt, _ = server._turn_prompt_with_summary(
                    session, "the actual user request"
                )
                instructions = server._system_instructions(session)

        self.assertLess(
            instructions.index("<personality_profile>"),
            instructions.index("Be warm."),
        )
        self.assertLess(
            turn_prompt.index("## Current mood"),
            turn_prompt.index("the actual user request"),
        )
        self.assertIn(
            "This mood is temporary expressive context. Use it subtly.", turn_prompt
        )

    async def test_mood_classification_uses_ephemeral_low_effort_no_tool_turn(
        self,
    ) -> None:
        server = main.CodexAppServer()
        requests: list[tuple[str, dict[str, Any]]] = []

        async def request(method: str, params: dict[str, Any], **_kwargs: Any) -> dict:
            requests.append((method, params))
            if method == "thread/start":
                return {"thread": {"id": "mood-thread"}}
            return {"turn": {"id": "mood-turn"}}

        server._ensure_running = AsyncMock()
        server._request = AsyncMock(side_effect=request)
        server._wait_for_turn = AsyncMock(
            return_value=(
                '{"changed":true,"label":"playful",'
                '"traits":"quietly amused and attentive","strength":0.58,'
                '"causes":["The user made a playful observation."]}'
            )
        )

        result = await server.classify_mood(
            "That was unexpectedly funny.",
            session_key=_mood_test_key("classified"),
            recent_context="The conversation was relaxed.",
            current_mood={
                "label": "neutral",
                "traits": "calm and observant",
                "strength": 0.50,
                "causes": [],
            },
        )

        self.assertEqual(result["label"] if result else None, "playful")
        self.assertEqual(
            [method for method, _ in requests], ["thread/start", "turn/start"]
        )
        thread_params = requests[0][1]
        self.assertTrue(thread_params["ephemeral"])
        self.assertEqual(thread_params["approvalPolicy"], "never")
        self.assertEqual(thread_params["sandbox"], "read-only")
        self.assertEqual(thread_params["runtimeWorkspaceRoots"], [])
        self.assertNotIn("dynamicTools", thread_params)
        self.assertEqual(requests[1][1]["effort"], "low")
        self.assertIn("<current_user_turn>", requests[1][1]["input"][0]["text"])
        self.assertNotIn("classified", " ".join(server._sessions.keys()))

    def test_transient_mood_has_traits_label_strength_and_causes(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("transient"))

        self.assertTrue(
            _apply_mood_event(
                server, session, "The deployment failed with an error.", now=100.0
            )
        )
        mood = server.mood_state(_mood_test_key("transient"), now=100.0)

        self.assertEqual(mood["label"], "concerned")
        self.assertGreater(mood["strength"], 0.0)
        self.assertTrue(mood["traits"])
        self.assertEqual(len(mood["causes"]), 1)
        self.assertIn("problem", mood["causes"][0])

    def test_trivial_and_duplicate_turns_do_not_move_mood_again(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("duplicates"))

        self.assertFalse(_apply_mood_event(server, session, "okay", now=100.0))
        self.assertTrue(
            _apply_mood_event(
                server, session, "Please fix the broken build.", now=101.0
            )
        )
        before = server.mood_state(_mood_test_key("duplicates"), now=101.0)

        self.assertFalse(
            _apply_mood_event(
                server, session, "Please fix the broken build.", now=102.0
            )
        )
        self.assertEqual(
            server.mood_state(_mood_test_key("duplicates"), now=102.0)["label"],
            before["label"],
        )

    def test_mood_strength_decays_by_three_percentage_points_per_minute(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("decay"))
        _apply_mood_event(server, session, "Please fix the broken build.", now=100.0)
        initial = session.mood
        assert initial is not None
        first_strength = initial.strength

        after_one_minute = server.mood_state(_mood_test_key("decay"), now=160.0)

        self.assertAlmostEqual(
            after_one_minute["strength"],
            first_strength - main.MOOD_DECAY_PER_MINUTE,
            places=8,
        )

    def test_neutral_mood_does_not_decay(self) -> None:
        server = main.CodexAppServer()

        first = server.mood_state(_mood_test_key("neutral"), now=100.0)
        second = server.mood_state(_mood_test_key("neutral"), now=100_000.0)

        self.assertEqual(first, second)

    def test_zero_strength_restores_profile_baseline_and_stops_decay(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("zero"))
        _apply_mood_event(server, session, "The build is broken.", now=100.0)
        baseline = session.mood
        assert baseline is not None
        baseline_traits = baseline.baseline_traits

        restored = server.mood_state(_mood_test_key("zero"), now=100.0 + 60 * 100)
        later = server.mood_state(_mood_test_key("zero"), now=100.0 + 60 * 200)

        self.assertEqual(restored["traits"], baseline_traits)
        self.assertEqual(restored["label"], "neutral")
        self.assertEqual(restored["strength"], 0.50)
        self.assertFalse(restored["transient"])
        self.assertEqual(later, restored)

    def test_new_meaningful_event_reactivates_decay_after_neutral(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("react"))
        _apply_mood_event(server, session, "The build is broken.", now=100.0)
        server.mood_state(_mood_test_key("react"), now=100.0 + 60 * 100)

        self.assertTrue(
            _apply_mood_event(
                server,
                session,
                "Everything is fixed now.",
                label="relieved",
                traits="lighter and relieved",
                strength=0.55,
                causes=["The user reported that the problem eased."],
                now=7_000.0,
            )
        )
        mood = server.mood_state(_mood_test_key("react"), now=7_060.0)

        self.assertEqual(mood["label"], "relieved")
        self.assertTrue(mood["transient"])
        self.assertLess(mood["strength"], 0.55)

    def test_mood_strength_and_causes_are_clamped_and_bounded(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("clamp"))
        _apply_mood_event(
            server,
            session,
            "a meaningful event",
            label="focused",
            traits="steady and focused",
            strength=2.0,
            causes=["one", "two", "three", "four"],
            now=100.0,
        )

        mood = server.mood_state(_mood_test_key("clamp"), now=100.0)
        self.assertEqual(mood["strength"], 1.0)
        self.assertEqual(mood["causes"], ["one", "two", "three"])

    def test_zero_event_strength_returns_to_neutral(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("zero-event"))
        _apply_mood_event(
            server,
            session,
            "another meaningful event",
            label="focused",
            traits="steady and focused",
            strength=-1.0,
            causes=["not retained"],
            now=100.0,
        )

        mood = server.mood_state(_mood_test_key("zero-event"), now=100.0)
        self.assertEqual(mood["label"], "neutral")
        self.assertEqual(mood["strength"], 0.50)
        self.assertFalse(mood["transient"])

    def test_stale_mood_causes_are_replaced(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("stale"))
        _apply_mood_event(server, session, "The build is broken.", now=100.0)
        self.assertEqual(len(session.mood.causes if session.mood else ()), 1)

        _apply_mood_event(
            server,
            session,
            "Everything is fixed now.",
            label="relieved",
            traits="lighter and relieved",
            strength=0.55,
            causes=["The user reported that the problem eased."],
            now=101.0,
        )

        self.assertEqual(
            server.mood_state(_mood_test_key("stale"), now=101.0)["causes"],
            ["The user reported that the problem eased."],
        )

    def test_mood_does_not_create_durable_memory_or_skill_updates(self) -> None:
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
                _apply_mood_event(
                    server,
                    server._session(_mood_test_key("durable")),
                    "Please fix this issue.",
                    now=100.0,
                )

            self.assertFalse((root / "theia" / "memories").exists())
            self.assertFalse((root / "theia" / "skills").exists())

    def test_mood_is_isolated_by_session_user_guild_and_channel(self) -> None:
        server = main.CodexAppServer()
        source_key = _mood_test_key("guild:1:channel:2:user:3")
        other_channel = _mood_test_key("guild:1:channel:4:user:3")
        other_user = _mood_test_key("guild:1:channel:2:user:5")
        other_guild = _mood_test_key("guild:6:channel:2:user:3")

        _apply_mood_event(
            server,
            server._session(source_key),
            "The request failed.",
            now=100.0,
        )

        self.assertEqual(server.mood_state(source_key, now=100.0)["label"], "concerned")
        for key in (other_channel, other_user, other_guild):
            with self.subTest(key=key):
                self.assertEqual(server.mood_state(key, now=100.0)["label"], "neutral")

    def test_transient_mood_does_not_reset_codex_session(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("no-reset"))
        session.thread_id = "existing-thread"
        session.instruction_fingerprint = server._instruction_fingerprint(session)

        _apply_mood_event(server, session, "Please fix this issue.", now=100.0)

        self.assertEqual(session.thread_id, "existing-thread")
        self.assertEqual(
            session.instruction_fingerprint, server._instruction_fingerprint(session)
        )

    async def test_personality_change_resets_mood_to_the_new_profile_baseline(
        self,
    ) -> None:
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
                await server.configure_personality(
                    "session",
                    name="warm",
                    attachment=SimpleNamespace(
                        filename="warm.md",
                        size=12,
                        read=AsyncMock(return_value=b"Be warm."),
                    ),
                )
                session = server._session("session")
                _apply_mood_event(server, session, "The build failed.", now=100.0)
                session.thread_id = "old-thread"

                await server.configure_personality(
                    "session",
                    name="formal",
                    attachment=SimpleNamespace(
                        filename="formal.md",
                        size=14,
                        read=AsyncMock(return_value=b"Be formal and precise."),
                    ),
                )
                mood = server.mood_state("session", now=100.0)

        self.assertIsNone(session.thread_id)
        self.assertIn("composed", mood["traits"])
        self.assertEqual(mood["label"], "neutral")
        self.assertEqual(mood["strength"], 0.50)

    async def test_persisted_mood_restores_with_elapsed_decay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "state.json"
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(state_path),
                },
            ):
                server = main.CodexAppServer()
                _apply_mood_event(
                    server,
                    server._session("session"),
                    "The build failed.",
                    now=100.0,
                )
                with patch("theia.server.core.time.time", return_value=160.0):
                    restored = main.CodexAppServer().mood_state("session", now=160.0)

        self.assertAlmostEqual(restored["strength"], 0.69, places=8)
        self.assertEqual(restored["label"], "concerned")

    async def test_internal_presence_recap_and_self_improvement_do_not_change_mood(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = main.CodexAppServer()
            server._state_path = root / "state.json"
            server._codex_home = root
            session = server._session(_mood_test_key("internal"))
            _apply_mood_event(server, session, "The build failed.", now=100.0)
            before = server.mood_state(_mood_test_key("internal"), now=100.0)
            server._ensure_running = AsyncMock()
            server._request = AsyncMock(
                side_effect=[
                    {"thread": {"id": "presence-thread"}},
                    {"turn": {"id": "presence-turn"}},
                ]
            )
            server._wait_for_turn = AsyncMock(
                return_value='{"activity_type":"none","text":"idle"}'
            )
            await server.generate_presence(
                "generic", session_key=_mood_test_key("internal")
            )

            server._request = AsyncMock(
                side_effect=[
                    {"thread": {"id": "recap-thread"}},
                    {"turn": {"id": "recap-turn"}},
                ]
            )
            server._wait_for_turn = AsyncMock(return_value='{"recap":"nothing"}')
            await server.generate_nightly_recap(
                "journal", session_key=_mood_test_key("internal")
            )

            review_channel = _Channel()
            review_channel.guild = _admin_guild()
            server._request = AsyncMock(
                side_effect=[
                    {"thread": {"id": "review-thread"}},
                    {"turn": {"id": "review-turn"}},
                ]
            )
            server._wait_for_turn = AsyncMock(return_value='{"updates":[]}')
            await server._run_self_improvement_review(
                session,
                "review this",
                "reviewed",
                channel=cast(Any, review_channel),
                user_id=7,
                user=SimpleNamespace(
                    id=7,
                    guild_permissions=SimpleNamespace(administrator=True),
                ),
                allow_tools=True,
            )

            self.assertEqual(
                server.mood_state(_mood_test_key("internal"), now=100.0), before
            )

    def test_mood_does_not_change_tool_or_permission_policy(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("policy"))
        before = server._thread_instruction_params(session, allow_tools=False)
        _apply_mood_event(server, session, "Please inspect this issue.", now=100.0)

        after = server._thread_instruction_params(session, allow_tools=False)
        self.assertEqual(after, before)
        self.assertEqual(server._approval_policy(False), "never")

    async def test_personality_is_injected_after_base_priors(self) -> None:
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
                attachment = SimpleNamespace(
                    filename="friendly.txt",
                    size=22,
                    read=AsyncMock(return_value=b"Use a warm tone."),
                )
                await server.configure_personality(
                    "session", name="friendly", attachment=attachment
                )
                server._ensure_running = AsyncMock()
                server._request = AsyncMock(return_value={"thread": {"id": "thread"}})
                await server._ensure_thread(server._session("session"))

        params = cast(Any, server._request.await_args).args[1]
        self.assertTrue(params["baseInstructions"].startswith(main.BASE_PRIORS))
        self.assertIn("style-only guidance", params["baseInstructions"])
        self.assertIn(
            "<personality_profile>\nUse a warm tone.", params["baseInstructions"]
        )
        self.assertIn("source code", params["developerInstructions"])
        self.assertIn("server administrator", params["developerInstructions"])

    async def test_changing_personality_resets_the_session_thread(self) -> None:
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
                session = server._session("session")
                session.thread_id = "old-thread"
                session.loaded = True
                session.instruction_fingerprint = "old-fingerprint"
                attachment = SimpleNamespace(
                    filename="formal.md",
                    size=13,
                    read=AsyncMock(return_value=b"Be formal."),
                )

                await server.configure_personality(
                    "session", name="formal", attachment=attachment
                )

                self.assertIsNone(session.thread_id)
                self.assertFalse(session.loaded)
                self.assertIsNone(session.instruction_fingerprint)

    async def test_instruction_fingerprint_resets_stale_threads(self) -> None:
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
            server._ensure_running = AsyncMock()
            server._request = AsyncMock(return_value={"thread": {"id": "new-thread"}})
            session = server._session("session")
            session.thread_id = "old-thread"
            session.loaded = True
            session.instruction_fingerprint = "stale"

            await server._ensure_thread(session)

            self.assertEqual(session.thread_id, "new-thread")
            await_args = cast(Any, server._request.await_args)
            self.assertEqual(await_args.args[0], "thread/start")
            self.assertEqual(
                await_args.args[1]["baseInstructions"],
                main.BASE_PRIORS,
            )

    async def test_approval_is_bound_to_user_and_cleared(self) -> None:
        server = main.CodexAppServer()
        channel = _Channel()
        channel.guild = _admin_guild()
        state = main._TurnState(thread_id="thread", channel=channel, user_id=7)
        server._turns["turn"] = state
        request = asyncio.create_task(
            server._server_request_result(
                "item/commandExecution/requestApproval",
                {
                    "threadId": "thread",
                    "turnId": "turn",
                    "itemId": "item",
                    "command": "hidden command",
                },
            )
        )
        await asyncio.sleep(0)
        self.assertEqual(len(channel.sent), 1)
        self.assertEqual(channel.sent[0]["embed"].title, "Approval needed")
        self.assertNotIn("hidden command", channel.sent[0]["embed"].description)
        self.assertNotIn("content", channel.sent[0])
        self.assertFalse(server.resolve_approval(8, True, channel))
        self.assertTrue(server.resolve_approval(7, True, channel))
        self.assertEqual(await request, {"decision": "accept"})
        self.assertFalse(server.resolve_approval(7, False, channel))
        self.assertFalse(server._pending_approvals)

    async def test_account_install_approval_uses_interaction_sender(self) -> None:
        sender = AsyncMock(return_value=SimpleNamespace(id=1))
        with patch.dict(
            os.environ,
            {
                main.ALWAYS_ADMIN_USERS_ENV: "7",
                "THEIA_APPROVAL_LEVEL": "high",
            },
        ):
            server = main.CodexAppServer()
            channel = _Channel()
            channel.id = 123
            state = main._TurnState(
                thread_id="thread",
                channel=channel,
                user_id=7,
                interaction_sender=sender,
                allow_tools=True,
                allow_discord_tools=False,
            )
            server._turns["turn"] = state
            request = asyncio.create_task(
                server._server_request_result(
                    "item/commandExecution/requestApproval",
                    {
                        "threadId": "thread",
                        "turnId": "turn",
                        "itemId": "item",
                        "command": "git status",
                    },
                )
            )
            await asyncio.sleep(0)

            sender.assert_awaited_once()
            self.assertEqual(channel.sent, [])
            sender_args = cast(Any, sender.await_args)
            self.assertEqual(sender_args.kwargs["embed"].title, "Approval needed")
            self.assertTrue(server.resolve_approval(7, False, channel))
            self.assertEqual(await request, {"decision": "decline"})
