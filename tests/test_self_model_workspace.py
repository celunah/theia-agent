# pylint: disable=wildcard-import,unused-wildcard-import,protected-access
"""Focused tests for grounded self-model and session-global workspace state."""

from tests.test_support import *


WORKSPACE_TEST_NAMESPACE = f"workspace-test-{time.time_ns()}"


def _workspace_key(name: str) -> str:
    return f"{WORKSPACE_TEST_NAMESPACE}:{name}"


def _operations(*entries: tuple[str, str, str]) -> list[dict[str, str]]:
    return [
        {"op": "upsert", "key": key, "category": category, "text": text}
        for key, category, text in entries
    ]


class SelfModelTests(AsyncBehaviorTestBase):
    @staticmethod
    def _healthy_transport(server: Any) -> None:
        server._process = SimpleNamespace(returncode=None)
        server._reader_task = SimpleNamespace(done=lambda: False)

    def test_self_model_uses_current_harness_facts(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_workspace_key("self-model"))
        session.mode = "voice"
        session.thread_id = "thread"
        session.loaded = True
        server._model = "gpt-test"

        snapshot = server._self_model_snapshot(
            session,
            allow_tools=False,
            allow_discord_tools=True,
            phase="starting",
        )

        self.assertEqual(snapshot["model"], "gpt-test")
        self.assertEqual(snapshot["mode"], "voice")
        self.assertEqual(snapshot["effective_tool_policy"], "safe read-only")
        self.assertEqual(snapshot["approval_level"], "never")
        self.assertTrue(snapshot["thread_bound"])
        self.assertNotIn("secret", repr(snapshot).casefold())
        self.assertNotIn(session.key, repr(snapshot))

    def test_capability_status_uses_live_harness_state(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_workspace_key("capabilities"))
        self._healthy_transport(server)
        server._memory_roots = (Path("/configured/memory"),)
        server._realtime_feature_enabled = True
        session.background_review_count = 1

        status = server._self_model_snapshot(
            session,
            allow_tools=True,
            allow_discord_tools=True,
            memory_retrieval_used=True,
        )["capability_status"]

        self.assertEqual(status["session_workspace"], "available, currently empty")
        self.assertEqual(status["workspace_entry_count"], 0)
        self.assertEqual(status["durable_memory"], "available")
        self.assertEqual(status["memory_retrieval"], "available, used this turn")
        self.assertEqual(status["image_input"], "available")
        self.assertEqual(status["audio_input"], "available")
        self.assertEqual(status["semantic_audio_understanding"], "available")
        self.assertEqual(status["video_input"], "unavailable")
        self.assertEqual(status["stt_provider"], "not configured")
        self.assertEqual(status["tts_provider"], "not configured")
        self.assertEqual(status["active_voice_provider"], "codex-realtime")
        self.assertEqual(status["codex_transport"], "healthy")
        self.assertFalse(status["process_recovering"])
        self.assertEqual(status["protected_actions"], "available, requires approval")
        self.assertEqual(status["discord_tools"], "available")
        self.assertEqual(status["background_review"], "active")

    def test_configured_custom_audio_is_distinguished_from_semantic_audio(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_workspace_key("custom-audio"))
        self._healthy_transport(server)
        server._audio = SimpleNamespace(
            transcription=SimpleNamespace(
                enabled=True, base_url="https://stt.example/v1"
            ),
            tts=SimpleNamespace(enabled=True, base_url="https://tts.example/v1"),
        )
        server._realtime_feature_enabled = True

        status = server._self_model_snapshot(
            session,
            allow_tools=True,
            allow_discord_tools=False,
        )["capability_status"]

        self.assertEqual(status["stt_provider"], "configured")
        self.assertEqual(status["tts_provider"], "configured")
        self.assertEqual(status["active_voice_provider"], "custom")
        self.assertEqual(status["audio_input"], "available")
        self.assertEqual(status["semantic_audio_understanding"], "unavailable")
        self.assertEqual(status["discord_tools"], "unavailable")

    def test_unavailable_capabilities_are_not_claimed(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_workspace_key("unavailable"))
        server._process = None
        server._reader_task = None
        server._memory_roots = ()
        server._realtime_feature_enabled = True
        server._audio = SimpleNamespace(
            transcription=SimpleNamespace(enabled=False, base_url=""),
            tts=SimpleNamespace(enabled=False, base_url=""),
        )

        snapshot = server._self_model_snapshot(
            session,
            allow_tools=False,
            allow_discord_tools=True,
        )
        status = snapshot["capability_status"]

        self.assertEqual(status["session_workspace"], "available, currently empty")
        self.assertEqual(status["durable_memory"], "unavailable")
        self.assertEqual(status["memory_retrieval"], "unavailable")
        self.assertEqual(status["image_input"], "currently unavailable")
        self.assertEqual(status["audio_input"], "currently unavailable")
        self.assertEqual(
            status["semantic_audio_understanding"], "currently unavailable"
        )
        self.assertEqual(status["video_input"], "unavailable")
        self.assertEqual(status["codex_transport"], "unavailable")
        self.assertNotIn("realtime voice", snapshot["capabilities"])
        self.assertEqual(
            status["protected_actions"], "unavailable under current tool policy"
        )
        self.assertEqual(status["discord_tools"], "unavailable")

    def test_protected_actions_are_never_reported_as_automatically_allowed(
        self,
    ) -> None:
        server = main.CodexAppServer()
        session = server._session(_workspace_key("approval-status"))
        status = server._self_model_snapshot(
            session,
            allow_tools=True,
            allow_discord_tools=False,
        )

        self.assertTrue(status["approval_required_for_protected_actions"])
        self.assertEqual(status["protected_actions"], "available, requires approval")
        self.assertNotIn("automatically allowed", server._render_self_model(status))

    def test_self_model_failure_does_not_block_prompt_construction(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_workspace_key("self-model-failure"))
        with patch.object(
            server,
            "_self_model_snapshot",
            side_effect=RuntimeError("self-model failed"),
        ):
            prompt, _ = server._turn_prompt_with_summary(
                session, "The normal request still needs an answer."
            )

        self.assertIn("The normal request still needs an answer.", prompt)
        self.assertNotIn("## Theia self-model", prompt)

    def test_background_review_state_is_session_scoped(self) -> None:
        server = main.CodexAppServer()
        active = server._session(_workspace_key("review-active"))
        idle = server._session(_workspace_key("review-idle"))
        active.background_review_count = 1

        active_status = server._self_model_snapshot(
            active,
            allow_tools=False,
            allow_discord_tools=False,
        )["capability_status"]
        idle_status = server._self_model_snapshot(
            idle,
            allow_tools=False,
            allow_discord_tools=False,
        )["capability_status"]

        self.assertEqual(active_status["background_review"], "active")
        self.assertEqual(idle_status["background_review"], "inactive")

    async def test_personality_metadata_is_used_without_copying_its_prompt(
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
                        size=80,
                        read=AsyncMock(
                            return_value=b"You are Cel, a warm guide. Hidden lore: moonlit."
                        ),
                    ),
                )
                session = server._session("session")
                snapshot = server._self_model_snapshot(
                    session,
                    allow_tools=True,
                    allow_discord_tools=False,
                )

        self.assertEqual(snapshot["character_name"], "Cel")
        self.assertEqual(snapshot["character_identifier"], "warm")
        self.assertNotIn("moonlit", repr(snapshot))

    def test_rendered_self_model_is_temporary_and_bounded(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_workspace_key("render"))
        rendered = server._render_self_model(
            server._self_model_snapshot(
                session,
                allow_tools=True,
                allow_discord_tools=True,
            )
        )
        self.assertIn("## Theia self-model", rendered)
        self.assertIn("read-only, harness-grounded", rendered)
        self.assertNotIn(session.key, rendered)

    def test_dynamic_context_is_before_the_user_input(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_workspace_key("prompt"))
        server._apply_workspace_delta(
            session,
            _operations(("goal", "goal", "Keep the workspace bounded")),
            base_generation=1,
            base_revision=0,
            now=time.time(),
        )
        prompt, _ = server._turn_prompt_with_summary(session, "The actual user request")
        self.assertLess(
            prompt.index("## Theia self-model"),
            prompt.index("## Session global workspace"),
        )
        self.assertLess(
            prompt.index("Capability status:"),
            prompt.index("## Session global workspace"),
        )
        self.assertLess(
            prompt.index("## Session global workspace"),
            prompt.index("The actual user request"),
        )


class WorkspaceStateTests(AsyncBehaviorTestBase):
    def test_workspace_isolated_and_does_not_reset_codex_thread(self) -> None:
        server = main.CodexAppServer()
        first = server._session(_workspace_key("first"))
        second = server._session(_workspace_key("second"))
        first.thread_id = "thread-one"

        self.assertTrue(
            server._apply_workspace_delta(
                first,
                _operations(("goal", "goal", "Finish the design")),
                base_generation=1,
                base_revision=0,
                now=time.time(),
            )
        )
        self.assertEqual(first.thread_id, "thread-one")
        self.assertEqual(len(server.session_workspace(first.key)["entries"]), 1)
        self.assertEqual(server.session_workspace(second.key)["entries"], [])

    def test_workspace_rejects_stale_results_and_bounds_notes(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_workspace_key("bounds"))
        entries = tuple(
            (f"note-{index}", "context_note", f"Note {index}") for index in range(40)
        )
        self.assertTrue(
            server._apply_workspace_delta(
                session,
                _operations(*entries),
                base_generation=1,
                base_revision=0,
                now=time.time(),
            )
        )
        snapshot = server.session_workspace(session.key)
        self.assertLessEqual(len(snapshot["entries"]), 24)
        self.assertFalse(
            server._apply_workspace_delta(
                session,
                _operations(("stale", "goal", "Must not apply")),
                base_generation=1,
                base_revision=0,
                now=time.time() + 1,
            )
        )
        self.assertNotIn(
            "stale",
            {
                entry["key"]
                for entry in server.session_workspace(session.key)["entries"]
            },
        )

    def test_workspace_parser_rejects_secrets_and_malformed_data(self) -> None:
        server = main.CodexAppServer()
        self.assertIsNone(server._parse_workspace_delta("not json"))
        self.assertIsNone(
            server._parse_workspace_delta(
                '{"operations":[{"op":"upsert","key":"token",'
                '"category":"goal","text":"secret=abc"}]}'
            )
        )
        self.assertEqual(server._parse_workspace_delta('{"operations":[]}'), [])

    async def test_workspace_review_is_ephemeral_low_effort_and_no_tool(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        requests: list[tuple[str, dict[str, Any]]] = []

        async def request(method: str, params: dict[str, Any], **_kwargs: Any) -> dict:
            requests.append((method, params))
            if method == "thread/start":
                return {"thread": {"id": "workspace-thread"}}
            return {"turn": {"id": "workspace-turn"}}

        server._request = AsyncMock(side_effect=request)
        server._wait_for_turn = AsyncMock(
            return_value=(
                '{"operations":[{"op":"upsert","key":"current_goal",'
                '"category":"goal","text":"Continue the design"}]}'
            )
        )
        session = server._session(_workspace_key("review"))
        session.lock = asyncio.Lock()
        server._schedule_workspace_review(
            session,
            "Continue the implementation.",
            "The design is ready for the next step.",
            recent_context="A bounded prior exchange.",
            self_model={"agent": "Theia", "mode": "text"},
        )
        task = session.workspace_review_task
        self.assertIsNotNone(task)
        assert task is not None
        await task

        self.assertEqual(
            [method for method, _ in requests], ["thread/start", "turn/start"]
        )
        thread_params = requests[0][1]
        self.assertTrue(thread_params["ephemeral"])
        self.assertEqual(thread_params["approvalPolicy"], "never")
        self.assertEqual(thread_params["sandbox"], "read-only")
        self.assertEqual(thread_params["runtimeWorkspaceRoots"], [])
        self.assertEqual(requests[1][1]["effort"], "low")
        self.assertNotIn("workspace-review", " ".join(server._sessions))
        self.assertEqual(
            server.session_workspace(session.key)["entries"][0]["key"],
            "current_goal",
        )

    async def test_workspace_review_failure_leaves_state_unchanged(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server._request = AsyncMock(side_effect=RuntimeError("worker failed"))
        session = server._session(_workspace_key("failure"))
        session.lock = asyncio.Lock()
        server._schedule_workspace_review(
            session,
            "A request",
            "A response",
            recent_context=None,
            self_model={"agent": "Theia"},
        )
        task = session.workspace_review_task
        assert task is not None
        await task
        self.assertEqual(server.session_workspace(session.key)["entries"], [])

    async def test_workspace_persists_and_restores_as_session_state(self) -> None:
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
                session = server._session("persistent")
                server._apply_workspace_delta(
                    session,
                    _operations(("decision", "decision", "Use session state")),
                    base_generation=1,
                    base_revision=0,
                    now=time.time(),
                )
                restored_server = main.CodexAppServer()
                restored = restored_server.session_workspace("persistent")

        self.assertEqual(restored["entries"][0]["key"], "decision")
        self.assertEqual(restored["entries"][0]["text"], "Use session state")

    def test_reset_clears_workspace_without_touching_personality_files(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_workspace_key("reset"))
        server._apply_workspace_delta(
            session,
            _operations(("goal", "goal", "Temporary goal")),
            base_generation=1,
            base_revision=0,
            now=time.time(),
        )
        assert session.workspace is not None
        generation = session.workspace.generation
        server._reset_session_thread(session)
        self.assertEqual(server.session_workspace(session.key)["entries"], [])
        assert session.workspace is not None
        self.assertEqual(session.workspace.generation, generation + 1)
