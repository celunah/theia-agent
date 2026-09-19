# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable,duplicate-code
from tests.test_support import *


class AsyncBehaviorTests(AsyncBehaviorTestBase):
    def test_self_improvement_treats_skills_as_a_first_class_update(self) -> None:
        instructions = main.CodexAppServer._self_improvement_developer_instructions(
            Path("/tmp/memories"),
            Path("/tmp/skills"),
            None,
        )
        prompt = main.CodexAppServer._self_improvement_prompt(
            "Use the release checklist.",
            "The release checklist is reusable for future deployments.",
        )

        self.assertIn("skills as a first-class outcome", instructions)
        self.assertIn("repeatable workflow, procedure, tool-use pattern", instructions)
        self.assertIn("create a new skill when no existing skill fits", instructions)
        self.assertIn(
            "memory, user-profile, skill, and personality updates separately", prompt
        )
        self.assertIn("update a matching skill or create a new one", prompt)

    async def test_self_improvement_can_create_private_memories_and_skills(
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
                memory_root = root / "theia" / "memories"
                skill_root = root / "theia" / "skills"
                updates = (
                    {
                        "kind": "memory",
                        "path": "MEMORY.md",
                        "content": "- The user prefers concise release notes.",
                    },
                    {
                        "kind": "user_profile",
                        "path": "USER.md",
                        "content": "- User works on Discord-native agents.",
                    },
                    {
                        "kind": "skill",
                        "path": "release-notes/SKILL.md",
                        "content": "# Release notes\nPrefer a short checklist.",
                    },
                )

                applied = server._apply_self_improvement_updates(
                    updates,
                    memory_root=memory_root,
                    skill_root=skill_root,
                    personality_path=None,
                )

            self.assertEqual(applied, 3)
            self.assertEqual(
                (memory_root / "MEMORY.md").read_text(encoding="utf-8"),
                updates[0]["content"] + "\n",
            )
            self.assertEqual(
                (memory_root / "USER.md").read_text(encoding="utf-8"),
                updates[1]["content"] + "\n",
            )
            self.assertEqual(
                (skill_root / "release-notes" / "SKILL.md").read_text(encoding="utf-8"),
                updates[2]["content"] + "\n",
            )

    async def test_self_improvement_rejects_skill_traversal_and_unsafe_names(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = main.CodexAppServer()
            memory_root = root / "memories"
            skill_root = root / "skills"
            outside = root / "outside"
            updates = (
                {
                    "kind": "skill",
                    "path": "../outside/SKILL.md",
                    "content": "must not escape",
                },
                {
                    "kind": "skill",
                    "path": "new skill/SKILL.md",
                    "content": "must use a safe name",
                },
                {
                    "kind": "skill",
                    "path": "nested/child/SKILL.md",
                    "content": "must stay direct-child",
                },
            )

            applied = server._apply_self_improvement_updates(
                updates,
                memory_root=memory_root,
                skill_root=skill_root,
                personality_path=None,
            )

            self.assertEqual(applied, 0)
            self.assertFalse(outside.exists())
            self.assertFalse(skill_root.exists())

    async def test_self_improvement_review_applies_structured_updates_for_admin_turn(
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
                personality_root = root / "theia" / "personalities"
                personality_root.mkdir(parents=True)
                personality_path = personality_root / "Cel.md"
                personality_path.write_text("Be warm.", encoding="utf-8")
                session = server._session("session")
                session.personality_name = "Cel"
                server._request = AsyncMock(
                    side_effect=(
                        {"thread": {"id": "review-thread"}},
                        {"turn": {"id": "review-turn"}},
                    )
                )
                server._wait_for_turn = AsyncMock(
                    return_value=json.dumps(
                        {
                            "updates": [
                                {
                                    "kind": "memory",
                                    "path": "MEMORY.md",
                                    "content": "- Keep release summaries concise.",
                                },
                                {
                                    "kind": "user_profile",
                                    "path": "USER.md",
                                    "content": "- User maintains Theia.",
                                },
                                {
                                    "kind": "skill",
                                    "path": "review/SKILL.md",
                                    "content": "# Review\nUse evidence.",
                                },
                                {
                                    "kind": "personality",
                                    "path": "active",
                                    "content": "Keep the tone warm and direct.",
                                },
                            ]
                        }
                    )
                )
                channel = _Channel()
                channel.guild = _admin_guild()

                applied = await server._run_self_improvement_review(
                    session,
                    "Please review the release.",
                    "The release is ready.",
                    channel=cast(Any, channel),
                    user_id=7,
                    user=SimpleNamespace(
                        id=7,
                        guild_permissions=SimpleNamespace(administrator=True),
                    ),
                    allow_tools=True,
                )

            self.assertEqual(applied, 4)
            self.assertIsNotNone(session.pending_self_improvement_summary)
            assert session.pending_self_improvement_summary is not None
            self.assertIn("Memory created", session.pending_self_improvement_summary)
            self.assertIn("Skill created", session.pending_self_improvement_summary)
            self.assertIn(
                "Personality updated", session.pending_self_improvement_summary
            )
            self.assertIn(
                "Keep release summaries concise.",
                session.pending_self_improvement_summary,
            )
            thread_params = cast(Any, server._request.await_args_list[0]).args[1]
            self.assertEqual(thread_params["sandbox"], "read-only")
            self.assertEqual(thread_params["approvalPolicy"], "never")
            self.assertTrue(thread_params["ephemeral"])
            self.assertNotIn(server._cwd, thread_params["runtimeWorkspaceRoots"])
            turn_params = cast(Any, server._request.await_args_list[1]).args[1]
            self.assertIn("outputSchema", turn_params)
            self.assertEqual(
                [message["content"] for message in channel.sent],
                [
                    "-# Memory created",
                    "-# Skill created",
                    "-# Personality updated",
                ],
            )
            self.assertIn(
                "Keep release summaries concise.",
                (root / "theia" / "memories" / "MEMORY.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "# Review",
                (root / "theia" / "skills" / "review" / "SKILL.md").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertIn(
                "warm and direct", personality_path.read_text(encoding="utf-8")
            )
            self.assertFalse(any(key.startswith("__") for key in server._sessions))
            persisted = json.loads((root / "state.json").read_text(encoding="utf-8"))
            self.assertFalse(any(key.startswith("__") for key in persisted["sessions"]))

    async def test_self_improvement_no_change_is_recorded_in_session_context(
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
                session = server._session("session")
                server._request = AsyncMock(
                    side_effect=(
                        {"thread": {"id": "review-thread"}},
                        {"turn": {"id": "review-turn"}},
                    )
                )
                server._wait_for_turn = AsyncMock(return_value='{"updates": []}')
                channel = _Channel()
                channel.guild = _admin_guild()

                applied = await server._run_self_improvement_review(
                    session,
                    "Request",
                    "Response",
                    channel=cast(Any, channel),
                    user_id=7,
                    user=None,
                    allow_tools=True,
                )

                self.assertEqual(applied, 0)
                self.assertEqual(
                    session.pending_self_improvement_summary,
                    "Self-improvement review completed. No durable updates were applied.",
                )

    async def test_self_improvement_summary_survives_restart(self) -> None:
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
                summary = "Self-improvement review completed. No durable updates were applied."
                server._session("session").pending_self_improvement_summary = summary
                server._persist_state()

                restarted = main.CodexAppServer()

            self.assertEqual(
                restarted._session("session").pending_self_improvement_summary,
                "Self-improvement review completed. No durable updates were applied.",
            )

    async def test_self_improvement_summary_is_injected_once_into_next_turn(
        self,
    ) -> None:
        server = main.CodexAppServer()
        server.account = {"type": "chatgpt"}
        server.requires_openai_auth = True
        server._ensure_running = AsyncMock()
        server.refresh_account = AsyncMock()
        server._select_reasoning_effort = AsyncMock(return_value="low")
        server._ensure_thread = AsyncMock()
        server._request = AsyncMock(return_value={"turn": {"id": "turn"}})
        server._wait_for_turn = AsyncMock(return_value="done")
        session = server._session("session")
        session.pending_self_improvement_summary = (
            "Self-improvement review completed. Applied durable updates:\n"
            "- Memory created: User prefers concise release notes."
        )

        result = await server.ask(
            "What changed?",
            session_key="session",
            channel=None,
            user_id=7,
            allow_tools=False,
        )

        self.assertEqual(result, "done")
        turn_params = cast(Any, server._request.await_args).args[1]
        turn_text = turn_params["input"][0]["text"]
        self.assertIn("Memory created: User prefers concise release notes.", turn_text)
        self.assertIn("What changed?", turn_text)
        self.assertLess(
            turn_text.index("Memory created"), turn_text.index("What changed?")
        )
        self.assertIsNone(session.pending_self_improvement_summary)

    async def test_self_improvement_summary_survives_failed_turn_start(self) -> None:
        server = main.CodexAppServer()
        server.account = {"type": "chatgpt"}
        server.requires_openai_auth = True
        server._ensure_running = AsyncMock()
        server.refresh_account = AsyncMock()
        server._select_reasoning_effort = AsyncMock(return_value="low")
        server._ensure_thread = AsyncMock()
        server._request = AsyncMock(
            side_effect=main.CodexAppServerError("turn unavailable")
        )
        session = server._session("session")
        session.pending_self_improvement_summary = "review summary"

        with self.assertRaises(main.CodexAppServerError):
            await server.ask(
                "What changed?",
                session_key="session",
                channel=None,
                user_id=7,
                allow_tools=False,
            )

        self.assertEqual(session.pending_self_improvement_summary, "review summary")

    async def test_self_improvement_failure_never_fails_the_completed_turn(
        self,
    ) -> None:
        server = main.CodexAppServer()
        server._request = AsyncMock(
            side_effect=main.CodexAppServerError("review unavailable")
        )
        channel = _Channel()
        channel.guild = _admin_guild()

        result = await server._run_self_improvement_review(
            server._session("session"),
            "request",
            "response",
            channel=cast(Any, channel),
            user_id=7,
            user=None,
            allow_tools=True,
        )

        self.assertEqual(result, 0)
        self.assertEqual(channel.sent, [])

    async def test_changing_model_resets_thread_before_next_request(self) -> None:
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
            server.available_models = AsyncMock(
                return_value=(
                    {"id": main.DEFAULT_CODEX_MODEL},
                    {"id": "gpt-test"},
                )
            )
            session = server._session("model-change")
            session.thread_id = "old-thread"
            session.loaded = True
            session.tool_policy = True
            session.instruction_fingerprint = server._instruction_fingerprint(session)

            await server.set_model("gpt-test")

            server._ensure_running = AsyncMock()
            server._request = AsyncMock(return_value={"thread": {"id": "new-thread"}})
            await server._ensure_thread(session)

        self.assertEqual(server.model_name(), "gpt-test")
        self.assertEqual(session.thread_id, "new-thread")
        request = cast(Any, server._request.await_args)
        self.assertEqual(request.args[0], "thread/start")
        self.assertEqual(request.args[1]["model"], "gpt-test")
