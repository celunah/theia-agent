# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable,duplicate-code
from tests.test_support import *


class AsyncBehaviorTests(AsyncBehaviorTestBase):
    async def test_personality_upload_selects_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attachment = SimpleNamespace(
                filename="calm.md",
                size=20,
                read=AsyncMock(return_value=b"Be calm and concise."),
            )
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(root / "state.json"),
                },
            ):
                server = main.CodexAppServer()
                selected = await server.configure_personality(
                    "session", name="calm", attachment=attachment
                )
                self.assertEqual(selected, "calm")
                self.assertEqual(server.active_personality("session"), "calm")
                self.assertEqual(server.personality_names(), ("calm",))

                restarted = main.CodexAppServer()
                self.assertEqual(restarted.active_personality("session"), "calm")
                self.assertEqual(
                    await restarted.configure_personality("session", name="none"),
                    None,
                )
                self.assertIsNone(restarted.active_personality("session"))

    async def test_personality_scopes_resolve_in_precedence_order_and_persist(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                "THEIA_HOME": str(root / "theia"),
                "THEIA_STATE": str(root / "state.json"),
            }
            with patch.dict(os.environ, environment):
                server = main.CodexAppServer()

                async def upload(name: str, text: bytes, user_id: int) -> None:
                    await server.configure_personality(
                        f"guild:42:channel:1:user:{user_id}",
                        name=name,
                        attachment=SimpleNamespace(
                            filename=f"{name}.md",
                            size=len(text),
                            read=AsyncMock(return_value=text),
                        ),
                        scope={
                            "global": "everyone",
                            "server": "server",
                            "me": "me",
                        }[name],
                        actor_user_id=user_id,
                        guild_id=42,
                    )

                await upload("global", b"Be global.", 1)
                await upload("server", b"Be server-wide.", 2)
                await upload("me", b"Be personal.", 9)

                self.assertEqual(
                    server.active_personality("guild:42:channel:5:user:9"), "me"
                )
                self.assertEqual(
                    server.active_personality("guild:42:channel:5:user:8"), "server"
                )
                self.assertEqual(
                    server.active_personality("guild:43:channel:5:user:8"), "global"
                )
                self.assertEqual(
                    server.personality_selection("guild:42:channel:5:user:9"),
                    {"scope": "me", "name": "me", "set_by": 9},
                )
                self.assertEqual(
                    server.personality_selection("guild:42:channel:5:user:8"),
                    {"scope": "server", "name": "server", "set_by": 2},
                )

                await server.configure_personality(
                    "guild:42:channel:5:user:9",
                    name="none",
                    scope="me",
                    actor_user_id=9,
                    guild_id=42,
                )
                self.assertEqual(
                    server.active_personality("guild:42:channel:5:user:9"), "server"
                )

                restarted = main.CodexAppServer()
                self.assertEqual(
                    restarted.active_personality("guild:42:channel:5:user:8"),
                    "server",
                )
                self.assertEqual(
                    restarted.active_personality("guild:43:channel:5:user:8"),
                    "global",
                )

    async def test_personality_server_scope_requires_a_server(self) -> None:
        server = main.CodexAppServer()
        with self.assertRaisesRegex(main.CodexAppServerError, "requires a server"):
            await server.configure_personality(
                "guild:0:channel:1:user:9",
                name="none",
                scope="server",
                actor_user_id=9,
            )

    async def test_personality_summary_uses_ephemeral_codex_and_memory_counts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory_root = root / "theia" / "memories"
            memory_root.mkdir(parents=True)
            (memory_root / "MEMORY.md").write_text(
                "- First memory about <@101>\n"
                "- Second memory about [Discord user id: 202]\n"
                "- Third memory about <@101>\n",
                encoding="utf-8",
            )
            (memory_root / "USER.md").write_text(
                "- The profile is also associated with <@202>.\n", encoding="utf-8"
            )
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
                    name="cel",
                    attachment=SimpleNamespace(
                        filename="cel.md",
                        size=40,
                        read=AsyncMock(
                            return_value=b"You are Celune, a calm guardian."
                        ),
                    ),
                )
                requests: list[tuple[str, dict[str, Any]]] = []

                async def request(
                    method: str, params: dict[str, Any], **_kwargs: Any
                ) -> dict[str, Any]:
                    requests.append((method, params))
                    if method == "thread/start":
                        return {"thread": {"id": "summary-thread"}}
                    return {"turn": {"id": "summary-turn"}}

                server._request = AsyncMock(side_effect=request)
                server._ensure_running = AsyncMock()
                server._wait_for_turn = AsyncMock(
                    return_value=(
                        '{"description":"Celune is a calm guardian who protects '
                        'others and responds with warm, observant precision."}'
                    )
                )
                summary = await server.personality_summary("session")

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(
            summary["description"],
            "Celune is a calm guardian who protects others and responds with warm, "
            "observant precision.",
        )
        self.assertEqual(summary["known_entries"], 4)
        self.assertEqual(summary["known_users"], 2)
        self.assertEqual(
            [method for method, _ in requests], ["thread/start", "turn/start"]
        )
        thread_params = requests[0][1]
        self.assertTrue(thread_params["ephemeral"])
        self.assertEqual(thread_params["approvalPolicy"], "never")
        self.assertEqual(thread_params["sandbox"], "read-only")
        self.assertEqual(thread_params["runtimeWorkspaceRoots"], [])
        self.assertEqual(thread_params["baseInstructions"], main.BASE_PRIORS)
        self.assertNotIn("dynamicTools", thread_params)
        turn_params = requests[1][1]
        self.assertEqual(turn_params["effort"], "low")
        self.assertIn(
            "You are Celune, a calm guardian.", turn_params["input"][0]["text"]
        )
        self.assertIn("outputSchema", turn_params)
        self.assertFalse(
            any(key.startswith("__personality_summary__") for key in server._sessions)
        )

    async def test_about_personality_recovers_one_active_profile_for_user_and_guild(
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
                attachment = SimpleNamespace(
                    filename="cel.md",
                    size=8,
                    read=AsyncMock(return_value=b"Be warm."),
                )
                await server.configure_personality(
                    "guild:42:channel:7:user:9",
                    name="Cel",
                    attachment=attachment,
                )

                self.assertEqual(
                    server.active_personality("guild:42:channel:8:user:9"),
                    "Cel",
                )
                inherited_session = server._session("guild:42:channel:8:user:9")
                instructions = server._system_instructions(
                    inherited_session, allow_tools=False
                )
                self.assertIn("Be warm.", instructions)
                self.assertIn(
                    "warm", server.mood_state(inherited_session.key)["traits"]
                )
                self.assertIsNotNone(
                    server._self_improvement_personality_path(inherited_session)
                )
                self.assertIsNone(
                    server.active_personality("guild:43:channel:8:user:9")
                )

                await server.configure_personality(
                    "guild:42:channel:8:user:9", name="none"
                )
                self.assertIsNone(
                    server.active_personality("guild:42:channel:8:user:9")
                )

    async def test_personality_requires_a_name_for_upload(self) -> None:
        server = main.CodexAppServer()
        attachment = SimpleNamespace(
            filename="calm.md",
            size=4,
            read=AsyncMock(return_value=b"Calm"),
        )
        with self.assertRaisesRegex(main.CodexAppServerError, "paired"):
            await server.configure_personality(
                "session", name=None, attachment=attachment
            )
        with self.assertRaisesRegex(main.CodexAppServerError, "cannot be used"):
            await server.configure_personality(
                "session", name="none", attachment=attachment
            )

    async def test_inherited_personality_reaches_ephemeral_workers(self) -> None:
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
                    "guild:42:channel:7:user:9",
                    name="Cel",
                    attachment=SimpleNamespace(
                        filename="cel.md",
                        size=8,
                        read=AsyncMock(return_value=b"Be warm."),
                    ),
                )
                target = "guild:42:channel:8:user:9"
                server._ensure_running = AsyncMock()
                server._request = AsyncMock(
                    side_effect=[
                        {"thread": {"id": "presence-thread"}},
                        {"turn": {"id": "presence-turn"}},
                        {"thread": {"id": "recap-thread"}},
                        {"turn": {"id": "recap-turn"}},
                    ]
                )
                server._wait_for_turn = AsyncMock(
                    side_effect=[
                        '{"activity_type":"none","text":"idle"}',
                        '{"recap":"The user reviewed the project."}',
                    ]
                )

                await server.generate_presence("generic", session_key=target)
                await server.generate_nightly_recap("journal", session_key=target)

                requests = server._request.await_args_list

        self.assertIn("Be warm.", requests[0].args[1]["baseInstructions"])
        self.assertIn("Be warm.", requests[2].args[1]["baseInstructions"])

    async def test_resting_mood_is_derived_from_the_active_personality(self) -> None:
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
                    name="cel",
                    attachment=SimpleNamespace(
                        filename="cel.md",
                        size=40,
                        read=AsyncMock(
                            return_value=b"Be warm, curious, and observant."
                        ),
                    ),
                )

                mood = server.mood_state("session")

        self.assertIn("warm", mood["traits"])
        self.assertIn("curious", mood["traits"])
        self.assertEqual(mood["label"], "neutral")
        self.assertEqual(mood["strength"], main.MOOD_BASELINE_STRENGTH)
        self.assertFalse(mood["transient"])
