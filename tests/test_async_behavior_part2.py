# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable,duplicate-code
from tests.test_support import *


class AsyncBehaviorTests(AsyncBehaviorTestBase):
    async def test_adaptive_reasoning_maps_assessment_to_supported_effort(self) -> None:
        server = main.CodexAppServer()
        models = (
            {
                "id": "test-model",
                "isDefault": True,
                "supportedReasoningEfforts": [
                    {"reasoningEffort": effort}
                    for effort in ("low", "medium", "high", "xhigh")
                ],
            },
        )
        server.available_models = AsyncMock(return_value=models)

        for assessment, expected in (
            ({"complexity": "simple", "requires_tool": False}, "low"),
            ({"complexity": "moderate", "requires_tool": True}, "medium"),
            ({"complexity": "complex", "requires_tool": True}, "high"),
            ({"complexity": "very_complex", "requires_tool": True}, "xhigh"),
        ):
            with self.subTest(assessment=assessment):
                server._assess_request = AsyncMock(return_value=assessment)
                self.assertEqual(
                    await server._select_reasoning_effort("task", ()), expected
                )

        server.available_models.return_value = (
            {
                "id": "test-model",
                "isDefault": True,
                "supportedReasoningEfforts": [
                    {"reasoningEffort": effort}
                    for effort in ("medium", "high", "xhigh", "max")
                ],
            },
        )
        server._assess_request = AsyncMock(
            return_value={"complexity": "very_complex", "requires_tool": True}
        )
        self.assertEqual(await server._select_reasoning_effort("task", ()), "max")

    async def test_assessment_failure_falls_back_to_medium(self) -> None:
        server = main.CodexAppServer()
        server.available_models = AsyncMock(return_value=())
        server._assess_request = AsyncMock(return_value=None)

        self.assertEqual(await server._select_reasoning_effort("task", ()), "medium")

    async def test_protocol_error_preserves_nested_codex_details(self) -> None:
        server = main.CodexAppServer()
        server._send = AsyncMock()
        request = asyncio.create_task(server._request("turn/start", {}))
        await asyncio.sleep(0)
        request_id = next(iter(server._pending))
        server._pending[request_id].set_result(
            {
                "id": request_id,
                "error": {
                    "message": "Request failed",
                    "data": {"statusCode": 404, "statusText": "Not Found"},
                },
            }
        )

        with self.assertRaisesRegex(main.CodexAppServerError, "404 Not Found"):
            await request

    async def test_non_adaptive_request_skips_assessment(self) -> None:
        with patch.dict(os.environ, {"CODEX_ADAPTIVE_REASONING": "false"}):
            server = main.CodexAppServer()
        server._assess_request = AsyncMock()

        self.assertEqual(await server._select_reasoning_effort("task", ()), "medium")
        server._assess_request.assert_not_awaited()

    async def test_assessment_is_ephemeral_and_hidden_from_user_turn(self) -> None:
        server = main.CodexAppServer()
        server._request = AsyncMock(
            side_effect=[
                {"thread": {"id": "assessment-thread"}},
                {"turn": {"id": "assessment-turn"}},
            ]
        )
        server._wait_for_turn = AsyncMock(
            return_value='{"complexity":"complex","requires_tool":true}'
        )

        result = await server._assess_request("inspect the project", (), effort="low")

        self.assertEqual(result, {"complexity": "complex", "requires_tool": True})
        thread_params = server._request.await_args_list[0].args[1]
        turn_params = server._request.await_args_list[1].args[1]
        self.assertTrue(thread_params["ephemeral"])
        self.assertEqual(thread_params["approvalPolicy"], "never")
        self.assertEqual(turn_params["effort"], "low")
        self.assertIn("outputSchema", turn_params)

    async def test_memory_retrieval_is_ephemeral_bounded_and_read_only(self) -> None:
        server = main.CodexAppServer()
        source = server._session("guild:42:channel:7:user:9")
        source.personality_name = "cel"
        server._memory_instructions = cast(
            Any, Mock(return_value="private memory about the release")
        )
        server._personality_instructions = cast(
            Any, Mock(return_value="warm and precise")
        )
        requests: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

        async def request(method: str, params: dict[str, Any], **kwargs: Any) -> dict:
            requests.append((method, params, kwargs))
            if method == "thread/start":
                return {"thread": {"id": "memory-thread"}}
            return {"turn": {"id": "memory-turn"}}

        server._request = AsyncMock(side_effect=request)
        server._ensure_running = AsyncMock()
        server._wait_for_turn = AsyncMock(
            return_value='{"matches":[{"summary":"The release was planned.","confidence":0.8}]}'
        )

        result = await server.generate_memory_retrieval(
            "What did we discuss earlier?",
            session_key=source.key,
            allow_tools=True,
        )

        self.assertEqual(
            result,
            {"matches": [{"summary": "The release was planned.", "confidence": 0.8}]},
        )
        self.assertEqual(
            [method for method, _, _ in requests], ["thread/start", "turn/start"]
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
        self.assertIn("outputSchema", turn_params)
        worker_input = turn_params["input"][0]["text"]
        self.assertIn("private memory about the release", worker_input)
        self.assertIn("warm and precise", worker_input)
        self.assertNotIn("memory-thread", server._sessions)

    async def test_memory_retrieval_does_not_cross_the_safe_tool_boundary(self) -> None:
        server = main.CodexAppServer()
        server._memory_instructions = cast(Any, Mock())

        result = await server.generate_memory_retrieval(
            "What did we discuss earlier?",
            session_key="session",
        )

        self.assertIsNone(result)
        cast(Any, server._memory_instructions).assert_not_called()

    async def test_thread_history_methods_use_codex_pagination(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server._request = AsyncMock(side_effect=[{"data": []}, {"data": []}])

        await server.list_thread_turns(
            "thread",
            cursor="older",
            limit=50,
            sort_direction="asc",
            items_view="full",
        )
        await server.list_thread_items(
            "thread",
            cursor="items",
            limit=25,
            sort_direction="asc",
            turn_id="turn",
        )

        self.assertEqual(
            server._request.await_args_list[0].args,
            (
                "thread/turns/list",
                {
                    "threadId": "thread",
                    "limit": 50,
                    "sortDirection": "asc",
                    "itemsView": "full",
                    "cursor": "older",
                },
            ),
        )
        self.assertEqual(
            server._request.await_args_list[1].args,
            (
                "thread/items/list",
                {
                    "threadId": "thread",
                    "limit": 25,
                    "sortDirection": "asc",
                    "cursor": "items",
                    "turnId": "turn",
                },
            ),
        )

    async def test_loaded_threads_reconcile_persisted_sessions(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server._request = AsyncMock(
            return_value={"data": ["loaded-thread", "another-thread"]}
        )
        session = server._session("discord-user")
        session.thread_id = "loaded-thread"
        session.loaded = False

        result = await server.loaded_threads()

        self.assertEqual(result["data"], ["loaded-thread", "another-thread"])
        self.assertEqual(server._loaded_thread_ids, {"loaded-thread", "another-thread"})
        self.assertTrue(session.loaded)
        self.assertEqual(
            server._request.await_args,
            (("thread/loaded/list", {}),),
        )

    async def test_provider_capabilities_are_cached_until_forced(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server._request = AsyncMock(
            return_value={"namespaceTools": True, "webSearch": True}
        )

        first = await server.provider_capabilities(
            model="gpt-test", model_provider="openai"
        )
        second = await server.provider_capabilities(
            model="gpt-test", model_provider="openai"
        )

        self.assertEqual(first, second)
        server._request.assert_awaited_once_with(
            "modelProvider/capabilities/read",
            {"model": "gpt-test", "modelProvider": "openai"},
        )

    async def test_thread_management_methods_update_local_state(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server._persist_state = lambda: None
        server._request = AsyncMock(
            side_effect=[
                {"thread": {"id": "thread"}},
                {"thread": {"id": "thread"}},
                {},
                {},
            ]
        )
        session = server._session("discord-user")
        session.thread_id = "thread"
        session.loaded = True
        server._loaded_thread_ids.add("thread")

        await server.set_thread_name("thread", "A useful name")
        await server.rollback_thread("thread", num_turns=2)
        await server.unarchive_thread("thread")
        await server.delete_thread("thread")

        self.assertEqual(
            server._request.await_args_list,
            [
                (("thread/name/set", {"threadId": "thread", "name": "A useful name"}),),
                (("thread/rollback", {"threadId": "thread", "numTurns": 2}),),
                (("thread/unarchive", {"threadId": "thread"}),),
                (("thread/delete", {"threadId": "thread"}),),
            ],
        )
        self.assertIsNone(session.thread_id)
        self.assertFalse(session.loaded)
        self.assertNotIn("thread", server._loaded_thread_ids)

    async def test_thread_rollback_and_delete_reject_active_turns(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server._request = AsyncMock()
        session = server._session("discord-user")
        session.thread_id = "thread"
        session.turn_id = "turn"

        with self.assertRaisesRegex(main.CodexAppServerError, "active Codex turn"):
            await server.rollback_thread("thread")
        with self.assertRaisesRegex(main.CodexAppServerError, "active Codex turn"):
            await server.delete_thread("thread")
        server._request.assert_not_awaited()

    async def test_retention_archives_sessions_after_thirty_days(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server._persist_state = lambda: None
        server._request = AsyncMock(return_value={})
        now = 1_000_000.0
        session = server._session("discord-user")
        session.thread_id = "thread"
        session.loaded = True
        session.last_activity_at = now - main.SESSION_ARCHIVE_AFTER - 1

        result = await server.enforce_retention(now=now)

        self.assertEqual(result, {"archived": 1, "deleted": 0})
        self.assertTrue(session.archived)
        self.assertFalse(session.loaded)
        server._request.assert_awaited_once_with(
            "thread/archive", {"threadId": "thread"}
        )

    async def test_retention_deletes_sessions_after_ninety_days(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server._persist_state = lambda: None
        server._request = AsyncMock(return_value={})
        now = 1_000_000.0
        session = server._session("discord-user")
        session.thread_id = "thread"
        session.loaded = False
        session.archived = True
        session.last_activity_at = now - main.SESSION_DELETE_AFTER - 1

        result = await server.enforce_retention(now=now)

        self.assertEqual(result, {"archived": 0, "deleted": 1})
        self.assertIsNone(session.thread_id)
        self.assertFalse(session.archived)
        server._request.assert_awaited_once_with(
            "thread/delete", {"threadId": "thread"}
        )

    async def test_retention_state_survives_restart(self) -> None:
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
                session = server._session("discord-user")
                session.thread_id = "thread"
                session.archived = True
                session.last_activity_at = 123.0
                server._persist_state()

                restarted = main.CodexAppServer()
                restored = restarted._session("discord-user")

        self.assertEqual(restored.thread_id, "thread")
        self.assertTrue(restored.archived)
        self.assertEqual(restored.last_activity_at, 123.0)

    def test_state_load_discards_temporary_and_unthreaded_mood_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "sessions": {
                            "real": {"thread_id": "real-thread"},
                            "__self_improvement__:old": {
                                "thread_id": "temporary-thread",
                                "last_activity_at": 10.0,
                            },
                            "mood-only": {
                                "mood": {
                                    "baseline_traits": "steady, attentive",
                                    "baseline_cause": "Resting affect.",
                                    "traits": "steady, attentive",
                                    "label": "neutral",
                                    "strength": 0.5,
                                    "causes": ["Resting affect."],
                                    "transient": False,
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(state_path),
                },
            ):
                server = main.CodexAppServer()

            self.assertIn("real", server._sessions)
            self.assertNotIn("__self_improvement__:old", server._sessions)
            self.assertNotIn("mood-only", server._sessions)
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(tuple(persisted["sessions"]), ("real",))

    async def test_retention_prunes_unthreaded_transient_sessions(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        session = server._session("mood-only")
        session.mood = server._new_mood_state(session)

        result = await server.enforce_retention()

        self.assertEqual(result, {"archived": 0, "deleted": 0})
        self.assertNotIn("mood-only", server._sessions)

    async def test_activity_unarchives_a_thirty_day_session(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server._persist_state = lambda: None
        server._request = AsyncMock(return_value={})
        now = 1_000_000.0
        session = server._session("discord-user")
        session.thread_id = "thread"
        session.archived = True
        session.last_activity_at = now - main.SESSION_ARCHIVE_AFTER - 1

        await server._prepare_session_for_activity(session, now=now)

        self.assertFalse(session.archived)
        self.assertFalse(session.loaded)
        self.assertEqual(session.last_activity_at, now)
        server._request.assert_awaited_once_with(
            "thread/unarchive", {"threadId": "thread"}
        )

    async def test_undo_rolls_back_one_completed_turn(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server._persist_state = lambda: None
        server._request = AsyncMock(return_value={})
        session = server._session("discord-user")
        session.thread_id = "thread"
        session.loaded = True
        session.last_activity_at = time.time()
        session.instruction_fingerprint = server._instruction_fingerprint(session)
        session.tool_policy = True

        await server.undo("discord-user")

        self.assertEqual(session.thread_id, "thread")
        server._request.assert_awaited_once_with(
            "thread/rollback", {"threadId": "thread", "numTurns": 1}
        )

    async def test_non_admin_thread_uses_read_only_tool_policy(self) -> None:
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
                server._ensure_running = AsyncMock()
                server._request = AsyncMock(return_value={"thread": {"id": "thread"}})
                await server._ensure_thread(
                    server._session("session"), allow_tools=False
                )

        params = cast(Any, server._request.await_args).args[1]
        self.assertEqual(params["approvalPolicy"], "never")
        self.assertEqual(params["sandbox"], "read-only")
        self.assertIn("non-administrator", params["developerInstructions"])
        self.assertEqual(
            params["runtimeWorkspaceRoots"],
            [str(Path(directory) / "theia" / "attachments")],
        )
        self.assertEqual(
            params["cwd"],
            str(Path(directory) / "theia" / "attachments"),
        )
        self.assertNotIn(
            server._cwd,
            params["runtimeWorkspaceRoots"],
        )
        self.assertNotIn(
            "persistent memory",
            server._thread_instruction_params(
                server._session("session"), allow_tools=False
            )["baseInstructions"],
        )

    async def test_voice_transcript_rechecks_current_administrator_access(self) -> None:
        member = SimpleNamespace(
            guild_permissions=SimpleNamespace(administrator=False),
        )
        guild = SimpleNamespace(id=42, get_member=lambda _user_id: member)
        channel = SimpleNamespace(
            id=7,
            guild=guild,
            send=AsyncMock(),
        )
        session = main.VoiceSession(
            session_key="voice",
            user_id=9,
            guild_id=42,
            voice_channel_id=8,
            text_channel=cast(Any, channel),
            allow_tools=True,
            on_transcript=AsyncMock(),
        )
        with (
            patch.object(main.bot.presence, "touch", new=AsyncMock()),
            patch.object(main.bot.codex, "status", return_value={"turn_id": None}),
            patch(
                "theia.bot.core._channel_context",
                new=AsyncMock(return_value=None),
            ),
            patch("theia.bot.core.handle_request", new=AsyncMock()) as handle,
        ):
            await _handle_voice_transcript(session, "speaker", "hello")
            await asyncio.sleep(0.05)

        self.assertFalse(cast(Any, handle.await_args).kwargs["allow_tools"])

    async def test_scheduled_requests_run_concurrently(self) -> None:
        """Keep multiple independent agentic requests active at the same time."""
        test_bot = main.TheiaBot()
        started: set[str] = set()
        release = asyncio.Event()

        async def request(name: str) -> None:
            started.add(name)
            await release.wait()

        try:
            test_bot.schedule_request(request("first"))
            test_bot.schedule_request(request("second"))
            for _ in range(20):
                if started == {"first", "second"}:
                    break
                await asyncio.sleep(0)
            self.assertEqual(started, {"first", "second"})
        finally:
            release.set()
            await test_bot._cancel_request_tasks()

    async def test_btw_acknowledges_before_agentic_request_completes(self) -> None:
        """Do not hold the slash-command callback open for the Codex turn."""
        source = _Channel()
        response = SimpleNamespace(defer=AsyncMock())
        interaction = SimpleNamespace(
            id=56,
            channel=source,
            response=response,
            followup=SimpleNamespace(send=AsyncMock()),
            user=SimpleNamespace(
                id=7, guild_permissions=SimpleNamespace(administrator=False)
            ),
        )
        started = asyncio.Event()
        release = asyncio.Event()

        async def request(*_args: Any, **_kwargs: Any) -> None:
            started.set()
            await release.wait()

        with (
            patch("theia.bot.core._require_login", new=AsyncMock(return_value=True)),
            patch("theia.bot.core.handle_request", new=request),
        ):
            await cast(Any, main.codex_btw.callback)(interaction, "keep working")
            self.assertFalse(started.is_set())
            await asyncio.wait_for(started.wait(), timeout=1)
            self.assertFalse(release.is_set())

        release.set()
        await main.bot._cancel_request_tasks()

    async def test_attachments_are_cached_as_codex_local_inputs(self) -> None:
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
                image = SimpleNamespace(
                    filename="photo.png",
                    content_type="image/png",
                    size=4,
                    read=AsyncMock(return_value=b"data"),
                )
                text = SimpleNamespace(
                    filename="notes.txt",
                    content_type="text/plain",
                    size=5,
                    read=AsyncMock(return_value=b"hello"),
                )
                prepared = await server._prepare_attachments((image, text))

                self.assertEqual(prepared[0]["type"], "localImage")
                self.assertIn("hello", prepared[1]["text"])
                cached = list((root / "theia" / "attachments").iterdir())
                self.assertEqual(len(cached), 2)

    async def test_attachment_cache_repairs_a_partial_or_corrupt_existing_file(
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
                cached = server._store_attachment("notes.txt", b"complete")
                cached.write_bytes(b"partial")

                repaired = server._store_attachment("notes.txt", b"complete")
                repaired_bytes = repaired.read_bytes()

        self.assertEqual(repaired, cached)
        self.assertEqual(repaired_bytes, b"complete")

    async def test_attachment_cache_rejects_a_request_that_exceeds_its_quota(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(root / "state.json"),
                    "THEIA_ATTACHMENT_CACHE_LIMIT_BYTES": "3",
                },
            ):
                server = main.CodexAppServer()
                with self.assertRaisesRegex(main.CodexAppServerError, "cache is full"):
                    server._store_attachment("notes.txt", b"data")

    def test_symlinked_outbound_paths_do_not_escape_the_configured_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.txt"
            inside = root / "inside"
            inside.mkdir()
            link = inside / "link.txt"
            outside.write_text("secret", encoding="utf-8")
            try:
                link.symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")

            self.assertFalse(_path_is_under(link, (inside,)))

    async def test_memory_snapshot_is_injected_and_hermes_roots_are_shared(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hermes = root / "hermes"
            memories = hermes / "memories"
            skills = hermes / "skills"
            memories.mkdir(parents=True)
            skills.mkdir(parents=True)
            (memories / "MEMORY.md").write_text(
                "Remember the project context.", encoding="utf-8"
            )
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(root / "state.json"),
                    "HERMES_HOME": str(hermes),
                },
            ):
                server = main.CodexAppServer()
                instructions = server._system_instructions(server._session("session"))

            self.assertIn(hermes / "memories", server._memory_roots)
            self.assertIn(hermes / "skills", server._skill_roots)
            self.assertIn("Remember the project context.", instructions)
