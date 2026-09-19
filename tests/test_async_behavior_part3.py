# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable,duplicate-code
from tests.test_support import *


class AsyncBehaviorTests(AsyncBehaviorTestBase):
    async def test_nightly_recaps_are_unique_and_isolated_by_user_and_server(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"THEIA_NIGHTLY_RECAP": "true"}):
                manager = main.NightlyRecapManager(root, timezone_name="UTC")
                occurred_at = datetime(2026, 9, 6, 18, 30, tzinfo=timezone.utc)
                manager.record_exchange(
                    user_id=7,
                    user_name="Alice",
                    guild_id=42,
                    channel_id=100,
                    session_key="guild:42:channel:100:user:7",
                    prompt="Plan the release.",
                    context=(
                        "Recent messages from this Discord channel:\n"
                        "Bob [Discord user id: 8]: The release is ready."
                    ),
                    response="I recorded the release plan.",
                    completed=True,
                    occurred_at=occurred_at,
                    request_id="alice-42",
                )
                manager.record_exchange(
                    user_id=7,
                    user_name="Alice",
                    guild_id=99,
                    channel_id=200,
                    session_key="guild:99:channel:200:user:7",
                    prompt="Check the other server.",
                    context="Carol [Discord user id: 9]: Server-specific context.",
                    response="I checked it.",
                    completed=True,
                    occurred_at=occurred_at,
                    request_id="alice-99",
                )
                manager.record_exchange(
                    user_id=10,
                    user_name="Dana",
                    guild_id=42,
                    channel_id=100,
                    session_key="guild:42:channel:100:user:10",
                    prompt="Review the incident.",
                    context="Alice [Discord user id: 7]: The incident is resolved.",
                    response="The incident is recorded.",
                    completed=True,
                    occurred_at=occurred_at,
                    request_id="dana-42",
                )
                prompts: list[tuple[str, str | None]] = []

                async def generate(prompt: str, session_key: str | None) -> str:
                    prompts.append((prompt, session_key))
                    return "The primary user worked through the day's major task."

                generated = await manager.process_due(
                    generate,
                    now=datetime(2026, 9, 7, tzinfo=timezone.utc),
                )
                restarted = main.NightlyRecapManager(root, timezone_name="UTC")

            self.assertEqual(generated, 3)
            self.assertEqual(len(prompts), 3)
            self.assertTrue(any("guild:42:user:7" in prompt for prompt, _ in prompts))
            alice_recap = restarted.context_for(user_id=7, guild_id=42)
            other_server_recap = restarted.context_for(user_id=7, guild_id=99)
            dana_recap = restarted.context_for(user_id=10, guild_id=42)
            self.assertIsNotNone(alice_recap)
            self.assertIsNotNone(other_server_recap)
            self.assertIsNotNone(dana_recap)
            assert alice_recap is not None
            assert other_server_recap is not None
            self.assertIn("2026-09-06", alice_recap)
            self.assertIn("18:30:00", alice_recap)
            self.assertIn("Bob (Discord user id: 8)", alice_recap)
            self.assertNotIn("Carol (Discord user id: 9)", alice_recap)
            self.assertIn("Carol (Discord user id: 9)", other_server_recap)
            self.assertNotIn("Dana (Discord user id: 10)", alice_recap)

    async def test_nightly_recap_failure_keeps_journal_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"THEIA_NIGHTLY_RECAP": "true"}):
                manager = main.NightlyRecapManager(root, timezone_name="UTC")
                manager.record_exchange(
                    user_id=7,
                    user_name="Alice",
                    guild_id=42,
                    channel_id=100,
                    session_key="session",
                    prompt="Remember this.",
                    context=None,
                    response="Done.",
                    completed=True,
                    occurred_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
                    request_id="retry-me",
                )

                async def fail(_prompt: str, _session_key: str | None) -> str:
                    raise RuntimeError("temporary failure")

                self.assertEqual(
                    await manager.process_due(
                        fail,
                        now=datetime(2026, 9, 7, tzinfo=timezone.utc),
                    ),
                    0,
                )
                self.assertEqual(len(manager._journal), 1)

                async def succeed(_prompt: str, _session_key: str | None) -> str:
                    return "The user made a durable decision."

                self.assertEqual(
                    await manager.process_due(
                        succeed,
                        now=datetime(2026, 9, 7, tzinfo=timezone.utc),
                    ),
                    1,
                )
                self.assertEqual(manager._journal, [])

    def test_nightly_recap_scheduler_uses_local_midnight(self) -> None:
        manager = main.NightlyRecapManager(
            Path(tempfile.mkdtemp()), timezone_name="UTC"
        )
        try:
            self.assertEqual(
                manager.seconds_until_midnight(
                    datetime(2026, 9, 6, 23, 59, 30, tzinfo=timezone.utc)
                ),
                30,
            )
        finally:
            manager.state_path.unlink(missing_ok=True)

    async def test_nightly_recap_generation_is_ephemeral_and_has_no_tools(
        self,
    ) -> None:
        server = main.CodexAppServer()
        requests: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

        async def request(method: str, params: dict[str, Any], **kwargs: Any) -> dict:
            requests.append((method, params, kwargs))
            if method == "thread/start":
                return {"thread": {"id": "recap-thread"}}
            return {"turn": {"id": "recap-turn"}}

        server._request = AsyncMock(side_effect=request)
        server._ensure_running = AsyncMock()
        server._wait_for_turn = AsyncMock(
            return_value='{"recap":"The release was completed."}'
        )
        result = await server.generate_nightly_recap(
            "Summarize this private journal.",
            session_key="guild:42:channel:100:user:7",
        )

        self.assertEqual(result, "The release was completed.")
        self.assertEqual(
            [method for method, _, _ in requests], ["thread/start", "turn/start"]
        )
        thread_params = requests[0][1]
        self.assertTrue(thread_params["ephemeral"])
        self.assertEqual(thread_params["approvalPolicy"], "never")
        self.assertEqual(thread_params["sandbox"], "read-only")
        self.assertEqual(thread_params["runtimeWorkspaceRoots"], [])
        self.assertNotIn("dynamicTools", thread_params)
        self.assertEqual(requests[1][1]["effort"], "low")
        self.assertIn("outputSchema", requests[1][1])
        self.assertNotIn("recap-thread", server._sessions)

    async def test_approval_request_includes_owner_checked_buttons(self) -> None:
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
                },
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        view = channel.sent[0]["view"]
        self.assertEqual(
            [button.label for button in view.children], ["Approve", "Deny"]
        )
        self.assertTrue(server.resolve_approval(7, False, channel))
        self.assertEqual(await request, {"decision": "decline"})

    async def test_approval_uses_request_user_when_member_cache_is_empty(self) -> None:
        server = main.CodexAppServer()
        channel = _Channel()
        user = SimpleNamespace(
            id=7,
            guild_permissions=SimpleNamespace(administrator=True),
        )
        channel.guild = SimpleNamespace(id=1, get_member=lambda _user_id: None)
        state = main._TurnState(
            thread_id="thread",
            channel=channel,
            user_id=user.id,
            user=user,
        )
        server._turns["turn"] = state
        request = asyncio.create_task(
            server._server_request_result(
                "item/commandExecution/requestApproval",
                {"threadId": "thread", "turnId": "turn", "itemId": "item"},
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertEqual(channel.sent[0]["embed"].title, "Approval needed")
        interaction = SimpleNamespace(
            user=user,
            response=SimpleNamespace(edit_message=AsyncMock()),
        )
        await cast(Any, channel.sent[0]["view"].children[0]).callback(interaction)
        self.assertEqual(await request, {"decision": "accept"})

    async def test_approval_falls_back_to_active_turn_when_ids_are_missing(
        self,
    ) -> None:
        server = main.CodexAppServer()
        channel = _Channel()
        channel.guild = _admin_guild()
        state = main._TurnState(thread_id="thread", channel=channel, user_id=7)
        server._turns["turn"] = state
        request = asyncio.create_task(
            server._server_request_result(
                "item/fileChange/requestApproval",
                {"threadId": "thread"},
                request_id="request-1",
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertEqual(channel.sent[0]["embed"].title, "Approval needed")
        self.assertTrue(server.resolve_approval(7, False, channel))
        self.assertEqual(await request, {"decision": "decline"})

    async def test_unavailable_approval_is_reported_in_discord(self) -> None:
        server = main.CodexAppServer()
        channel = _Channel()
        channel.guild = _admin_guild()
        state = main._TurnState(
            thread_id="thread", channel=channel, user_id=7, allow_tools=False
        )
        server._turns["turn"] = state

        result = await server._server_request_result(
            "item/commandExecution/requestApproval",
            {"threadId": "thread", "turnId": "turn", "itemId": "item"},
        )

        self.assertEqual(result, {"decision": "decline"})
        self.assertEqual(len(channel.sent), 1)
        self.assertEqual(channel.sent[0]["embed"].title, "Approval needed")
        self.assertIn("cannot be approved", channel.sent[0]["embed"].description)
        self.assertNotIn("view", channel.sent[0])

    async def test_approval_lost_admin_access_is_reported_in_discord(self) -> None:
        server = main.CodexAppServer()
        channel = _Channel()
        channel.guild = _admin_guild(administrator=False)
        request_user = SimpleNamespace(
            id=7,
            guild_permissions=SimpleNamespace(administrator=True),
        )
        state = main._TurnState(
            thread_id="thread", channel=channel, user_id=7, user=request_user
        )
        server._turns["turn"] = state

        result = await server._server_request_result(
            "item/commandExecution/requestApproval",
            {"threadId": "thread", "turnId": "turn", "itemId": "item"},
        )

        self.assertEqual(result, {"decision": "decline"})
        self.assertEqual(len(channel.sent), 1)
        self.assertIn("administrator access", channel.sent[0]["embed"].description)
        self.assertFalse(state.allow_tools)

    async def test_approval_request_describes_the_action_without_raw_details(
        self,
    ) -> None:
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
                    "reason": "I need to run `git status` in /workspace/project.",
                },
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        description = channel.sent[0]["embed"].description
        self.assertEqual(description, "I need to run in")
        self.assertEqual(
            channel.sent[0]["embed"].footer.text,
            "You can also use /approve or /deny.",
        )
        self.assertNotIn("git status", description)
        self.assertNotIn("/workspace/project", description)
        self.assertTrue(server.resolve_approval(7, False, channel))
        self.assertEqual(await request, {"decision": "decline"})

    async def test_medium_approval_level_auto_approves_safe_commands(self) -> None:
        with patch.dict(os.environ, {"THEIA_APPROVAL_LEVEL": "medium"}):
            server = main.CodexAppServer()
        channel = _Channel()
        channel.guild = _admin_guild()
        state = main._TurnState(thread_id="thread", channel=channel, user_id=7)
        server._turns["turn"] = state

        result = await server._server_request_result(
            "item/commandExecution/requestApproval",
            {
                "threadId": "thread",
                "turnId": "turn",
                "itemId": "item",
                "command": "git status",
            },
        )

        self.assertEqual(result, {"decision": "accept"})
        self.assertEqual(channel.sent, [])

    async def test_high_approval_level_still_surfaces_safe_commands(self) -> None:
        with patch.dict(os.environ, {"THEIA_APPROVAL_LEVEL": "high"}):
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
                    "command": "git status",
                },
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertEqual(len(channel.sent), 1)
        self.assertTrue(server.resolve_approval(7, False, channel))
        self.assertEqual(await request, {"decision": "decline"})

    async def test_low_approval_level_keeps_very_dangerous_commands_manual(
        self,
    ) -> None:
        with patch.dict(os.environ, {"THEIA_APPROVAL_LEVEL": "low"}):
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
                    "command": "rm -rf /tmp/example",
                },
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertEqual(len(channel.sent), 1)
        self.assertTrue(server.resolve_approval(7, False, channel))
        self.assertEqual(await request, {"decision": "decline"})

    async def test_low_approval_level_auto_approves_an_ordinary_command(self) -> None:
        with patch.dict(os.environ, {"THEIA_APPROVAL_LEVEL": "low"}):
            server = main.CodexAppServer()
        channel = _Channel()
        channel.guild = _admin_guild()
        state = main._TurnState(thread_id="thread", channel=channel, user_id=7)
        server._turns["turn"] = state

        result = await server._server_request_result(
            "item/commandExecution/requestApproval",
            {
                "threadId": "thread",
                "turnId": "turn",
                "itemId": "item",
                "command": "echo hello",
            },
        )

        self.assertEqual(result, {"decision": "accept"})
        self.assertEqual(channel.sent, [])

    async def test_low_approval_level_keeps_interpreters_manual(self) -> None:
        with patch.dict(os.environ, {"THEIA_APPROVAL_LEVEL": "low"}):
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
                    "command": "python -c 'print(1)'",
                },
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertEqual(len(channel.sent), 1)
        self.assertTrue(server.resolve_approval(7, False, channel))
        self.assertEqual(await request, {"decision": "decline"})

    async def test_approval_button_declines_after_administrator_access_is_revoked(
        self,
    ) -> None:
        server = main.CodexAppServer()
        channel = _Channel()
        member = SimpleNamespace(
            id=7,
            guild_permissions=SimpleNamespace(administrator=True),
        )
        channel.guild = SimpleNamespace(id=1, get_member=lambda _user_id: member)
        state = main._TurnState(thread_id="thread", channel=channel, user_id=7)
        server._turns["turn"] = state
        request = asyncio.create_task(
            server._server_request_result(
                "item/commandExecution/requestApproval",
                {"threadId": "thread", "turnId": "turn", "itemId": "item"},
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        member.guild_permissions.administrator = False
        interaction = SimpleNamespace(
            user=SimpleNamespace(id=7),
            response=SimpleNamespace(edit_message=AsyncMock()),
        )
        await cast(Any, channel.sent[0]["view"].children[0]).callback(interaction)

        self.assertEqual(await request, {"decision": "decline"})
        self.assertFalse(state.allow_tools)

    async def test_dynamic_discord_tool_rechecks_current_administrator_access(
        self,
    ) -> None:
        server = main.CodexAppServer()
        channel = _Channel()
        channel.guild = _admin_guild()
        state = main._TurnState(
            thread_id="thread", channel=channel, user_id=7, allow_tools=True
        )
        channel.guild.get_member(7).guild_permissions.administrator = False

        result = await server._dynamic_tool_call(
            state,
            {
                "tool": "send_message",
                "namespace": "discord",
                "arguments": {"content": "should not send"},
            },
        )

        self.assertFalse(result["success"])
        self.assertEqual(channel.sent, [])
        self.assertFalse(state.allow_tools)

    async def test_dynamic_discord_tool_rechecks_cached_member_for_stored_user(
        self,
    ) -> None:
        server = main.CodexAppServer()
        channel = _Channel()
        cached_member = SimpleNamespace(
            id=7,
            guild_permissions=SimpleNamespace(administrator=False),
        )
        request_user = SimpleNamespace(
            id=7,
            guild_permissions=SimpleNamespace(administrator=True),
        )
        channel.guild = SimpleNamespace(
            id=1,
            get_member=lambda user_id: cached_member if user_id == 7 else None,
        )
        state = main._TurnState(
            thread_id="thread",
            channel=channel,
            user_id=7,
            user=request_user,
            allow_tools=True,
        )

        result = await server._dynamic_tool_call(
            state,
            {
                "tool": "send_message",
                "namespace": "discord",
                "arguments": {"content": "should not send"},
            },
        )

        self.assertFalse(result["success"])
        self.assertEqual(channel.sent, [])
        self.assertFalse(state.allow_tools)

    async def test_dynamic_discord_send_failure_returns_unsuccessful_result(
        self,
    ) -> None:
        server = main.CodexAppServer()
        channel = _FailingSendChannel()
        channel.guild = _admin_guild()
        state = main._TurnState(
            thread_id="thread", channel=channel, user_id=7, allow_tools=True
        )

        result = await server._dynamic_tool_call(
            state,
            {
                "tool": "send_message",
                "namespace": "discord",
                "arguments": {"content": "message"},
            },
        )

        self.assertFalse(result["success"])
        self.assertEqual(
            result["contentItems"][0]["text"],
            "Discord could not send the message.",
        )

    def test_safe_text_redacts_windows_and_unc_paths(self) -> None:
        windows = r"failed at C:\Users\user\PrivateProject\secret.db"
        unc = r"failed at \\server\share\PrivateProject\secret.db"

        for sanitizer in (
            main._safe_intermediate_text,
            main._safe_approval_reason,
            main._safe_error_reason,
        ):
            rendered_windows = sanitizer(windows)
            rendered_unc = sanitizer(unc)
            self.assertNotIn("PrivateProject", rendered_windows)
            self.assertNotIn("secret.db", rendered_windows)
            self.assertNotIn("PrivateProject", rendered_unc)
            self.assertNotIn("secret.db", rendered_unc)

    async def test_non_admin_dynamic_discord_tool_is_rejected(self) -> None:
        server = main.CodexAppServer()
        state = main._TurnState(
            thread_id="thread", channel=_Channel(), user_id=7, allow_tools=False
        )
        result = await server._dynamic_tool_call(
            state,
            {
                "tool": "send_message",
                "namespace": "discord",
                "arguments": {"content": "should not send"},
            },
        )
        self.assertFalse(result["success"])
        self.assertEqual(state.channel.sent, [])

    async def test_thread_tool_is_available_without_explicit_user_intent(self) -> None:
        server = main.CodexAppServer()
        channel = _Channel()
        channel.guild = _admin_guild()
        thread = _Channel()
        thread.guild = channel.guild
        channel.create_thread = AsyncMock(return_value=thread)
        server.set_thread_name = AsyncMock()
        state = main._TurnState(
            thread_id="thread",
            channel=channel,
            user_id=7,
            allow_tools=True,
        )

        result = await server._dynamic_tool_call(
            state,
            {
                "tool": "create_thread",
                "namespace": "discord",
                "arguments": {
                    "name": "Requested by Codex",
                    "opening_message": "I am organizing this request in a thread.",
                },
            },
        )

        self.assertTrue(result["success"])
        channel.create_thread.assert_awaited_once_with(
            name="Requested by Codex", auto_archive_duration=1440
        )

    async def test_thread_tool_creates_and_routes_the_active_session(self) -> None:
        server = main.CodexAppServer()
        channel = _Channel()
        channel.guild = _admin_guild()
        thread = _Channel()
        thread.guild = channel.guild
        thread.edit = AsyncMock()
        source_message = SimpleNamespace(
            channel=channel,
            create_thread=AsyncMock(return_value=thread),
        )
        changed: list[object] = []
        events: list[tuple[str, dict]] = []

        async def on_event(event: str, payload: dict) -> None:
            events.append((event, payload))

        server.set_thread_name = AsyncMock()
        state = main._TurnState(
            thread_id="thread",
            session=server._session("source"),
            channel=channel,
            user_id=7,
            allow_tools=True,
            thread_source=source_message,
            user_prompt="Create a thread for this request.",
            on_channel_change=changed.append,
            on_event=on_event,
        )

        result = await server._dynamic_tool_call(
            state,
            {
                "threadId": "thread",
                "turnId": "turn",
                "callId": "call",
                "tool": "create_thread",
                "namespace": "discord",
                "arguments": {
                    "name": "  A\nuseful thread  ",
                    "opening_message": "I am moving this request into its own thread.",
                },
            },
        )

        self.assertTrue(result["success"])
        thread.edit.assert_awaited_once_with(name="A useful thread")
        server.set_thread_name.assert_awaited_once_with("thread", "A useful thread")
        source_message.create_thread.assert_awaited_once_with(
            name="A useful thread", auto_archive_duration=1440
        )
        self.assertIs(state.channel, thread)
        self.assertEqual(changed, [thread])
        self.assertEqual(
            events,
            [
                (
                    "thread_opening",
                    {
                        "type": "agentMessage",
                        "phase": "commentary",
                        "text": "I am moving this request into its own thread.",
                    },
                )
            ],
        )
        self.assertEqual(thread.sent, [])

        repeated = await server._dynamic_tool_call(
            state,
            {
                "tool": "create_thread",
                "namespace": "discord",
                "arguments": {"name": "A second thread"},
            },
        )
        self.assertTrue(repeated["success"])
        self.assertEqual(
            repeated["contentItems"][0]["text"],
            "Thread setup is complete. Continue with the user's request now; "
            "do not mention thread setup or call create_thread again.",
        )
        source_message.create_thread.assert_awaited_once()
        self.assertEqual(len(thread.sent), 0)

    def test_admin_thread_params_include_discord_thread_tool(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("transient"))

        admin_params = server._thread_instruction_params(session, True)
        restricted_params = server._thread_instruction_params(session, False)

        self.assertEqual(admin_params["dynamicTools"][0]["name"], "discord")
        self.assertEqual(
            admin_params["dynamicTools"][0]["tools"][0]["name"],
            "create_thread",
        )
        self.assertIn(
            "same base priors, active personality, user request",
            admin_params["developerInstructions"],
        )
        self.assertIn(
            "all applicable user formatting requirements",
            admin_params["dynamicTools"][0]["tools"][0]["inputSchema"]["properties"][
                "opening_message"
            ]["description"],
        )
        self.assertNotIn("dynamicTools", restricted_params)

    def test_rebinding_a_created_thread_preserves_the_session_after_restart(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            with patch.dict(os.environ, {"THEIA_STATE": str(state_path)}):
                server = main.CodexAppServer()
                session = server._session("source")
                session.thread_id = "codex-thread"
                server.rebind_session("source", "thread-channel")

                self.assertIs(
                    server._session("source"), server._session("thread-channel")
                )
                server._persist_state()
                restarted = main.CodexAppServer()

            self.assertEqual(
                restarted._session("thread-channel").thread_id,
                "codex-thread",
            )
            self.assertIs(
                restarted._session("source"),
                restarted._session("thread-channel"),
            )

    async def test_message_claims_are_persisted_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            with patch.dict(os.environ, {"THEIA_STATE": str(state_path)}):
                server = main.CodexAppServer()
                self.assertTrue(server.claim_message("message-1"))
                self.assertFalse(server.claim_message("message-1"))
                server.complete_message("message-1")
                restarted = main.CodexAppServer()
                self.assertFalse(restarted.claim_message("message-1"))

    def test_assessment_parser_rejects_untrusted_or_incomplete_output(self) -> None:
        self.assertIsNone(main.CodexAppServer._parse_assessment("not json"))
        self.assertIsNone(
            main.CodexAppServer._parse_assessment(
                '{"complexity":"complex","requires_tool":"yes"}'
            )
        )
        self.assertEqual(
            main.CodexAppServer._parse_assessment(
                '```json\n{"complexity":"simple","requires_tool":false}\n```'
            ),
            {"complexity": "simple", "requires_tool": False},
        )

    async def test_user_turn_uses_effort_after_assessment(self) -> None:
        server = main.CodexAppServer()
        server.account = {"type": "chatgpt"}
        server.requires_openai_auth = True
        server._ensure_running = AsyncMock()
        server.refresh_account = AsyncMock()
        server._select_reasoning_effort = AsyncMock(return_value="high")
        server._ensure_thread = AsyncMock()
        server._request = AsyncMock(return_value={"turn": {"id": "turn"}})
        server._wait_for_turn = AsyncMock(return_value="done")

        result = await server.ask(
            "inspect the project",
            session_key="session",
            channel=None,
            user_id=7,
        )

        self.assertEqual(result, "done")
        self.assertEqual(cast(Any, server._request.await_args).args[0], "turn/start")
        turn_params = cast(Any, server._request.await_args).args[1]
        self.assertEqual(turn_params["effort"], "high")
        self.assertEqual(turn_params["model"], main.DEFAULT_CODEX_MODEL)

    async def test_completed_admin_turn_schedules_self_improvement_in_background(
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
        review = AsyncMock()
        server._run_self_improvement_review = review
        channel = _Channel()
        channel.guild = _admin_guild()

        result = await server.ask(
            "inspect the project",
            session_key="session",
            channel=cast(Any, channel),
            user_id=7,
            allow_tools=True,
        )
        await asyncio.sleep(0)

        self.assertEqual(result, "done")
        review.assert_awaited_once()
