# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable,duplicate-code
from tests.test_support import *


class AsyncBehaviorTests(AsyncBehaviorTestBase):
    async def test_unhandled_interaction_gets_a_restart_safe_acknowledgement(
        self,
    ) -> None:
        response = SimpleNamespace(
            is_done=lambda: False,
            send_message=AsyncMock(),
        )
        interaction = SimpleNamespace(response=response)
        with patch("theia.bot.core.STALE_INTERACTION_FALLBACK_DELAY", 0):
            await main.bot._acknowledge_unhandled_interaction(cast(Any, interaction))

        response.send_message.assert_awaited_once_with(
            "This control expired or was interrupted by a restart. "
            "Please start a new request.",
            ephemeral=True,
        )

    async def test_btw_without_prompt_opens_the_shared_request_modal(self) -> None:
        interaction = SimpleNamespace(
            user=SimpleNamespace(id=7),
            channel=_Channel(),
            response=SimpleNamespace(
                is_done=lambda: False,
                send_modal=AsyncMock(),
            ),
        )
        with (
            patch.object(main.bot, "customizations", None),
            patch("theia.bot.core._require_login", new=AsyncMock(return_value=True)),
        ):
            await cast(Any, main.codex_btw.callback)(interaction)

        interaction.response.send_modal.assert_awaited_once()
        self.assertIsInstance(
            interaction.response.send_modal.await_args.args[0], main._PromptModal
        )

    async def test_about_command_fetches_data_and_sends_a_private_embed(self) -> None:
        guild = SimpleNamespace(id=42)
        channel = SimpleNamespace(id=7, guild=guild)
        interaction = SimpleNamespace(
            id=1,
            channel=channel,
            user=SimpleNamespace(
                id=9,
                name="username",
                mention="@username",
            ),
            response=SimpleNamespace(defer=AsyncMock()),
            followup=SimpleNamespace(send=AsyncMock()),
        )
        with (
            patch.object(main.bot.presence, "touch", new=AsyncMock()),
            patch.object(
                main.bot.codex,
                "account_details",
                new=AsyncMock(return_value={"account": {"planType": "plus"}}),
            ) as account_details,
            patch.object(
                main.bot.codex,
                "codex_cli_version",
                return_value="0.153.0",
            ),
            patch.object(main.bot.codex, "mode", return_value="text"),
            patch.object(main.bot.codex, "active_personality", return_value="Cel"),
            patch("theia.bot.core._theia_revision", return_value="a1b2c3d"),
        ):
            await cast(Any, main.codex_about.callback)(interaction)

        account_details.assert_awaited_once_with()
        interaction.response.defer.assert_awaited_once_with(ephemeral=True)
        kwargs = interaction.followup.send.await_args.kwargs
        self.assertTrue(kwargs["ephemeral"])
        self.assertEqual(
            [(field.name, field.value) for field in kwargs["embed"].fields],
            [
                ("Theia Agent", "1.0.2 (a1b2c3d)"),
                ("Codex CLI", "0.153.0"),
                ("Account", "@username"),
                ("Plan", "Plus ($20/mo)"),
                ("Mode", "Text"),
                ("Personality", "Cel"),
            ],
        )

    async def test_personality_root_shows_the_active_character_card_privately(
        self,
    ) -> None:
        guild = SimpleNamespace(id=42)
        channel = SimpleNamespace(id=7, guild=guild)
        interaction = SimpleNamespace(
            channel=channel,
            user=SimpleNamespace(id=9, name="username"),
            response=SimpleNamespace(defer=AsyncMock()),
            followup=SimpleNamespace(send=AsyncMock()),
        )
        summary = {
            "name": "cel",
            "identifier": "cel",
            "character_name": "Celune",
            "description": "A calm lunar guardian.",
            "known_entries": 12,
            "known_users": 3,
            "scope": "me",
            "set_by": 9,
        }
        with (
            patch.object(main.bot.presence, "touch", new=AsyncMock()),
            patch.object(
                main.bot.codex,
                "personality_summary",
                new=AsyncMock(return_value=summary),
            ),
            patch.object(
                main.bot.codex,
                "mood_state",
                return_value={"label": "neutral", "strength": 0.5},
            ),
            patch.object(
                main.bot.rich_presence,
                "_current_activity",
                discord.CustomActivity("watching the moon"),
            ),
        ):
            await cast(Any, main.codex_personality.callback)(interaction)

        interaction.response.defer.assert_awaited_once_with(ephemeral=True)
        kwargs = interaction.followup.send.await_args.kwargs
        self.assertTrue(kwargs["ephemeral"])
        embed = kwargs["embed"]
        self.assertEqual(embed.title, "Celune (cel)")
        self.assertEqual(embed.description, "A calm lunar guardian.")
        self.assertEqual(
            [(field.name, field.value) for field in embed.fields],
            [
                ("Known Entries", "12"),
                ("Scope", "me"),
                ("Set by", "Discord user ID: 9"),
                ("Known Users", "3"),
                ("Mood", "Neutral (50%)"),
                ("Presence", "watching the moon"),
            ],
        )
        self.assertEqual(
            embed.footer.text,
            "Add or change the character with `/personality <file> <slug>`.",
        )

    async def test_personality_shared_scopes_require_administrator_access(self) -> None:
        interaction = SimpleNamespace(
            channel=SimpleNamespace(id=7, guild=SimpleNamespace(id=42)),
            guild=SimpleNamespace(id=42),
            user=SimpleNamespace(
                id=9,
                guild_permissions=SimpleNamespace(administrator=False),
            ),
            response=SimpleNamespace(defer=AsyncMock(), is_done=lambda: True),
            followup=SimpleNamespace(send=AsyncMock()),
        )
        configure = AsyncMock()
        with (
            patch.object(main.bot.presence, "touch", new=AsyncMock()),
            patch.object(main.bot.codex, "configure_personality", new=configure),
        ):
            await cast(Any, main.codex_personality.callback)(
                interaction,
                name="cel",
                scope=discord.app_commands.Choice(name="server", value="server"),
            )

        configure.assert_not_awaited()
        self.assertTrue(interaction.followup.send.await_args.kwargs["ephemeral"])
        self.assertIn(
            "server or everyone personalities",
            interaction.followup.send.await_args.kwargs["embed"].description,
        )

    async def test_login_messages_cover_each_authentication_path(self) -> None:
        """Expose distinct status messages for cached, imported, and device auth."""
        cases = (
            ({"login_cached": True}, "Already logged in"),
            ({"login_imported": True}, "Cached authentication imported"),
            (
                {"verificationUrl": "https://example.test/device", "userCode": "ABC"},
                "Device code required",
            ),
        )
        for result, expected_title in cases:
            send = AsyncMock()
            with (
                patch.object(main.bot.presence, "touch", new=AsyncMock()),
                patch.object(
                    main.bot.codex, "begin_login", new=AsyncMock(return_value=result)
                ),
                patch.object(main.bot.codex, "mark_authenticated"),
            ):
                await main.handle_login(_Channel(), send, user_id=7)

            embed = cast(Any, send.await_args).kwargs["embed"]
            self.assertEqual(embed.title, expected_title)

    async def test_login_field_labels_and_footer_are_customizable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = main.FrontendCustomizationStore(Path(directory) / "frontend.json")
            store.set(42, "login_verification_link", "label", "Open link")
            store.set(42, "login_code", "label", "Device code")
            store.set(42, "login_visibility_footer", "label", "Private")
            channel = _Channel()
            channel.guild = SimpleNamespace(id=42, name="Example")
            send = AsyncMock()
            with (
                patch.object(main.bot, "customizations", store),
                patch.object(main.bot.presence, "touch", new=AsyncMock()),
                patch.object(
                    main.bot.codex,
                    "begin_login",
                    new=AsyncMock(
                        return_value={
                            "verificationUrl": "https://example.test/device",
                            "userCode": "ABC",
                        }
                    ),
                ),
            ):
                await main.handle_login(channel, send, user_id=7)

        embed = cast(Any, send.await_args).kwargs["embed"]
        self.assertEqual(
            [field.name for field in embed.fields], ["Open link", "Device code"]
        )
        self.assertEqual(embed.footer.text, "Private")

    async def test_login_without_a_usable_cache_starts_device_code_flow(self) -> None:
        """Start device-code authentication when neither cache is usable."""
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server.refresh_account = AsyncMock()
        server._request = AsyncMock(
            return_value={
                "loginId": "login-1",
                "verificationUrl": "https://example.test/device",
                "userCode": "ABC",
            }
        )
        server.account = None
        server.requires_openai_auth = True

        result = await server.begin_login(_Channel(), 7)

        self.assertEqual(result["loginId"], "login-1")
        server._request.assert_awaited_once_with(
            "account/login/start",
            {"type": "chatgptDeviceCode"},
        )

    async def test_login_completion_reports_authentication_completed(self) -> None:
        """Report the final user-visible status after device authentication succeeds."""
        server = main.CodexAppServer()
        server._login_id = "login-1"
        server._login_channel = cast(Any, _Channel())
        server._login_user_id = 7

        with (
            patch.object(server, "mark_authenticated"),
            patch.object(server, "_background_send") as background_send,
        ):
            server._handle_notification(
                {
                    "method": "account/login/completed",
                    "params": {"loginId": "login-1", "success": True},
                }
            )

        embed = background_send.call_args.args[1]
        self.assertEqual(embed.title, "Authentication completed")

    async def test_customization_command_is_admin_only_and_server_scoped(self) -> None:
        guild = SimpleNamespace(id=42, name="Example")
        channel = SimpleNamespace(id=7, name="general", guild=guild)
        response = SimpleNamespace(
            is_done=lambda: False,
            send_message=AsyncMock(),
        )
        interaction = SimpleNamespace(
            guild=guild,
            channel=channel,
            response=response,
            user=SimpleNamespace(
                id=9,
                name="member",
                display_name="Member",
                guild_permissions=SimpleNamespace(administrator=False),
            ),
        )

        await cast(Any, main.codex_customize.callback)(
            interaction,
            "usage",
            "title",
            "Should not save",
        )

        embed = response.send_message.await_args.kwargs["embed"]
        self.assertEqual(embed.title, "Administrator access required")
        self.assertTrue(response.send_message.await_args.kwargs["ephemeral"])

        with tempfile.TemporaryDirectory() as directory:
            store = main.FrontendCustomizationStore(Path(directory) / "frontend.json")
            admin_response = SimpleNamespace(
                is_done=lambda: False,
                send_message=AsyncMock(),
            )
            admin = SimpleNamespace(
                guild=guild,
                channel=channel,
                response=admin_response,
                user=SimpleNamespace(
                    id=1,
                    name="admin",
                    display_name="Admin",
                    guild_permissions=SimpleNamespace(administrator=True),
                ),
            )
            with patch.object(main.bot, "customizations", store):
                await cast(Any, main.codex_customize.callback)(
                    admin,
                    "/usage",
                    "title",
                    "Usage for {server}",
                )

            self.assertNotIn("ephemeral", admin_response.send_message.await_args.kwargs)
            self.assertEqual(
                store.render(
                    42,
                    "usage",
                    "title",
                    "Usage",
                    context={"server": "Example"},
                ),
                "Usage for Example",
            )

    async def test_debug_command_is_admin_only_and_starts_live_refresh(self) -> None:
        guild = SimpleNamespace(id=42)
        channel = SimpleNamespace(id=7, guild=guild)
        response = SimpleNamespace(send_message=AsyncMock())
        message = SimpleNamespace(edit=AsyncMock())
        interaction = SimpleNamespace(
            guild=guild,
            channel=channel,
            response=response,
            original_response=AsyncMock(return_value=message),
            user=SimpleNamespace(
                id=1,
                name="admin",
                guild_permissions=SimpleNamespace(administrator=True),
            ),
        )
        state = {
            "runtime": {"process": "running", "authenticated": True},
            "configuration": {"model": "gpt-5.6-luna", "approval_level": "high"},
            "session": {"mode": "text", "personality": "Cel", "mood": {}},
            "counts": {},
            "usage": {},
        }
        with (
            patch.object(main.bot.codex, "debug_state", return_value=state),
            patch.object(main.bot, "schedule_debug_refresh") as schedule,
        ):
            await cast(Any, main.codex_debug.callback)(interaction)

        kwargs = response.send_message.await_args.kwargs
        self.assertTrue(kwargs["ephemeral"])
        self.assertIsInstance(kwargs["view"], main._DebugView)
        schedule.assert_called_once_with(
            message,
            kwargs["view"],
            session_key_value=main.session_key(channel, 1),
            channel=channel,
            user=interaction.user,
        )

    async def test_debug_command_rejects_non_administrators(self) -> None:
        guild = SimpleNamespace(id=42)
        channel = SimpleNamespace(id=7, guild=guild)
        response = SimpleNamespace(
            is_done=lambda: False,
            send_message=AsyncMock(),
        )
        interaction = SimpleNamespace(
            guild=guild,
            channel=channel,
            response=response,
            user=SimpleNamespace(
                id=2,
                guild_permissions=SimpleNamespace(administrator=False),
            ),
        )
        with patch.object(main.bot, "schedule_debug_refresh") as schedule:
            await cast(Any, main.codex_debug.callback)(interaction)

        self.assertEqual(
            response.send_message.await_args.kwargs["embed"].title,
            "Administrator access required",
        )
        schedule.assert_not_called()

    async def test_model_selection_confirmation_is_public(self) -> None:
        interaction = SimpleNamespace(
            guild=SimpleNamespace(id=42),
            channel=SimpleNamespace(id=7, guild=SimpleNamespace(id=42)),
            user=SimpleNamespace(id=9),
            response=SimpleNamespace(defer=AsyncMock()),
            followup=SimpleNamespace(send=AsyncMock()),
        )
        with (
            patch.object(main.bot.presence, "touch", new=AsyncMock()),
            patch.object(main.bot.codex, "is_authenticated", return_value=True),
            patch.object(main.bot.codex, "set_model", new=AsyncMock()),
        ):
            await cast(Any, main.codex_model.callback)(interaction, "gpt-test")

        interaction.response.defer.assert_awaited_once_with()
        self.assertNotIn("ephemeral", interaction.followup.send.await_args.kwargs)

    async def test_restart_confirmation_is_public(self) -> None:
        interaction = SimpleNamespace(
            guild=SimpleNamespace(id=42),
            channel=SimpleNamespace(id=7, guild=SimpleNamespace(id=42)),
            user=SimpleNamespace(
                id=9,
                guild_permissions=SimpleNamespace(administrator=True),
            ),
            response=SimpleNamespace(send_message=AsyncMock()),
        )
        with (
            patch.object(main.bot, "_restart_task", None),
            patch("theia.bot.core._restart_in_place", new=AsyncMock()),
        ):
            await cast(Any, main.codex_restart.callback)(interaction)
            restart_task = main.bot._restart_task
            if restart_task is not None:
                await restart_task

        self.assertNotIn(
            "ephemeral", interaction.response.send_message.await_args.kwargs
        )

    async def test_status_customization_changes_only_discord_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = main.FrontendCustomizationStore(Path(directory) / "frontend.json")
            store.set(42, "thinking", "label", "Working ({status})")
            store.set(42, "intermediate", "content", "Update: {text}")
            calls: list[dict] = []

            async def send(**kwargs):
                calls.append(kwargs)
                return _Message()

            delivery = main._ResponseDelivery(
                send,
                {},
                owner_id=7,
                customizer=store,
                guild_id=42,
            )
            await delivery.on_event("tool_activity", {})
            self.assertEqual(calls[0]["content"], "-# Working (Thinking)")
            await delivery.on_event(
                "item_completed",
                {
                    "type": "agentMessage",
                    "phase": "commentary",
                    "text": "Checking the request.",
                },
            )

            assert delivery.status_message is not None
            status_message = cast(Any, delivery.status_message)
            self.assertEqual(
                status_message.edits[-1]["content"],
                "-# Update: Checking the request.",
            )

    async def test_self_improvement_statuses_are_customizable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = main.FrontendCustomizationStore(Path(directory) / "frontend.json")
            store.set(42, "personality_updated", "label", "Style refined")
            server = main.CodexAppServer()
            server.set_frontend_customizer(store)
            channel = _Channel()
            channel.guild = SimpleNamespace(id=42)

            await server._notify_self_improvement(
                cast(Any, channel), ["Personality updated"]
            )

        self.assertEqual(channel.sent[0]["content"], "-# Style refined")

    def test_explicit_codex_cli_override_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            local_cli = root / "codex"
            local_cli.write_text("#!/bin/sh\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "THEIA_CODEX_CLI": str(local_cli),
                    "CODEX_CWD": str(workspace),
                },
            ):
                server = main.CodexAppServer()
                self.assertEqual(server._codex_executable(), str(local_cli.resolve()))

    async def test_typing_indicator_wraps_active_request(self) -> None:
        channel = _TypingChannel()
        async with main._typing_indicator(channel):
            self.assertTrue(channel.typing_started)
        self.assertFalse(channel.typing_started)

    async def test_first_request_names_an_existing_discord_thread(self) -> None:
        channel = SimpleNamespace(id=42, edit=AsyncMock())
        with (
            patch("theia.bot.core._is_thread", return_value=True),
            patch.object(main.bot.codex, "is_participating_thread", return_value=False),
        ):
            await main._name_new_response_thread(channel, "  Fix the\nthread   title  ")

        channel.edit.assert_awaited_once_with(name="Codex: Fix the thread title")

    async def test_existing_participating_thread_keeps_its_name(self) -> None:
        channel = SimpleNamespace(id=42, edit=AsyncMock())
        with (
            patch("theia.bot.core._is_thread", return_value=True),
            patch.object(main.bot.codex, "is_participating_thread", return_value=True),
        ):
            await main._name_new_response_thread(channel, "A later request")

        channel.edit.assert_not_awaited()

    async def test_restart_replaces_the_current_process_with_same_invocation(
        self,
    ) -> None:
        close = AsyncMock()
        with (
            patch.object(main.bot, "close", new=close),
            patch("theia.bot.core.os.execv") as execv,
            patch("theia.bot.core.sys.executable", "/usr/bin/python"),
            patch("theia.bot.core.sys.argv", ["main.py", "--test"]),
        ):
            await main._restart_in_place(delay=0)

        close.assert_awaited_once_with()
        execv.assert_called_once_with(
            "/usr/bin/python", ["/usr/bin/python", "main.py", "--test"]
        )

    @unittest.skipIf(
        os.name == "nt",
        "POSIX executable path fixture is not portable to Windows",
    )
    async def test_restart_uses_the_original_binary_for_compiled_processes(
        self,
    ) -> None:
        close = AsyncMock()
        with (
            patch.object(main.bot, "close", new=close),
            patch("theia.bot.core.os.execv") as execv,
            patch("theia.bot.core.sys.executable", "/tmp/onefile-runtime/theia"),
            patch("theia.bot.core.sys.argv", ["/opt/theia", "--test"]),
            patch("theia.bot.voice.__compiled__", object(), create=True),
        ):
            await main._restart_in_place(delay=0)

        close.assert_awaited_once_with()
        execv.assert_called_once_with("/opt/theia", ["/opt/theia", "--test"])

    def test_thread_request_detection_requires_an_explicit_request(self) -> None:
        self.assertTrue(main._user_requested_thread("Please create a thread for this."))
        self.assertTrue(main._user_requested_thread("Can you create me a thread?"))
        self.assertTrue(
            main._user_requested_thread("Could you create a Discord thread?")
        )
        self.assertTrue(main._user_requested_thread("Can you make a separate thread?"))
        self.assertTrue(
            main._user_requested_thread("Move this conversation into a thread.")
        )
        self.assertTrue(main._user_requested_thread("I want a thread for this."))
        self.assertTrue(main._user_requested_thread("Thread this request."))
        self.assertTrue(main._user_requested_thread("Keep this in its own space."))
        self.assertTrue(
            main._user_requested_thread("Split this off into a separate discussion.")
        )
        self.assertTrue(main._user_requested_thread("Give this its own conversation."))
        self.assertFalse(main._user_requested_thread("What did we say recently?"))
        self.assertFalse(main._user_requested_thread("Please do not create a thread."))
        self.assertFalse(main._user_requested_thread("How do I create a thread?"))
        self.assertFalse(
            main._user_requested_thread("Can you explain how to create a thread?")
        )

    async def test_auto_thread_stays_in_source_channel_without_explicit_request(
        self,
    ) -> None:
        source = _Channel()
        source.guild = SimpleNamespace(id=42)
        message = SimpleNamespace(
            channel=source,
            create_thread=AsyncMock(),
        )
        with patch.dict(os.environ, {"THEIA_AUTO_THREAD": "true"}):
            response_channel = await main._maybe_create_response_thread(
                message,
                "Please answer this request.",
            )

        self.assertIs(response_channel, source)
        message.create_thread.assert_not_awaited()

    async def test_auto_thread_uses_requested_thread(self) -> None:
        source = _Channel()
        source.guild = SimpleNamespace(id=42)
        thread = _Channel()
        message = SimpleNamespace(
            channel=source,
            create_thread=AsyncMock(return_value=thread),
        )
        with patch.dict(os.environ, {"THEIA_AUTO_THREAD": "true"}):
            response_channel = await main._maybe_create_response_thread(
                message,
                "Please create a thread for this request.",
            )

        self.assertIs(response_channel, thread)
        message.create_thread.assert_awaited_once_with(
            name="Codex: Please create a thread for this request.",
            auto_archive_duration=1440,
        )
        self.assertEqual(thread.sent, [])

    async def test_auto_thread_is_enabled_by_default(self) -> None:
        """Create requested response threads when no environment override exists."""
        source = _Channel()
        source.guild = SimpleNamespace(id=42)
        thread = _Channel()
        message = SimpleNamespace(
            channel=source,
            create_thread=AsyncMock(return_value=thread),
        )
        with patch.dict(os.environ, {}, clear=True):
            response_channel = await main._maybe_create_response_thread(
                message,
                "Please create a thread for this request.",
            )

        self.assertIs(response_channel, thread)
        message.create_thread.assert_awaited_once()

    async def test_requested_thread_can_be_created_from_a_slash_command_channel(
        self,
    ) -> None:
        source = _Channel()
        source.guild = SimpleNamespace(id=42)
        thread = _Channel()
        source.create_thread = AsyncMock(return_value=thread)
        with patch.dict(os.environ, {"THEIA_AUTO_THREAD": "true"}):
            response_channel = await main._maybe_create_response_thread(
                source,
                "Create a Discord thread for this request.",
            )

        self.assertIs(response_channel, thread)
        source.create_thread.assert_awaited_once_with(
            name="Codex: Create a Discord thread for this request.",
            auto_archive_duration=1440,
        )
        self.assertEqual(thread.sent, [])

    async def test_btw_requested_thread_targets_the_followup_in_that_thread(
        self,
    ) -> None:
        source = _Channel()
        source.id = 42
        source.guild = SimpleNamespace(id=99)
        thread = _Channel()
        thread.id = 43
        source.create_thread = AsyncMock(return_value=thread)
        response = SimpleNamespace(defer=AsyncMock())
        interaction = SimpleNamespace(
            id=55,
            channel=source,
            response=response,
            followup=SimpleNamespace(send=AsyncMock()),
            user=SimpleNamespace(
                id=7,
                guild_permissions=SimpleNamespace(administrator=False),
            ),
        )
        with (
            patch("theia.bot.core._require_login", new=AsyncMock(return_value=True)),
            patch(
                "theia.bot.core._is_thread",
                side_effect=lambda channel: channel is thread,
            ),
            patch("theia.bot.core._name_new_response_thread", new=AsyncMock()),
            patch("theia.bot.core.handle_request", new=AsyncMock()) as handle,
            patch.dict(os.environ, {"THEIA_AUTO_THREAD": "true"}),
        ):
            await cast(Any, main.codex_btw.callback)(
                interaction,
                "Create a thread for this request.",
            )
            await asyncio.sleep(0.05)

        handle.assert_awaited_once()
        await_args = cast(Any, handle.await_args)
        self.assertIs(await_args.kwargs["channel"], thread)
        self.assertIs(await_args.kwargs["thread"], thread)
        main.bot._participating_threads.discard(thread.id)
        main.bot.codex._discord_threads.discard(thread.id)

    async def test_account_install_btw_starts_a_normal_turn_without_threads(
        self,
    ) -> None:
        source = _Channel()
        source.id = 42
        source.guild = SimpleNamespace(id=99)
        response = SimpleNamespace(defer=AsyncMock())
        edit_original = AsyncMock(return_value=SimpleNamespace(id=1))
        interaction = SimpleNamespace(
            id=57,
            channel=source,
            response=response,
            followup=SimpleNamespace(send=AsyncMock()),
            edit_original_response=edit_original,
            user=SimpleNamespace(
                id=7,
                guild_permissions=SimpleNamespace(administrator=True),
            ),
            is_user_integration=lambda: True,
            is_guild_integration=lambda: False,
        )
        with (
            patch("theia.bot.core._require_login", new=AsyncMock(return_value=True)),
            patch(
                "theia.bot.core._maybe_create_response_thread", new=AsyncMock()
            ) as create_thread,
            patch("theia.bot.core.handle_request", new=AsyncMock()) as handle,
        ):
            await cast(Any, main.codex_btw.callback)(interaction, "start this session")
            await asyncio.sleep(0.05)

        create_thread.assert_not_awaited()
        request = cast(Any, handle.await_args).kwargs
        self.assertFalse(request["allow_tools"])
        self.assertFalse(request["allow_discord_tools"])
        self.assertIs(request["channel"], source)
        sender = request["interaction_sender"]
        await sender(content="normal turn response")
        edit_original.assert_awaited_once_with(content="normal turn response")

    async def test_account_install_login_does_not_grant_server_access(self) -> None:
        guild = SimpleNamespace(id=99)
        interaction = SimpleNamespace(
            channel=_Channel(),
            guild=guild,
            response=SimpleNamespace(defer=AsyncMock()),
            followup=SimpleNamespace(send=AsyncMock()),
            user=SimpleNamespace(
                id=7,
                guild_permissions=SimpleNamespace(administrator=True),
            ),
            is_user_integration=lambda: True,
            is_guild_integration=lambda: False,
        )
        with patch("theia.bot.core.handle_login", new=AsyncMock()) as login:
            await cast(Any, main.codex_login.callback)(interaction)

        self.assertFalse(cast(Any, login.await_args).kwargs["grant_server"])
        self.assertIsNotNone(cast(Any, login.await_args).kwargs["on_complete_send"])

    async def test_auto_thread_creation_failure_falls_back_to_source_channel(
        self,
    ) -> None:
        source = _Channel()
        source.guild = SimpleNamespace(id=42)
        message = SimpleNamespace(
            channel=source,
            create_thread=AsyncMock(side_effect=RuntimeError("thread unavailable")),
        )
        with patch.dict(os.environ, {"THEIA_AUTO_THREAD": "true"}):
            response_channel = await main._maybe_create_response_thread(
                message,
                "Create a thread for this request.",
            )

        self.assertIs(response_channel, source)
        message.create_thread.assert_awaited_once()

    async def test_bare_mention_uses_recent_context_nudge(self) -> None:
        """Route a mention without message text instead of dropping it."""
        channel = _Channel()
        author = SimpleNamespace(
            id=7,
            bot=False,
            guild_permissions=SimpleNamespace(administrator=False),
        )
        theia_user = SimpleNamespace(id=123)
        message = SimpleNamespace(
            id=88,
            channel=channel,
            author=author,
            content="<@123>",
            mentions=[theia_user],
            attachments=[],
        )
        scheduled: list[Any] = []
        with (
            patch.object(main.bot._connection, "user", theia_user),
            patch.object(main.bot.presence, "touch", new=AsyncMock()),
            patch.object(main.bot.codex, "is_authenticated", return_value=True),
            patch.object(
                main.bot,
                "schedule_request",
                side_effect=scheduled.append,
            ),
            patch(
                "theia.bot.core._message_context",
                new=AsyncMock(return_value="recent context"),
            ),
            patch("theia.bot.core.handle_request", new=AsyncMock()) as handle,
        ):
            await on_message(cast(Any, message))
            self.assertEqual(len(scheduled), 1)
            await scheduled.pop()

        handle.assert_awaited_once()
        await_args = cast(Any, handle.await_args)
        self.assertEqual(
            await_args.args[1], "Please respond to the recent conversation context."
        )
        self.assertEqual(await_args.kwargs["context"], "recent context")

    async def test_recent_channel_context_is_oldest_to_newest_and_skips_status(
        self,
    ) -> None:
        def message(
            message_id: int, author: str, content: str, *, bot: bool = False
        ) -> SimpleNamespace:
            return SimpleNamespace(
                id=message_id,
                author=SimpleNamespace(
                    id=message_id + 100,
                    display_name=author,
                    bot=bot,
                ),
                content=content,
                attachments=[],
                mentions=[],
            )

        history = _HistoryChannel(
            [
                message(1, "Alice", "first message"),
                message(2, "Theia", "-# Thinking", bot=True),
                message(3, "Bob", "most recent message"),
            ]
        )
        current = message(4, "Alice", "What did we say recently?")
        current.channel = history

        context = await main._message_context(current)

        assert context is not None
        self.assertIn("Recent messages from this Discord channel", context)
        self.assertLess(
            context.index("first message"), context.index("most recent message")
        )
        self.assertNotIn("Thinking", context)
        self.assertNotIn("What did we say recently?", context)
        self.assertIn("Alice [Discord user id:", context)
        self.assertEqual(history.history_calls[0]["limit"], 12)

    async def test_recent_channel_context_is_bounded_and_keeps_newest_messages(
        self,
    ) -> None:
        def message(message_id: int) -> SimpleNamespace:
            return SimpleNamespace(
                id=message_id,
                author=SimpleNamespace(display_name="User", bot=False),
                content=f"message {message_id}",
                attachments=[],
                mentions=[],
            )

        history = _HistoryChannel([message(item) for item in range(1, 8)])
        current = message(8)
        current.channel = history
        with patch.dict(
            os.environ,
            {"THEIA_CONTEXT_MESSAGES": "3", "THEIA_CONTEXT_MAX_CHARACTERS": "45"},
        ):
            context = await main._message_context(current)

        assert context is not None
        self.assertNotIn("message 1", context)
        self.assertIn("message 7", context)
        self.assertLessEqual(len(context.split("\n", 1)[1]), 45)

    async def test_backfill_logs_channel_id_when_history_is_forbidden(self) -> None:
        channel = _ForbiddenHistoryChannel(321)
        known_channels = dict(main.bot._known_channels)
        main.bot._known_channels.clear()
        try:
            with (
                patch.object(
                    main.bot.codex,
                    "channel_checkpoints",
                    return_value=(channel.id,),
                ),
                patch.object(
                    main.bot.codex,
                    "channel_checkpoint",
                    return_value=99,
                ),
                patch.object(main.bot, "get_channel", return_value=channel),
                self.assertLogs("theia.codex", level="INFO") as logs,
            ):
                await main.bot.backfill_after_resume()
        finally:
            main.bot._known_channels.clear()
            main.bot._known_channels.update(known_channels)

        self.assertTrue(
            any(
                "channel_id=321" in message and "error=Forbidden" in message
                for message in logs.output
            )
        )

    async def test_login_uses_cached_account(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server.refresh_account = AsyncMock()
        server.account = {"type": "chatgpt"}
        server.requires_openai_auth = True
        server._persist_state = lambda: None

        result = await server.begin_login(_Channel(), 7)

        self.assertEqual(result, {"login_cached": True})
        self.assertTrue(server.is_authenticated(7))

    async def test_login_reports_an_imported_auth_cache(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server.refresh_account = AsyncMock()
        server.account = {"type": "chatgpt"}
        server.requires_openai_auth = True
        server._auth_imported = True
        server._persist_state = lambda: None

        result = await server.begin_login(_Channel(), 7)

        self.assertEqual(result, {"login_imported": True})
        self.assertFalse(server._auth_imported)

    async def test_admin_cached_login_authorizes_the_server(self) -> None:
        server = main.CodexAppServer()
        server._ensure_running = AsyncMock()
        server.refresh_account = AsyncMock()
        server.account = {"type": "chatgpt"}
        server.requires_openai_auth = True
        server._persist_state = lambda: None

        result = await server.begin_login(
            _Channel(),
            7,
            guild_id=42,
            grant_server=True,
        )

        self.assertEqual(result, {"login_cached": True})
        self.assertTrue(server.is_authenticated(7, 42))
        self.assertTrue(server.is_authenticated(99, 42))
        self.assertFalse(server.is_authenticated(99, 43))

    def test_server_login_grant_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            with patch.dict(os.environ, {"THEIA_STATE": str(state_path)}):
                server = main.CodexAppServer()
                server.mark_server_authenticated(42)
                restarted = main.CodexAppServer()

        self.assertTrue(restarted.is_authenticated(99, 42))
        self.assertFalse(restarted.is_authenticated(99, 43))

    def test_corrupt_session_state_is_quarantined_before_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            state_path.write_text("{broken", encoding="utf-8")
            with patch.dict(os.environ, {"THEIA_STATE": str(state_path)}):
                server = main.CodexAppServer()
                backups = tuple(state_path.parent.glob("state.json.corrupt-*"))
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_text(encoding="utf-8"), "{broken")
                self.assertFalse(state_path.exists())

                server.mark_authenticated(41)
                self.assertEqual(
                    json.loads(state_path.read_text(encoding="utf-8"))[
                        "authenticated_users"
                    ],
                    [41],
                )
