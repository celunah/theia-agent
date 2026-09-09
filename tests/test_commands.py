# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable,duplicate-code
from tests.test_support import *


class CommandSurfaceTests(unittest.TestCase):
    def test_default_codex_model_is_configured(self) -> None:
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

        self.assertEqual(server.model_name(), "gpt-5.6-luna")
        self.assertEqual(server.status("model-default")["model"], "gpt-5.6-luna")

    def test_approval_level_defaults_to_high_and_accepts_configured_values(
        self,
    ) -> None:
        with patch.dict(os.environ, {"THEIA_APPROVAL_LEVEL": "medium"}):
            server = main.CodexAppServer()
        self.assertEqual(server.approval_level(), "medium")

        with patch.dict(os.environ, {"THEIA_APPROVAL_LEVEL": "unsupported"}):
            fallback = main.CodexAppServer()
        self.assertEqual(fallback.approval_level(), main.DEFAULT_APPROVAL_LEVEL)

    def test_always_admin_users_parse_and_override_discord_permissions(self) -> None:
        with patch.dict(
            os.environ,
            {main.ALWAYS_ADMIN_USERS_ENV: "42, 99,invalid,0,42"},
        ):
            self.assertEqual(
                main._configured_user_ids(main.ALWAYS_ADMIN_USERS_ENV),
                frozenset({42, 99}),
            )
            configured_user = SimpleNamespace(
                id=42,
                guild_permissions=SimpleNamespace(administrator=False),
            )
            regular_user = SimpleNamespace(
                id=7,
                guild_permissions=SimpleNamespace(administrator=False),
            )

            self.assertTrue(
                main._is_server_admin(cast(Any, configured_user), _Channel())
            )
            self.assertFalse(main._is_server_admin(cast(Any, regular_user), _Channel()))
            self.assertTrue(main._is_server_admin(cast(Any, configured_user), None))

            channel = _Channel()
            self.assertTrue(
                main.CodexAppServer._has_current_server_admin_access(
                    channel,
                    42,
                    current_user=configured_user,
                )
            )
            self.assertFalse(
                main.CodexAppServer._has_current_server_admin_access(
                    channel,
                    7,
                    current_user=regular_user,
                )
            )

    def test_environment_loads_from_compiled_executable_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "dist" / "theia"
            executable.parent.mkdir()
            dotenv_path = executable.parent / ".env"
            dotenv_path.write_text("TOKEN=private-token\n", encoding="utf-8")
            with (
                patch.object(core_module.sys, "argv", [str(executable)]),
                patch.object(
                    core_module.sys,
                    "executable",
                    str(root / ".venv" / "bin" / "python"),
                ),
                patch.object(core_module.Path, "cwd", return_value=root / "working"),
                patch.object(core_module, "load_dotenv") as load_dotenv,
            ):
                core_module._load_environment()

        load_dotenv.assert_called_once_with(dotenv_path, override=False)

    def test_embedded_build_revision_is_used_when_git_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "theia"
            package.mkdir()
            (package / "build-revision.txt").write_text("a1b2c3d\n", encoding="ascii")
            with (
                patch.object(core_module, "__file__", str(package / "core.py")),
                patch.dict(os.environ, {"THEIA_COMMIT": "stale00"}),
            ):
                self.assertEqual(core_module._theia_revision(), "a1b2c3d")

    def test_nuitka_build_embeds_the_build_revision(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                build_nuitka.importlib.util,
                "find_spec",
                return_value=object(),
            ),
            patch.object(
                build_nuitka,
                "_git_revision",
                return_value="a1b2c3d",
            ),
            patch.object(
                build_nuitka.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0),
            ) as run,
        ):
            self.assertEqual(
                build_nuitka.build_executable(Path(directory), "theia"),
                0,
            )

        command = run.call_args.args[0]
        self.assertTrue(
            any(argument.endswith("=theia/build-revision.txt") for argument in command)
        )

    def test_only_requested_slash_commands_are_registered(self) -> None:
        names = {command.name for command in main.bot.tree.get_commands()}
        self.assertEqual(
            names,
            {
                "login",
                "about",
                "usage",
                "credits",
                "approve",
                "deny",
                "stop",
                "undo",
                "btw",
                "skill",
                "personality",
                "model",
                "mode",
                "restart",
                "customize",
                "debug",
            },
        )
        self.assertEqual(main.bot.command_prefix, ())
        self.assertIsNone(main.bot.help_command)
        self.assertIs(main.CodexBot, main.TheiaBot)

    def test_commands_refer_to_codex_not_the_harness(self) -> None:
        for command in main.bot.tree.get_commands():
            self.assertNotIn("Theia", getattr(command, "description", ""))

    def test_commands_support_guild_and_account_installations(self) -> None:
        for command in main.bot.tree.get_commands():
            payload = command.to_dict(main.bot.tree)
            self.assertEqual(payload["integration_types"], [0, 1])
            self.assertEqual(payload["contexts"], [0, 1, 2])

    def test_user_only_install_keeps_regular_users_on_safe_policy(self) -> None:
        interaction = SimpleNamespace(
            user=SimpleNamespace(id=7),
            guild=SimpleNamespace(id=42),
            is_user_integration=lambda: True,
            is_guild_integration=lambda: False,
        )
        self.assertTrue(main._is_user_only_install(cast(Any, interaction)))
        with patch.dict(os.environ, {main.ALWAYS_ADMIN_USERS_ENV: ""}):
            self.assertFalse(main._interaction_allows_tools(cast(Any, interaction)))

    def test_guild_install_keeps_server_admin_tool_policy(self) -> None:
        user = SimpleNamespace(
            id=7,
            guild_permissions=SimpleNamespace(administrator=True),
        )
        interaction = SimpleNamespace(
            user=user,
            channel=SimpleNamespace(guild=SimpleNamespace(id=42)),
            guild=SimpleNamespace(id=42),
            is_user_integration=lambda: False,
            is_guild_integration=lambda: True,
        )
        self.assertFalse(main._is_user_only_install(cast(Any, interaction)))
        self.assertTrue(main._interaction_allows_tools(cast(Any, interaction)))

    def test_codex_logger_is_concise_colored_and_namespaced(self) -> None:
        logger = logging.getLogger("theia.codex")
        self.assertEqual(logger.name, "theia.codex")
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(
            sum(
                getattr(handler, "_theia_codex_handler", False)
                for handler in logger.handlers
            ),
            1,
        )
        handler = next(
            handler
            for handler in logger.handlers
            if getattr(handler, "_theia_codex_handler", False)
        )
        record = logger.makeRecord(
            logger.name,
            logging.INFO,
            __file__,
            1,
            "Codex turn started",
            (),
            None,
        )
        self.assertIn("\x1b[", handler.format(record))

    def test_base_priors_are_identity_neutral(self) -> None:
        self.assertNotIn("Codex", main.BASE_PRIORS)
        self.assertNotIn("Theia", main.BASE_PRIORS)

    def test_base_priors_make_ordinary_conversation_spoken_first(self) -> None:
        self.assertIn("spoken-first delivery", main.BASE_PRIORS)
        self.assertIn("acknowledge\nthe user's request directly", main.BASE_PRIORS)
        self.assertIn("one thought at a time", main.BASE_PRIORS)
        self.assertIn("short,\nnatural paragraphs", main.BASE_PRIORS)
        self.assertIn("without filler, forced slang", main.BASE_PRIORS)
        self.assertIn("headings, and lists in ordinary conversation", main.BASE_PRIORS)

    def test_base_priors_keep_technical_answers_complete(self) -> None:
        self.assertIn("For code,\nreviews, procedures", main.BASE_PRIORS)
        self.assertIn("explicit requests for detail", main.BASE_PRIORS)
        self.assertIn("expand as needed", main.BASE_PRIORS)
        self.assertIn(
            "preserve\nimportant facts and complete reasoning", main.BASE_PRIORS
        )

    def test_medium_is_the_non_adaptive_default(self) -> None:
        self.assertEqual(main.DEFAULT_REASONING_EFFORT, "medium")

    def test_codex_stdio_limit_allows_large_restored_thread_events(self) -> None:
        with patch.dict(
            os.environ,
            {"THEIA_CODEX_STDIO_LIMIT": str(main.MAX_CODEX_STDIO_LIMIT * 2)},
        ):
            server = main.CodexAppServer()

        self.assertEqual(server._stdio_limit, main.MAX_CODEX_STDIO_LIMIT)

    def test_codex_child_environment_excludes_theia_and_provider_secrets(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TOKEN": "discord-secret",
                "STT_TOKEN": "stt-secret",
                "TTS_TOKEN": "tts-secret",
                "OPENAI_API_KEY": "openai-secret",
                "CODEX_API_KEY": "codex-secret",
            },
        ):
            server = main.CodexAppServer()

        for name in (
            "TOKEN",
            "STT_TOKEN",
            "TTS_TOKEN",
            "OPENAI_API_KEY",
            "CODEX_API_KEY",
        ):
            self.assertNotIn(name, server._codex_environment)
        self.assertEqual(
            server._codex_environment["CODEX_HOME"], str(server._codex_home)
        )

    def test_thread_name_is_compact_and_single_line(self) -> None:
        name = main._thread_name("  Review the\nnew   gateway behavior  ")

        self.assertEqual(name, "Codex: Review the new gateway behavior")
        self.assertLessEqual(len(main._thread_name("x" * 200)), 100)

    def test_app_server_uses_private_codex_home_for_installs(self) -> None:
        private_home = Path("/tmp/codex-discord-private")
        global_home = Path("/tmp/codex-global")
        with patch.dict(
            os.environ,
            {
                "CODEX_DISCORD_HOME": str(private_home),
                "CODEX_HOME": str(global_home),
                "CODEX_MEMORY_ROOTS": "",
                "CODEX_SKILL_ROOTS": "",
            },
        ):
            server = main.CodexAppServer()

        self.assertEqual(server._codex_home, private_home.resolve())
        self.assertEqual(
            server._codex_environment["CODEX_HOME"], str(private_home.resolve())
        )
        self.assertIn(
            private_home.resolve() / "memories" / "hermes", server._memory_roots
        )
        self.assertIn(global_home.resolve() / "skills", server._skill_roots)
        self.assertNotIn(
            global_home.resolve() / "skills", server._shared_workspace_roots
        )

    def test_theia_identity_and_home_overrides(self) -> None:
        private_home = Path("/tmp/theia-test-home")
        state_path = Path("/tmp/theia-test-state.json")
        with patch.dict(
            os.environ,
            {
                "THEIA_HOME": str(private_home),
                "THEIA_STATE": str(state_path),
                "CODEX_DISCORD_HOME": "/tmp/legacy-home",
                "CODEX_DISCORD_STATE": "/tmp/legacy-state.json",
            },
        ):
            server = main.CodexAppServer()

        self.assertEqual(main.AGENT_NAME, "Theia")
        self.assertEqual(main.AGENT_DISPLAY_NAME, "Theia Agent")
        self.assertEqual(server._codex_home, private_home.resolve())
        self.assertEqual(server._state_path, state_path)

    def test_theia_home_migrates_legacy_runtime_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_home = root / "theia"
            legacy_home = root / "legacy"
            legacy_home.mkdir()
            (legacy_home / "auth.json").write_text("auth", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(private_home),
                    "CODEX_DISCORD_HOME": str(legacy_home),
                },
            ):
                server = main.CodexAppServer()
                private_home.mkdir(exist_ok=True)
                server._migrate_legacy_home()

            self.assertEqual(
                (private_home / "auth.json").read_text(encoding="utf-8"), "auth"
            )
            self.assertEqual(
                (legacy_home / "auth.json").read_text(encoding="utf-8"), "auth"
            )

    def test_private_home_bootstraps_existing_global_auth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_home = root / "private"
            global_home = root / "global"
            global_home.mkdir()
            auth = '{"auth_mode":"chatgpt","tokens":{"access_token":"private-test"}}'
            (global_home / "auth.json").write_text(auth, encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "CODEX_DISCORD_HOME": str(private_home),
                    "CODEX_HOME": str(global_home),
                },
            ):
                server = main.CodexAppServer()
                private_home.mkdir(exist_ok=True)
                self.assertTrue(server._import_global_auth())

            self.assertEqual(
                (private_home / "auth.json").read_text(encoding="utf-8"), auth
            )
            self.assertEqual(
                (global_home / "auth.json").read_text(encoding="utf-8"), auth
            )

    def test_invalid_private_auth_can_be_replaced_by_global_auth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_home = root / "private"
            global_home = root / "global"
            private_home.mkdir()
            global_home.mkdir()
            (private_home / "auth.json").write_text("invalid", encoding="utf-8")
            auth = '{"auth_mode":"chatgpt","tokens":{"access_token":"global"}}'
            (global_home / "auth.json").write_text(auth, encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "CODEX_DISCORD_HOME": str(private_home),
                    "CODEX_HOME": str(global_home),
                },
            ):
                server = main.CodexAppServer()
                self.assertTrue(server._import_global_auth(force=True))

            self.assertEqual(
                (private_home / "auth.json").read_text(encoding="utf-8"), auth
            )

    @unittest.skipIf(
        os.name == "nt",
        "POSIX file permission semantics are not enforced on Windows",
    )
    def test_private_auth_file_uses_restricted_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_home = root / "private"
            global_home = root / "global"
            global_home.mkdir()
            auth = '{"auth_mode":"chatgpt","tokens":{"access_token":"private-test"}}'
            (global_home / "auth.json").write_text(auth, encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "CODEX_DISCORD_HOME": str(private_home),
                    "CODEX_HOME": str(global_home),
                },
            ):
                server = main.CodexAppServer()
                private_home.mkdir(exist_ok=True)
                server._import_global_auth()

            self.assertEqual(
                (private_home / "auth.json").stat().st_mode & 0o777,
                0o600,
            )

    def test_private_home_defaults_codex_web_search_to_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            private_home = Path(directory) / "private"
            with patch.dict(
                os.environ,
                {"THEIA_HOME": str(private_home), "THEIA_WEB_SEARCH": ""},
            ):
                server = main.CodexAppServer()
                private_home.mkdir(exist_ok=True)
                server._ensure_web_search_config()

            self.assertEqual(
                (private_home / "config.toml").read_text(encoding="utf-8"),
                'web_search = "indexed"\n',
            )

    @unittest.skipIf(
        os.name == "nt",
        "POSIX file permission semantics are not enforced on Windows",
    )
    def test_private_config_file_uses_restricted_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            private_home = Path(directory) / "private"
            with patch.dict(
                os.environ,
                {"THEIA_HOME": str(private_home), "THEIA_WEB_SEARCH": ""},
            ):
                server = main.CodexAppServer()
                private_home.mkdir(exist_ok=True)
                server._ensure_web_search_config()

            self.assertEqual(
                (private_home / "config.toml").stat().st_mode & 0o777,
                0o600,
            )

    def test_explicit_codex_web_search_mode_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            private_home = Path(directory) / "private"
            private_home.mkdir()
            config = private_home / "config.toml"
            original = (
                'web_search = "indexed"\n\n'
                '[projects."/tmp/project"]\n'
                'trust_level = "trusted"\n'
            )
            config.write_text(original, encoding="utf-8")
            with patch.dict(
                os.environ,
                {"THEIA_HOME": str(private_home), "THEIA_WEB_SEARCH": ""},
            ):
                server = main.CodexAppServer()
                server._ensure_web_search_config()

            self.assertEqual(config.read_text(encoding="utf-8"), original)

    def test_web_search_environment_override_updates_private_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            private_home = Path(directory) / "private"
            private_home.mkdir()
            config = private_home / "config.toml"
            config.write_text(
                'web_search = "live"\n\n[projects."/tmp/project"]\n',
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"THEIA_HOME": str(private_home), "THEIA_WEB_SEARCH": "disabled"},
            ):
                server = main.CodexAppServer()
                server._ensure_web_search_config()

            self.assertTrue(
                config.read_text(encoding="utf-8").startswith(
                    'web_search = "disabled"\n'
                )
            )

    def test_change_status_requires_completed_event_and_known_root(self) -> None:
        memory = (Path("/shared/memories"),)
        skills = (Path("/shared/skills"),)
        changed = {
            "type": "fileChange",
            "status": "completed",
            "changes": [
                {"path": "/shared/memories/MEMORY.md", "kind": {"type": "add"}},
                {"path": "/shared/skills/demo/SKILL.md", "kind": {"type": "update"}},
            ],
        }
        self.assertEqual(
            main._verified_change_status(changed, memory, skills),
            ["Memory created", "Skill updated"],
        )
        changed["status"] = "inProgress"
        self.assertEqual(main._verified_change_status(changed, memory, skills), [])

    def test_symbol_only_intermediate_text_is_omitted(self) -> None:
        self.assertEqual(main._safe_intermediate_text("  ***  •  —  …  "), "")
        self.assertEqual(
            main._safe_intermediate_text("Inspecting the project — …"),
            "Inspecting the project — …",
        )

    def test_mentions_are_converted_to_prompts(self) -> None:
        self.assertEqual(main._mention_prompt("<@123> hello there", 123), "hello there")
        self.assertEqual(
            main._mention_prompt("<@!123> hello there", 123), "hello there"
        )

    def test_error_reason_hides_protocol_prefix_and_paths(self) -> None:
        reason = main._safe_error_reason(
            "Codex turn/start failed: service unavailable at /tmp/private/result.json"
        )
        self.assertEqual(reason, "service unavailable at")
        self.assertEqual(
            main._safe_error_reason("Run `/login` first."), "Run `/login` first."
        )

    def test_error_reason_preserves_nested_codex_status_details(self) -> None:
        reason = main._safe_error_reason(
            {
                "message": "Request failed",
                "codexErrorInfo": {
                    "httpStatusCode": 404,
                    "statusText": "Not Found",
                },
            }
        )
        self.assertEqual(reason, "Request failed; 404 Not Found")

    def test_error_reason_redacts_credentials_while_preserving_status(self) -> None:
        reason = main._safe_error_reason(
            "Request failed: HTTP 404 Not Found api_key=secret-value"
        )
        self.assertIn("404 Not Found", reason)
        self.assertNotIn("secret-value", reason)

    def test_credits_use_named_limits(self) -> None:
        embed = main._credits_embed(
            {
                "rateLimits": {
                    "credits": {"balance": 12},
                    "primary": {"usedPercent": 10, "resetsAt": 0},
                    "secondary": {"usedPercent": 20, "resetsAt": 0},
                }
            }
        )
        self.assertEqual(
            {field.name for field in embed.fields},
            {"Balance", "Status", "5-hour limit", "Weekly limit"},
        )

    def test_usage_and_credit_field_labels_are_customizable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = main.FrontendCustomizationStore(Path(directory) / "frontend.json")
            for target, value in (
                ("usage_lifetime_tokens", "Lifetime"),
                ("usage_peak_daily_tokens", "Daily peak"),
                ("usage_current_streak", "Current"),
                ("usage_longest_streak", "Longest"),
                ("usage_longest_running_turn", "Slowest"),
                ("credits_balance", "Credits"),
                ("credits_status", "State"),
                ("credits_five_hour_limit", "Five hour"),
                ("credits_weekly_limit", "Seven day"),
            ):
                store.set(42, target, "label", value)
            channel = SimpleNamespace(
                id=7, guild=SimpleNamespace(id=42, name="Example")
            )
            with patch.object(main.bot, "customizations", store):
                usage = main._usage_embed(
                    {
                        "summary": {
                            "lifetimeTokens": 1,
                            "peakDailyTokens": 2,
                            "currentStreakDays": 3,
                            "longestStreakDays": 4,
                            "longestRunningTurnSec": 5,
                        }
                    },
                    channel=channel,
                )
                credit_embed = main._credits_embed(
                    {
                        "rateLimits": {
                            "credits": {"balance": 12},
                            "primary": {"usedPercent": 10, "resetsAt": 0},
                            "secondary": {"usedPercent": 20, "resetsAt": 0},
                        }
                    },
                    channel=channel,
                )

        self.assertEqual(
            [field.name for field in usage.fields],
            ["Lifetime", "Daily peak", "Current", "Longest", "Slowest"],
        )
        self.assertEqual(
            [field.name for field in credit_embed.fields],
            ["Credits", "State", "Five hour", "Seven day"],
        )

    def test_usage_reports_only_claimed_theia_thread_tokens(self) -> None:
        server = main.CodexAppServer()
        server._persist_state = lambda: None
        server._usage_threads.clear()
        server._usage_daily.clear()
        server._usage_tracked_since = None
        session = server._session("discord-user")
        session.thread_id = "theia-thread"
        server._claim_usage_thread("theia-thread")
        second_session = server._session("another-discord-user")
        second_session.thread_id = "another-theia-thread"
        server._claim_usage_thread("another-theia-thread")

        server._handle_notification(
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "threadId": "theia-thread",
                    "tokenUsage": {
                        "total": {
                            "inputTokens": 30,
                            "outputTokens": 12,
                            "totalTokens": 42,
                        }
                    },
                },
            }
        )
        server._handle_notification(
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "threadId": "another-theia-thread",
                    "tokenUsage": {"total": {"totalTokens": 8}},
                },
            }
        )
        server._handle_notification(
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "threadId": "account-wide-thread",
                    "tokenUsage": {"total": {"totalTokens": 9_900_000_000}},
                },
            }
        )

        usage = asyncio.run(server.usage())

        self.assertEqual(usage["scope"], "theia")
        self.assertEqual(usage["summary"]["lifetimeTokens"], 50)
        self.assertEqual(usage["summary"]["totalCumulativeTokens"], 50)
        self.assertEqual(usage["summary"]["peakDailyTokens"], 50)

    def test_usage_embed_labels_theia_scope(self) -> None:
        embed = main._usage_embed(
            {
                "scope": "theia",
                "summary": {
                    "lifetimeTokens": 42,
                    "totalCumulativeTokens": 42,
                    "peakDailyTokens": 42,
                    "currentStreakDays": 1,
                    "longestStreakDays": 1,
                    "longestRunningTurnSec": 2,
                },
            }
        )

        self.assertEqual(
            embed.description, "Usage tracked from Theia's conversation threads."
        )
        self.assertEqual(embed.fields[0].name, "Total cumulative tokens")
        self.assertEqual(embed.fields[0].value, "42")

    def test_usage_embed_rounds_longest_running_turn_to_whole_seconds(self) -> None:
        embed = main._usage_embed(
            {
                "summary": {
                    "lifetimeTokens": 42,
                    "longestRunningTurnSec": 12.75,
                }
            }
        )

        self.assertEqual(embed.fields[-1].value, "13 seconds")

    def test_personality_autocomplete_uses_available_profiles(self) -> None:
        with patch.object(
            main.bot.codex, "personality_names", return_value=("calm", "formal")
        ):
            choices = asyncio.run(
                main.personality_autocomplete(SimpleNamespace(), "cal")
            )
        self.assertEqual(
            [(choice.name, choice.value) for choice in choices], [("calm", "calm")]
        )

    def test_personality_summary_extracts_bounded_character_card_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_root = root / "personalities"
            profile_root.mkdir()
            (profile_root / "cel.md").write_text(
                "# Celune\n\n"
                "You are Celune, a calm and curious lunar guardian.\n\n"
                "Speak with a warm, observant response style.\n",
                encoding="utf-8",
            )

            summary = main.PersonalityStore(root).summary("cel")

        self.assertEqual(summary.name, "cel")
        self.assertEqual(summary.identifier, "cel")
        self.assertEqual(summary.character_name, "Celune")

    def test_personality_summary_embed_has_default_and_customized_fields(self) -> None:
        summary = {
            "name": "cel",
            "identifier": "cel",
            "character_name": "Celune",
            "description": "A calm lunar guardian.",
            "known_entries": 12,
            "known_users": 3,
            "scope": "me",
            "set_by": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            store = main.FrontendCustomizationStore(Path(directory) / "frontend.json")
            for target, value in (
                ("personality_known_entries", "Lore"),
                ("personality_known_users", "People"),
                ("personality_scope", "Where"),
                ("personality_set_by", "Author"),
                ("personality_mood", "Affect"),
                ("personality_presence", "Activity"),
                ("personality_footer", "Use {command} to change {character_name}"),
            ):
                store.set(42, target, "label", value)
            channel = SimpleNamespace(id=7, guild=SimpleNamespace(id=42))
            with patch.object(main.bot, "customizations", store):
                embed = main._personality_summary_embed(
                    summary,
                    {"label": "neutral", "strength": 0.5},
                    "watching the moon",
                    channel=channel,
                    user=cast(Any, SimpleNamespace(name="Luna", id=9)),
                )

        self.assertEqual(embed.title, "Celune (cel)")
        self.assertEqual(embed.description, "A calm lunar guardian.")
        self.assertEqual(
            [(field.name, field.value) for field in embed.fields],
            [
                ("Lore", "12"),
                ("Where", "me"),
                ("Author", "Legacy selection"),
                ("People", "3"),
                ("Affect", "Neutral (50%)"),
                ("Activity", "watching the moon"),
            ],
        )
        self.assertEqual(embed.footer.text, "Use /personality to change Celune")

    def test_model_autocomplete_uses_codex_models(self) -> None:
        with patch.object(
            main.bot.codex,
            "available_models",
            new=AsyncMock(
                return_value=(
                    {"id": "gpt-test", "name": "Test model"},
                    {"id": "other"},
                )
            ),
        ):
            choices = asyncio.run(main.model_autocomplete(SimpleNamespace(), "test"))
        self.assertEqual(
            [(choice.name, choice.value) for choice in choices],
            [("Test model (gpt-test)", "gpt-test")],
        )

    def test_frontend_customization_renders_templates_per_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = main.FrontendCustomizationStore(Path(directory) / "frontend.json")
            store.set(
                42,
                "/usage",
                "title",
                "Usage for {server}",
            )
            store.set(42, "usage", "content", "Balance for {user}: {text}")

            self.assertEqual(
                store.render(
                    42,
                    "command:usage",
                    "title",
                    "Usage",
                    context={"server": "Example", "user": "Alice"},
                ),
                "Usage for Example",
            )
            self.assertEqual(
                store.render(
                    42,
                    "usage",
                    "content",
                    "Usage tracked from Theia's conversation threads.",
                    context={"server": "Example", "user": "Alice"},
                ),
                "Balance for Alice: Usage tracked from Theia's conversation threads.",
            )
            self.assertEqual(
                store.render(7, "usage", "title", "Usage"),
                "Usage",
            )

    def test_frontend_customization_persists_separately_and_can_reset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frontend.json"
            store = main.FrontendCustomizationStore(path)
            store.set(42, "label:thinking", "label", "Working")
            self.assertEqual(store.color(42, "label:thinking", 0x123456), 0x123456)

            restarted = main.FrontendCustomizationStore(path)
            self.assertEqual(
                restarted.render(42, "thinking", "label", "Thinking"),
                "Working",
            )
            _, _, reset = restarted.set(42, "thinking", "label", "default")
            self.assertTrue(reset)
            self.assertEqual(
                restarted.render(42, "thinking", "label", "Thinking"),
                "Thinking",
            )

    def test_frontend_customization_validates_placeholders_and_colors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = main.FrontendCustomizationStore(Path(directory) / "frontend.json")
            with self.assertRaisesRegex(main.CustomizationError, "Unknown placeholder"):
                store.set(42, "usage", "title", "{secret}")
            with self.assertRaisesRegex(main.CustomizationError, "hex value"):
                store.set(42, "usage", "color", "not-a-color")
            store.set(42, "usage", "color", "#000000")
            self.assertEqual(store.color(42, "usage", 0xFFFFFF), 0)

    def test_corrupt_frontend_customization_is_quarantined_before_recovery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frontend.json"
            path.write_text("{broken", encoding="utf-8")

            store = main.FrontendCustomizationStore(path)

            backups = tuple(path.parent.glob("frontend.json.corrupt-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), "{broken")
            self.assertFalse(path.exists())

            store.set(42, "usage", "title", "Recovered")
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["guilds"]["42"][
                    "command:usage"
                ]["title"],
                "Recovered",
            )

    def test_customization_autocomplete_includes_commands_and_labels(self) -> None:
        choices = asyncio.run(
            main.customization_target_autocomplete(SimpleNamespace(), "think")
        )
        self.assertEqual(
            [(choice.name, choice.value) for choice in choices],
            [("Label: Thinking", "label:thinking")],
        )

    def test_customization_lists_all_frontend_elements(self) -> None:
        expected = {
            name
            for group in (
                (
                    "choose_button",
                    "approve_button",
                    "deny_button",
                    "previous_button",
                    "next_button",
                    "other_button",
                    "answer_button",
                    "decline_button",
                ),
                ("input_modal_title", "json_response", "text_input_label"),
                (
                    "login_verification_link",
                    "login_code",
                    "login_visibility_footer",
                ),
                (
                    "usage_lifetime_tokens",
                    "usage_peak_daily_tokens",
                    "usage_current_streak",
                    "usage_longest_streak",
                    "usage_longest_running_turn",
                ),
                (
                    "credits_balance",
                    "credits_status",
                    "credits_five_hour_limit",
                    "credits_weekly_limit",
                ),
                (
                    "about_theia_agent",
                    "about_codex_cli",
                    "about_account",
                    "about_plan",
                    "about_mode",
                    "about_personality",
                ),
                (
                    "personality_known_entries",
                    "personality_known_users",
                    "personality_mood",
                    "personality_presence",
                    "personality_footer",
                ),
                (
                    "debug_runtime",
                    "debug_configuration",
                    "debug_session",
                    "debug_counts",
                    "debug_usage",
                    "debug_live_footer",
                    "debug_stop_updates",
                ),
                ("image_follow_up",),
            )
            for name in group
        }
        self.assertTrue(expected.issubset(set(main.LABEL_TARGETS)))

    def test_frontend_embed_customization_does_not_change_default_without_server(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = main.FrontendCustomizationStore(Path(directory) / "frontend.json")
            store.set(42, "usage", "title", "Custom usage")
            default = main._command_embed(
                "Usage",
                "Usage tracked from Theia's conversation threads.",
                target="command:usage",
                guild_id=7,
                customizer=store,
            )
            customized = main._command_embed(
                "Usage",
                "Usage tracked from Theia's conversation threads.",
                target="command:usage",
                guild_id=42,
                customizer=store,
            )
            store.set(42, "request_failed", "label", "Failure")
            label_customized = main._command_embed(
                "Request failed",
                "The request failed.",
                target="label:request_failed",
                guild_id=42,
                customizer=store,
            )

        self.assertEqual(default.title, "Usage")
        self.assertEqual(customized.title, "Custom usage")
        self.assertEqual(label_customized.title, "Failure")

    def test_about_embed_contains_only_the_requested_runtime_details(self) -> None:
        user = SimpleNamespace(name="username", mention="<@123456789>")
        with patch("theia.bot.core._theia_revision", return_value="a1b2c3d"):
            embed = main._about_embed(
                account={"planType": "plus"},
                cli_version="0.153.0",
                mode="text",
                personality="Cel",
                user=cast(Any, user),
            )

        self.assertEqual(embed.title, "About Theia")
        self.assertEqual(
            [(field.name, field.value) for field in embed.fields],
            [
                ("Theia Agent", "1.0.2 (a1b2c3d)"),
                ("Codex CLI", "0.153.0"),
                ("Account", "@username"),
                ("Plan", "Plus ($20/mo)"),
                ("Mode", "Text"),
                ("Personality", "Cel"),
            ],
        )

    def test_about_embed_customizes_field_labels(self) -> None:
        user = SimpleNamespace(name="username", mention="<@123456789>")
        with tempfile.TemporaryDirectory() as directory:
            store = main.FrontendCustomizationStore(Path(directory) / "frontend.json")
            for target, value in (
                ("about_theia_agent", "Agent"),
                ("about_codex_cli", "CLI"),
                ("about_account", "User"),
                ("about_plan", "Subscription"),
                ("about_mode", "Interaction"),
                ("about_personality", "Style"),
            ):
                store.set(42, target, "label", value)
            with patch.object(main.bot, "customizations", store):
                embed = main._about_embed(
                    account={"planType": "plus"},
                    cli_version="0.153.0",
                    mode="text",
                    personality="Cel",
                    channel=SimpleNamespace(
                        id=7, guild=SimpleNamespace(id=42, name="Example")
                    ),
                    user=cast(Any, user),
                )

        self.assertEqual(
            [field.name for field in embed.fields],
            ["Agent", "CLI", "User", "Subscription", "Interaction", "Style"],
        )

    def test_pagination_buttons_use_frontend_customization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = main.FrontendCustomizationStore(Path(directory) / "frontend.json")
            store.set(42, "previous_button", "label", "Back")
            store.set(42, "next_button", "label", "Forward")
            view = main._PaginatorView(
                ["first", "second"],
                owner_id=7,
                customizer=store,
                guild_id=42,
            )

        self.assertEqual(
            [getattr(item, "label", None) for item in view.children],
            ["Back", "Forward"],
        )

    def test_interaction_views_use_restart_safe_component_ids(self) -> None:
        views = (
            main._PaginatorView(["first", "second"], owner_id=7),
            main._ImageResultView(
                7,
                (Path("/tmp/generated.png"),),
                on_action=AsyncMock(),
            ),
            main._DecisionView(
                7,
                [("Approve", "accept", discord.ButtonStyle.success)],
            ),
            main._DebugView(7),
            main._FormView(7, prompt="Provide JSON"),
            main._UserInputView(
                7,
                [{"id": "answer", "question": "Answer", "options": [{"label": "Yes"}]}],
            ),
        )

        for view in views:
            self.assertTrue(view.children)
            self.assertTrue(
                all(
                    str(getattr(item, "custom_id", "")).startswith("theia:")
                    for item in view.children
                )
            )

    def test_persistent_view_store_restores_a_paginator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "views.json"
            store = _PersistentViewStore(path)
            view = main._PaginatorView(["first", "second"], owner_id=7)
            store.register(view, SimpleNamespace(id=123))

            restored_store = _PersistentViewStore(path)
            restored: list[Any] = []

            def add_view(candidate: Any, *, message_id: int) -> None:
                restored.append((candidate, message_id))

            fake_bot = SimpleNamespace(
                add_view=add_view,
            )
            restored_store.restore(
                cast(Any, fake_bot),
                main.bot._restore_persistent_view,
            )

        self.assertEqual(len(restored), 1)
        restored_view, message_id = restored[0]
        self.assertEqual(message_id, 123)
        self.assertTrue(restored_view.is_persistent())
        self.assertEqual(restored_view.persistence_token, view.persistence_token)

    def test_input_modals_use_frontend_customization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = main.FrontendCustomizationStore(Path(directory) / "frontend.json")
            store.set(42, "input_modal_title", "label", "Your input")
            store.set(42, "json_response", "label", "Payload")
            store.set(42, "text_input_label", "label", "Response")
            store.set(42, "choose_button", "label", "Select")
            channel = SimpleNamespace(guild=SimpleNamespace(id=42))
            form = main._FormView(
                7,
                prompt="Provide JSON",
                channel=cast(Any, channel),
                customizer=store,
            )
            json_modal = main._JsonModal(
                form,
                7,
                title="Codex input",
                prompt="Provide JSON",
            )
            user_input = main._UserInputView(
                7,
                [
                    {
                        "id": "details",
                        "header": "Details",
                        "question": "Explain",
                        "options": [{}],
                    }
                ],
                channel=cast(Any, channel),
                customizer=store,
            )
            text_modal = main._TextModal(
                user_input,
                7,
                user_input.current_question,
            )

        self.assertEqual(json_modal.title, "Your input")
        self.assertEqual(json_modal.value.to_component_dict()["label"], "Payload")
        self.assertEqual(text_modal.title, "Your input")
        self.assertEqual(text_modal.answer.to_component_dict()["label"], "Response")
        self.assertEqual(getattr(user_input.children[0], "label", None), "Select")

    def test_prompt_modal_is_shared_by_follow_up_flows(self) -> None:
        modal = main._PromptModal(
            7,
            on_submit=AsyncMock(),
            channel=SimpleNamespace(guild=SimpleNamespace(id=42)),
            title="Send a request",
        )

        self.assertEqual(modal.title, "Send a request")
        self.assertEqual(modal.prompt.to_component_dict()["label"], "Request")

    def test_codex_cli_version_is_read_from_the_selected_executable(self) -> None:
        server = main.CodexAppServer()
        with (
            patch.object(server, "_codex_executable", return_value="/tmp/codex"),
            patch(
                "theia.server.lifecycle.subprocess.run",
                return_value=SimpleNamespace(stdout="codex-cli 0.153.0\n", stderr=""),
            ) as run,
        ):
            self.assertEqual(server.codex_cli_version(), "0.153.0")

        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["/tmp/codex", "--version"])
