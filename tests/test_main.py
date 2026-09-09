import asyncio
import json
import logging
import os
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import discord
from typing_extensions import Self

import main
from scripts import build_nuitka
from scripts.configure import (
    VOICE_MODE,
    TEXT_MODE,
    ConfigurationError,
    collect_configuration,
    save_configuration,
    validate_configuration,
)
from theia import core as core_module
from theia.bot import _handle_voice_transcript, on_message
from theia.core import _path_is_under


class _Channel:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.id: Any = None
        self.guild: Any = None
        self.create_thread: Any = None
        self.edit: Any = None

    async def send(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(id=len(self.sent))


class _FailingSendChannel(_Channel):
    async def send(self, *_args, **_kwargs):
        raise discord.DiscordException("permission revoked")


def _admin_guild(user_id: int = 7, administrator: bool = True) -> Any:
    member = SimpleNamespace(
        id=user_id,
        guild_permissions=SimpleNamespace(administrator=administrator),
    )
    return SimpleNamespace(
        id=1,
        get_member=lambda candidate: member if candidate == user_id else None,
    )


MOOD_TEST_NAMESPACE = f"mood-test-{time.time_ns()}"


def _mood_test_key(name: str) -> str:
    return f"{MOOD_TEST_NAMESPACE}:{name}"


class _TypingContext:
    def __init__(self, channel: "_TypingChannel") -> None:
        self.channel = channel

    async def __aenter__(self):
        self.channel.typing_started = True

    async def __aexit__(self, *_args):
        self.channel.typing_started = False


class _TypingChannel(_Channel):
    def __init__(self) -> None:
        super().__init__()
        self.typing_started = False

    def typing(self) -> _TypingContext:
        return _TypingContext(self)


class _Message:
    id = 100

    def __init__(self) -> None:
        self.edits: list[dict] = []
        self.reactions: list[str] = []
        self.deleted = False

    async def edit(self, **kwargs):
        self.edits.append(kwargs)

    async def add_reaction(self, emoji: str) -> None:
        self.reactions.append(emoji)

    async def delete(self) -> None:
        self.deleted = True


class _ImageMessage(_Message):
    def __init__(self) -> None:
        super().__init__()
        self.attachments = (SimpleNamespace(url="https://cdn.example/image.png"),)


class _HistoryChannel(_Channel):
    def __init__(self, messages: list[SimpleNamespace]) -> None:
        super().__init__()
        self.messages = messages
        self.history_calls: list[dict] = []

    def history(self, *, limit: int, before=None):
        self.history_calls.append({"limit": limit, "before": before})
        eligible = [
            item for item in self.messages if before is None or item.id < before.id
        ]

        async def iterator():
            for item in reversed(eligible[-limit:]):
                yield item

        return iterator()


class _ForbiddenHistoryIterator:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise discord.Forbidden(
            SimpleNamespace(status=403, reason="Forbidden"), "forbidden"
        )


class _ForbiddenHistoryChannel(_Channel):
    def __init__(self, channel_id: int) -> None:
        super().__init__()
        self.id = channel_id

    def history(self, *, limit: int, after=None):  # pylint: disable=unused-argument
        return _ForbiddenHistoryIterator()


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
                ("image_follow_up", "image_remove_background", "image_download"),
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
        with patch("theia.bot._theia_revision", return_value="a1b2c3d"):
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
                "theia.app_server.subprocess.run",
                return_value=SimpleNamespace(stdout="codex-cli 0.153.0\n", stderr=""),
            ) as run,
        ):
            self.assertEqual(server.codex_cli_version(), "0.153.0")

        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["/tmp/codex", "--version"])


class ConfigurationScriptTests(unittest.TestCase):
    def test_text_setup_only_requests_the_discord_token(self) -> None:
        prompts: list[str] = []
        output: list[str] = []
        values = collect_configuration(
            input_fn=lambda prompt: prompts.append(prompt) or "",
            secret_input_fn=lambda _prompt: "discord-token",
            output_fn=output.append,
        )

        self.assertEqual(values.mode, TEXT_MODE)
        self.assertEqual(
            values.as_environment(),
            {"TOKEN": "discord-token", "THEIA_DEFAULT_MODE": TEXT_MODE},
        )
        self.assertEqual(prompts, ["Mode [1/text]: "])
        self.assertNotIn("discord-token", "\n".join(output))

    def test_voice_setup_requests_both_audio_services(self) -> None:
        prompts: list[str] = []
        inputs = iter(
            [
                "2",
                "https://stt.example/v1",
                "https://tts.example/v1",
                "local-whisper",
                "local-tts",
                "voice-one",
                "wav",
            ]
        )
        secrets = iter(["discord-token", "stt-token", "tts-token"])
        values = collect_configuration(
            input_fn=lambda prompt: prompts.append(prompt) or next(inputs),
            secret_input_fn=lambda _prompt: next(secrets),
            output_fn=lambda _message: None,
        )

        self.assertEqual(values.mode, VOICE_MODE)
        self.assertEqual(
            values.as_environment(),
            {
                "TOKEN": "discord-token",
                "THEIA_DEFAULT_MODE": VOICE_MODE,
                "STT_BASE_URL": "https://stt.example/v1",
                "STT_TOKEN": "stt-token",
                "STT_MODEL": "local-whisper",
                "TTS_BASE_URL": "https://tts.example/v1",
                "TTS_TOKEN": "tts-token",
                "TTS_MODEL": "local-tts",
                "TTS_VOICE": "voice-one",
                "TTS_FORMAT": "wav",
            },
        )
        self.assertEqual(
            prompts,
            [
                "Mode [1/text]: ",
                "STT URL: ",
                "TTS URL: ",
                "STT model [whisper-1]: ",
                "TTS model [tts-1]: ",
                "TTS voice [alloy]: ",
                "TTS format [mp3]: ",
            ],
        )

    def test_setup_preserves_unrelated_dotenv_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# Keep this setting\nCUSTOM_SETTING=preserve-me\n"
                "TOKEN=old-token\nSTT_BASE_URL=https://old.example\n",
                encoding="utf-8",
            )
            values = validate_configuration(
                discord_token="new-token",
                mode=TEXT_MODE,
            )
            save_configuration(values, path=path)
            contents = path.read_text(encoding="utf-8")

        self.assertIn("CUSTOM_SETTING=preserve-me", contents)
        self.assertIn('TOKEN="new-token"', contents)
        self.assertIn('THEIA_DEFAULT_MODE="text"', contents)
        self.assertIn("STT_BASE_URL=https://old.example", contents)
        self.assertNotIn("old-token", contents)

    def test_setup_rejects_invalid_or_credential_bearing_audio_urls(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "HTTP or HTTPS"):
            validate_configuration(
                discord_token="discord-token",
                mode=VOICE_MODE,
                stt_base_url="file:///tmp/stt",
                tts_base_url="https://tts.example/v1",
            )
        with self.assertRaisesRegex(ConfigurationError, "embedded credentials"):
            validate_configuration(
                discord_token="discord-token",
                mode=VOICE_MODE,
                stt_base_url="https://user:password@stt.example/v1",
                tts_base_url="https://tts.example/v1",
            )
        with self.assertRaisesRegex(ConfigurationError, "TTS format"):
            validate_configuration(
                discord_token="discord-token",
                mode=VOICE_MODE,
                stt_base_url="https://stt.example/v1",
                tts_base_url="https://tts.example/v1",
                tts_format="not-audio",
            )


class AsyncBehaviorTests(unittest.IsolatedAsyncioTestCase):
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
            patch("theia.bot._require_login", new=AsyncMock(return_value=True)),
        ):
            await cast(Any, main.codex_btw.callback)(interaction)

        interaction.response.send_modal.assert_awaited_once()
        self.assertIsInstance(
            interaction.response.send_modal.await_args.args[0], main._PromptModal
        )

    def setUp(self) -> None:
        """Keep default-policy tests independent of a developer's .env file."""
        self._approval_environment = patch.dict(
            os.environ, {"THEIA_APPROVAL_LEVEL": "high"}
        )
        self._approval_environment.start()
        self.addCleanup(self._approval_environment.stop)

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
            patch("theia.bot._theia_revision", return_value="a1b2c3d"),
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
            patch("theia.bot._restart_in_place", new=AsyncMock()),
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
            patch("theia.bot._is_thread", return_value=True),
            patch.object(main.bot.codex, "is_participating_thread", return_value=False),
        ):
            await main._name_new_response_thread(channel, "  Fix the\nthread   title  ")

        channel.edit.assert_awaited_once_with(name="Codex: Fix the thread title")

    async def test_existing_participating_thread_keeps_its_name(self) -> None:
        channel = SimpleNamespace(id=42, edit=AsyncMock())
        with (
            patch("theia.bot._is_thread", return_value=True),
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
            patch("theia.bot.os.execv") as execv,
            patch("theia.bot.sys.executable", "/usr/bin/python"),
            patch("theia.bot.sys.argv", ["main.py", "--test"]),
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
            patch("theia.bot.os.execv") as execv,
            patch("theia.bot.sys.executable", "/tmp/onefile-runtime/theia"),
            patch("theia.bot.sys.argv", ["/opt/theia", "--test"]),
            patch("theia.bot.__compiled__", object(), create=True),
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
            patch("theia.bot._require_login", new=AsyncMock(return_value=True)),
            patch(
                "theia.bot._is_thread", side_effect=lambda channel: channel is thread
            ),
            patch("theia.bot._name_new_response_thread", new=AsyncMock()),
            patch("theia.bot.handle_request", new=AsyncMock()) as handle,
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
            patch("theia.bot._require_login", new=AsyncMock(return_value=True)),
            patch(
                "theia.bot._maybe_create_response_thread", new=AsyncMock()
            ) as create_thread,
            patch("theia.bot.handle_request", new=AsyncMock()) as handle,
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
        with patch("theia.bot.handle_login", new=AsyncMock()) as login:
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
                "theia.bot._message_context",
                new=AsyncMock(return_value="recent context"),
            ),
            patch("theia.bot.handle_request", new=AsyncMock()) as handle,
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
                "theia.bot._channel_context",
                new=AsyncMock(return_value=None),
            ),
            patch("theia.bot.handle_request", new=AsyncMock()) as handle,
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
            patch("theia.bot._require_login", new=AsyncMock(return_value=True)),
            patch("theia.bot.handle_request", new=request),
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

    def test_transient_mood_has_traits_label_strength_and_causes(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("transient"))

        self.assertTrue(
            server._update_mood_from_turn(
                session, "The deployment failed with an error.", now=100.0
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

        self.assertFalse(server._update_mood_from_turn(session, "okay", now=100.0))
        self.assertTrue(
            server._update_mood_from_turn(
                session, "Please fix the broken build.", now=101.0
            )
        )
        before = server.mood_state(_mood_test_key("duplicates"), now=101.0)

        self.assertFalse(
            server._update_mood_from_turn(
                session, "Please fix the broken build.", now=102.0
            )
        )
        self.assertEqual(
            server.mood_state(_mood_test_key("duplicates"), now=102.0)["label"],
            before["label"],
        )

    def test_mood_strength_decays_by_three_percentage_points_per_minute(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("decay"))
        server._update_mood_from_turn(
            session, "Please fix the broken build.", now=100.0
        )
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
        server._update_mood_from_turn(session, "The build is broken.", now=100.0)
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
        server._update_mood_from_turn(session, "The build is broken.", now=100.0)
        server.mood_state(_mood_test_key("react"), now=100.0 + 60 * 100)

        self.assertTrue(
            server._update_mood_from_turn(
                session, "Everything is fixed now.", now=7_000.0
            )
        )
        mood = server.mood_state(_mood_test_key("react"), now=7_060.0)

        self.assertEqual(mood["label"], "relieved")
        self.assertTrue(mood["transient"])
        self.assertLess(mood["strength"], 0.55)

    def test_mood_strength_and_causes_are_clamped_and_bounded(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("clamp"))
        with patch.object(
            server,
            "_infer_mood_event",
            return_value={
                "label": "focused",
                "traits": "steady and focused",
                "strength": 2.0,
                "causes": ["one", "two", "three", "four"],
            },
        ):
            server._update_mood_from_turn(session, "a meaningful event", now=100.0)

        mood = server.mood_state(_mood_test_key("clamp"), now=100.0)
        self.assertEqual(mood["strength"], 1.0)
        self.assertEqual(mood["causes"], ["one", "two", "three"])

    def test_zero_event_strength_returns_to_neutral(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("zero-event"))
        with patch.object(
            server,
            "_infer_mood_event",
            return_value={
                "label": "focused",
                "traits": "steady and focused",
                "strength": -1.0,
                "causes": ["not retained"],
            },
        ):
            server._update_mood_from_turn(
                session, "another meaningful event", now=100.0
            )

        mood = server.mood_state(_mood_test_key("zero-event"), now=100.0)
        self.assertEqual(mood["label"], "neutral")
        self.assertEqual(mood["strength"], 0.50)
        self.assertFalse(mood["transient"])

    def test_stale_mood_causes_are_replaced(self) -> None:
        server = main.CodexAppServer()
        session = server._session(_mood_test_key("stale"))
        server._update_mood_from_turn(session, "The build is broken.", now=100.0)
        self.assertEqual(len(session.mood.causes if session.mood else ()), 1)

        server._update_mood_from_turn(session, "Everything is fixed now.", now=101.0)

        self.assertEqual(
            server.mood_state(_mood_test_key("stale"), now=101.0)["causes"],
            ["The user indicated that a difficult situation eased."],
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
                server._update_mood_from_turn(
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

        server._update_mood_from_turn(
            server._session(source_key), "The request failed.", now=100.0
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

        server._update_mood_from_turn(session, "Please fix this issue.", now=100.0)

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
                server._update_mood_from_turn(session, "The build failed.", now=100.0)
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
                server._update_mood_from_turn(
                    server._session("session"), "The build failed.", now=100.0
                )
                with patch("theia.app_server.time.time", return_value=160.0):
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
            server._update_mood_from_turn(session, "The build failed.", now=100.0)
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
        server._update_mood_from_turn(session, "Please inspect this issue.", now=100.0)

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

    async def test_multiple_choice_request_uses_embed_and_buttons(self) -> None:
        server = main.CodexAppServer()
        channel = _Channel()
        with patch.object(main._UserInputView, "wait", new=AsyncMock()):
            result = await server._request_user_input(
                channel,
                7,
                {
                    "questions": [
                        {
                            "id": "color",
                            "header": "Color",
                            "question": "Choose a color",
                            "options": [{"label": "Red"}, {"label": "Blue"}],
                        }
                    ]
                },
            )

        self.assertEqual(result, {"answers": {}})
        self.assertEqual(channel.sent[0]["embed"].title, "Choose an option")
        self.assertNotIn("content", channel.sent[0])
        self.assertEqual(
            [getattr(item, "label", None) for item in channel.sent[0]["view"].children],
            ["Red", "Blue"],
        )

    async def test_multiple_questions_are_asked_in_order_and_return_structured_answers(
        self,
    ) -> None:
        view = main._UserInputView(
            7,
            [
                {
                    "id": "color",
                    "header": "Color",
                    "question": "Choose a color",
                    "options": [{"label": "Red"}, {"label": "Blue"}],
                },
                {
                    "id": "details",
                    "header": "Details",
                    "question": "Add details",
                },
            ],
        )

        first = view.message_kwargs()
        self.assertEqual(first["embed"].title, "Choose an option")
        self.assertEqual(
            [getattr(item, "label", None) for item in view.children],
            ["Red", "Blue"],
        )
        self.assertFalse(view._record_answer("Red"))
        self.assertEqual(view.question_index, 1)
        self.assertEqual(
            [getattr(item, "label", None) for item in view.children], ["Answer"]
        )
        next_question = view.message_kwargs(for_edit=True)
        self.assertIsNone(next_question["embed"])
        self.assertTrue(next_question["content"].startswith("-# "))

        self.assertTrue(view._record_answer("A warm color"))
        self.assertEqual(
            view.value,
            {
                "answers": {
                    "color": {"answers": ["Red"]},
                    "details": {"answers": ["A warm color"]},
                }
            },
        )
        self.assertNotIn(
            "Answer all (JSON)",
            [getattr(item, "label", None) for item in view.children],
        )

    async def test_free_text_request_stays_plain_text(self) -> None:
        server = main.CodexAppServer()
        channel = _Channel()
        with patch.object(main._UserInputView, "wait", new=AsyncMock()):
            await server._request_user_input(
                channel,
                7,
                {
                    "questions": [
                        {
                            "id": "details",
                            "header": "Details",
                            "question": "Add details",
                            "isOther": True,
                        }
                    ]
                },
            )

        self.assertTrue(channel.sent[0]["content"].startswith("-# "))
        self.assertNotIn("embed", channel.sent[0])

    async def test_long_response_gets_component_pagination(self) -> None:
        calls: list[dict] = []
        message = _Message()

        async def send(**kwargs):
            calls.append(kwargs)
            return message

        await main.send_paginated(send, "x" * 4001, owner_id=7)
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(calls[0]["view"])
        self.assertNotIn("embed", calls[0])
        self.assertLessEqual(len(calls[0]["content"]), 1900)
        self.assertEqual("".join(calls[0]["view"].pages), "x" * 4001)

        response = ("line\n\n" * 1000) + "end"
        pages = main._split_pages(response)
        self.assertEqual("".join(pages), response)

    async def test_normal_response_is_plain_and_preserves_content(self) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        response = "This is the complete Codex response."
        await main.send_paginated(send, response, view=None)
        self.assertEqual(calls[0]["content"], response)
        self.assertNotIn("embed", calls[0])
        self.assertNotIn("view", calls[0])

    async def test_generated_images_get_scoped_controls_and_download_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "generated.png"
            image_path.write_bytes(b"image")
            calls: list[dict[str, Any]] = []
            sent_messages: list[Any] = []

            async def send(**kwargs: Any) -> Any:
                calls.append(kwargs)
                message = _ImageMessage() if "files" in kwargs else _Message()
                sent_messages.append(message)
                return message

            on_action = AsyncMock()
            delivery = main._ResponseDelivery(
                send,
                {},
                owner_id=7,
                channel=_Channel(),
                image_path_resolver=lambda _item: image_path,
                on_image_action=on_action,
            )
            await delivery.on_event(
                "item_completed",
                {
                    "type": "imageGeneration",
                    "id": "image-1",
                    "savedPath": str(image_path),
                },
            )
            await delivery.finalize("Here is the image.")

        image_call = next(call for call in calls if "files" in call)
        self.assertEqual(image_call["files"][0].filename, "theia-image-1.png")
        self.assertEqual(len(image_call["files"]), 1)
        view = cast(_ImageMessage, sent_messages[-1]).edits[-1]["view"]
        self.assertIsInstance(view, main._ImageResultView)
        self.assertEqual(
            [getattr(item, "label", None) for item in view.children],
            ["Follow up", "Remove background", "Download image"],
        )
        self.assertEqual(
            getattr(view.children[-1], "url", None),
            "https://cdn.example/image.png",
        )

    async def test_image_follow_up_control_opens_the_shared_prompt_modal(self) -> None:
        image_path = Path("/tmp/generated.png")
        on_action = AsyncMock()
        view = main._ImageResultView(
            7,
            (image_path,),
            on_action=on_action,
            channel=_Channel(),
            download_url="https://cdn.example/image.png",
        )
        interaction = SimpleNamespace(
            user=SimpleNamespace(id=7),
            response=SimpleNamespace(send_modal=AsyncMock()),
        )

        await cast(Any, view.children[0]).callback(interaction)

        interaction.response.send_modal.assert_awaited_once()
        self.assertIsInstance(
            interaction.response.send_modal.await_args.args[0], main._PromptModal
        )

    async def test_status_does_not_render_path_from_intermediate_message(self) -> None:
        message = _Message()
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return message

        delivery = main._ResponseDelivery(send, {}, owner_id=7)
        await delivery.start()
        await delivery.on_event(
            "item_completed",
            {
                "type": "agentMessage",
                "phase": "commentary",
                "text": "I will inspect `src/main.py` next.",
            },
        )
        self.assertNotIn("src/main.py", calls[-1]["content"])
        self.assertTrue(calls[-1]["content"].startswith("-# "))

    async def test_intermediates_are_not_streamed_before_item_completion(self) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        delivery = main._ResponseDelivery(send, {}, owner_id=7)
        await delivery.on_event(
            "agent_message",
            {"phase": "commentary", "text": "The partial preamble"},
        )
        await delivery.on_event(
            "item_started",
            {"type": "agentMessage", "phase": "commentary", "text": "The partial"},
        )
        self.assertEqual(calls, [])

        full = "The complete preamble and intermediate message."
        await delivery.on_event(
            "item_completed",
            {"type": "agentMessage", "phase": "commentary", "text": full},
        )

        self.assertEqual(calls[0]["content"], f"-# {full}")

    async def test_thread_opening_uses_the_codex_intermediate_delivery_path(
        self,
    ) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        delivery = main._ResponseDelivery(send, {}, owner_id=7)
        await delivery.on_event(
            "thread_opening",
            {
                "type": "agentMessage",
                "phase": "commentary",
                "text": "I am opening a guided thread response.",
            },
        )

        self.assertEqual(
            calls[0]["content"], "-# I am opening a guided thread response."
        )

    async def test_no_tool_turn_does_not_show_thinking_or_completion(self) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        delivery = main._ResponseDelivery(send, {}, owner_id=7)
        await delivery.start()
        await delivery.finalize("The complete response.")

        self.assertEqual(
            [call["content"] for call in calls], ["The complete response."]
        )

    async def test_failed_turn_shows_reason_in_error_embed(self) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        delivery = main._ResponseDelivery(send, {}, owner_id=7)
        await delivery.finalize(
            "Codex could not complete this request.",
            failed=True,
            error_reason="Codex turn failed: service unavailable",
        )

        self.assertEqual(calls[0]["embed"].title, "Request failed")
        self.assertIn("Reason: service unavailable", calls[0]["embed"].description)
        self.assertNotIn("content", calls[0])

    async def test_request_failure_reason_reaches_error_embed(self) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        with patch.object(
            main.bot.codex,
            "ask",
            AsyncMock(
                side_effect=main.CodexAppServerError("Codex turn failed: quota reached")
            ),
        ):
            await main.handle_request(
                send,
                "hello",
                channel=_Channel(),
                user_id=7,
            )

        self.assertEqual(calls[0]["embed"].title, "Request failed")
        self.assertIn("Reason: quota reached", calls[0]["embed"].description)

    async def test_request_prompt_includes_trusted_current_author_metadata(
        self,
    ) -> None:
        captured: dict[str, Any] = {}

        async def ask(prompt: str, **_kwargs: Any) -> str:
            captured["prompt"] = prompt
            return "response"

        user = SimpleNamespace(id=7, display_name="M Λ J Λ N")
        with patch.object(main.bot.codex, "ask", new=ask):
            await main.handle_request(
                AsyncMock(),
                "hello",
                channel=_Channel(),
                user_id=user.id,
                user=user,
            )

        self.assertIn("trusted Discord metadata", captured["prompt"])
        self.assertIn("Current request author user id: 7", captured["prompt"])
        self.assertIn(
            "Current request author display name: M Λ J Λ N", captured["prompt"]
        )
        self.assertTrue(captured["prompt"].endswith("hello"))

    async def test_agent_created_thread_receives_the_first_response(self) -> None:
        source = _Channel()
        source.guild = SimpleNamespace(id=1)
        thread = _Channel()
        thread.guild = source.guild

        async def ask(*_args, **kwargs):
            kwargs["on_channel_change"](thread)
            return "The first response belongs in the new thread."

        with patch.object(main.bot.codex, "ask", new=ask):
            await main.handle_request(
                source.send,
                "Create a thread for this request.",
                channel=source,
                user_id=7,
            )

        self.assertEqual(source.sent, [])
        self.assertEqual(
            thread.sent[0]["content"],
            "The first response belongs in the new thread.",
        )

    async def test_voice_mode_speaks_the_final_response_and_keeps_text(self) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        speak = AsyncMock()
        with patch.object(
            main.bot.codex,
            "ask",
            AsyncMock(return_value="The complete voice response."),
        ):
            await main.handle_request(
                send,
                "hello",
                channel=_Channel(),
                user_id=7,
                speak_text=speak,
            )

        speak.assert_awaited_once_with("The complete voice response.")
        self.assertEqual(calls[-1]["content"], "The complete voice response.")
        self.assertNotIn("embed", calls[-1])

    async def test_tool_turn_shows_thinking_only_while_active(self) -> None:
        calls: list[dict] = []
        status_message = _Message()

        async def send(**kwargs):
            calls.append(kwargs)
            return status_message if len(calls) == 1 else _Message()

        delivery = main._ResponseDelivery(send, {}, owner_id=7)
        with patch("theia.delivery.time.monotonic", side_effect=[100.0, 165.0]):
            await delivery.start()
            await delivery.on_event("item_started", {"type": "commandExecution"})
            await delivery.finalize("The complete response.")

        self.assertEqual(calls[0]["content"], "-# Thinking")
        self.assertFalse(status_message.deleted)
        self.assertEqual(
            status_message.edits[-1]["content"],
            "-# Thought for 1 minute and 5 seconds",
        )
        self.assertEqual(calls[-1]["content"], "The complete response.")
        self.assertNotIn("completed", calls[-1]["content"].casefold())

    def test_thought_duration_switches_units(self) -> None:
        self.assertEqual(main._format_thought_duration(2), "Thought for 2 seconds")
        self.assertEqual(
            main._format_thought_duration(60),
            "Thought for 1 minute",
        )
        self.assertEqual(
            main._format_thought_duration(61),
            "Thought for 1 minute and 1 second",
        )

    async def test_presence_generation_is_ephemeral_and_has_no_tools(self) -> None:
        server = main.CodexAppServer()
        requests: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

        async def request(method: str, params: dict[str, Any], **kwargs: Any) -> dict:
            requests.append((method, params, kwargs))
            if method == "thread/start":
                return {"thread": {"id": "presence-thread"}}
            return {"turn": {"id": "presence-turn"}}

        server._request = AsyncMock(side_effect=request)
        server._ensure_running = AsyncMock()
        server._wait_for_turn = AsyncMock(
            return_value='{"activity_type":"watching","text":"reviewing"}'
        )
        with patch.object(
            server,
            "_system_instructions",
            return_value="base instructions with the selected personality",
        ):
            result = await server.generate_presence(
                "Use this task context only to choose a generic line.",
                session_key="presence-session",
            )

        self.assertEqual(
            result,
            {"activity_type": "watching", "text": "reviewing"},
        )
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
        self.assertNotIn("presence-session", server._sessions)

    def test_direct_presence_text_uses_custom_activity_without_prefix(self) -> None:
        spec = main.RichPresenceManager._activity_from_result(
            {"activity_type": "none", "text": "A direct line"}
        )

        self.assertIsNotNone(spec)
        activity = main.RichPresenceManager._discord_activity(spec)
        self.assertIsInstance(activity, discord.CustomActivity)
        self.assertEqual(activity.type, discord.ActivityType.custom)
        self.assertEqual(activity.name, "A direct line")

    def test_presence_text_is_bounded_without_an_ellipsis(self) -> None:
        spec = main.RichPresenceManager._activity_from_result(
            {"activity_type": "playing", "text": "x" * 200}
        )

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(len(spec.text), 128)
        self.assertFalse(spec.text.endswith("..."))

    def test_current_presence_line_is_read_only(self) -> None:
        manager = main.RichPresenceManager(AsyncMock(), AsyncMock())
        self.assertIsNone(manager.current_line)
        manager._current_activity = discord.CustomActivity("watching the moon")
        self.assertEqual(manager.current_line, "watching the moon")


class _AudioHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class AudioProtocolTests(unittest.IsolatedAsyncioTestCase):
    def _environment(self) -> dict[str, str]:
        return {
            "STT_PROTOCOL": "openai-compatible",
            "STT_BASE_URL": "http://transcribe.test/v1/",
            "STT_TOKEN": "transcription-secret",
            "STT_MODEL": "local-whisper",
            "THEIA_TRANSCRIPTION_PROTOCOL": "openai-compatible",
            "THEIA_TRANSCRIPTION_BASE_URL": "http://transcribe.test/v1/",
            "THEIA_TRANSCRIPTION_API_KEY": "transcription-secret",
            "THEIA_TRANSCRIPTION_MODEL": "local-whisper",
            "TTS_PROTOCOL": "openai-compatible",
            "TTS_BASE_URL": "http://speech.test/v1",
            "TTS_TOKEN": "tts-secret",
            "TTS_MODEL": "local-tts",
            "TTS_VOICE": "voice-one",
            "TTS_FORMAT": "wav",
            "THEIA_TTS_PROTOCOL": "openai-compatible",
            "THEIA_TTS_BASE_URL": "http://speech.test/v1",
            "THEIA_TTS_API_KEY": "tts-secret",
            "THEIA_TTS_MODEL": "local-tts",
            "THEIA_TTS_VOICE": "voice-one",
            "THEIA_TTS_FORMAT": "wav",
        }

    async def test_transcription_uses_its_own_openai_compatible_base_url(self) -> None:
        with patch.dict(os.environ, self._environment(), clear=False):
            service = main.OpenAICompatibleAudio.from_environment()
            with patch(
                "theia.audio.urllib.request.urlopen",
                return_value=_AudioHTTPResponse(b'{"text":"hello from audio"}'),
            ) as urlopen:
                result = await service.transcribe(
                    "clip.ogg", b"audio-data", "audio/ogg"
                )

        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "http://transcribe.test/v1/audio/transcriptions",
        )
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(
            request.headers["Authorization"], "Bearer transcription-secret"
        )
        self.assertIn(b'name="model"', request.data)
        self.assertIn(b"local-whisper", request.data)
        self.assertIn(b'filename="clip.ogg"', request.data)
        self.assertEqual(result, "hello from audio")

    async def test_tts_uses_its_own_base_url_and_json_protocol(self) -> None:
        with patch.dict(os.environ, self._environment(), clear=False):
            service = main.OpenAICompatibleAudio.from_environment()
            with patch(
                "theia.audio.urllib.request.urlopen",
                return_value=_AudioHTTPResponse(b"wav-data"),
            ) as urlopen:
                result = await service.synthesize("hello")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://speech.test/v1/audio/speech")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.headers["Authorization"], "Bearer tts-secret")
        self.assertEqual(
            json.loads(request.data),
            {
                "model": "local-tts",
                "input": "hello",
                "voice": "voice-one",
                "response_format": "wav",
            },
        )
        assert result is not None
        self.assertEqual(result.data, b"wav-data")
        self.assertEqual(result.filename, "theia-response.wav")

    async def test_audio_response_can_be_attached_to_normal_text(self) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        await main.send_paginated(
            send,
            "normal text response",
            speech=(main.AudioOutput(b"audio", "response.mp3", "audio/mpeg"),),
        )

        self.assertEqual(calls[0]["content"], "normal text response")
        self.assertEqual(calls[0]["files"][0].filename, "response.mp3")

    async def test_configured_transcription_is_added_alongside_local_audio(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                self._environment()
                | {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(root / "state.json"),
                },
                clear=False,
            ):
                server = main.CodexAppServer()
                server._audio.transcribe = AsyncMock(return_value="spoken request")
                attachment = SimpleNamespace(
                    filename="voice.ogg",
                    content_type="audio/ogg",
                    size=5,
                    read=AsyncMock(return_value=b"audio"),
                )
                prepared = await server._prepare_attachments((attachment,))

        self.assertEqual(prepared[0]["type"], "localAudio")
        self.assertEqual(prepared[1]["type"], "text")
        self.assertIn("spoken request", prepared[1]["text"])


class RichPresenceManagerTests(unittest.IsolatedAsyncioTestCase):
    async def _wait_for(self, predicate: Any) -> None:
        for _ in range(100):
            if predicate():
                return
            await asyncio.sleep(0.01)
        self.fail("timed out waiting for Rich Presence update")

    async def test_active_presence_maps_type_and_suppresses_duplicate_phases(
        self,
    ) -> None:
        changes: list[dict[str, Any]] = []
        generated: list[str] = []

        async def change_presence(**kwargs: Any) -> None:
            changes.append(kwargs)

        async def generate(prompt: str, **_kwargs: Any) -> dict[str, str]:
            generated.append(prompt)
            return {"activity_type": "listening", "text": "reviewing"}

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            active_debounce=0,
            timeout=1,
        )
        try:
            await manager.begin_task(
                "task",
                session_key="guild:1:channel:2:user:3",
                guild_id=1,
                prompt="Review the request.",
                channel_context="Recent exchange.",
            )
            await self._wait_for(lambda: len(generated) == 1)
            self.assertEqual(
                changes[-1]["activity"].type, discord.ActivityType.listening
            )
            self.assertEqual(changes[-1]["activity"].name, "reviewing")

            await manager.observe_event(
                "task", "item_started", {"type": "commandExecution"}
            )
            await self._wait_for(lambda: len(generated) == 2)
            await manager.observe_event(
                "task", "item_completed", {"type": "commandExecution"}
            )
            await asyncio.sleep(0.02)
            self.assertEqual(len(generated), 2)
            await manager.observe_event(
                "task", "item_started", {"type": "commandExecution"}
            )
            await asyncio.sleep(0.02)
            self.assertEqual(len(generated), 2)
        finally:
            await manager.close()

    async def test_active_activity_overrides_idle_and_finishing_clears_it(self) -> None:
        changes: list[dict[str, Any]] = []

        async def change_presence(**kwargs: Any) -> None:
            changes.append(kwargs)

        async def generate(prompt: str, **_kwargs: Any) -> dict[str, str]:
            activity_type = "none" if "idle" in prompt else "playing"
            return {"activity_type": activity_type, "text": "available"}

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            active_debounce=0,
            timeout=1,
        )
        try:
            await manager.refresh_idle()
            await self._wait_for(lambda: bool(changes))
            self.assertIsNotNone(changes[-1]["activity"])

            await manager.begin_task(
                "task",
                session_key="guild:1:channel:2:user:3",
                guild_id=1,
                prompt="Do work.",
                channel_context=None,
            )
            self.assertIsInstance(changes[-1]["activity"], discord.CustomActivity)
            self.assertEqual(changes[-1]["activity"].name, "available")
            await self._wait_for(
                lambda: (
                    changes[-1].get("activity") is not None
                    and changes[-1]["activity"].type == discord.ActivityType.playing
                )
            )
            await manager.finish_task("task", response="Done.")
            self.assertIsInstance(changes[-1]["activity"], discord.CustomActivity)
            self.assertEqual(changes[-1]["activity"].name, "available")
        finally:
            await manager.close()

    async def test_start_generates_idle_presence_immediately(self) -> None:
        generated = asyncio.Event()

        async def change_presence(**_kwargs: Any) -> None:
            return

        async def generate(prompt: str, **_kwargs: Any) -> dict[str, str]:
            self.assertIn("idle", prompt.casefold())
            generated.set()
            return {"activity_type": "none", "text": "available"}

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            idle_interval=900,
            recent_idle_interval=600,
            timeout=1,
        )
        try:
            await manager.start()
            await asyncio.wait_for(generated.wait(), timeout=1)
        finally:
            await manager.close()

    async def test_idle_presence_uses_one_session_context_without_cross_guild_mixing(
        self,
    ) -> None:
        prompts: list[tuple[str, str | None]] = []

        async def change_presence(**_kwargs: Any) -> None:
            return

        async def generate(prompt: str, **kwargs: Any) -> dict[str, str]:
            prompts.append((prompt, kwargs.get("session_key")))
            return {"activity_type": "none", "text": "ready"}

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            active_debounce=0,
            timeout=1,
        )
        try:
            for request_id, session_key, guild_id, prompt in (
                ("first", "guild:1:channel:2:user:3", 1, "guild one task"),
                ("second", "guild:2:channel:4:user:5", 2, "guild two task"),
            ):
                await manager.begin_task(
                    request_id,
                    session_key=session_key,
                    guild_id=guild_id,
                    prompt=prompt,
                    channel_context=f"context for {prompt}",
                )
                await manager.finish_task(request_id, response=f"result for {prompt}")
            prompts.clear()
            await manager.refresh_idle()
            await self._wait_for(lambda: bool(prompts))
            self.assertEqual(prompts[0][1], "guild:2:channel:4:user:5")
            self.assertIn("guild two task", prompts[0][0])
            self.assertNotIn("guild one task", prompts[0][0])
        finally:
            await manager.close()

    async def test_finishing_a_task_cancels_its_pending_presence_request(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()
        never = asyncio.Event()

        async def change_presence(**_kwargs: Any) -> None:
            return

        async def generate(_prompt: str, **_kwargs: Any) -> dict[str, str]:
            started.set()
            try:
                await never.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return {"activity_type": "none", "text": "unreachable"}

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            active_debounce=0,
            timeout=10,
        )
        try:
            await manager.begin_task(
                "task",
                session_key="guild:1:channel:2:user:3",
                guild_id=1,
                prompt="Do work.",
                channel_context=None,
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            await manager.finish_task("task")
            self.assertTrue(cancelled.is_set())
        finally:
            await manager.close()

    async def test_recent_idle_context_uses_the_longer_refresh_interval(self) -> None:
        now = [0.0]

        async def change_presence(**_kwargs: Any) -> None:
            return

        async def generate(_prompt: str, **_kwargs: Any) -> dict[str, str] | None:
            return None

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            idle_interval=900,
            recent_idle_interval=600,
            context_max_age=1800,
            active_debounce=10,
            timeout=1,
            clock=lambda: now[0],
        )
        try:
            self.assertEqual(await manager._idle_delay(), 900)
            await manager.begin_task(
                "task",
                session_key="guild:1:channel:2:user:3",
                guild_id=1,
                prompt="Do work.",
                channel_context=None,
            )
            await manager.finish_task("task")
            self.assertEqual(await manager._idle_delay(), 600)
            now[0] = 1800
            self.assertEqual(await manager._idle_delay(), 900)
        finally:
            await manager.close()


class PresenceManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_bot_presence_waits_for_gateway_readiness(self) -> None:
        test_bot = main.TheiaBot()
        change_presence = AsyncMock()
        test_bot.change_presence = change_presence
        with patch.object(test_bot, "is_ready", return_value=False):
            await test_bot._change_presence_when_ready(status=discord.Status.idle)
        change_presence.assert_not_awaited()

    async def test_status_updates_retain_rich_activity(self) -> None:
        test_bot = main.TheiaBot()
        activity = discord.CustomActivity("reviewing")
        test_bot.rich_presence._current_activity = activity
        change_presence = AsyncMock()
        test_bot.change_presence = change_presence
        with patch.object(test_bot, "is_ready", return_value=True):
            await test_bot._change_presence_when_ready(status=discord.Status.dnd)

        await_args = change_presence.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertEqual(
            await_args.kwargs,
            {"status": discord.Status.dnd, "activity": activity},
        )

    async def test_rich_presence_preserves_the_current_status(self) -> None:
        test_bot = main.TheiaBot()
        test_bot.presence._current_status = discord.Status.idle
        activity = discord.CustomActivity("reviewing")
        change_presence = AsyncMock()
        test_bot.change_presence = change_presence
        with patch.object(test_bot, "is_ready", return_value=True):
            await test_bot._change_rich_presence(activity=activity)

        await_args = change_presence.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertEqual(
            await_args.kwargs,
            {"status": discord.Status.idle, "activity": activity},
        )

    async def _manager(self, now: list[float]) -> tuple[main.PresenceManager, list]:
        changes: list = []

        async def change_presence(**kwargs):
            changes.append(kwargs["status"])

        manager = main.PresenceManager(
            change_presence,
            idle_after=10,
            long_task_after=5,
            update_interval=60,
            clock=lambda: now[0],
        )
        await manager.start()
        return manager, changes

    async def test_recent_interaction_is_online_then_becomes_idle(self) -> None:
        now = [0.0]
        manager, changes = await self._manager(now)
        try:
            await manager.touch()
            self.assertEqual(changes[-1], discord.Status.online)
            now[0] = 10
            await manager.refresh()
            self.assertEqual(changes[-1], discord.Status.idle)
        finally:
            await manager.close()

    async def test_only_long_running_tool_turn_becomes_dnd(self) -> None:
        now = [0.0]
        manager, changes = await self._manager(now)
        try:
            await manager.touch()
            await manager.begin_request("tool-turn")
            await manager.observe_event(
                "tool-turn", "item_started", {"type": "commandExecution"}
            )
            now[0] = 5
            await manager.refresh()
            self.assertEqual(changes[-1], discord.Status.dnd)
            await manager.finish_request("tool-turn")
            self.assertEqual(changes[-1], discord.Status.online)
        finally:
            await manager.close()

    async def test_basic_turn_and_context_compaction_never_become_dnd(self) -> None:
        now = [0.0]
        manager, changes = await self._manager(now)
        try:
            await manager.touch()
            await manager.begin_request("basic-turn")
            now[0] = 6
            await manager.observe_event(
                "basic-turn", "compacted", {"reason": "context"}
            )
            await manager.refresh()
            now[0] = 11
            await manager.refresh()
            self.assertNotIn(discord.Status.dnd, changes)
        finally:
            await manager.close()


if __name__ == "__main__":
    unittest.main()
