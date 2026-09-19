from discord import app_commands

# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable
from tests.test_support import *
from theia.server.policy import _SELF_IMPROVEMENT_HISTORY_LIMIT


class SelfImprovementAuditTests(AsyncBehaviorTestBase):
    def _personality_fixture(self, root: Path) -> tuple[main.CodexAppServer, Path]:
        server = main.CodexAppServer()
        personality_root = root / "theia" / "personalities"
        personality_root.mkdir(parents=True)
        path = personality_root / "Cel.md"
        path.write_text("Be warm.", encoding="utf-8")
        return server, path

    def test_applied_update_has_safe_audit_record_and_personality_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(root / "state.json"),
                },
            ):
                server, personality_path = self._personality_fixture(root)
                applied = server._apply_self_improvement_updates(
                    [
                        {
                            "kind": "personality",
                            "path": "active",
                            "content": "Keep answers warm and direct.",
                        }
                    ],
                    memory_root=root / "theia" / "memories",
                    skill_root=root / "theia" / "skills",
                    personality_path=personality_path,
                )
                record = server.self_improvement_history(limit=1)[0]
                backup = server._self_improvement_revision_path(record["id"])

            self.assertEqual(applied, 1)
            self.assertEqual(record["category"], "personality")
            self.assertEqual(record["target"], "personality:Cel")
            self.assertEqual(record["status"], "applied")
            self.assertEqual(len(record["previous_content_hash"]), 64)
            self.assertEqual(len(record["new_content_hash"]), 64)
            self.assertNotIn("Keep answers warm", record["reason"])
            self.assertEqual(backup.read_text(encoding="utf-8"), "Be warm.")

    def test_personality_write_failure_preserves_source_and_records_rejection(
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
                server, personality_path = self._personality_fixture(root)
                with patch.object(
                    server, "_atomic_self_improvement_write", return_value=False
                ):
                    applied = server._apply_self_improvement_updates(
                        [
                            {
                                "kind": "personality",
                                "path": "active",
                                "content": "This must not be applied.",
                            }
                        ],
                        memory_root=root / "theia" / "memories",
                        skill_root=root / "theia" / "skills",
                        personality_path=personality_path,
                    )
                record = server.self_improvement_history(limit=1)[0]

            self.assertEqual(applied, 0)
            self.assertEqual(personality_path.read_text(encoding="utf-8"), "Be warm.")
            self.assertEqual(record["status"], "rejected")
            self.assertIn("saved safely", record["reason"])

    async def test_personality_revert_is_atomic_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(root / "state.json"),
                },
            ):
                server, personality_path = self._personality_fixture(root)
                server._apply_self_improvement_updates(
                    [
                        {
                            "kind": "personality",
                            "path": "active",
                            "content": "Use a calm delivery.",
                        }
                    ],
                    memory_root=root / "theia" / "memories",
                    skill_root=root / "theia" / "skills",
                    personality_path=personality_path,
                )
                change_id = server.self_improvement_history(limit=1)[0]["id"]
                result = await server.revert_self_improvement(
                    change_id, super_admin=True
                )
                preview = server.self_improvement_preview(change_id)
                history = server.self_improvement_history(limit=3)

            self.assertEqual(result["status"], "reverted")
            self.assertEqual(personality_path.read_text(encoding="utf-8"), "Be warm.")
            self.assertEqual(preview["status"], "reverted")
            self.assertFalse(preview["revertible"])
            self.assertEqual(history[0]["status"], "reverted")
            self.assertEqual(history[0]["category"], "personality")
            self.assertNotIn("target_name", preview)

    async def test_revert_rejects_later_personality_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(root / "state.json"),
                },
            ):
                server, personality_path = self._personality_fixture(root)
                update = {
                    "kind": "personality",
                    "path": "active",
                    "content": "Use a calm delivery.",
                }
                server._apply_self_improvement_updates(
                    [update],
                    memory_root=root / "theia" / "memories",
                    skill_root=root / "theia" / "skills",
                    personality_path=personality_path,
                )
                change_id = server.self_improvement_history(limit=1)[0]["id"]
                personality_path.write_text(
                    personality_path.read_text(encoding="utf-8") + "\nManual change.\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    main.CodexAppServerError, "changed after this update"
                ):
                    await server.revert_self_improvement(change_id, super_admin=True)

            self.assertIn("Manual change", personality_path.read_text(encoding="utf-8"))

    async def test_malformed_review_and_secret_rejection_do_not_modify_targets(
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
                server, personality_path = self._personality_fixture(root)
                self.assertEqual(server._parse_self_improvement("not json"), [])
                applied = server._apply_self_improvement_updates(
                    [
                        {
                            "kind": "personality",
                            "path": "active",
                            "content": "Store this api_key: secret-value.",
                        }
                    ],
                    memory_root=root / "theia" / "memories",
                    skill_root=root / "theia" / "skills",
                    personality_path=personality_path,
                )
                record = server.self_improvement_history(limit=1)[0]

            self.assertEqual(applied, 0)
            self.assertEqual(record["status"], "rejected")
            self.assertEqual(personality_path.read_text(encoding="utf-8"), "Be warm.")
            self.assertNotIn("secret-value", record["reason"])

    def test_history_persists_restores_and_drops_old_records(self) -> None:
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
                with patch(
                    "theia.server.self_improvement.time.time_ns", return_value=123
                ):
                    for _ in range(_SELF_IMPROVEMENT_HISTORY_LIMIT + 5):
                        server._append_self_improvement_audit(
                            category="memory",
                            target="memory:MEMORY.md",
                            status="applied",
                            reason="Validated durable update applied atomically.",
                        )
                server._persist_state()
                restored = main.CodexAppServer()

            self.assertEqual(
                len(server._self_improvement_history),
                _SELF_IMPROVEMENT_HISTORY_LIMIT,
            )
            self.assertEqual(
                len(restored._self_improvement_history),
                _SELF_IMPROVEMENT_HISTORY_LIMIT,
            )
            self.assertEqual(
                restored._self_improvement_history[-1]["id"],
                server._self_improvement_history[-1]["id"],
            )
            self.assertNotIn("private", state_path.read_text(encoding="utf-8"))

    async def test_revert_control_is_not_available_to_a_regular_administrator(
        self,
    ) -> None:
        guild = SimpleNamespace(id=42)
        channel = SimpleNamespace(id=7, guild=guild)
        response = SimpleNamespace(send_message=AsyncMock())
        interaction = SimpleNamespace(
            guild=guild,
            channel=channel,
            response=response,
            user=SimpleNamespace(
                id=9,
                guild_permissions=SimpleNamespace(administrator=True),
            ),
        )
        with (
            patch.dict(
                os.environ,
                {
                    main.SUPER_ADMIN_USERS_ENV: "",
                    main.ALWAYS_ADMIN_USERS_ENV: "",
                },
            ),
            patch.object(main.bot.presence, "touch", new=AsyncMock()),
        ):
            await main.codex_improvements.callback(
                interaction,
                app_commands.Choice(name="revert", value="revert"),
                "imp-12345678",
            )

        self.assertEqual(
            response.send_message.await_args.kwargs["embed"].title,
            "Super Admin access required",
        )
