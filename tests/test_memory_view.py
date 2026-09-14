# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable
from tests.test_support import *


class MemoryRecordTests(unittest.TestCase):
    def _server(self, _root: Path) -> main.CodexAppServer:
        return main.CodexAppServer()

    def test_records_are_scoped_and_include_safe_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memories = root / "memories"
            (memories / "users" / "9").mkdir(parents=True)
            (memories / "servers" / "42").mkdir(parents=True)
            (memories / "servers" / "99").mkdir(parents=True)
            (memories / "users" / "9" / "USER.md").write_text(
                "- User preference for concise answers.\n", encoding="utf-8"
            )
            (memories / "servers" / "42" / "MEMORY.md").write_text(
                "- Current server release plan.\n", encoding="utf-8"
            )
            (memories / "servers" / "99" / "MEMORY.md").write_text(
                "- Unrelated server secret plan.\n", encoding="utf-8"
            )
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(root / "state.json"),
                    "CODEX_MEMORY_ROOTS": str(memories),
                },
            ):
                server = self._server(root)
                key = "guild:42:channel:7:user:9"
                own = server.memory_view(
                    key,
                    "me",
                    actor_user_id=9,
                    actor_guild_id=42,
                    server_admin=False,
                    super_admin=False,
                )
                current_server = server.memory_view(
                    key,
                    "server",
                    actor_user_id=9,
                    actor_guild_id=42,
                    server_admin=True,
                    super_admin=False,
                )
                with self.assertRaisesRegex(main.CodexAppServerError, "Super Admin"):
                    server.memory_view(
                        key,
                        "everyone",
                        actor_user_id=9,
                        actor_guild_id=42,
                        server_admin=True,
                        super_admin=False,
                    )
                all_records = server.memory_view(
                    key,
                    "everyone",
                    actor_user_id=9,
                    actor_guild_id=42,
                    server_admin=True,
                    super_admin=True,
                )

        self.assertEqual(len(own["records"]), 1)
        self.assertEqual(own["records"][0]["source_category"], "user_memory")
        self.assertEqual(own["records"][0]["scope"], "user scope")
        self.assertEqual(own["records"][0]["display_metadata"]["updated"], "recently")
        self.assertEqual(len(current_server["records"]), 1)
        self.assertEqual(
            current_server["records"][0]["source_category"], "server_memory"
        )
        self.assertEqual(len(all_records["records"]), 3)
        self.assertNotIn("/memories/", str(all_records))
        self.assertNotIn("secret", own["records"][0]["text"].casefold())

    def test_search_inspect_and_confirmed_forget_preserve_other_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memories = root / "memories" / "users" / "9"
            memories.mkdir(parents=True)
            source = memories / "USER.md"
            source.write_text(
                "- Keep this alpha fact.\n- Keep this beta fact.\n", encoding="utf-8"
            )
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(root / "state.json"),
                    "CODEX_MEMORY_ROOTS": str(root / "memories"),
                },
            ):
                server = self._server(root)
                key = "guild:42:channel:7:user:9"
                found = server.memory_view(
                    key,
                    "me",
                    search="alpha",
                    actor_user_id=9,
                    actor_guild_id=42,
                    server_admin=False,
                    super_admin=False,
                )
                record_id = found["records"][0]["record_id"]
                inspected = server.memory_record(
                    key,
                    record_id,
                    "me",
                    actor_user_id=9,
                    actor_guild_id=42,
                    server_admin=False,
                    super_admin=False,
                )
                with self.assertRaisesRegex(
                    main.CodexAppServerError, "Explicit confirmation"
                ):
                    server.forget_memory(
                        key,
                        record_id,
                        "me",
                        actor_user_id=9,
                        actor_guild_id=42,
                        server_admin=False,
                        super_admin=False,
                    )
                server.forget_memory(
                    key,
                    record_id,
                    "me",
                    confirmed=True,
                    actor_user_id=9,
                    actor_guild_id=42,
                    server_admin=False,
                    super_admin=False,
                )
                audit = json.loads(
                    (root / "theia" / "memory-audit.json").read_text(encoding="utf-8")
                )
                remaining = source.read_text(encoding="utf-8")

        self.assertIn("alpha", inspected["text"])
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["action"], "forget")
        self.assertNotIn("alpha", remaining)
        self.assertIn("beta", remaining)

    def test_edit_and_persistence_failure_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memories = root / "memories" / "users" / "9"
            memories.mkdir(parents=True)
            source = memories / "USER.md"
            source.write_text("- Original preference.\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(root / "state.json"),
                    "CODEX_MEMORY_ROOTS": str(root / "memories"),
                },
            ):
                server = self._server(root)
                key = "guild:42:channel:7:user:9"
                record_id = server.memory_view(
                    key,
                    "me",
                    actor_user_id=9,
                    actor_guild_id=42,
                    server_admin=False,
                    super_admin=False,
                )["records"][0]["record_id"]
                with (
                    patch(
                        "theia.server.personality_state.append_audit",
                        return_value=False,
                    ),
                    self.assertRaisesRegex(
                        main.CodexAppServerError, "could not be changed"
                    ),
                ):
                    server.edit_memory(
                        key,
                        record_id,
                        "New preference.",
                        "me",
                        confirmed=True,
                        actor_user_id=9,
                        actor_guild_id=42,
                        server_admin=False,
                        super_admin=False,
                    )
                unchanged = source.read_text(encoding="utf-8")
                server.edit_memory(
                    key,
                    record_id,
                    "New preference.",
                    "me",
                    confirmed=True,
                    actor_user_id=9,
                    actor_guild_id=42,
                    server_admin=False,
                    super_admin=False,
                )
                updated = source.read_text(encoding="utf-8")

        self.assertEqual(unchanged, "- Original preference.\n")
        self.assertIn("New preference.", updated)

    def test_malformed_sources_are_ignored_and_legacy_markdown_is_readable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memories = root / "memories"
            memories.mkdir(parents=True)
            (memories / "MEMORY.md").write_bytes(b"\xff\xfe")
            (memories / "USER.md").write_text(
                "# User\n\nA legacy paragraph.\n\nAnother paragraph.",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(root / "state.json"),
                    "CODEX_MEMORY_ROOTS": str(memories),
                },
            ):
                server = self._server(root)
                result = server.memory_view(
                    "guild:42:channel:7:user:9",
                    "everyone",
                )

        self.assertEqual(result["total_entries"], 2)
        self.assertTrue(
            all(record["scope"] == "legacy/unscoped" for record in result["records"])
        )


class MemoryViewTests(unittest.TestCase):
    def test_memory_embed_supports_customized_total_and_page_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            customizer = main.FrontendCustomizationStore(
                Path(directory) / "frontend.json"
            )
            customizer.set(42, "memory_total_entries", "label", "Entries ({count})")
            customizer.set(42, "memory_page", "label", "Memory page {page}/{pages}")
            customizer.set(42, "memory_search", "label", "Find")
            customizer.set(42, "memory_forget", "label", "Remove")
            customizer.set(42, "memory_edit", "label", "Change")
            view = main._MemoryView(
                ["first memory", "second memory"],
                character_name="Celune",
                owner_id=7,
                total_entries=2,
                customizer=customizer,
                guild_id=42,
            )

        embed = view.embed()
        self.assertIn("Entries (2): 2", embed.description or "")
        self.assertEqual(embed.footer.text, "Memory page 1/2")
        self.assertEqual(
            [getattr(item, "label", None) for item in view.children],
            ["Previous", "Next", "Find", "Remove", "Change"],
        )
        self.assertEqual(
            [getattr(item, "style", None) for item in view.children],
            [
                discord.ButtonStyle.secondary,
                discord.ButtonStyle.secondary,
                discord.ButtonStyle.secondary,
                discord.ButtonStyle.danger,
                discord.ButtonStyle.primary,
            ],
        )

    def test_persistent_view_store_restores_a_memory_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "views.json"
            store = _PersistentViewStore(path)
            view = main._MemoryView(
                ["first memory", "second memory"],
                character_name="Celune",
                character_slug="cel",
                scope="server",
                owner_id=7,
                total_entries=2,
            )
            store.register(view, SimpleNamespace(id=456))

            restored_store = _PersistentViewStore(path)
            restored: list[Any] = []

            def add_view(candidate: Any, *, message_id: int) -> None:
                restored.append((candidate, message_id))

            fake_bot = SimpleNamespace(add_view=add_view)
            restored_store.restore(
                cast(Any, fake_bot),
                main.bot._restore_persistent_view,
            )

        self.assertEqual(len(restored), 1)
        restored_view, message_id = restored[0]
        self.assertEqual(message_id, 456)
        self.assertTrue(restored_view.is_persistent())
        self.assertEqual(restored_view.owner_id, 7)
        self.assertEqual(restored_view.embed().title, "Celune's Memory")
        self.assertIn("first memory", restored_view.embed().description)

    def test_memory_view_search_and_controls_are_owner_locked(self) -> None:
        view = main._MemoryView(
            [
                {
                    "record_id": "0123456789abcdef01234567",
                    "text": "Alpha memory",
                    "source_category": "user_memory",
                    "display_metadata": {
                        "source": "user memory",
                        "scope": "current user",
                        "updated": "recently",
                    },
                },
                {
                    "record_id": "fedcba9876543210fedcba98",
                    "text": "Beta memory",
                    "source_category": "user_memory",
                    "display_metadata": {
                        "source": "user memory",
                        "scope": "current user",
                        "updated": "recently",
                    },
                },
            ],
            character_name="Celune",
            owner_id=7,
        )
        view.search_query = "alpha"
        view._sync_buttons()
        self.assertIn("Alpha memory", view.embed().description or "")
        self.assertNotIn("Beta memory", view.embed().description or "")
        response = SimpleNamespace(send_message=AsyncMock())
        allowed = asyncio.run(
            view.interaction_check(
                cast(
                    discord.Interaction,
                    SimpleNamespace(user=SimpleNamespace(id=8), response=response),
                )
            )
        )

        self.assertFalse(allowed)
        response.send_message.assert_awaited_once()
        self.assertTrue(response.send_message.await_args.kwargs["ephemeral"])
