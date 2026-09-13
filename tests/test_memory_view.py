# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable
from tests.test_support import *


class MemoryViewTests(unittest.TestCase):
    def test_memory_embed_supports_customized_total_and_page_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            customizer = main.FrontendCustomizationStore(
                Path(directory) / "frontend.json"
            )
            customizer.set(42, "memory_total_entries", "label", "Entries ({count})")
            customizer.set(42, "memory_page", "label", "Memory page {page}/{pages}")
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
