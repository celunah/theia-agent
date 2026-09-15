from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import main
from theia.server.codex_update import CodexUpdateResult


class CodexDatabaseRepairTests(unittest.IsolatedAsyncioTestCase):
    def test_sqlite_startup_detection_is_narrow(self) -> None:
        server = main.CodexAppServer()

        server._stderr_tail = [
            "Error: failed to initialize sqlite state runtime under private home"
        ]
        self.assertTrue(server._has_sqlite_startup_failure())

        server._stderr_tail = ["Error: failed to authenticate with Codex"]
        self.assertFalse(server._has_sqlite_startup_failure())

    def test_database_backup_preserves_only_sqlite_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                "os.environ",
                {"THEIA_HOME": str(root), "CODEX_HOME": str(root / "global")},
            ):
                server = main.CodexAppServer()
            server._codex_home = root
            (root / "state_5.sqlite").write_text("state", encoding="utf-8")
            (root / "state_5.sqlite-wal").write_text("wal", encoding="utf-8")
            (root / "auth.json").write_text("auth", encoding="utf-8")

            backup = server._backup_codex_databases()

            self.assertIsNotNone(backup)
            assert backup is not None
            self.assertEqual(
                sorted(path.name for path in backup.iterdir()),
                ["state_5.sqlite", "state_5.sqlite-wal"],
            )
            self.assertFalse((root / "state_5.sqlite").exists())
            self.assertEqual((root / "auth.json").read_text(encoding="utf-8"), "auth")

    async def test_sqlite_startup_repair_retries_once_and_keeps_auth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                "os.environ",
                {"THEIA_HOME": str(root), "CODEX_HOME": str(root / "global")},
            ):
                server = main.CodexAppServer()
            server._codex_home = root
            server._codex_environment["CODEX_HOME"] = str(root)
            (root / "state_5.sqlite").write_text("state", encoding="utf-8")
            (root / "auth.json").write_text("auth", encoding="utf-8")

            process = SimpleNamespace(
                returncode=None,
                stdin=None,
                stdout=None,
                stderr=None,
            )
            request_count = 0

            async def request_side_effect(*_args: object, **_kwargs: object) -> dict:
                nonlocal request_count
                request_count += 1
                if request_count == 1:
                    server._stderr_tail = ["failed to initialize sqlite state runtime"]
                    raise main.CodexAppServerError("The Codex App Server exited.")
                return {}

            async def close_process() -> None:
                server._process = None

            server._request = AsyncMock(side_effect=request_side_effect)
            server._send = AsyncMock()
            server._refresh_realtime_capability = AsyncMock()
            server._configure_shared_roots = AsyncMock()
            server.refresh_account = AsyncMock()
            server.refresh_skills = AsyncMock()
            server.loaded_threads = AsyncMock(return_value=())
            server._start_memory_watchdog = Mock()
            server._maybe_update_codex = AsyncMock(
                return_value=CodexUpdateResult("disabled")
            )
            server._codex_executable = Mock(return_value="codex")
            server.codex_cli_version = Mock(return_value="0.153.0")
            server._read_output = AsyncMock()
            server._read_stderr = AsyncMock()
            server._close_locked = AsyncMock(side_effect=close_process)

            with patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=process),
            ) as launch:
                await server.start()

            self.assertEqual(launch.await_count, 2)
            self.assertEqual(request_count, 2)
            self.assertEqual((root / "auth.json").read_text(encoding="utf-8"), "auth")
            backups = tuple(root.glob("codex-database-repair-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(
                (backups[0] / "state_5.sqlite").read_text(encoding="utf-8"),
                "state",
            )


if __name__ == "__main__":
    unittest.main()
