"""Startup storage ownership and Docker configuration tests."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from theia import permissions


ROOT = Path(__file__).resolve().parents[1]


class StoragePermissionTests(unittest.TestCase):
    """Verify storage repair and fail-closed startup behavior."""

    @unittest.skipUnless(os.name == "posix", "container ownership is POSIX-only")
    def test_container_root_repairs_ownership_before_dropping_privileges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            workspace = root / "workspace"
            state = runtime / "sessions.json"
            state.parent.mkdir()
            state.write_text("{}", encoding="utf-8")

            with (
                patch.dict(
                    os.environ,
                    {
                        "THEIA_CONTAINER": "1",
                        "THEIA_UID": "1234",
                        "THEIA_GID": "2345",
                    },
                ),
                patch.object(permissions.os, "geteuid", side_effect=[0, 1234]),
                patch.object(permissions.os, "getegid", side_effect=[0, 2345]),
                patch.object(permissions.os, "chown") as chown,
                patch.object(permissions.os, "setgroups") as setgroups,
                patch.object(permissions.os, "setgid") as setgid,
                patch.object(permissions.os, "setuid") as setuid,
            ):
                repaired = permissions.prepare_runtime_storage(
                    runtime, workspace, state
                )

            self.assertTrue(repaired)
            self.assertGreaterEqual(chown.call_count, 3)
            self.assertTrue(
                all(call.args[1:3] == (1234, 2345) for call in chown.call_args_list)
            )
            setgroups.assert_called_once_with([])
            setgid.assert_called_once_with(2345)
            setuid.assert_called_once_with(1234)

    @unittest.skipUnless(os.name == "posix", "container ownership is POSIX-only")
    def test_chown_failure_is_fatal_when_access_is_still_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            workspace = root / "workspace"

            with (
                patch.dict(
                    os.environ,
                    {
                        "THEIA_CONTAINER": "1",
                        "THEIA_UID": "1234",
                        "THEIA_GID": "2345",
                    },
                ),
                patch.object(permissions.os, "geteuid", side_effect=[0, 1234]),
                patch.object(permissions.os, "getegid", side_effect=[0, 2345]),
                patch.object(permissions.os, "chown", side_effect=PermissionError),
                patch.object(permissions.os, "setgroups"),
                patch.object(permissions.os, "setgid"),
                patch.object(permissions.os, "setuid"),
                patch.object(permissions.os, "access", return_value=False),
                self.assertRaisesRegex(
                    permissions.StoragePermissionError,
                    "storage permissions remain unusable",
                ),
            ):
                permissions.prepare_runtime_storage(runtime, workspace)

    def test_codex_startup_exposes_fatal_storage_failure_to_lighthouse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            error = permissions.StoragePermissionError(
                "workspace", "storage permissions remain unusable"
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "THEIA_HOME": str(root / "runtime"),
                        "THEIA_STATE": str(root / "runtime" / "sessions.json"),
                        "CODEX_CWD": str(root / "workspace"),
                    },
                ),
                patch(
                    "theia.server.state.prepare_runtime_storage",
                    side_effect=error,
                ),
                self.assertLogs("theia.codex", level="CRITICAL") as captured,
            ):
                server = main.CodexAppServer()

            self.assertIn("FATAL", "\n".join(captured.output))
            self.assertEqual(server.startup_snapshot()["status"], "degraded")
            self.assertEqual(server.startup_snapshot()["severity"], "FATAL")
            with self.assertRaisesRegex(main.CodexAppServerError, "FATAL"):
                asyncio.run(server.start())

    def test_compose_prevents_implicit_bind_source_creation(self) -> None:
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertEqual(compose.count("create_host_path: false"), 2)
        self.assertNotIn("\n    user:", compose)
        self.assertIn('THEIA_CONTAINER: "1"', compose)
        self.assertIn("USER root", dockerfile)
        self.assertIn("THEIA_CONTAINER=1", dockerfile)


if __name__ == "__main__":
    unittest.main()
