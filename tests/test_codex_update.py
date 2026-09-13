"""Focused tests for staged Codex CLI updates."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import main
from theia.server.codex_update import CodexUpdater


class CodexUpdaterTests(unittest.IsolatedAsyncioTestCase):
    def _updater(self, root: Path, *, enabled: bool = True) -> CodexUpdater:
        return CodexUpdater(
            root,
            cwd=root,
            environment={"CODEX_HOME": str(root)},
            enabled=enabled,
            interval=86400,
            timeout=30,
        )

    @staticmethod
    def _write_cli(directory: Path, version: str) -> Path:
        executable = directory / "codex"
        executable.write_text(
            f'#!/bin/sh\nif [ "$1" = "--version" ]; then echo "codex {version}"; fi\n',
            encoding="utf-8",
        )
        executable.chmod(0o700)
        return executable

    def _installer_side_effect(self, version: str) -> object:
        def install(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess:
            if command and command[-1] == "--version":
                reported_version = "0.199.0" if "0.199.0" in command[0] else version
                return subprocess.CompletedProcess(
                    command, 0, stdout=f"codex {reported_version}\n", stderr=""
                )
            install_dir = Path(kwargs["env"]["CODEX_INSTALL_DIR"])  # type: ignore[index]
            self._write_cli(install_dir, version)
            return subprocess.CompletedProcess(command, 0)

        return install

    def test_disabled_update_does_not_download_or_select_a_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            updater = self._updater(Path(directory), enabled=False)
            updater._download_installer = Mock()  # type: ignore[method-assign]

            result = updater._maybe_update_sync(force=True)

        self.assertEqual(result.status, "disabled")
        updater._download_installer.assert_not_called()
        self.assertIsNone(updater.active_executable())

    def test_update_stages_candidate_and_persists_private_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            updater = self._updater(root)
            updater._download_installer = Mock(return_value=b"installer")  # type: ignore[method-assign]
            updater._verify_app_server = Mock(return_value=True)  # type: ignore[method-assign]

            with patch(
                "theia.server.codex_update.subprocess.run",
                side_effect=self._installer_side_effect("0.200.0"),
            ):
                result = updater._maybe_update_sync(force=True)

            manifest = json.loads(updater.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(result.status, "updated")
            self.assertEqual(result.version, "0.200.0")
            self.assertTrue(result.activated)
            self.assertEqual(updater.active_executable(), result.executable)
            self.assertEqual(manifest["active_version"], "0.200.0")
            self.assertEqual(Path(manifest["active_dir"]).parent, Path("."))
            self.assertTrue(updater.status()["managed_install"])

    def test_recent_success_skips_another_network_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            updater = self._updater(root)
            updater._download_installer = Mock(return_value=b"installer")  # type: ignore[method-assign]
            updater._verify_app_server = Mock(return_value=True)  # type: ignore[method-assign]

            with patch(
                "theia.server.codex_update.subprocess.run",
                side_effect=self._installer_side_effect("0.200.0"),
            ):
                updater._maybe_update_sync(force=True)

            updater._download_installer.reset_mock()
            result = updater._maybe_update_sync(force=False)

        self.assertEqual(result.status, "skipped")
        updater._download_installer.assert_not_called()

    def test_failed_candidate_is_not_activated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            updater = self._updater(root)
            updater._download_installer = Mock(return_value=b"installer")  # type: ignore[method-assign]
            updater._verify_app_server = Mock(return_value=False)  # type: ignore[method-assign]

            with patch(
                "theia.server.codex_update.subprocess.run",
                side_effect=self._installer_side_effect("0.200.0"),
            ):
                result = updater._maybe_update_sync(force=True)

        self.assertEqual(result.status, "failed")
        self.assertIsNone(updater.active_executable())

    def test_rollback_restores_the_previous_managed_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            updater = self._updater(root)
            previous_dir = updater.install_root / "0.199.0-old"
            previous_dir.mkdir(parents=True)
            previous = self._write_cli(previous_dir, "0.199.0")
            updater._write_manifest(
                {
                    "active_dir": previous_dir.name,
                    "active_version": "0.199.0",
                    "previous_dir": None,
                    "previous_version": None,
                    "last_check_at": 0,
                }
            )
            updater._download_installer = Mock(return_value=b"installer")  # type: ignore[method-assign]
            updater._verify_app_server = Mock(return_value=True)  # type: ignore[method-assign]

            with patch(
                "theia.server.codex_update.subprocess.run",
                side_effect=self._installer_side_effect("0.200.0"),
            ):
                result = updater._maybe_update_sync(force=True)

            self.assertTrue(updater.rollback(result.install_dir))
            self.assertEqual(updater.active_executable(), str(previous))

    async def test_explicit_cli_path_is_not_replaced_by_managed_update(self) -> None:
        with patch.dict(os.environ, {"THEIA_CODEX_CLI": "/explicit/codex"}):
            server = main.CodexAppServer()
            server._codex_updater.maybe_update = AsyncMock()  # type: ignore[method-assign]
            result = await server._maybe_update_codex()

        self.assertEqual(result.status, "skipped")
        server._codex_updater.maybe_update.assert_not_called()


class CodexConfigurationTests(unittest.TestCase):
    def test_auto_update_configuration_is_opt_in_and_bounded(self) -> None:
        with patch.dict(
            os.environ,
            {
                main.CODEX_AUTO_UPDATE_ENV: "true",
                main.CODEX_UPDATE_INTERVAL_ENV: "300",
                main.CODEX_UPDATE_TIMEOUT_ENV: "20",
            },
        ):
            server = main.CodexAppServer()

        self.assertTrue(server._codex_auto_update)
        self.assertEqual(server._codex_update_interval, 300)
        self.assertEqual(server._codex_update_timeout, 20)


if __name__ == "__main__":
    unittest.main()
