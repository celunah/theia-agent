"""Opt-in, staged updates for the Codex CLI."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


CODEX_AUTO_UPDATE_ENV = "THEIA_CODEX_AUTO_UPDATE"
CODEX_UPDATE_INTERVAL_ENV = "THEIA_CODEX_UPDATE_INTERVAL"
CODEX_UPDATE_TIMEOUT_ENV = "THEIA_CODEX_UPDATE_TIMEOUT"
DEFAULT_CODEX_AUTO_UPDATE = False
DEFAULT_CODEX_UPDATE_INTERVAL = 24 * 60 * 60.0
DEFAULT_CODEX_UPDATE_TIMEOUT = 120.0
CODEX_INSTALLER_URL = "https://chatgpt.com/codex/install.sh"
CODEX_INSTALLER_WINDOWS_URL = "https://chatgpt.com/codex/install.ps1"
CODEX_INSTALLER_MAX_BYTES = 1024 * 1024
CODEX_NPM_PACKAGE = "@openai/codex"
_VERSION_RE = re.compile(r"(?<![0-9])([0-9]+\.[0-9]+\.[0-9]+)(?![0-9])")
_INSTALL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")


@dataclass(frozen=True)
class CodexUpdateResult:
    """Describe one bounded automatic update attempt."""

    status: str
    version: str | None = None
    executable: str | None = None
    install_dir: str | None = None
    activated: bool = False


def _version_from_output(output: str) -> str | None:
    match = _VERSION_RE.search(output)
    return match.group(1) if match else None


def _find_executable(root: Path) -> Path | None:
    """Find a Codex command produced by an installer in ``root``."""
    for directory in (root, root / "bin", root / "node_modules" / ".bin"):
        for name in ("codex.exe", "codex.cmd", "codex"):
            candidate = directory / name
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                continue
    return None


class CodexUpdater:
    """Stage and activate official standalone Codex installations."""

    def __init__(
        self,
        home: Path,
        *,
        cwd: Path,
        environment: Mapping[str, str],
        enabled: bool,
        interval: float,
        timeout: float,
    ) -> None:
        self.home = home.resolve()
        self.cwd = cwd.resolve()
        self.environment = dict(environment)
        self.enabled = enabled
        self.interval = max(0.0, interval)
        self.timeout = max(10.0, timeout)
        self.install_root = self.home / "codex-versions"
        self.manifest_path = self.home / "codex-update.json"
        self._lock = threading.Lock()

    def active_executable(self) -> str | None:
        """Return the validated managed executable selected by the manifest."""
        manifest = self._load_manifest()
        install_dir = self._manifest_directory(manifest.get("active_dir"))
        executable = _find_executable(install_dir) if install_dir else None
        if (
            executable is not None
            and os.name != "nt"
            and not os.access(executable, os.X_OK)
        ):
            executable = None
        return str(executable) if executable is not None else None

    def status(self) -> dict[str, object]:
        """Return sanitized updater state for administrator diagnostics."""
        manifest = self._load_manifest()
        active_dir = self._manifest_directory(manifest.get("active_dir"))
        return {
            "enabled": self.enabled,
            "managed_version": str(manifest.get("active_version") or "") or None,
            "managed_install": active_dir is not None
            and _find_executable(active_dir) is not None,
            "last_check_at": manifest.get("last_check_at"),
        }

    async def maybe_update(self, *, force: bool = False) -> CodexUpdateResult:
        """Run an update off the event loop and keep failures non-fatal."""
        try:
            return await asyncio.to_thread(self._maybe_update_sync, force=force)
        except Exception:  # noqa: BLE001 - an updater failure must not block startup
            return CodexUpdateResult("failed")

    def rollback(self, install_dir: str | None) -> bool:
        """Restore the previous managed installation after startup failure."""
        if not install_dir:
            return False
        with self._lock:
            manifest = self._load_manifest()
            current = self._manifest_directory(manifest.get("active_dir"))
            if current is None or current.name != Path(install_dir).name:
                return False
            previous = self._manifest_directory(manifest.get("previous_dir"))
            if previous is not None and _find_executable(previous) is not None:
                manifest["active_dir"] = previous.name
                manifest["active_version"] = manifest.get("previous_version")
            else:
                manifest["active_dir"] = None
                manifest["active_version"] = None
            manifest["previous_dir"] = None
            manifest["previous_version"] = None
            self._write_manifest(manifest)
            return True

    def _maybe_update_sync(self, *, force: bool) -> CodexUpdateResult:
        if not self.enabled:
            return CodexUpdateResult("disabled")
        with self._lock:
            manifest = self._load_manifest()
            active = self.active_executable()
            if not force and active and self._check_is_fresh(manifest):
                return CodexUpdateResult(
                    "skipped",
                    version=str(manifest.get("active_version") or "") or None,
                    executable=active,
                    install_dir=str(
                        self._manifest_directory(manifest.get("active_dir")) or ""
                    ),
                )
            try:
                result = self._install(manifest, active)
            except (OSError, subprocess.SubprocessError, ValueError):
                return CodexUpdateResult("failed")
            if result.status in {"updated", "unchanged"}:
                manifest = self._load_manifest()
                manifest["last_check_at"] = time.time()
                self._write_manifest(manifest)
            return result

    def _install(
        self, manifest: dict[str, object], active: str | None
    ) -> CodexUpdateResult:
        self.home.mkdir(parents=True, exist_ok=True)
        self.home.chmod(0o700)
        self.install_root.mkdir(parents=True, exist_ok=True)
        self.install_root.chmod(0o700)
        script = self._download_installer()
        staging = Path(
            tempfile.mkdtemp(prefix=".codex-stage-", dir=str(self.install_root))
        )
        final_dir: Path | None = None
        try:
            environment = dict(self.environment)
            environment.update(
                {
                    "CODEX_HOME": str(self.home),
                    "CODEX_INSTALL_DIR": str(staging),
                    "CODEX_NON_INTERACTIVE": "1",
                }
            )
            if self._use_npm_installer():
                completed = self._run_npm_installer(staging, environment)
            else:
                script = self._download_installer()
                command = self._installer_command()
                completed = subprocess.run(
                    command,
                    input=script,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=str(self.cwd if self.cwd.is_dir() else self.home),
                    env=environment,
                    check=False,
                    timeout=self.timeout,
                )
            if completed.returncode != 0:
                return CodexUpdateResult("failed")
            candidate = _find_executable(staging)
            version = self._cli_version(candidate, environment)
            if candidate is None or version is None:
                return CodexUpdateResult("failed")
            if not self._verify_app_server(candidate, environment):
                return CodexUpdateResult("failed")
            active_version = _version_from_output(
                self._version_output(Path(active), environment) if active else ""
            )
            if active and active_version == version:
                return CodexUpdateResult(
                    "unchanged", version=version, executable=active
                )

            final_dir = self.install_root / f"{version}-{uuid.uuid4().hex[:10]}"
            staging.rename(final_dir)
            final_executable = _find_executable(final_dir)
            if final_executable is None:
                return CodexUpdateResult("failed")
            previous_dir = self._manifest_directory(manifest.get("active_dir"))
            self._write_manifest(
                {
                    "active_dir": final_dir.name,
                    "active_version": version,
                    "previous_dir": previous_dir.name if previous_dir else None,
                    "previous_version": manifest.get("active_version"),
                    "last_check_at": time.time(),
                }
            )
            return CodexUpdateResult(
                "updated",
                version=version,
                executable=str(final_executable),
                install_dir=str(final_dir),
                activated=True,
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if final_dir is not None and not self._manifest_uses(final_dir):
                shutil.rmtree(final_dir, ignore_errors=True)

    def _use_npm_installer(self) -> bool:
        """Use npm in Docker so the updated package files persist with Theia."""
        return self.environment.get("THEIA_CONTAINER", "").strip().casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _run_npm_installer(
        self, staging: Path, environment: Mapping[str, str]
    ) -> subprocess.CompletedProcess[bytes]:
        npm = shutil.which("npm")
        if npm is None:
            raise OSError("npm is unavailable")
        command = [
            npm,
            "install",
            "--prefix",
            str(staging),
            "--save-exact",
            "--ignore-scripts",
            "--no-audit",
            "--no-fund",
            f"{CODEX_NPM_PACKAGE}@latest",
        ]
        npm_environment = dict(environment)
        npm_environment["NPM_CONFIG_CACHE"] = str(self.home / ".npm-cache")
        npm_environment["NPM_CONFIG_UPDATE_NOTIFIER"] = "false"
        return subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(self.cwd if self.cwd.is_dir() else self.home),
            env=npm_environment,
            check=False,
            timeout=self.timeout,
        )

    def _download_installer(self) -> bytes:
        installer_url = (
            CODEX_INSTALLER_WINDOWS_URL if os.name == "nt" else CODEX_INSTALLER_URL
        )
        request = urllib.request.Request(
            installer_url,
            headers={"User-Agent": "Theia-Codex-Updater"},
        )
        with urllib.request.urlopen(
            request, timeout=min(self.timeout, 30.0)
        ) as response:
            script = response.read(CODEX_INSTALLER_MAX_BYTES + 1)
        if len(script) > CODEX_INSTALLER_MAX_BYTES:
            raise ValueError("Codex installer is too large")
        return script

    @staticmethod
    def _installer_command() -> list[str]:
        if os.name == "nt":
            powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
            if powershell is None:
                raise OSError("PowerShell is unavailable")
            return [powershell, "-NoProfile", "-NonInteractive", "-Command", "-"]
        shell = shutil.which("sh")
        if shell is None:
            raise OSError("POSIX shell is unavailable")
        return [str(shell)]

    def _cli_version(
        self, executable: Path | None, environment: Mapping[str, str]
    ) -> str | None:
        if executable is None:
            return None
        return _version_from_output(self._version_output(executable, environment))

    def _version_output(self, executable: Path, environment: Mapping[str, str]) -> str:
        try:
            result = subprocess.run(
                [str(executable), "--version"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                env=environment,
                text=True,
                check=False,
                timeout=min(10.0, self.timeout),
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return f"{result.stdout}\n{result.stderr}"

    def _verify_app_server(
        self, executable: Path, environment: Mapping[str, str]
    ) -> bool:
        try:
            with subprocess.Popen(
                [str(executable), "app-server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=str(self.cwd if self.cwd.is_dir() else self.home),
                env=environment,
                bufsize=0,
            ) as process:
                return self._verify_app_server_process(process)
        except OSError:
            return False

    def _verify_app_server_process(self, process: subprocess.Popen) -> bool:
        """Exchange one initialize request with a candidate app-server."""
        lines: queue.Queue[bytes] = queue.Queue(maxsize=16)

        def collect_output() -> None:
            if process.stdout is None:
                return
            try:
                for line in process.stdout:
                    with contextlib.suppress(queue.Full):
                        lines.put_nowait(line)
            except (OSError, ValueError):
                return

        reader = threading.Thread(target=collect_output, daemon=True)
        reader.start()
        valid = False
        try:
            if process.stdin is None:
                return False
            request = {
                "method": "initialize",
                "id": 1,
                "params": {
                    "clientInfo": {
                        "name": "theia-updater",
                        "title": "Theia Codex updater",
                        "version": "1.0.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            }
            process.stdin.write(
                (json.dumps(request, separators=(",", ":")) + "\n").encode()
            )
            process.stdin.flush()
            deadline = time.monotonic() + min(15.0, self.timeout)
            while time.monotonic() < deadline:
                try:
                    raw_line = lines.get(timeout=max(0.05, deadline - time.monotonic()))
                except queue.Empty:
                    break
                try:
                    response = json.loads(raw_line.decode("utf-8", errors="replace"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if response.get("id") == 1:
                    valid = "error" not in response and isinstance(
                        response.get("result"), dict
                    )
                    break
        except (BrokenPipeError, OSError, ValueError):
            valid = False
        finally:
            if process.poll() is None:
                with contextlib.suppress(OSError):
                    process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                    process.wait(timeout=5)
            if process.poll() is None:
                with contextlib.suppress(OSError):
                    process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                    process.wait(timeout=5)
        return valid

    def _check_is_fresh(self, manifest: Mapping[str, object]) -> bool:
        value = manifest.get("last_check_at", 0.0)
        if not isinstance(value, (int, float, str)):
            return False
        try:
            last_check = float(value)
        except (TypeError, ValueError):
            return False
        return time.time() - last_check < self.interval

    def _manifest_directory(self, value: object) -> Path | None:
        if not isinstance(value, str) or not _INSTALL_NAME_RE.fullmatch(value):
            return None
        candidate = self.install_root / value
        try:
            if candidate.parent != self.install_root or not candidate.is_dir():
                return None
        except OSError:
            return None
        return candidate

    def _load_manifest(self) -> dict[str, object]:
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_manifest(self, manifest: Mapping[str, object]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".codex-update-", suffix=".tmp", dir=str(self.home)
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(
                json.dumps(dict(manifest), separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            temporary.chmod(0o600)
            os.replace(temporary, self.manifest_path)
        finally:
            with contextlib.suppress(OSError):
                temporary.unlink()

    def _manifest_uses(self, install_dir: Path) -> bool:
        manifest = self._load_manifest()
        active = self._manifest_directory(manifest.get("active_dir"))
        return active is not None and active == install_dir
