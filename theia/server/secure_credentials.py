"""Encrypted credential lifecycle integration for the Codex runtime."""

from __future__ import annotations

import asyncio
import contextlib
import getpass
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..audio import OpenAICompatibleAudio
from ..audio_provider import QwenAudioProvider
from .vault import (
    CredentialVault,
    InvalidPassphrase,
    VaultError,
    legacy_environment_credentials,
    legacy_secret_environment_names,
    scrub_legacy_dotenv,
    scrub_legacy_environment,
    vault_path,
)


class SecureCredentialLifecycleMixin:
    """Add encrypted vault setup, unlock, runtime wiring, and lock teardown."""

    if TYPE_CHECKING:
        _codex_home: Path
        _global_codex_home: Path
        _codex_environment: dict[str, str]
        _pending: dict[int, Any]
        _realtime_sessions: dict[str, Any]
        _lifecycle_lock: asyncio.Lock

        async def _close_locked(self) -> None: ...

    def enable_secure_credentials(self) -> None:
        """Require the encrypted credential vault for the real launcher."""
        configured = os.getenv("THEIA_VAULT_KEYCHAIN", "false")
        keychain_enabled = configured.strip().casefold() in {"1", "true", "yes", "on"}
        self._credential_vault = CredentialVault(
            vault_path(self._codex_home),
            keychain_enabled=keychain_enabled,
        )
        self._secure_credentials_enabled = True
        self._credentials_ready = False
        self._credential_environment: dict[str, str] = {}
        self._vault_auth_written = False
        self._credential_watchdog_task: asyncio.Task[None] | None = None
        self._credential_last_activity = time.monotonic()
        try:
            self._credential_idle_timeout = max(
                0.0, float(os.getenv("THEIA_VAULT_IDLE_TIMEOUT", "0"))
            )
        except ValueError:
            self._credential_idle_timeout = 0.0

    def secure_vault_snapshot(self) -> dict[str, Any] | None:
        """Return the lock-only dashboard projection when secure mode is active."""
        if not getattr(self, "_secure_credentials_enabled", False):
            return None
        return self._credential_vault.snapshot()

    def _legacy_auth_payload(self) -> str:
        for path in (
            self._codex_home / "auth.json",
            self._global_codex_home / "auth.json",
        ):
            try:
                if path.is_file() and not path.is_symlink():
                    return path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
        return ""

    async def _read_vault_passphrase(self, prompt: str = "") -> str:
        return await asyncio.to_thread(getpass.getpass, prompt)

    async def _create_secure_vault(self) -> None:
        vault = self._credential_vault
        legacy = legacy_environment_credentials()
        auth = self._legacy_auth_payload()
        if not legacy and not auth:
            raise VaultError(
                "The vault is not initialized and no legacy credentials are available."
            )
        vault.set_input_hint(
            "Create a vault passphrase in the terminal and press Enter."
        )
        vault.record_event("INFO", "Vault setup required")
        first = await self._read_vault_passphrase("")
        try:
            second = await self._read_vault_passphrase("")
            if first != second:
                vault.record_event("WARNING", "Passphrases did not match")
                raise VaultError("The vault passphrases did not match.")
            vault.create(
                {"environment": legacy, "codex_auth": auth},
                first,
            )
            if vault.keychain_enabled and not vault.store_passphrase_key(first):
                vault.record_event("WARNING", "OS keychain unavailable")
            self._scrub_legacy_credentials()
        finally:
            del first
            with contextlib.suppress(UnboundLocalError):
                del second

    def _write_vault_auth(self, auth: str) -> None:
        if not auth:
            return
        target = self._codex_home / "auth.json"
        temporary_name: str | None = None
        try:
            self._codex_home.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".auth.", suffix=".tmp", dir=self._codex_home, text=True
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(auth)
                stream.flush()
                os.fsync(stream.fileno())
            with contextlib.suppress(OSError):
                os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, target)
            with contextlib.suppress(OSError):
                target.chmod(0o600)
            self._vault_auth_written = True
        except OSError as exc:
            raise VaultError(
                "The private Codex authentication cache could not be written."
            ) from exc
        finally:
            if temporary_name is not None:
                with contextlib.suppress(OSError):
                    Path(temporary_name).unlink()

    def _persist_vault_auth(self) -> None:
        """Capture a newly-created private Codex login inside the encrypted vault."""
        if not getattr(self, "_secure_credentials_enabled", False) or not getattr(
            self, "_credentials_ready", False
        ):
            return
        path = self._codex_home / "auth.json"
        try:
            if not path.is_file() or path.is_symlink():
                return
            auth = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return
        try:
            current = self._credential_vault.credentials().get("codex_auth")
            if current == auth:
                return
            self._credential_vault.update({"codex_auth": auth})
        except VaultError:
            return
        self._vault_auth_written = True

    def apply_secure_credentials(self) -> None:
        """Configure providers from decrypted credentials without environment export."""
        payload = self._credential_vault.credentials()
        environment = payload.get("environment", {})
        if not isinstance(environment, dict):
            raise VaultError("The credential vault environment payload is invalid.")
        self._credential_environment = {
            key: value
            for key, value in environment.items()
            if isinstance(key, str) and isinstance(value, str)
        }
        self._audio = OpenAICompatibleAudio.from_environment(
            self._credential_environment
        )
        self.__dict__["_qwen_audio"] = QwenAudioProvider.from_environment(
            self._credential_environment
        )
        self.__dict__.pop("_qwen_perception_client", None)
        for name in legacy_secret_environment_names():
            self._codex_environment.pop(name, None)
        auth = payload.get("codex_auth")
        if isinstance(auth, str):
            self._write_vault_auth(auth)
        self._scrub_legacy_credentials()
        self._credentials_ready = True
        self._credential_last_activity = time.monotonic()
        self._start_credential_watchdog()

    def _clear_provider_credentials(self) -> None:
        """Drop provider objects that may retain decrypted credential strings."""
        self._audio = OpenAICompatibleAudio.from_environment({})
        self.__dict__["_qwen_audio"] = QwenAudioProvider.from_environment({})
        self.__dict__.pop("_qwen_perception_client", None)

    def _scrub_legacy_credentials(self) -> None:
        """Remove one-time plaintext migration sources after encrypted import."""
        scrub_legacy_environment()
        paths: list[Path] = []
        configured = os.getenv("THEIA_ENV_FILE")
        if configured:
            paths.append(Path(configured).expanduser())
        roots = (
            Path.cwd(),
            Path(sys.argv[0]).expanduser().resolve().parent,
            Path(sys.executable).expanduser().resolve().parent,
            Path(__file__).resolve().parents[2],
        )
        paths.extend(root / ".env" for root in roots)
        scrub_legacy_dotenv(tuple(paths))

    def _start_credential_watchdog(self) -> None:
        if self._credential_idle_timeout <= 0 or self._credential_watchdog_task:
            return
        self._credential_watchdog_task = asyncio.create_task(
            self._credential_watchdog()
        )

    async def _credential_watchdog(self) -> None:
        interval = max(1.0, min(60.0, self._credential_idle_timeout / 4))
        try:
            while self._credentials_ready:
                await asyncio.sleep(interval)
                if (
                    self._pending
                    or self._realtime_sessions
                    or time.monotonic() - self._credential_last_activity
                    < self._credential_idle_timeout
                ):
                    continue
                await self.lock_secure_credentials(reason="Vault idle timeout")
                return
        finally:
            if self._credential_watchdog_task is asyncio.current_task():
                self._credential_watchdog_task = None

    def touch_secure_credentials(self) -> None:
        """Record provider activity for the optional inactivity lock."""
        if getattr(self, "_secure_credentials_enabled", False) and getattr(
            self, "_credentials_ready", False
        ):
            self._credential_last_activity = time.monotonic()

    async def unlock_secure_credentials(self) -> None:
        """Unlock or initialize the vault before any provider process starts."""
        if not getattr(self, "_secure_credentials_enabled", False):
            return
        vault = self._credential_vault
        mode = os.getenv("THEIA_VAULT_UNLOCK_MODE", "interactive").strip().casefold()
        if mode not in {"interactive", "auto", "unattended"}:
            mode = "interactive"
        if not vault.initialized:
            await self._create_secure_vault()
            self.apply_secure_credentials()
            return
        if mode in {"auto", "unattended"}:
            vault.record_event("INFO", "Checking OS keychain")
            if vault.keychain_enabled and vault.unlock_keychain():
                vault.record_event("INFO", "Vault unlocked from OS keychain")
                self.apply_secure_credentials()
                return
            vault.record_event("WARNING", "No usable keychain credential found")
            if mode == "unattended":
                raise VaultError("No usable OS keychain credential was found.")
        vault.record_event("INFO", "Passphrase required")
        while True:
            try:
                passphrase = await self._read_vault_passphrase("")
                try:
                    try:
                        vault.unlock(passphrase)
                    except InvalidPassphrase:
                        vault.record_event("WARNING", "Invalid passphrase")
                        continue
                    if vault.keychain_enabled and not vault.store_passphrase_key(
                        passphrase
                    ):
                        vault.record_event("WARNING", "OS keychain unavailable")
                finally:
                    del passphrase
                self.apply_secure_credentials()
                return
            except EOFError as exc:
                raise VaultError(
                    "Interactive vault unlock requires a local terminal."
                ) from exc

    def secure_credential(self, name: str) -> str:
        """Read one unlocked credential without exposing the complete payload."""
        if getattr(self, "_secure_credentials_enabled", False) and not getattr(
            self, "_credentials_ready", False
        ):
            raise VaultError("The credential vault is locked.")
        if getattr(self, "_secure_credentials_enabled", False):
            self.touch_secure_credentials()
            return self._credential_environment.get(name, "")
        return os.getenv(name, "").strip()

    async def lock_secure_credentials(self, *, reason: str = "Vault locked") -> None:
        """Stop the Codex process and discard decrypted vault material."""
        if not getattr(self, "_secure_credentials_enabled", False):
            return
        watchdog = self._credential_watchdog_task
        if watchdog is not None and watchdog is not asyncio.current_task():
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog
        self._credential_watchdog_task = None
        async with self._lifecycle_lock:
            self._credential_vault.record_event("INFO", reason)
            await self._close_locked()
            self._credential_vault.lock(reason=reason)
            self._credential_environment.clear()
            self._credentials_ready = False
            self._clear_provider_credentials()
            if self._vault_auth_written:
                with contextlib.suppress(OSError):
                    (self._codex_home / "auth.json").unlink()
                self._vault_auth_written = False
