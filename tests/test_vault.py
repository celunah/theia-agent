"""Encrypted credential vault contract tests."""

# pylint: disable=consider-using-with

from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import keyring

import main

from theia.server.vault import (
    CredentialVault,
    InvalidPassphrase,
    VaultError,
    _read_protected_passphrase_file,
)


class VaultTests(unittest.TestCase):
    """Verify authenticated persistence and complete lock transitions."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="theia-vault-")
        self.addCleanup(self._temporary.cleanup)
        self.path = Path(self._temporary.name) / "credentials.vault"
        self.credentials = {
            "environment": {
                "TOKEN": "discord-secret",
                "THEIA_QWEN_PERCEPTION_API_KEY": "qwen-secret",
            },
            "codex_auth": '{"tokens":{"access_token":"codex-secret"}}',
        }

    def test_round_trip_change_and_lock(self) -> None:
        vault = CredentialVault(self.path)
        vault.create(self.credentials, "correct horse battery")
        self.assertFalse(vault.locked)
        self.assertEqual(vault.credentials(), self.credentials)

        vault.lock()
        self.assertTrue(vault.locked)
        with self.assertRaises(InvalidPassphrase):
            vault.unlock("incorrect horse battery")
        self.assertTrue(vault.locked)

        vault.unlock("correct horse battery")
        vault.change_passphrase("correct horse battery", "new correct battery")
        vault.lock()
        vault.unlock("new correct battery")
        self.assertEqual(vault.credentials(), self.credentials)

    def test_tampering_is_rejected_without_unlocking(self) -> None:
        vault = CredentialVault(self.path)
        vault.create(self.credentials, "correct horse battery")
        vault.lock()
        envelope = json.loads(self.path.read_text(encoding="utf-8"))
        ciphertext = envelope["payload"]["ciphertext"]
        envelope["payload"]["ciphertext"] = (
            "A" if ciphertext[0] != "A" else "B"
        ) + ciphertext[1:]
        self.path.write_text(json.dumps(envelope), encoding="utf-8")

        with self.assertRaises(InvalidPassphrase):
            vault.unlock("correct horse battery")
        self.assertTrue(vault.locked)

    def test_keychain_stores_only_the_derived_wrapping_key(self) -> None:
        vault = CredentialVault(self.path, keychain_enabled=True)
        vault.create(self.credentials, "correct horse battery")
        stored: dict[tuple[str, str], str] = {}

        with (
            patch("theia.server.vault._system_keyring", return_value=object()),
            patch.object(
                keyring,
                "set_password",
                side_effect=lambda service, user, value: stored.__setitem__(
                    (service, user), value
                ),
            ),
            patch.object(
                keyring,
                "get_password",
                side_effect=lambda service, user: stored.get((service, user)),
            ),
        ):
            self.assertTrue(vault.store_passphrase_key("correct horse battery"))
            vault.lock()
            self.assertTrue(vault.unlock_keychain())

        self.assertEqual(vault.credentials(), self.credentials)
        self.assertNotIn("correct horse battery", stored.values())

    @unittest.skipUnless(hasattr(os, "geteuid"), "requires POSIX ownership checks")
    def test_recovery_file_must_be_owner_only_regular_file(self) -> None:
        path = Path(self._temporary.name) / "vault-passphrase"
        path.write_text("correct horse battery\n", encoding="utf-8")
        path.chmod(0o640)
        self.assertIsNone(_read_protected_passphrase_file(path))

        path.chmod(0o600)
        self.assertEqual(_read_protected_passphrase_file(path), "correct horse battery")
        link = Path(self._temporary.name) / "vault-passphrase-link"
        link.symlink_to(path)
        self.assertIsNone(_read_protected_passphrase_file(link))

    def test_update_reencrypts_payload_without_changing_unlock_contract(self) -> None:
        vault = CredentialVault(self.path)
        vault.create(self.credentials, "correct horse battery")
        vault.update({"codex_auth": '{"access_token":"redacted"}'})
        vault.lock()
        vault.unlock("correct horse battery")

        self.assertEqual(
            vault.credentials()["codex_auth"], '{"access_token":"redacted"}'
        )
        self.assertEqual(
            vault.credentials()["environment"], self.credentials["environment"]
        )


class SecureLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_unattended_start_repairs_an_invalid_keychain_entry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="theia-secure-recovery-") as directory:
            root = Path(directory)
            recovery_file = root / "vault-passphrase"
            recovery_file.write_text("correct horse battery\n", encoding="utf-8")
            recovery_file.chmod(0o600)
            vault_path = root / "credentials.vault"
            CredentialVault(vault_path, keychain_enabled=True).create(
                {
                    "environment": {"TOKEN": "discord-secret"},
                    "codex_auth": "{}",
                },
                "correct horse battery",
            )
            vault_id = json.loads(vault_path.read_text(encoding="utf-8"))["vault_id"]
            stored = {vault_id: base64.urlsafe_b64encode(b"\x01" * 32).decode("ascii")}

            with (
                patch.dict(
                    os.environ,
                    {
                        "THEIA_HOME": str(root),
                        "CODEX_HOME": str(root / "global-codex"),
                        "THEIA_VAULT_KEYCHAIN": "true",
                        "THEIA_VAULT_UNLOCK_MODE": "unattended",
                        "THEIA_VAULT_PASSPHRASE_FILE": str(recovery_file),
                    },
                    clear=False,
                ),
                patch("theia.server.vault._system_keyring", return_value=object()),
                patch.object(
                    keyring,
                    "get_password",
                    side_effect=lambda _service, username: stored.get(username),
                ),
                patch.object(
                    keyring,
                    "set_password",
                    side_effect=lambda _service, username, value: stored.__setitem__(
                        username, value
                    ),
                ),
            ):
                server = main.CodexAppServer()
                server.enable_secure_credentials()
                server._read_vault_passphrase = AsyncMock(  # type: ignore[method-assign]
                    side_effect=AssertionError("unattended recovery prompted")
                )

                await server.unlock_secure_credentials()

                self.assertTrue(server._credentials_ready)
                self.assertEqual(server.secure_credential("TOKEN"), "discord-secret")
                self.assertEqual(len(stored), 1)
                self.assertNotIn("correct horse battery", stored.values())
                self.assertTrue(
                    any(
                        "restored from recovery file" in event["detail"]
                        for event in server._credential_vault.events()
                    )
                )
                self.assertTrue(
                    all(
                        "correct horse battery" not in event["detail"]
                        for event in server._credential_vault.events()
                    )
                )
                server._credential_vault.lock()
                self.assertTrue(server._credential_vault.unlock_keychain())
                server._read_vault_passphrase.assert_not_awaited()

    async def test_unattended_start_rejects_missing_or_wrong_recovery_file(
        self,
    ) -> None:
        for recovery_value in (None, "wrong horse battery"):
            with (
                self.subTest(
                    recovery_value="missing" if recovery_value is None else "wrong"
                ),
                tempfile.TemporaryDirectory(
                    prefix="theia-secure-recovery-failure-"
                ) as directory,
            ):
                root = Path(directory)
                recovery_file = root / "vault-passphrase"
                if recovery_value is not None:
                    recovery_file.write_text(recovery_value, encoding="utf-8")
                    recovery_file.chmod(0o600)
                CredentialVault(root / "credentials.vault").create(
                    {"environment": {"TOKEN": "discord-secret"}},
                    "correct horse battery",
                )
                with (
                    patch.dict(
                        os.environ,
                        {
                            "THEIA_HOME": str(root),
                            "CODEX_HOME": str(root / "global-codex"),
                            "THEIA_VAULT_KEYCHAIN": "true",
                            "THEIA_VAULT_UNLOCK_MODE": "unattended",
                            "THEIA_VAULT_PASSPHRASE_FILE": str(recovery_file),
                        },
                        clear=False,
                    ),
                    patch(
                        "theia.server.vault._system_keyring",
                        return_value=object(),
                    ),
                    patch.object(keyring, "get_password", return_value=None),
                    patch.object(keyring, "set_password") as set_password,
                ):
                    server = main.CodexAppServer()
                    server.enable_secure_credentials()
                    server._read_vault_passphrase = AsyncMock()  # type: ignore[method-assign]

                    with self.assertRaises(VaultError):
                        await server.unlock_secure_credentials()

                    server._read_vault_passphrase.assert_not_awaited()
                    set_password.assert_not_called()

    async def test_first_secure_startup_creates_vault_and_scrubs_legacy_token(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="theia-secure-") as directory:
            root = Path(directory)
            with patch.dict(
                "os.environ",
                {
                    "THEIA_HOME": str(root),
                    "TOKEN": "discord-secret",
                    "THEIA_VAULT_KEYCHAIN": "false",
                },
                clear=False,
            ):
                server = main.CodexAppServer()
                server.enable_secure_credentials()
                server._read_vault_passphrase = AsyncMock(  # type: ignore[method-assign]
                    side_effect=["correct horse battery", "correct horse battery"]
                )

                await server.unlock_secure_credentials()

                self.assertTrue(server._credentials_ready)
                self.assertEqual(server.secure_credential("TOKEN"), "discord-secret")
                self.assertNotIn("TOKEN", os.environ)
                self.assertTrue((root / "credentials.vault").is_file())

                await server.lock_secure_credentials(reason="test lock")
                self.assertFalse(server._credentials_ready)
                with self.assertRaises(VaultError):
                    server.secure_credential("TOKEN")


if __name__ == "__main__":
    unittest.main()
