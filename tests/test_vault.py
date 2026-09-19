"""Encrypted credential vault contract tests."""

# pylint: disable=consider-using-with

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import keyring

import main

from theia.server.vault import CredentialVault, InvalidPassphrase, VaultError


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
