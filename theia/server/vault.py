"""Encrypted credentials storage and local unlock primitives."""

from __future__ import annotations

import base64
import contextlib
import json
import os
import re
import secrets
import tempfile
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import keyring
from argon2.low_level import Type, hash_secret_raw
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


VAULT_FORMAT = "theia-auth-vault"
VAULT_VERSION = 1
VAULT_FILE_ENV = "THEIA_VAULT_PATH"
VAULT_UNLOCK_MODE_ENV = "THEIA_VAULT_UNLOCK_MODE"
VAULT_KEYCHAIN_ENV = "THEIA_VAULT_KEYCHAIN"
VAULT_KEYCHAIN_SERVICE = "theia-agent-vault"
VAULT_MIN_PASSPHRASE_LENGTH = 12
VAULT_MAX_BYTES = 4 * 1024 * 1024
VAULT_EVENT_LIMIT = 12
_KDF_MEMORY_KIB = 64 * 1024
_KDF_TIME_COST = 3
_KDF_PARALLELISM = 1
_KEY_LENGTH = 32
_SALT_LENGTH = 16
_NONCE_LENGTH = 12
_SECRET_ENV_NAMES = (
    "TOKEN",
    "DISCORD_TOKEN",
    "THEIA_DISCORD_TOKEN",
    "STT_TOKEN",
    "THEIA_TRANSCRIPTION_API_KEY",
    "TTS_TOKEN",
    "THEIA_TTS_API_KEY",
    "THEIA_QWEN_AUDIO_TOKEN",
    "THEIA_QWEN_PERCEPTION_API_KEY",
    "DASHSCOPE_API_KEY",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
)
_KEY_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")
_DOTENV_ASSIGNMENT_RE = re.compile(
    r"^\s*(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*="
)


class VaultError(RuntimeError):
    """Base error for safe vault setup, unlock, and persistence failures."""


class InvalidPassphrase(VaultError):
    """The supplied passphrase did not authenticate the vault."""


class VaultCorruptError(VaultError):
    """The vault file is malformed or failed authenticated decryption."""


class VaultLockedError(VaultError):
    """A credential operation was requested while the vault is locked."""


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _b64decode(value: Any, *, length: int | None = None) -> bytes:
    if not isinstance(value, str):
        raise VaultCorruptError("The vault encoding is invalid.")
    try:
        decoded = base64.urlsafe_b64decode(value.encode("ascii"))
    except (ValueError, UnicodeError) as exc:
        raise VaultCorruptError("The vault encoding is invalid.") from exc
    if length is not None and len(decoded) != length:
        raise VaultCorruptError("The vault encoding is invalid.")
    return decoded


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _zeroize(value: bytearray | None) -> None:
    if value is None:
        return
    for index, _ in enumerate(value):
        value[index] = 0


def _validate_passphrase(passphrase: str) -> str:
    if not isinstance(passphrase, str) or len(passphrase) < VAULT_MIN_PASSPHRASE_LENGTH:
        raise VaultError(
            f"The vault passphrase must be at least {VAULT_MIN_PASSPHRASE_LENGTH} characters."
        )
    if "\x00" in passphrase:
        raise VaultError("The vault passphrase contains an invalid character.")
    return passphrase


def _derive_kek(passphrase: str, salt: bytes) -> bytearray:
    _validate_passphrase(passphrase)
    return bytearray(
        hash_secret_raw(
            passphrase.encode("utf-8"),
            salt,
            time_cost=_KDF_TIME_COST,
            memory_cost=_KDF_MEMORY_KIB,
            parallelism=_KDF_PARALLELISM,
            hash_len=_KEY_LENGTH,
            type=Type.ID,
        )
    )


def _system_keyring() -> Any | None:
    """Return a supported OS keyring backend, never an insecure fallback."""
    try:
        backend = keyring.get_keyring()
    except Exception:  # noqa: BLE001 - keychain availability is optional
        return None
    module = type(backend).__module__
    supported = (
        "keyring.backends.Windows",
        "keyring.backends.macOS",
        "keyring.backends.SecretService",
        "keyring.backends.kwallet",
    )
    return backend if module.startswith(supported) else None


def _keychain_get(vault_id: str) -> bytearray | None:
    backend = _system_keyring()
    if backend is None:
        return None
    try:
        encoded = keyring.get_password(VAULT_KEYCHAIN_SERVICE, vault_id)
        if not encoded:
            return None
        return bytearray(_b64decode(encoded, length=_KEY_LENGTH))
    except Exception:  # noqa: BLE001 - do not expose backend details
        return None


def _keychain_set(vault_id: str, key: bytes) -> bool:
    backend = _system_keyring()
    if backend is None:
        return False
    try:
        keyring.set_password(
            VAULT_KEYCHAIN_SERVICE,
            vault_id,
            _b64encode(key),
        )
        return True
    except Exception:  # noqa: BLE001 - do not expose backend details
        return False


def _keychain_delete(vault_id: str) -> None:
    backend = _system_keyring()
    if backend is None:
        return
    try:
        keyring.delete_password(VAULT_KEYCHAIN_SERVICE, vault_id)
    except Exception:  # noqa: BLE001 - deletion is best effort
        return


def vault_path(home: Path, configured: str | None = None) -> Path:
    """Resolve the encrypted credential store below the private runtime home."""
    selected = configured or os.getenv(VAULT_FILE_ENV)
    path = Path(selected).expanduser() if selected else home / "credentials.vault"
    return path.absolute()


def legacy_environment_credentials(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Collect supported legacy secrets for one-time encrypted migration."""
    source = os.environ if environment is None else environment
    return {
        name: value.strip()
        for name in _SECRET_ENV_NAMES
        if (value := source.get(name)) is not None and value.strip()
    }


def legacy_secret_environment_names() -> tuple[str, ...]:
    """Return names that must not remain in Theia's child environment."""
    return _SECRET_ENV_NAMES


def scrub_legacy_environment() -> None:
    """Remove migrated secret values from this process environment."""
    for name in _SECRET_ENV_NAMES:
        os.environ.pop(name, None)


def scrub_legacy_dotenv(paths: tuple[Path, ...]) -> None:
    """Remove migrated secret assignments from explicitly discovered dotenv files."""
    secret_names = frozenset(_SECRET_ENV_NAMES)
    for path in dict.fromkeys(paths):
        try:
            if not path.is_file() or path.is_symlink():
                continue
            original = path.read_text(encoding="utf-8")
            lines = original.splitlines(keepends=True)
            retained = [
                line
                for line in lines
                if (
                    (match := _DOTENV_ASSIGNMENT_RE.match(line)) is None
                    or match.group("key") not in secret_names
                )
            ]
            updated = "".join(retained)
            if updated == original:
                continue
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(updated)
                    stream.flush()
                    os.fsync(stream.fileno())
                with contextlib.suppress(OSError):
                    os.chmod(temporary_name, 0o600)
                os.replace(temporary_name, path)
                with contextlib.suppress(OSError):
                    path.chmod(0o600)
            finally:
                with contextlib.suppress(OSError):
                    Path(temporary_name).unlink()
        except (OSError, UnicodeError):
            continue


class CredentialVault:
    """Small encrypted credential store with passphrase and OS-keychain unlock."""

    def __init__(self, path: Path, *, keychain_enabled: bool = False) -> None:
        self.path = path.expanduser().absolute()
        self.keychain_enabled = keychain_enabled
        self._credentials: dict[str, Any] | None = None
        self._dek: bytearray | None = None
        self._vault_id: str | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=VAULT_EVENT_LIMIT)
        self._reason = "Vault unlock required"
        self._input_hint = "Type the vault passphrase in the terminal and press Enter."

    @property
    def locked(self) -> bool:
        return self._credentials is None or self._dek is None

    @property
    def initialized(self) -> bool:
        try:
            return self.path.is_file() and not self.path.is_symlink()
        except OSError:
            return False

    @property
    def reason(self) -> str:
        if not self.initialized:
            return "Vault setup required"
        return self._reason

    @property
    def input_hint(self) -> str:
        return self._input_hint

    def set_input_hint(self, value: str) -> None:
        """Set bounded user-facing guidance without exposing secret material."""
        self._input_hint = re.sub(r"[\x00-\x1f\x7f]", "", value).strip()[:180]

    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._events)

    def record_event(self, severity: str, detail: str) -> None:
        normalized = severity.upper()
        if normalized not in {"INFO", "WARNING", "ERROR"}:
            normalized = "INFO"
        safe_detail = re.sub(r"[\x00-\x1f\x7f]", "", detail).strip()[:120]
        self._events.append(
            {
                "timestamp": time.time(),
                "severity": normalized,
                "detail": safe_detail,
            }
        )

    def snapshot(self) -> dict[str, Any]:
        """Return only safe lock state and unlock diagnostics."""
        return {
            "status": "locked" if self.locked else "unlocked",
            "reason": self.reason,
            "input_hint": self.input_hint if self.locked else "",
            "events": self.events(),
        }

    def _aad(self, envelope: Mapping[str, Any]) -> bytes:
        return _canonical_json(
            {
                "format": envelope.get("format"),
                "version": envelope.get("version"),
                "vault_id": envelope.get("vault_id"),
            }
        )

    def _read(self) -> dict[str, Any]:
        if not self.initialized:
            raise VaultError("The encrypted credential vault has not been initialized.")
        try:
            data = self.path.read_bytes()
        except OSError as exc:
            raise VaultError(
                "The encrypted credential vault could not be read."
            ) from exc
        if len(data) > VAULT_MAX_BYTES:
            raise VaultCorruptError("The encrypted credential vault is too large.")
        try:
            envelope = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VaultCorruptError(
                "The encrypted credential vault is malformed."
            ) from exc
        if not isinstance(envelope, dict):
            raise VaultCorruptError("The encrypted credential vault is malformed.")
        if (
            envelope.get("format") != VAULT_FORMAT
            or envelope.get("version") != VAULT_VERSION
        ):
            raise VaultCorruptError(
                "The encrypted credential vault version is unsupported."
            )
        vault_id = envelope.get("vault_id")
        if not isinstance(vault_id, str) or not re.fullmatch(r"[0-9a-f]{32}", vault_id):
            raise VaultCorruptError(
                "The encrypted credential vault identity is invalid."
            )
        self._vault_id = vault_id
        return envelope

    def _write(self, envelope: Mapping[str, Any]) -> None:
        parent = self.path.parent
        if self.path.is_symlink():
            raise VaultError(
                "The encrypted credential vault must not be a symbolic link."
            )
        try:
            parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                parent.chmod(0o700)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=parent,
                text=False,
            )
            try:
                encoded = _canonical_json(dict(envelope))
                if len(encoded) > VAULT_MAX_BYTES:
                    raise VaultError("The encrypted credential vault is too large.")
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                with contextlib.suppress(OSError):
                    os.chmod(temporary_name, 0o600)
                os.replace(temporary_name, self.path)
                with contextlib.suppress(OSError):
                    self.path.chmod(0o600)
            finally:
                with contextlib.suppress(OSError):
                    Path(temporary_name).unlink()
        except OSError as exc:
            raise VaultError(
                "The encrypted credential vault could not be written."
            ) from exc

    @staticmethod
    def _payload(credentials: Mapping[str, Any]) -> bytes:
        clean: dict[str, Any] = {}
        for key, value in credentials.items():
            if not isinstance(key, str) or not _KEY_NAME_RE.fullmatch(key):
                raise VaultError("The credential name is invalid.")
            if isinstance(value, dict):
                if any(
                    not isinstance(child_key, str)
                    or not _KEY_NAME_RE.fullmatch(child_key)
                    or not isinstance(child_value, str)
                    for child_key, child_value in value.items()
                ):
                    raise VaultError("The credential value is invalid.")
                clean[key] = dict(value)
            elif isinstance(value, str):
                clean[key] = value
            else:
                raise VaultError("The credential value is invalid.")
        return _canonical_json(clean)

    @staticmethod
    def _parse_payload(plaintext: bytes) -> dict[str, Any]:
        try:
            value = json.loads(plaintext.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VaultCorruptError("The credential payload is malformed.") from exc
        if not isinstance(value, dict):
            raise VaultCorruptError("The credential payload is malformed.")
        return value

    def _envelope(
        self,
        credentials: Mapping[str, Any],
        dek: bytes | bytearray,
        kek: bytes | bytearray,
        *,
        vault_id: str,
        salt: bytes,
    ) -> dict[str, Any]:
        envelope: dict[str, Any] = {
            "format": VAULT_FORMAT,
            "version": VAULT_VERSION,
            "vault_id": vault_id,
            "kdf": {
                "name": "argon2id",
                "salt": _b64encode(salt),
                "memory_kib": _KDF_MEMORY_KIB,
                "time_cost": _KDF_TIME_COST,
                "parallelism": _KDF_PARALLELISM,
            },
        }
        aad = self._aad(envelope)
        wrap_nonce = secrets.token_bytes(_NONCE_LENGTH)
        payload_nonce = secrets.token_bytes(_NONCE_LENGTH)
        envelope["wrapped_dek"] = {
            "nonce": _b64encode(wrap_nonce),
            "ciphertext": _b64encode(
                AESGCM(bytes(kek)).encrypt(wrap_nonce, dek, aad + b"/dek")
            ),
        }
        envelope["payload"] = {
            "nonce": _b64encode(payload_nonce),
            "ciphertext": _b64encode(
                AESGCM(bytes(dek)).encrypt(
                    payload_nonce,
                    self._payload(credentials),
                    aad + b"/payload",
                )
            ),
        }
        return envelope

    def _unwrap(
        self, envelope: Mapping[str, Any], kek: bytes | bytearray
    ) -> tuple[bytearray, dict[str, Any]]:
        wrapped = envelope.get("wrapped_dek")
        payload = envelope.get("payload")
        if not isinstance(wrapped, dict) or not isinstance(payload, dict):
            raise VaultCorruptError("The encrypted credential vault is incomplete.")
        try:
            wrap_nonce = _b64decode(wrapped.get("nonce"), length=_NONCE_LENGTH)
            wrapped_dek = _b64decode(wrapped.get("ciphertext"))
            payload_nonce = _b64decode(payload.get("nonce"), length=_NONCE_LENGTH)
            ciphertext = _b64decode(payload.get("ciphertext"))
            aad = self._aad(envelope)
            dek = bytearray(
                AESGCM(bytes(kek)).decrypt(wrap_nonce, wrapped_dek, aad + b"/dek")
            )
            plaintext = AESGCM(bytes(dek)).decrypt(
                payload_nonce,
                ciphertext,
                aad + b"/payload",
            )
            credentials = self._parse_payload(plaintext)
            return dek, credentials
        except InvalidTag as exc:
            raise InvalidPassphrase("The vault passphrase was not accepted.") from exc
        except (TypeError, ValueError) as exc:
            raise VaultCorruptError(
                "The encrypted credential vault is invalid."
            ) from exc

    def create(self, credentials: Mapping[str, Any], passphrase: str) -> None:
        """Create and unlock a new vault without overwriting an existing file."""
        if self.initialized:
            raise VaultError("The encrypted credential vault already exists.")
        salt = secrets.token_bytes(_SALT_LENGTH)
        kek = _derive_kek(passphrase, salt)
        dek = bytearray(secrets.token_bytes(_KEY_LENGTH))
        vault_id = secrets.token_hex(16)
        try:
            envelope = self._envelope(
                credentials, dek, kek, vault_id=vault_id, salt=salt
            )
            self._write(envelope)
            self._vault_id = vault_id
            self._credentials = dict(credentials)
            self._dek = dek
            self._reason = "Vault unlocked"
        except Exception:
            _zeroize(dek)
            raise
        finally:
            _zeroize(kek)

    def unlock(self, passphrase: str) -> None:
        """Authenticate and decrypt the complete vault before changing state."""
        envelope = self._read()
        kdf = envelope.get("kdf")
        if not isinstance(kdf, dict) or kdf.get("name") != "argon2id":
            raise VaultCorruptError("The vault key derivation format is unsupported.")
        salt = _b64decode(kdf.get("salt"), length=_SALT_LENGTH)
        kek = _derive_kek(passphrase, salt)
        dek: bytearray | None = None
        try:
            dek, credentials = self._unwrap(envelope, kek)
            self._credentials = credentials
            self._dek = dek
            self._reason = "Vault unlocked"
        except Exception:
            _zeroize(dek)
            self._credentials = None
            self._dek = None
            raise
        finally:
            _zeroize(kek)

    def unlock_keychain(self) -> bool:
        """Try the configured OS keychain without changing state on failure."""
        if not self.initialized:
            return False
        envelope = self._read()
        vault_id = str(envelope["vault_id"])
        kek = _keychain_get(vault_id)
        if kek is None:
            return False
        dek: bytearray | None = None
        try:
            dek, credentials = self._unwrap(envelope, kek)
            self._credentials = credentials
            self._dek = dek
            self._reason = "Vault unlocked from OS keychain"
            return True
        except InvalidPassphrase:
            return False
        finally:
            _zeroize(kek)
            if self.locked:
                _zeroize(dek)

    def store_passphrase_key(self, passphrase: str) -> bool:
        """Save the current passphrase-derived KEK in the OS keychain."""
        if self.locked or self._vault_id is None:
            raise VaultLockedError(
                "Unlock the vault before configuring keychain access."
            )
        envelope = self._read()
        kdf = envelope.get("kdf")
        if not isinstance(kdf, dict):
            return False
        kek = _derive_kek(passphrase, _b64decode(kdf.get("salt"), length=_SALT_LENGTH))
        try:
            return _keychain_set(self._vault_id, bytes(kek))
        finally:
            _zeroize(kek)

    def change_passphrase(self, old: str, new: str) -> None:
        """Atomically rewrap the existing DEK under a newly derived KEK."""
        envelope = self._read()
        kdf = envelope.get("kdf")
        if not isinstance(kdf, dict):
            raise VaultCorruptError("The vault key derivation format is invalid.")
        old_kek = _derive_kek(old, _b64decode(kdf.get("salt"), length=_SALT_LENGTH))
        new_salt = secrets.token_bytes(_SALT_LENGTH)
        new_kek = _derive_kek(new, new_salt)
        dek: bytearray | None = None
        try:
            dek, _ = self._unwrap(envelope, old_kek)
            updated = dict(envelope)
            updated["kdf"] = {
                "name": "argon2id",
                "salt": _b64encode(new_salt),
                "memory_kib": _KDF_MEMORY_KIB,
                "time_cost": _KDF_TIME_COST,
                "parallelism": _KDF_PARALLELISM,
            }
            aad = self._aad(updated)
            nonce = secrets.token_bytes(_NONCE_LENGTH)
            updated["wrapped_dek"] = {
                "nonce": _b64encode(nonce),
                "ciphertext": _b64encode(
                    AESGCM(bytes(new_kek)).encrypt(nonce, bytes(dek), aad + b"/dek")
                ),
            }
            self._write(updated)
            self._reason = "Vault unlocked"
            if self.keychain_enabled and self._vault_id is not None:
                _keychain_set(self._vault_id, bytes(new_kek))
        finally:
            _zeroize(old_kek)
            _zeroize(new_kek)
            _zeroize(dek)

    def update(self, values: Mapping[str, Any]) -> None:
        """Atomically replace selected decrypted values without changing the DEK."""
        if self._credentials is None or self._dek is None:
            raise VaultLockedError("The credential vault is locked.")
        current = dict(self._credentials)
        current.update(values)
        self._payload(current)
        envelope = self._read()
        aad = self._aad(envelope)
        nonce = secrets.token_bytes(_NONCE_LENGTH)
        updated = dict(envelope)
        updated["payload"] = {
            "nonce": _b64encode(nonce),
            "ciphertext": _b64encode(
                AESGCM(bytes(self._dek)).encrypt(
                    nonce,
                    self._payload(current),
                    aad + b"/payload",
                )
            ),
        }
        self._write(updated)
        self._credentials = json.loads(json.dumps(current))

    def credentials(self) -> dict[str, Any]:
        """Return a short-lived copy for provider configuration."""
        if self._credentials is None:
            raise VaultLockedError("The credential vault is locked.")
        return json.loads(json.dumps(self._credentials))

    def lock(self, *, reason: str = "Vault locked") -> None:
        """Drop retained credential references and best-effort zeroize key bytes."""
        credentials = self._credentials
        self._credentials = None
        if credentials is not None:
            credentials.clear()
        dek, self._dek = self._dek, None
        _zeroize(dek)
        self._reason = reason[:120]
