"""Interactively create or update Theia's local environment configuration."""

from __future__ import annotations

import argparse
import contextlib
import getpass
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_ENV = "THEIA_ENV_FILE"
TEXT_MODE = "text"
VOICE_MODE = "voice"
_ASSIGNMENT = re.compile(
    r"^(?P<prefix>\s*(?:export\s+)?)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*="
)
_MAX_TOKEN_LENGTH = 1024
_MAX_URL_LENGTH = 2048
_MAX_PARAMETER_LENGTH = 256
_SUPPORTED_TTS_FORMATS = frozenset({"mp3", "opus", "aac", "flac", "wav", "pcm"})


class ConfigurationError(ValueError):
    """A configuration value is missing or unsafe to persist."""


@dataclass(frozen=True, slots=True)
class ConfigurationValues:
    """Validated values collected by the setup wizard."""

    discord_token: str
    mode: str
    stt_base_url: str = ""
    stt_token: str = ""
    stt_model: str = "whisper-1"
    tts_base_url: str = ""
    tts_token: str = ""
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"
    tts_format: str = "mp3"
    realtime_model: str = ""
    realtime_voice: str = ""

    def as_environment(self) -> dict[str, str]:
        """Return values using Theia's supported environment variable names."""
        values = {
            "TOKEN": self.discord_token,
            "THEIA_DEFAULT_MODE": self.mode,
        }
        if self.mode == VOICE_MODE:
            values.update(
                {
                    "STT_BASE_URL": self.stt_base_url,
                    "STT_TOKEN": self.stt_token,
                    "TTS_BASE_URL": self.tts_base_url,
                    "TTS_TOKEN": self.tts_token,
                }
            )
            if self.stt_base_url and self.tts_base_url:
                values.update(
                    {
                        "STT_MODEL": self.stt_model,
                        "TTS_MODEL": self.tts_model,
                        "TTS_VOICE": self.tts_voice,
                        "TTS_FORMAT": self.tts_format,
                    }
                )
            else:
                values.update(
                    {
                        "THEIA_REALTIME_MODEL": self.realtime_model,
                        "THEIA_REALTIME_VOICE": self.realtime_voice,
                    }
                )
        return values


def _required(value: str, label: str, *, maximum: int) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ConfigurationError(f"{label} is required.")
    if len(cleaned) > maximum or any(char in cleaned for char in "\r\n"):
        raise ConfigurationError(f"{label} is not valid.")
    return cleaned


def _optional_token(value: str, label: str) -> str:
    cleaned = value.strip()
    if len(cleaned) > _MAX_TOKEN_LENGTH or any(char in cleaned for char in "\r\n"):
        raise ConfigurationError(f"{label} is not valid.")
    return cleaned


def _optional_parameter(value: str, label: str) -> str:
    cleaned = value.strip()
    if len(cleaned) > _MAX_PARAMETER_LENGTH or any(char in cleaned for char in "\r\n"):
        raise ConfigurationError(f"{label} is not valid.")
    return cleaned


def _parameter(value: str, label: str, default: str) -> str:
    return _optional_parameter(value, label) or default


def _tts_format(value: str) -> str:
    cleaned = _parameter(value, "The TTS format", "mp3").casefold()
    if cleaned not in _SUPPORTED_TTS_FORMATS:
        supported = ", ".join(sorted(_SUPPORTED_TTS_FORMATS))
        raise ConfigurationError(f"The TTS format must be one of: {supported}.")
    return cleaned


def _url(value: str, label: str) -> str:
    cleaned = _required(value, label, maximum=_MAX_URL_LENGTH)
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError(f"{label} must be an HTTP or HTTPS URL.")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigurationError(f"{label} must not contain embedded credentials.")
    return cleaned.rstrip("/")


def validate_configuration(
    *,
    discord_token: str,
    mode: str,
    stt_base_url: str = "",
    stt_token: str = "",
    stt_model: str = "",
    tts_base_url: str = "",
    tts_token: str = "",
    tts_model: str = "",
    tts_voice: str = "",
    tts_format: str = "",
    realtime_model: str = "",
    realtime_voice: str = "",
) -> ConfigurationValues:
    """Validate setup input without logging or displaying credential values."""
    selected_mode = mode.strip().casefold()
    if selected_mode not in {TEXT_MODE, VOICE_MODE}:
        raise ConfigurationError("Choose either text or voice mode.")
    token = _required(
        discord_token,
        "The Discord bot token",
        maximum=_MAX_TOKEN_LENGTH,
    )
    if selected_mode == TEXT_MODE:
        return ConfigurationValues(discord_token=token, mode=selected_mode)
    stt_value = stt_base_url.strip()
    tts_value = tts_base_url.strip()
    if bool(stt_value) != bool(tts_value):
        raise ConfigurationError(
            "Provide both audio service URLs, or leave both blank for Codex Realtime."
        )
    realtime_model_value = _optional_parameter(realtime_model, "The Realtime model")
    realtime_voice_value = _optional_parameter(realtime_voice, "The Realtime voice")
    return ConfigurationValues(
        discord_token=token,
        mode=selected_mode,
        stt_base_url=_url(stt_value, "The STT URL") if stt_value else "",
        stt_token=_optional_token(stt_token, "The STT token"),
        stt_model=_parameter(stt_model, "The STT model", "whisper-1"),
        tts_base_url=_url(tts_value, "The TTS URL") if tts_value else "",
        tts_token=_optional_token(tts_token, "The TTS token"),
        tts_model=_parameter(tts_model, "The TTS model", "tts-1"),
        tts_voice=_parameter(tts_voice, "The TTS voice", "alloy"),
        tts_format=_tts_format(tts_format),
        realtime_model=realtime_model_value,
        realtime_voice=realtime_voice_value,
    )


def configuration_path(explicit: str | Path | None = None) -> Path:
    """Return the environment file used by setup and normal Theia startup."""
    if explicit is not None:
        return Path(explicit).expanduser().absolute()
    configured = os.getenv(ENV_FILE_ENV)
    if configured:
        return Path(configured).expanduser().absolute()
    return (PROJECT_ROOT / ".env").absolute()


def _dotenv_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _updated_dotenv(existing: str, values: dict[str, str]) -> str:
    lines = existing.splitlines(keepends=True)
    updated: set[str] = set()
    result: list[str] = []
    for line in lines:
        if line.endswith("\r\n"):
            newline = "\r\n"
            body = line[:-2]
        elif line.endswith(("\n", "\r")):
            newline = line[-1]
            body = line[:-1]
        else:
            newline = ""
            body = line
        match = _ASSIGNMENT.match(body)
        key = match.group("key") if match else ""
        if key in values:
            prefix = match.group("prefix") if match else ""
            result.append(f"{prefix}{key}={_dotenv_value(values[key])}{newline}")
            updated.add(key)
        else:
            result.append(line)

    missing = [key for key in values if key not in updated]
    if missing:
        if result and not result[-1].endswith(("\n", "\r")):
            result.append("\n")
        result.extend(f"{key}={_dotenv_value(values[key])}\n" for key in missing)
    return "".join(result)


def _read_existing(path: Path) -> str:
    if not path.exists():
        return ""
    if path.is_symlink():
        raise ConfigurationError("The environment file must not be a symbolic link.")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(
            "The existing environment file could not be read."
        ) from exc


def _atomic_write(path: Path, contents: str) -> None:
    temporary_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
        temporary_name = None
        with contextlib.suppress(OSError):
            path.chmod(0o600)
    except OSError as exc:
        raise ConfigurationError("The environment file could not be written.") from exc
    finally:
        if temporary_name is not None:
            with contextlib.suppress(OSError):
                Path(temporary_name).unlink()


def save_configuration(
    values: ConfigurationValues,
    *,
    path: str | Path | None = None,
) -> Path:
    """Atomically update the local environment file with validated values."""
    target = configuration_path(path)
    if target.is_symlink():
        raise ConfigurationError("The environment file must not be a symbolic link.")
    contents = _updated_dotenv(_read_existing(target), values.as_environment())
    _atomic_write(target, contents)
    return target


def _choose_mode(
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    output_fn("Choose Theia's default interaction mode:")
    output_fn("  1. Text (Discord messages only)")
    output_fn("  2. Voice (Discord plus Codex Realtime or custom STT/TTS)")
    while True:
        choice = input_fn("Mode [1/text]: ").strip().casefold()
        if choice in {"", "1", TEXT_MODE}:
            return TEXT_MODE
        if choice in {"2", VOICE_MODE}:
            return VOICE_MODE
        output_fn("Please choose 1/text or 2/voice.")


def collect_configuration(
    *,
    input_fn: Callable[[str], str] | None = None,
    secret_input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> ConfigurationValues:
    """Collect the mode and credentials needed before Theia can start."""
    read = input if input_fn is None else input_fn
    read_secret = getpass.getpass if secret_input_fn is None else secret_input_fn
    write = print if output_fn is None else output_fn
    mode = _choose_mode(read, write)
    token = read_secret("Discord bot token (input hidden): ")
    if mode == TEXT_MODE:
        return validate_configuration(discord_token=token, mode=mode)
    stt_url = read("STT URL (blank for Codex Realtime): ")
    tts_url = read("TTS URL (blank for Codex Realtime): ")
    stt_token = (
        read_secret("STT token (blank if not required): ") if stt_url.strip() else ""
    )
    tts_token = (
        read_secret("TTS token (blank if not required): ") if tts_url.strip() else ""
    )
    if stt_url.strip():
        stt_model = read("STT model [whisper-1]: ")
        tts_model = read("TTS model [tts-1]: ")
        tts_voice = read("TTS voice [alloy]: ")
        tts_format = read("TTS format [mp3]: ")
        realtime_model = ""
        realtime_voice = ""
    else:
        stt_model = ""
        tts_model = ""
        tts_voice = ""
        tts_format = ""
        realtime_model = read("Realtime model (blank for Codex default): ")
        realtime_voice = read("Realtime voice (blank for Codex default): ")
    return validate_configuration(
        discord_token=token,
        mode=mode,
        stt_base_url=stt_url,
        stt_token=stt_token,
        stt_model=stt_model,
        tts_base_url=tts_url,
        tts_token=tts_token,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_format=tts_format,
        realtime_model=realtime_model,
        realtime_voice=realtime_voice,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the setup wizard and save its result for the next Theia launch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        help="Environment file to update instead of the project .env file.",
    )
    args = parser.parse_args(argv)
    try:
        values = collect_configuration()
        target = save_configuration(values, path=args.env_file)
    except KeyboardInterrupt:
        print("Configuration cancelled.")
        return 130
    except ConfigurationError as exc:
        print(f"Configuration failed: {exc}")
        return 1
    print(f"Configuration saved to {target}.")
    print("Start Theia with `python main.py` to apply the new Discord token.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
