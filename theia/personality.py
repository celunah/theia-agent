"""Validation and storage for Theia personality prompt profiles."""

import contextlib
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote
from typing import Any, cast
import unicodedata

PERSONALITY_SUFFIXES = frozenset({".md", ".markdown", ".text", ".txt"})
MAX_PERSONALITY_BYTES = 128 * 1024
MAX_PERSONALITY_NAME_LENGTH = 80
MAX_SUMMARY_DESCRIPTION_LENGTH = 600
MAX_SUMMARY_ITEM_LENGTH = 160
MAX_SUMMARY_ITEMS = 6
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
_IDENTIFIER_RE = re.compile(r"[^a-z0-9]+")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+?)\s*$")
_SUMMARY_SECTIONS = {"known entries", "known users"}


class PersonalityError(ValueError):
    """A personality profile could not be selected or stored."""


@dataclass(frozen=True)
class PersonalityProfile:
    """A validated personality name and its private prompt-file path."""

    name: str
    path: Path


@dataclass(frozen=True)
class PersonalitySummary:
    """A bounded, presentation-safe summary of one personality profile."""

    name: str
    identifier: str
    character_name: str
    description: str
    known_entries: tuple[str, ...]
    known_users: tuple[str, ...]


def _clean_summary_text(value: str, limit: int) -> str:
    """Remove formatting and mention syntax from profile-derived UI text."""
    text = re.sub(r"<@!?\d+>", "@user", value)
    text = re.sub(r"<@&\d+>", "@role", text)
    text = re.sub(r"<#\d+>", "#channel", text)
    text = re.sub(r"@(?:everyone|here)\b", "at everyone", text, flags=re.IGNORECASE)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*_~]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _heading(line: str) -> str | None:
    match = _HEADING_RE.match(line)
    if match is None:
        return None
    return _clean_summary_text(re.sub(r"\s+#+\s*$", "", match.group(1)), 200)


def _profile_identifier(name: str) -> str:
    normalized = (
        unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    )
    identifier = _IDENTIFIER_RE.sub("-", normalized.casefold()).strip("-")
    return identifier or "character"


def _character_name(name: str, prompt: str) -> str:
    match = re.search(
        r"^\s*(?:you are|i am)\s+([^,.:!\n]+)", prompt, re.IGNORECASE | re.MULTILINE
    )
    if match:
        return _clean_summary_text(match.group(1), 120)
    for line in prompt.splitlines():
        heading = _heading(line)
        if heading:
            return heading
    return _clean_summary_text(name, 120)


def _description(prompt: str) -> str:
    """Use the opening prose as a concise local profile description."""
    for block in re.split(r"\n\s*\n", prompt):
        lines = block.splitlines()
        if not lines:
            continue
        first_heading = next((_heading(line) for line in lines if line.strip()), None)
        if first_heading and first_heading.casefold() in _SUMMARY_SECTIONS:
            continue
        prose = []
        for line in lines:
            if _heading(line):
                continue
            line = re.sub(r"^\s*>\s?", "", line)
            line = _LIST_ITEM_RE.sub(r"\1", line)
            if line.strip():
                prose.append(line.strip())
        value = _clean_summary_text(" ".join(prose), MAX_SUMMARY_DESCRIPTION_LENGTH)
        if value:
            return value
    return "No character description is available."


def _section_items(prompt: str, section_name: str) -> tuple[str, ...]:
    items: list[str] = []
    active = False
    for line in prompt.splitlines():
        heading = _heading(line)
        if heading:
            active = heading.casefold() == section_name.casefold()
            continue
        if not active or not line.strip():
            continue
        match = _LIST_ITEM_RE.match(line)
        value = match.group(1) if match else line.strip()
        value = _clean_summary_text(value, MAX_SUMMARY_ITEM_LENGTH)
        if value and value not in items:
            items.append(value)
        if len(items) >= MAX_SUMMARY_ITEMS:
            break
    return tuple(items)


class PersonalityStore:
    """Validate, enumerate, and persist Markdown personality profiles."""

    def __init__(self, root: Path) -> None:
        """Use a ``personalities`` directory beneath the private runtime root."""
        self.root = root / "personalities"

    @staticmethod
    def normalize_name(value: str | None) -> str:
        """Normalize a profile name and reject unsafe or reserved values."""
        name = " ".join((value or "").strip().split())
        if not name:
            raise PersonalityError("A personality name is required.")
        if len(name) > MAX_PERSONALITY_NAME_LENGTH:
            raise PersonalityError("Personality names must be 80 characters or fewer.")
        if name.casefold() in {"none", "default", "neutral"}:
            raise PersonalityError(
                "`none` clears the active personality and cannot name a file."
            )
        if name in {".", ".."} or any(
            character in name for character in ("/", "\\", "\x00")
        ):
            raise PersonalityError("That personality name is not valid.")
        if any(ord(character) < 32 for character in name):
            raise PersonalityError("That personality name is not valid.")
        return name

    @staticmethod
    def is_clear_name(value: str | None) -> bool:
        """Return whether a command value requests clearing the active profile."""
        return (value or "").strip().casefold() in {"none", "default", "neutral"}

    def _path_for(self, name: str) -> Path:
        return self.root / f"{quote(name, safe='._-')}.md"

    def _profile_from_path(self, path: Path) -> PersonalityProfile | None:
        suffix = path.suffix.casefold()
        if suffix not in PERSONALITY_SUFFIXES or not path.is_file():
            return None
        name = unquote(path.name[: -len(path.suffix)])
        try:
            name = self.normalize_name(name)
        except PersonalityError:
            return None
        return PersonalityProfile(name=name, path=path)

    def profiles(self) -> tuple[PersonalityProfile, ...]:
        """Return valid profiles in stable case-insensitive display order."""
        try:
            paths = tuple(self.root.iterdir())
        except OSError:
            return ()
        profiles: list[PersonalityProfile] = []
        seen: set[str] = set()
        for path in sorted(paths, key=lambda item: item.name.casefold()):
            profile = self._profile_from_path(path)
            if profile is None or profile.name.casefold() in seen:
                continue
            seen.add(profile.name.casefold())
            profiles.append(profile)
        return tuple(sorted(profiles, key=lambda item: item.name.casefold()))

    def names(self) -> tuple[str, ...]:
        """Return the names of all readable personality profiles."""
        return tuple(profile.name for profile in self.profiles())

    def resolve(self, value: str | None) -> PersonalityProfile | None:
        """Resolve a selected name to a profile, returning ``None`` when absent."""
        name = self.normalize_name(value)
        wanted = name.casefold()
        return next(
            (
                profile
                for profile in self.profiles()
                if profile.name.casefold() == wanted
            ),
            None,
        )

    def read(self, value: str | None) -> tuple[str, str]:
        """Read a selected profile and return its canonical name and prompt text."""
        profile = self.resolve(value)
        if profile is None:
            raise PersonalityError("That personality profile is not available.")
        try:
            text = profile.path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise PersonalityError(
                "That personality profile could not be read."
            ) from exc
        text = text.strip()
        if not text:
            raise PersonalityError("That personality profile is empty.")
        if "\x00" in text:
            raise PersonalityError("That personality profile is not valid text.")
        return profile.name, text

    def summary(self, value: str | None) -> PersonalitySummary:
        """Return safe character-card data extracted from a selected profile."""
        name, prompt = self.read(value)
        return PersonalitySummary(
            name=name,
            identifier=_profile_identifier(name),
            character_name=_character_name(name, prompt),
            description=_description(prompt),
            known_entries=_section_items(prompt, "Known Entries"),
            known_users=_section_items(prompt, "Known Users"),
        )

    async def upload(self, attachment: object, value: str | None) -> str:
        """Download, validate, and store an uploaded personality prompt."""
        name = self.normalize_name(value)
        filename = str(getattr(attachment, "filename", "") or "")
        suffix = Path(filename).suffix.casefold()
        if suffix not in PERSONALITY_SUFFIXES:
            raise PersonalityError(
                "The personality file must be Markdown or plain text."
            )
        size = getattr(attachment, "size", None)
        if isinstance(size, int) and size > MAX_PERSONALITY_BYTES:
            raise PersonalityError("The personality file is too large.")
        read = getattr(attachment, "read", None)
        if not callable(read):
            raise PersonalityError("The personality file could not be read.")
        try:
            read_async = cast(Callable[[], Awaitable[Any]], read)
            raw = await read_async()
        except Exception as exc:
            raise PersonalityError("The personality file could not be read.") from exc
        if not isinstance(raw, bytes) or len(raw) > MAX_PERSONALITY_BYTES:
            raise PersonalityError("The personality file is too large or invalid.")
        try:
            text = raw.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise PersonalityError("The personality file must be UTF-8 text.") from exc
        if not text:
            raise PersonalityError("The personality file is empty.")
        if "\x00" in text:
            raise PersonalityError("The personality file must contain text only.")

        existing = self.resolve(name)
        stored_name = existing.name if existing is not None else name
        path = self._path_for(stored_name)
        temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.root.chmod(0o700)
            temporary.write_text(text + "\n", encoding="utf-8")
            temporary.chmod(0o600)
            temporary.replace(path)
        except OSError as exc:
            raise PersonalityError(
                "The personality profile could not be saved."
            ) from exc
        finally:
            with contextlib.suppress(OSError):
                temporary.unlink()
        return stored_name
