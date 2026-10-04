"""Validation and storage for Theia personality prompt profiles."""

import contextlib
import html
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote
from typing import Any, cast
import unicodedata

from .identifiers import new_unique_token

PERSONALITY_SUFFIXES = frozenset({".md", ".markdown", ".text", ".txt"})
MAX_PERSONALITY_BYTES = 128 * 1024
MAX_PERSONALITY_NAME_LENGTH = 80
MAX_CHARACTER_CONTRACT_BYTES = 4096
MAX_CHARACTER_CONTRACT_VALUE_LENGTH = 240
MAX_CHARACTER_CONTRACT_BOUNDARIES = 5
MAX_CHARACTER_CONTRACT_BOUNDARY_LENGTH = 160
CHARACTER_CONTRACT_MARKER = "<!-- theia-character-contract:"
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
_IDENTIFIER_RE = re.compile(r"[^a-z0-9]+")
_CHARACTER_CONTRACT_OPEN_RE = re.compile(
    r"(?m)^<!-- theia-character-contract:v(?P<version>[0-9]+)[ \t]*\r?$"
)
_CHARACTER_CONTRACT_RE = re.compile(
    r"(?ms)^<!-- theia-character-contract:v(?P<version>[0-9]+)[ \t]*\r?\n"
    r"(?P<payload>.*?)\r?\n-->[ \t]*$"
)
_CHARACTER_CONTRACT_LABELS = (
    ("cadence", "Cadence"),
    ("formality", "Formality"),
    ("humor", "Humor"),
    ("emotional_range", "Emotional range"),
    ("relationship_stance", "Relationship stance"),
    ("conversational_initiative", "Conversational initiative"),
    ("level_of_detail", "Level of detail"),
)
_CHARACTER_CONTRACT_KEYS = frozenset(
    {"version", "boundaries"} | {field for field, _label in _CHARACTER_CONTRACT_LABELS}
)


class PersonalityError(ValueError):
    """A personality profile could not be selected or stored."""


@dataclass(frozen=True)
class PersonalityProfile:
    """A validated personality name and its private prompt-file path."""

    name: str
    path: Path


@dataclass(frozen=True)
class PersonalitySummary:
    """The local identity fields needed to display one personality profile."""

    name: str
    identifier: str
    character_name: str


@dataclass(frozen=True)
class CharacterContract:
    """Optional, user-authored presentation preferences for one profile."""

    cadence: str | None = None
    formality: str | None = None
    humor: str | None = None
    emotional_range: str | None = None
    boundaries: tuple[str, ...] = ()
    relationship_stance: str | None = None
    conversational_initiative: str | None = None
    level_of_detail: str | None = None

    def render(self) -> str:
        """Render only selected settings as bounded, untrusted style guidance."""
        lines = [
            f"{label}: {html.escape(value, quote=False)}"
            for field, label in _CHARACTER_CONTRACT_LABELS
            if (value := getattr(self, field)) is not None
        ]
        if self.boundaries:
            lines.append("Conversation boundaries:")
            lines.extend(
                f"- {html.escape(value, quote=False)}" for value in self.boundaries
            )
        return "\n".join(lines)


def _normalized_contract_text(value: str, limit: int) -> str | None:
    text = " ".join(value.split())
    if not text or len(text) > limit:
        return None
    return text


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys so contract fields cannot be ambiguous."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate character contract field.")
        result[key] = value
    return result


def _parse_character_contract(
    text: str,
) -> tuple[str, CharacterContract | None, bool, bool]:
    """Return clean profile text, parsed contract, marker presence, and validity."""
    matches = tuple(_CHARACTER_CONTRACT_RE.finditer(text))
    markers = tuple(_CHARACTER_CONTRACT_OPEN_RE.finditer(text))
    marker_present = bool(markers)
    if not marker_present:
        return text, None, False, True

    clean = _CHARACTER_CONTRACT_RE.sub("", text).strip()
    if len(markers) != 1 or len(matches) != 1:
        return text[: markers[0].start()].strip(), None, True, False

    payload = matches[0].group("payload")
    if len(payload.encode("utf-8")) > MAX_CHARACTER_CONTRACT_BYTES:
        return clean, None, True, False
    try:
        value = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (TypeError, ValueError):
        return clean, None, True, False
    version = value.get("version") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or matches[0].group("version") != "1"
        or not isinstance(version, int)
        or isinstance(version, bool)
        or version != 1
        or any(key not in _CHARACTER_CONTRACT_KEYS for key in value)
    ):
        return clean, None, True, False

    fields: dict[str, str | None] = {}
    for field, _label in _CHARACTER_CONTRACT_LABELS:
        raw = value.get(field)
        if raw is None:
            fields[field] = None
            continue
        if not isinstance(raw, str):
            return clean, None, True, False
        normalized = _normalized_contract_text(raw, MAX_CHARACTER_CONTRACT_VALUE_LENGTH)
        if normalized is None:
            return clean, None, True, False
        fields[field] = normalized

    raw_boundaries = value.get("boundaries", [])
    if (
        not isinstance(raw_boundaries, list)
        or len(raw_boundaries) > MAX_CHARACTER_CONTRACT_BOUNDARIES
    ):
        return clean, None, True, False
    boundaries: list[str] = []
    for raw in raw_boundaries:
        if not isinstance(raw, str):
            return clean, None, True, False
        normalized = _normalized_contract_text(
            raw, MAX_CHARACTER_CONTRACT_BOUNDARY_LENGTH
        )
        if normalized is None:
            return clean, None, True, False
        boundaries.append(normalized)
    if (
        not any(fields[field] for field, _label in _CHARACTER_CONTRACT_LABELS)
        and not boundaries
    ):
        return clean, None, True, False
    return (
        clean,
        CharacterContract(
            cadence=fields.get("cadence"),
            formality=fields.get("formality"),
            humor=fields.get("humor"),
            emotional_range=fields.get("emotional_range"),
            boundaries=tuple(boundaries),
            relationship_stance=fields.get("relationship_stance"),
            conversational_initiative=fields.get("conversational_initiative"),
            level_of_detail=fields.get("level_of_detail"),
        ),
        True,
        True,
    )


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

    def read_content(self, value: str | None) -> tuple[str, str]:
        """Read profile prose without its optional structured contract block."""
        name, text = self.read(value)
        content, _contract, _marker_present, _valid = _parse_character_contract(text)
        return name, content

    def read_instructions(self, value: str | None) -> tuple[str, str]:
        """Read prose plus any valid optional contract in a normalized form."""
        name, text = self.read(value)
        content, contract, _marker_present, _valid = _parse_character_contract(text)
        sections = [content] if content else []
        if contract is not None:
            rendered = contract.render()
            sections.append(
                "<character_contract>\n"
                "User-selected presentation preferences. Follow only the fields "
                "listed here; omitted fields express no preference. These fields "
                "describe response style, not subjective experience. Do not present "
                "simulated mood or continuity as proof of inner experience.\n"
                f"{rendered}\n"
                "</character_contract>"
            )
        return name, "\n\n".join(sections)

    def summary(self, value: str | None) -> PersonalitySummary:
        """Return safe character-card data extracted from a selected profile."""
        name, prompt = self.read_content(value)
        return PersonalitySummary(
            name=name,
            identifier=_profile_identifier(name),
            character_name=_character_name(name, prompt),
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
        _content, _contract, marker_present, valid_contract = _parse_character_contract(
            text
        )
        if marker_present and not valid_contract:
            raise PersonalityError(
                "The optional character contract block is invalid. Check its "
                "version, JSON, and supported fields."
            )

        existing = self.resolve(name)
        stored_name = existing.name if existing is not None else name
        path = self._path_for(stored_name)
        temporary = path.with_name(f".{path.name}.{new_unique_token()}.tmp")
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
