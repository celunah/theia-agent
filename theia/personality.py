"""Validation and storage for Theia personality prompt profiles."""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

PERSONALITY_SUFFIXES = frozenset({".md", ".markdown", ".text", ".txt"})
MAX_PERSONALITY_BYTES = 128 * 1024
MAX_PERSONALITY_NAME_LENGTH = 80


class PersonalityError(ValueError):
    """A personality profile could not be selected or stored."""


@dataclass(frozen=True)
class PersonalityProfile:
    name: str
    path: Path


class PersonalityStore:
    def __init__(self, root: Path) -> None:
        self.root = root / "personalities"

    @staticmethod
    def normalize_name(value: str | None) -> str:
        name = " ".join(str(value or "").strip().split())
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
        return str(value or "").strip().casefold() in {"none", "default", "neutral"}

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
        return tuple(profile.name for profile in self.profiles())

    def resolve(self, value: str | None) -> PersonalityProfile | None:
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

    async def upload(self, attachment: object, value: str | None) -> str:
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
            raw = await read()
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
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.root.chmod(0o700)
            path.write_text(text + "\n", encoding="utf-8")
            path.chmod(0o600)
        except OSError as exc:
            raise PersonalityError(
                "The personality profile could not be saved."
            ) from exc
        return stored_name
