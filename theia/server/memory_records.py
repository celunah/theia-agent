"""Bounded, private records built from Theia's memory-compatible sources."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core import _redact_private_paths, _truncate
from ..identifiers import new_unique_token

MEMORY_RECORD_MAX_CHARACTERS = 3500
MEMORY_RECORD_MAX_COUNT = 512
MEMORY_AUDIT_MAX_ENTRIES = 256
MEMORY_AUDIT_MAX_BYTES = 1024 * 1024
MEMORY_AUDIT_FILE = "memory-audit.json"
_SCOPE_RE = re.compile(r"^(?:user|server):[1-9][0-9]*$")
_SCOPE_MARKER_RE = re.compile(
    r"<!--\s*(?:theia[-_ ])?scope\s*[:=]\s*"
    r"(user:[1-9][0-9]*|server:[1-9][0-9]*|everyone)\s*-->",
    re.IGNORECASE,
)
_CREDENTIAL_RE = re.compile(
    r"(?i)\b(?:bearer\s+[A-Za-z0-9._~+/=-]+|"
    r"(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|secret|"
    r"authorization|credential)\s*[:=]\s*[^\s,;]+|sk-[A-Za-z0-9_-]{16,})"
)
_DISCORD_ID_RE = re.compile(
    r"<@!?\d+>|(?:discord\s+user\s+id|user_id)\s*[:=]\s*\d+",
    re.IGNORECASE,
)
_POSIX_PATH_RE = re.compile(
    r"(?<!https:)(?<![A-Za-z0-9_])(?:file://)?/"
    r"[A-Za-z0-9._~:/@%+\-=]+"
)


def safe_memory_text(value: Any, limit: int = MEMORY_RECORD_MAX_CHARACTERS) -> str:
    """Bound memory text and remove values unsuitable for Discord display."""
    text = str(value or "").replace("\x00", "").strip()
    if not text:
        return ""
    text = _CREDENTIAL_RE.sub("[redacted]", text)
    text = _redact_private_paths(text)
    text = _POSIX_PATH_RE.sub("[private path]", text)
    text = _DISCORD_ID_RE.sub("[Discord user]", text)
    text = "\n".join(
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    )
    return _truncate(text.strip(), limit)


def normalize_memory_scope(value: Any) -> str:
    """Return only a supported internal scope key."""
    if not isinstance(value, str):
        return ""
    scope = re.sub(r"\s+", "", value.strip().casefold())
    return scope if scope == "everyone" or _SCOPE_RE.fullmatch(scope) else ""


def display_memory_scope(scope: str) -> str:
    """Return a safe human label without exposing a raw Discord identifier."""
    if scope == "everyone":
        return "everyone"
    if scope.startswith("user:"):
        return "user scope"
    if scope.startswith("server:"):
        return "server scope"
    return "legacy/unscoped"


def memory_scope_selector(
    value: Any,
    *,
    user_id: int | None,
    guild_id: int | None,
) -> tuple[str, str] | None:
    """Resolve a command selector to a kind and private target key."""
    requested = str(value or "me").strip().casefold()
    if requested == "me":
        return ("user", f"user:{user_id}") if user_id and user_id > 0 else None
    if requested == "server":
        return ("server", f"server:{guild_id}") if guild_id and guild_id > 0 else None
    if requested == "everyone":
        return "everyone", "everyone"
    if requested.startswith("user:") and normalize_memory_scope(requested):
        return "user", requested
    if requested.startswith("server:") and normalize_memory_scope(requested):
        return "server", requested
    return None


def _safe_timestamp(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    timestamp = float(value)
    return timestamp if 0 < timestamp < 4102444800 else None


def memory_timestamp(value: Any) -> float | None:
    """Parse a bounded timestamp from current or persisted view data."""
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    return _safe_timestamp(value)


def memory_display_date(value: Any) -> str:
    """Return a stable UTC calendar date for a memory timestamp."""
    timestamp = memory_timestamp(value)
    if timestamp is None:
        return "unknown"
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()


def _record_id(
    *,
    source_file: str,
    scope: str,
    text: str,
    ordinal: int,
) -> str:
    value = "\0".join((source_file, scope, str(ordinal), text))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _source_scope(path: Path, root: Path, fallback: str = "legacy") -> str:
    try:
        parts = (
            path.resolve(strict=False)
            .relative_to(root.resolve(strict=False))
            .parts[:-1]
        )
    except (OSError, ValueError):
        parts = ()
    for index, part in enumerate(parts[:-1]):
        candidate = parts[index + 1]
        if part.casefold() in {"user", "users", "me"} and candidate.isdigit():
            return f"user:{int(candidate)}" if int(candidate) > 0 else fallback
        if (
            part.casefold() in {"server", "servers", "guild", "guilds"}
            and candidate.isdigit()
        ):
            return f"server:{int(candidate)}" if int(candidate) > 0 else fallback
    if any(part.casefold() in {"everyone", "global", "shared"} for part in parts):
        return "everyone"
    return fallback


def _file_scope(path: Path, default: str) -> str:
    try:
        prefix = path.read_text(encoding="utf-8-sig")[:2000]
    except (OSError, UnicodeDecodeError):
        return default
    match = _SCOPE_MARKER_RE.search(prefix)
    if match is None:
        return default
    return normalize_memory_scope(match.group(1)) or default


def _markdown_blocks(text: str) -> list[tuple[str, int, int]]:
    """Extract display blocks while retaining offsets for safe single-record edits."""
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    starts = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+\S", line)
    ]
    blocks: list[tuple[str, int, int]] = []
    if starts:
        for ordinal, start in enumerate(starts):
            end_line = starts[ordinal + 1] if ordinal + 1 < len(starts) else len(lines)
            for index in range(start + 1, end_line):
                if lines[index].lstrip().startswith("#"):
                    end_line = index
                    break
            raw = "".join(lines[start:end_line])
            display = "\n".join(
                line.strip()
                for line in raw.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
            if display:
                end = offsets[end_line] if end_line < len(offsets) else len(text)
                blocks.append((display, offsets[start], end))
        return blocks
    cursor = 0
    for raw in re.split(r"(\n\s*\n)", text):
        if raw.isspace() or not raw.strip():
            cursor += len(raw)
            continue
        display = "\n".join(
            line.strip()
            for line in raw.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if display:
            blocks.append((display, cursor, cursor + len(raw)))
        cursor += len(raw)
    return blocks


@dataclass(frozen=True)
class MemoryRecord:
    """One bounded memory-like value with private mutation coordinates."""

    record_id: str
    text: str
    scope: str
    character_name: str
    character_slug: str
    source_file: str
    source_category: str
    created_at: float | None = None
    updated_at: float | None = None
    confidence: float | None = None
    ordinal: int = 0
    source_path: Path | None = field(default=None, repr=False, compare=False)
    source_root: Path | None = field(default=None, repr=False, compare=False)
    start_offset: int | None = field(default=None, repr=False, compare=False)
    end_offset: int | None = field(default=None, repr=False, compare=False)
    source_kind: str = field(default="markdown", repr=False, compare=False)
    source_key: str | None = field(default=None, repr=False, compare=False)
    scope_keys: tuple[str, ...] = field(default=(), repr=False, compare=False)

    @property
    def origin(self) -> str:
        """Return the stable source category used by compatibility callers."""
        return self.source_category

    @property
    def display_scope(self) -> str:
        """Return the privacy-safe scope label shown in Discord."""
        return display_memory_scope(self.scope)

    @property
    def display_metadata(self) -> dict[str, str]:
        """Return bounded source and date metadata for the memory card."""
        return {
            "source": self.source_category.replace("_", " "),
            "scope": self.display_scope,
            "updated": memory_display_date(
                self.updated_at if self.updated_at is not None else self.created_at
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize only safe fields for the Discord memory view."""
        return {
            "record_id": self.record_id,
            "text": self.text,
            "scope": self.display_scope,
            "character_name": self.character_name,
            "character_slug": self.character_slug,
            "source_file": self.source_file,
            "source_category": self.source_category,
            "origin": self.origin,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "confidence": self.confidence,
            "display_metadata": self.display_metadata,
        }


def markdown_records(
    path: Path,
    *,
    root: Path,
    character_name: str,
    character_slug: str,
    source_category: str,
    default_scope: str = "legacy",
) -> list[MemoryRecord]:
    """Parse a compatible Markdown file into bounded records."""
    try:
        if path.is_symlink() or not path.is_file():
            return []
        text = path.read_text(encoding="utf-8-sig")
        updated_at = _safe_timestamp(path.stat().st_mtime)
    except (OSError, UnicodeDecodeError):
        return []
    scope = _file_scope(path, _source_scope(path, root, default_scope))
    records: list[MemoryRecord] = []
    for ordinal, (display, start, end) in enumerate(_markdown_blocks(text)):
        bounded = safe_memory_text(display)
        if not bounded:
            continue
        category = source_category
        if scope.startswith("server:") and source_category == "character_memory":
            category = "server_memory"
        record_id = _record_id(
            source_file=path.name,
            scope=scope,
            text=display,
            ordinal=ordinal,
        )
        records.append(
            MemoryRecord(
                record_id=record_id,
                text=bounded,
                scope=scope,
                character_name=character_name,
                character_slug=character_slug,
                source_file=path.name,
                source_category=category,
                created_at=updated_at,
                updated_at=updated_at,
                ordinal=ordinal,
                source_path=path,
                source_root=root,
                start_offset=start,
                end_offset=end,
                scope_keys=(scope,),
            )
        )
    return records


def workspace_record(
    *,
    key: str,
    text: str,
    scope: str,
    character_name: str,
    character_slug: str,
    created_at: float | None,
    updated_at: float | None,
    scope_keys: tuple[str, ...] = (),
    source_key: str | None = None,
) -> MemoryRecord | None:
    """Build a safe memory record from one session workspace entry."""
    bounded = safe_memory_text(text)
    if not bounded:
        return None
    record_id = _record_id(
        source_file="session workspace",
        scope=scope,
        text=text,
        ordinal=0,
    )
    return MemoryRecord(
        record_id=record_id,
        text=bounded,
        scope=scope,
        character_name=character_name,
        character_slug=character_slug,
        source_file="session workspace",
        source_category="workspace",
        created_at=created_at,
        updated_at=updated_at,
        source_key=source_key or key,
        scope_keys=scope_keys or (scope,),
    )


def recap_record(
    *,
    source_scope: str,
    day: str,
    text: str,
    generated_at: float | None,
    character_name: str,
    character_slug: str,
    scope_keys: tuple[str, ...] = (),
    source_key: str | None = None,
) -> MemoryRecord | None:
    """Build a safe memory record from one generated nightly recap."""
    bounded = safe_memory_text(text, MEMORY_RECORD_MAX_CHARACTERS)
    if not bounded:
        return None
    record_id = _record_id(
        source_file="nightly-recaps.json",
        scope=source_scope,
        text=f"{day}\0{text}",
        ordinal=0,
    )
    return MemoryRecord(
        record_id=record_id,
        text=bounded,
        scope=source_scope,
        character_name=character_name,
        character_slug=character_slug,
        source_file="nightly-recaps.json",
        source_category="recap",
        created_at=generated_at,
        updated_at=generated_at,
        source_kind="recap",
        source_key=source_key or day,
        scope_keys=scope_keys or (source_scope,),
    )


def append_audit(
    root: Path,
    *,
    record: MemoryRecord,
    action: str,
    replacement: str | None,
) -> bool:
    """Write a private recoverable pre-change snapshot before mutation."""
    if action not in {"forget", "edit"}:
        return False
    path = root / MEMORY_AUDIT_FILE
    try:
        raw = path.read_text(encoding="utf-8") if path.exists() else "[]"
        entries = json.loads(raw)
        if not isinstance(entries, list):
            return False
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    entry = {
        "audit_id": hashlib.sha256(
            f"{new_unique_token()}:{record.record_id}".encode()
        ).hexdigest()[:24],
        "record_id": record.record_id,
        "action": action,
        "text": record.text,
        "replacement": safe_memory_text(replacement) if replacement else None,
        "scope": record.scope,
        "source_file": record.source_file,
        "source_category": record.source_category,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entries.append(entry)
    entries = entries[-MEMORY_AUDIT_MAX_ENTRIES:]
    encoded = json.dumps(entries, indent=2, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > MEMORY_AUDIT_MAX_BYTES:
        return False
    temporary = path.with_suffix(".tmp")
    try:
        root.mkdir(parents=True, exist_ok=True)
        root.chmod(0o700)
        temporary.write_text(encoded, encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)
        return True
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        return False
