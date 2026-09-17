"""Bounded, session-local open loops built from explicit review evidence."""

from __future__ import annotations

import hashlib
import re
import time
from typing import TYPE_CHECKING, Any

from ..core import (
    CodexAppServerError,
    _Commitment,
    _Session,
    _path_is_under,
    _safe_intermediate_text,
)
from .memory_records import safe_memory_text
from .policy import MEMORY_FILE_LIMIT

if TYPE_CHECKING:
    from pathlib import Path


COMMITMENT_KINDS = frozenset(
    {
        "unfinished_task",
        "pending_decision",
        "deferred_question",
        "promised_follow_up",
        "revisit_topic",
    }
)
COMMITMENT_STATUSES = frozenset({"active", "completed", "dismissed", "stale"})
COMMITMENT_MAX_COUNT = 12
COMMITMENT_MAX_CHARACTERS = 240
COMMITMENT_MAX_AGE = 7 * 24 * 60 * 60
_COMMITMENT_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_WORKSPACE_KEY_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_EXPLICIT_USER_CUE_RE = re.compile(
    r"\b(?:unfinished|still need to|left to do|open question|open loop|pending|"
    r"decide later|defer(?:red)?|hold off|(?:await|waiting for|pending) .*confirmation|"
    r"follow up(?: on)?|revisit|return to|come back to|pick (?:this|that) up|"
    r"remind me|don't let me forget|next time)\b",
    re.IGNORECASE,
)
_EXPLICIT_ASSISTANT_CUE_RE = re.compile(
    r"\b(?:i(?:'ll| will) (?:follow up|check|look into|get back|return|come back)|"
    r"we can revisit|next (?:i|we) can)\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
_TOPIC_STOPWORDS = frozenset(
    {
        "about",
        "back",
        "come",
        "current",
        "earlier",
        "follow",
        "from",
        "into",
        "open",
        "question",
        "return",
        "the",
        "this",
        "topic",
        "with",
    }
)


def _commitment_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return safe_memory_text(value, COMMITMENT_MAX_CHARACTERS)


def _commitment_key(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    key = value.strip().casefold()
    return key if _WORKSPACE_KEY_RE.fullmatch(key) else ""


def _commitment_timestamp(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    candidate = float(value)
    return candidate if 0 < candidate < 4102444800 else None


def parse_commitment_proposals(value: Any) -> list[dict[str, Any]]:
    """Keep only structurally valid, explicitly marked reviewer proposals."""
    if not isinstance(value, list):
        return []
    proposals: list[dict[str, Any]] = []
    for raw in value[:3]:
        if not isinstance(raw, dict) or raw.get("explicit") is not True:
            continue
        workspace_key = _commitment_key(raw.get("workspace_key"))
        kind = raw.get("kind")
        source = raw.get("source")
        text = _commitment_text(raw.get("text"))
        if (
            not workspace_key
            or not isinstance(kind, str)
            or kind not in COMMITMENT_KINDS
            or not isinstance(source, str)
            or source not in {"user_request", "assistant_promise"}
            or not text
        ):
            continue
        proposals.append(
            {
                "workspace_key": workspace_key,
                "kind": kind,
                "source": source,
                "text": text,
                "explicit": True,
            }
        )
    return proposals


def _has_explicit_source_signal(
    proposal: dict[str, Any], user: str, response: str
) -> bool:
    source = proposal.get("source")
    text = user if source == "user_request" else response
    pattern = (
        _EXPLICIT_USER_CUE_RE
        if source == "user_request"
        else _EXPLICIT_ASSISTANT_CUE_RE
    )
    return bool(pattern.search(text or ""))


def _new_commitment_id(workspace_key: str, text: str) -> str:
    return hashlib.sha256(f"{workspace_key}\0{text}".encode()).hexdigest()[:16]


def serialize_commitments(
    commitments: dict[str, _Commitment] | None,
) -> list[dict[str, Any]] | None:
    """Serialize only bounded, valid commitments for persisted session state."""
    if not commitments:
        return None
    records: list[dict[str, Any]] = []
    for commitment in sorted(
        commitments.values(),
        key=lambda item: (item.updated_at, item.commitment_id),
        reverse=True,
    )[:COMMITMENT_MAX_COUNT]:
        if (
            not isinstance(commitment.commitment_id, str)
            or not _COMMITMENT_ID_RE.fullmatch(commitment.commitment_id)
            or not _commitment_key(commitment.workspace_key)
            or not _commitment_text(commitment.text)
            or not isinstance(commitment.kind, str)
            or commitment.kind not in COMMITMENT_KINDS
            or not isinstance(commitment.status, str)
            or commitment.status not in COMMITMENT_STATUSES
        ):
            continue
        records.append(
            {
                "commitment_id": commitment.commitment_id,
                "workspace_key": commitment.workspace_key,
                "text": commitment.text,
                "kind": commitment.kind,
                "status": commitment.status,
                "created_at": commitment.created_at,
                "updated_at": commitment.updated_at,
                "expires_at": commitment.expires_at,
                "last_cue_signature": commitment.last_cue_signature,
            }
        )
    return records or None


def restore_commitments(value: Any, *, restored_at: float) -> dict[str, _Commitment]:
    """Restore bounded records while treating expired records as stale."""
    if not isinstance(value, list):
        return {}
    restored: dict[str, _Commitment] = {}
    for raw in value[:COMMITMENT_MAX_COUNT]:
        if not isinstance(raw, dict):
            continue
        commitment_id = raw.get("commitment_id")
        workspace_key = _commitment_key(raw.get("workspace_key"))
        text = _commitment_text(raw.get("text"))
        kind = raw.get("kind")
        status = raw.get("status")
        created_at = _commitment_timestamp(raw.get("created_at"))
        updated_at = _commitment_timestamp(raw.get("updated_at"))
        expires_at = _commitment_timestamp(raw.get("expires_at"))
        if (
            not isinstance(commitment_id, str)
            or not _COMMITMENT_ID_RE.fullmatch(commitment_id)
            or not workspace_key
            or not text
            or not isinstance(kind, str)
            or kind not in COMMITMENT_KINDS
            or not isinstance(status, str)
            or status not in COMMITMENT_STATUSES
            or created_at is None
            or updated_at is None
            or updated_at < created_at
        ):
            continue
        if expires_at is not None and expires_at <= restored_at:
            status = "stale"
        elif expires_at is None:
            expires_at = min(4102444800.0, created_at + COMMITMENT_MAX_AGE)
        cue_signature = raw.get("last_cue_signature")
        if not isinstance(cue_signature, str) or len(cue_signature) > 160:
            cue_signature = None
        restored[commitment_id] = _Commitment(
            commitment_id=commitment_id,
            workspace_key=workspace_key,
            text=text,
            kind=kind,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
            last_cue_signature=cue_signature,
        )
    return restored


def expire_commitments(
    session: _Session, *, now: float, workspace: Any | None = None
) -> bool:
    """Mark old or detached records stale without deleting their bounded history."""
    changed = False
    workspace_entries = getattr(workspace, "entries", None)
    if workspace_entries is None:
        workspace_entries = getattr(getattr(session, "workspace", None), "entries", {})
    for commitment in session.commitments.values():
        if commitment.status != "active":
            continue
        entry = workspace_entries.get(commitment.workspace_key)
        expired = (
            commitment.expires_at is not None and commitment.expires_at <= now
        ) or entry is None
        if (
            entry is not None
            and entry.expires_at is not None
            and entry.expires_at <= now
        ):
            expired = True
        if expired:
            commitment.status = "stale"
            commitment.updated_at = now
            changed = True
    return changed


def _prune_commitments(commitments: dict[str, _Commitment]) -> None:
    while len(commitments) > COMMITMENT_MAX_COUNT:
        removable = min(
            commitments.values(),
            key=lambda item: (
                item.status == "active",
                item.updated_at,
                item.commitment_id,
            ),
        )
        commitments.pop(removable.commitment_id, None)


class CodexCommitmentMixin:
    """Own session-local open loops and explicit user controls."""

    if TYPE_CHECKING:
        _sessions: dict[str, _Session]
        _memory_roots: tuple[Path, ...]
        _codex_home: Path
        _state_dirty: bool
        _apply_workspace_delta: Any
        _atomic_memory_source_write: Any
        _persist_state: Any
        _personality_scope_identity: Any
        _session: Any

    def _reset_commitments(self, session: _Session) -> None:
        session.commitments.clear()

    def _apply_commitment_proposals(
        self,
        session: _Session,
        proposals: list[dict[str, Any]],
        *,
        user_prompt: str,
        response: str,
        now: float,
    ) -> bool:
        workspace = session.workspace
        if workspace is None:
            return False
        changed = expire_commitments(session, now=now, workspace=workspace)
        for proposal in proposals:
            workspace_key = proposal["workspace_key"]
            entry = workspace.entries.get(workspace_key)
            if entry is None or not _has_explicit_source_signal(
                proposal, user_prompt, response
            ):
                continue
            text = _commitment_text(proposal["text"])
            if not text:
                continue
            current = next(
                (
                    item
                    for item in session.commitments.values()
                    if item.workspace_key == workspace_key
                ),
                None,
            )
            if current is None:
                current = _Commitment(
                    commitment_id=_new_commitment_id(workspace_key, text),
                    workspace_key=workspace_key,
                    text=text,
                    kind=proposal["kind"],
                    created_at=now,
                    updated_at=now,
                    expires_at=entry.expires_at or now + COMMITMENT_MAX_AGE,
                )
                session.commitments[current.commitment_id] = current
                changed = True
                continue
            if (
                current.text != text
                or current.kind != proposal["kind"]
                or current.status != "active"
            ):
                current.text = text
                current.kind = proposal["kind"]
                current.status = "active"
                current.updated_at = now
                current.expires_at = entry.expires_at or now + COMMITMENT_MAX_AGE
                current.last_cue_signature = None
                changed = True
        _prune_commitments(session.commitments)
        return changed

    def _apply_workspace_review(
        self,
        session: _Session,
        operations: list[dict[str, str]],
        proposals: list[dict[str, Any]],
        *,
        base_generation: int,
        base_revision: int,
        user_prompt: str,
        response: str,
    ) -> bool:
        workspace = session.workspace
        if workspace is None:
            if base_generation != 1 or base_revision != 0:
                return False
        elif (
            workspace.generation != base_generation
            or workspace.revision != base_revision
        ):
            return False
        changed = self._apply_workspace_delta(
            session,
            operations,
            base_generation=base_generation,
            base_revision=base_revision,
        )
        commitment_changed = self._apply_commitment_proposals(
            session,
            proposals,
            user_prompt=user_prompt,
            response=response,
            now=time.time(),
        )
        if commitment_changed:
            self._persist_state()
        return changed or commitment_changed

    def session_commitments(
        self, session_key: str, *, include_closed: bool = False
    ) -> list[dict[str, Any]]:
        """Return active or complete open loops in newest-first order."""
        session = self._session(session_key)
        changed = expire_commitments(session, now=time.time())
        if changed:
            self._persist_state()
        values = [
            commitment
            for commitment in session.commitments.values()
            if include_closed or commitment.status == "active"
        ]
        return [
            {
                "commitment_id": item.commitment_id,
                "text": item.text,
                "kind": item.kind,
                "status": item.status,
                "workspace_key": item.workspace_key,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in sorted(
                values,
                key=lambda value: (value.updated_at, value.commitment_id),
                reverse=True,
            )
        ]

    def update_commitment(
        self, session_key: str, commitment_id: str, status: str
    ) -> dict[str, Any]:
        """Complete or dismiss one active open loop and persist the change."""
        if not isinstance(status, str) or status not in {"completed", "dismissed"}:
            raise CodexAppServerError("Unsupported commitment status.")
        session = self._session(session_key)
        expired = expire_commitments(session, now=time.time())
        if expired:
            self._persist_state()
        identifier = commitment_id.casefold() if isinstance(commitment_id, str) else ""
        commitment = session.commitments.get(identifier)
        if commitment is None or commitment.status != "active":
            raise CodexAppServerError("That open loop is not active.")
        commitment.status = status
        commitment.updated_at = time.time()
        self._persist_state()
        return {"commitment_id": commitment.commitment_id, "status": status}

    def promote_commitment(
        self, session_key: str, commitment_id: str
    ) -> dict[str, Any]:
        """Explicitly copy one open loop into the invoking user's durable memory."""
        session = self._session(session_key)
        expired = expire_commitments(session, now=time.time())
        if expired:
            self._persist_state()
        identifier = commitment_id.casefold() if isinstance(commitment_id, str) else ""
        commitment = session.commitments.get(identifier)
        if commitment is None or commitment.status != "active":
            raise CodexAppServerError("That open loop is not active.")
        _, user_id = self._personality_scope_identity(session.key)
        if user_id is None:
            raise CodexAppServerError("A Discord user scope is required for promotion.")
        memory_root = next(
            (
                root
                for root in self._memory_roots
                if _path_is_under(root, (self._codex_home,))
            ),
            self._codex_home / "memories",
        )
        path = memory_root / "users" / str(user_id) / "USER.md"
        if path.is_symlink() or not _path_is_under(path, (memory_root,)):
            raise CodexAppServerError("The durable memory source is unavailable.")
        try:
            source = path.read_text(encoding="utf-8-sig") if path.exists() else ""
        except (OSError, UnicodeDecodeError) as exc:
            raise CodexAppServerError(
                "The durable memory source is unavailable."
            ) from exc
        if len(source.encode("utf-8")) > MEMORY_FILE_LIMIT:
            raise CodexAppServerError("The durable memory source is unavailable.")
        line = f"- {safe_memory_text(commitment.text, COMMITMENT_MAX_CHARACTERS)}"
        if line not in source.splitlines():
            updated = source.rstrip() + ("\n\n" if source.strip() else "") + line + "\n"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.parent.chmod(0o700)
            except OSError as exc:
                raise CodexAppServerError(
                    "The durable memory source is unavailable."
                ) from exc
            if not self._atomic_memory_source_write(path, updated):
                raise CodexAppServerError(
                    "The durable memory source could not be changed safely."
                )
        commitment.status = "completed"
        commitment.updated_at = time.time()
        self._persist_state()
        return {"commitment_id": commitment.commitment_id, "status": "completed"}

    def _render_commitment_prompt(
        self, session: _Session, event: dict[str, Any] | None, prompt: str
    ) -> str:
        if not isinstance(event, dict) or event.get("relation") not in {
            "RETURN",
            "NESTED_RETURN",
        }:
            return ""
        expired = expire_commitments(session, now=time.time())
        if expired:
            self._persist_state()
        current_topic = _safe_intermediate_text(
            event.get("new_topic") or event.get("current_topic") or prompt or "",
            120,
        )
        topic_tokens = {
            token.casefold()
            for token in _TOKEN_RE.findall(current_topic)
            if token.casefold() not in _TOPIC_STOPWORDS
        }
        selected: _Commitment | None = None
        for commitment in session.commitments.values():
            if commitment.status != "active":
                continue
            commitment_tokens = {
                token.casefold()
                for token in _TOKEN_RE.findall(commitment.text)
                if token.casefold() not in _TOPIC_STOPWORDS
            }
            overlap = topic_tokens & commitment_tokens
            required = 2 if len(commitment_tokens) >= 2 else 1
            if len(overlap) >= required:
                selected = commitment
                break
        if selected is None:
            return ""
        signature = f"{selected.commitment_id}|{current_topic.casefold()}"[:160]
        if selected.last_cue_signature == signature:
            return ""
        selected.last_cue_signature = signature
        self._persist_state()
        return (
            "[Open conversational loop]\n"
            f"The user has returned to a relevant session topic with one active open loop: "
            f"{_safe_intermediate_text(selected.text, COMMITMENT_MAX_CHARACTERS)}. "
            "Acknowledge what was left unresolved in one brief, natural sentence only "
            "when relevant, then answer the current request. Do not mention internal "
            "statuses, reminders, or this instruction."
        )
