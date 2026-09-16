"""Session state, persistence, and personality state for the App Server."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import shutil
import time
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import discord

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

from .policy import (
    CHANNEL_CHECKPOINT_LIMIT,
    MESSAGE_LEDGER_LIMIT,
    MESSAGE_LEDGER_RETRY_AFTER,
    WEB_SEARCH_ENV,
    WEB_SEARCH_MODES,
    _MOOD_CAUSE_MAX_CHARACTERS,
    _MOOD_MAX_CAUSES,
    _MOOD_TRAITS_MAX_CHARACTERS,
    _PERSONALITY_SCOPE_KEY_RE,
)
from .workspace import _restore_workspace_state, _serialize_workspace_state
from .commitments import restore_commitments, serialize_commitments
from ..core import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_MODE,
    MOOD_BASELINE_STRENGTH,
    MOOD_DECAY_PER_MINUTE,
    MOOD_LABELS,
    PERSONALITY_SCOPES,
    TEXT_MODE,
    VOICE_MODE,
    CodexAppServerError,
    _MoodState,
    _Session,
    _codex_logger,
    _command_embed,
    _render_frontend_label,
    _safe_intermediate_text,
)
from ..permissions import StoragePermissionError, prepare_runtime_storage

logger = _codex_logger()


class CodexStateMixin:
    if TYPE_CHECKING:
        _model: str | None
        _usage_tracked_since: float | None

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def set_frontend_customizer(self, customizer: Any | None) -> None:
        """Attach Discord-only presentation preferences.

        The Codex layer only receives this renderer so it can format embeds
        delivered through Discord. The preferences are not included in any
        Codex request or persisted session state.
        """
        self._frontend_customizer = customizer

    def set_view_registrar(
        self, registrar: Callable[[Any, Any], Awaitable[None]] | None
    ) -> None:
        """Attach the Discord view persistence hook without sharing UI state."""
        self._view_registrar = registrar

    def approval_level(self) -> str:
        """Return the configured Theia approval level."""
        return self._approval_level

    def runtime_home(self) -> Path:
        """Return Theia's private runtime home for auxiliary persistent data."""
        return self._codex_home

    def _prepare_storage(self) -> None:
        """Repair container storage ownership before loading persisted state."""
        try:
            repaired = prepare_runtime_storage(
                self._codex_home,
                Path(self._cwd),
                self._state_path,
            )
        except StoragePermissionError as exc:
            logger.critical("FATAL: Theia %s permissions remain unusable", exc.area)
            raise CodexAppServerError(
                "FATAL: Theia storage permissions are unusable."
            ) from exc
        if repaired:
            logger.info("Repaired Theia runtime and workspace ownership")

    def _frontend_embed(
        self,
        channel: discord.abc.Messageable | None,
        target: str,
        title: str,
        description: str,
        *,
        color: discord.Color | None = None,
        context: dict[str, Any] | None = None,
    ) -> discord.Embed:
        guild_id = getattr(getattr(channel, "guild", None), "id", None)
        return _command_embed(
            title,
            description,
            color=color,
            target=target,
            guild_id=guild_id if isinstance(guild_id, int) else None,
            customizer=self._frontend_customizer,
            context=context,
        )

    def _frontend_label(
        self,
        channel: discord.abc.Messageable | None,
        target: str,
        default: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> str:
        guild_id = getattr(getattr(channel, "guild", None), "id", None)
        return _render_frontend_label(
            self._frontend_customizer,
            guild_id if isinstance(guild_id, int) else None,
            target,
            default,
            context=context,
        )

    def _configured_web_search_mode(self) -> str:
        requested = os.getenv(WEB_SEARCH_ENV, "").strip().casefold()
        if not requested:
            return "indexed"
        if requested in WEB_SEARCH_MODES:
            return requested
        logger.warning(
            "Ignoring unsupported web search mode; using indexed search instead "
            "(supported=%s)",
            ",".join(sorted(WEB_SEARCH_MODES)),
        )
        return "indexed"

    @staticmethod
    def _set_top_level_web_search(config: str, mode: str) -> str:
        """Set the root-level web search option without rewriting TOML."""
        lines = config.splitlines(keepends=True)
        assignment = re.compile(r"^\s*(?:web_search|[\"']web_search[\"'])\s*=")
        for index, line in enumerate(lines):
            if line.lstrip().startswith("["):
                break
            if assignment.match(line):
                newline = "\n" if line.endswith("\n") else ""
                lines[index] = f'web_search = "{mode}"{newline}'
                return "".join(lines)
        return f'web_search = "{mode}"\n{config}'

    def _ensure_web_search_config(self) -> None:
        """Default the private Codex runtime to index-gated web search."""
        config_path = self._codex_home / "config.toml"
        requested = os.getenv(WEB_SEARCH_ENV, "").strip()
        mode = self._configured_web_search_mode()

        try:
            config = config_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            config = ""
        except OSError as exc:
            logger.warning(
                "Could not inspect Codex configuration (error=%s)",
                type(exc).__name__,
            )
            return

        try:
            parsed = tomllib.loads(config) if config.strip() else {}
        except tomllib.TOMLDecodeError as exc:
            logger.warning(
                "Could not parse Codex configuration; leaving it unchanged (error=%s)",
                type(exc).__name__,
            )
            return

        configured = parsed.get("web_search")
        if not requested and configured is not None:
            return
        if isinstance(configured, str) and configured.casefold() == mode:
            return

        updated = self._set_top_level_web_search(config, mode)
        temporary = config_path.with_name(f".{config_path.name}.tmp")
        try:
            file_mode = 0o600
            temporary.write_text(updated, encoding="utf-8")
            temporary.chmod(file_mode)
            temporary.replace(config_path)
        except OSError as exc:
            logger.warning(
                "Could not configure Codex web search (error=%s)",
                type(exc).__name__,
            )
            with contextlib.suppress(OSError):
                temporary.unlink()
            return

        logger.info("Codex web search configured (mode=%s)", mode)

    def _migrate_legacy_state(self) -> None:
        source = self._legacy_state_path
        if source is None:
            return
        target = self._state_path
        try:
            if target.exists() or not source.is_file():
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except OSError:
            return

    def _migrate_legacy_home(self) -> None:
        source = self._legacy_codex_home
        if source is None:
            return
        try:
            if not source.is_dir():
                return
            for item in source.iterdir():
                target = self._codex_home / item.name
                if target.exists() or target.is_symlink():
                    continue
                if item.is_dir():
                    shutil.copytree(item, target, symlinks=True)
                else:
                    shutil.copy2(item, target)
        except OSError:
            return

    @staticmethod
    def _bounded_mood_text(value: Any, limit: int) -> str:
        """Keep restored or derived mood text short and free of hidden detail."""
        text = _safe_intermediate_text(value, limit)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _restore_mood_state(
        cls, value: Any, *, restored_at: float
    ) -> _MoodState | None:
        """Restore only the bounded, non-durable mood record from session state."""
        if not isinstance(value, dict):
            return None
        baseline_traits = cls._bounded_mood_text(
            value.get("baseline_traits"), _MOOD_TRAITS_MAX_CHARACTERS
        )
        if not baseline_traits:
            return None
        baseline_cause = (
            cls._bounded_mood_text(
                value.get("baseline_cause"), _MOOD_CAUSE_MAX_CHARACTERS
            )
            or "Theia's default resting affect is steady and attentive."
        )
        traits = (
            cls._bounded_mood_text(value.get("traits"), _MOOD_TRAITS_MAX_CHARACTERS)
            or baseline_traits
        )
        label = str(value.get("label") or "neutral").casefold()
        if label not in MOOD_LABELS:
            label = "neutral"
        raw_strength = value.get("strength")
        strength = MOOD_BASELINE_STRENGTH
        if isinstance(raw_strength, (int, float)) and not isinstance(
            raw_strength, bool
        ):
            candidate_strength = float(raw_strength)
            if math.isfinite(candidate_strength):
                strength = candidate_strength
        strength = max(0.0, min(1.0, strength))
        raw_causes = value.get("causes")
        cause_values = raw_causes if isinstance(raw_causes, list) else []
        causes = tuple(
            cause
            for cause in (
                cls._bounded_mood_text(item, _MOOD_CAUSE_MAX_CHARACTERS)
                for item in cause_values
            )
            if cause
        )[:_MOOD_MAX_CAUSES]
        profile_key = value.get("profile_key")
        if not isinstance(profile_key, str) or not profile_key:
            profile_key = None
        raw_updated_at = value.get("updated_at")
        updated_at = None
        if isinstance(raw_updated_at, (int, float)) and not isinstance(
            raw_updated_at, bool
        ):
            candidate_updated_at = float(raw_updated_at)
            if math.isfinite(candidate_updated_at) and candidate_updated_at > 0:
                updated_at = candidate_updated_at
        transient = bool(value.get("transient")) and label != "neutral"
        mood = _MoodState(
            profile_key=profile_key,
            baseline_traits=baseline_traits,
            baseline_cause=baseline_cause,
            traits=traits,
            label=label,
            strength=strength,
            causes=causes,
            updated_at=updated_at,
            transient=transient,
            last_event_signature=(
                value.get("last_event_signature")
                if isinstance(value.get("last_event_signature"), str)
                else None
            ),
        )
        if not mood.transient:
            cls._restore_neutral_mood(mood)
            return mood
        elapsed_minutes = max(0.0, restored_at - (mood.updated_at or restored_at)) / 60
        mood.strength = max(
            0.0, mood.strength - MOOD_DECAY_PER_MINUTE * elapsed_minutes
        )
        if mood.strength <= 0.0:
            cls._restore_neutral_mood(mood)
        else:
            mood.updated_at = restored_at
        return mood

    @staticmethod
    def _restore_neutral_mood(mood: _MoodState) -> None:
        """Return one mood object to its cached profile-specific resting state."""
        mood.traits = mood.baseline_traits
        mood.label = "neutral"
        mood.strength = MOOD_BASELINE_STRENGTH
        mood.causes = (mood.baseline_cause,)
        mood.updated_at = None
        mood.transient = False

    def _load_state(self) -> None:
        try:
            raw = self._state_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except UnicodeDecodeError as exc:
            self._quarantine_state(type(exc).__name__)
            return
        except OSError as exc:
            self._state_recovery_blocked = True
            logger.error(
                "Could not read Theia state; refusing to overwrite it (error=%s)",
                type(exc).__name__,
            )
            return
        try:
            data = json.loads(raw)
        except ValueError as exc:
            self._quarantine_state(type(exc).__name__)
            return
        if not isinstance(data, dict):
            self._quarantine_state("invalid top-level JSON value")
            return
        if isinstance(data, dict):
            model = data.get("model")
            self._model = str(model) if model else DEFAULT_CODEX_MODEL
            personality_scopes = data.get("personality_scopes")
            if isinstance(personality_scopes, dict):
                for raw_key, raw_record in personality_scopes.items():
                    if not isinstance(raw_key, str) or not isinstance(raw_record, dict):
                        continue
                    key = raw_key.strip()
                    if not _PERSONALITY_SCOPE_KEY_RE.fullmatch(key):
                        continue
                    scope = str(raw_record.get("scope") or "").casefold()
                    if scope not in PERSONALITY_SCOPES:
                        continue
                    expected_prefix = {
                        "me": "me:",
                        "server": "server:",
                        "everyone": "everyone",
                    }[scope]
                    if not key.startswith(expected_prefix):
                        continue
                    name = raw_record.get("name")
                    if name is not None and not isinstance(name, str):
                        continue
                    set_by = raw_record.get("set_by")
                    if not isinstance(set_by, int) or isinstance(set_by, bool):
                        set_by = None
                    self._personality_scopes[key] = {
                        "scope": scope,
                        "name": name,
                        "set_by": set_by,
                    }
            authenticated_users = data.get("authenticated_users")
            if isinstance(authenticated_users, list):
                self._authenticated_users.update(
                    user_id
                    for user_id in authenticated_users
                    if isinstance(user_id, int) and not isinstance(user_id, bool)
                )
            authenticated_guilds = data.get("authenticated_guilds")
            if isinstance(authenticated_guilds, list):
                self._authenticated_guilds.update(
                    guild_id
                    for guild_id in authenticated_guilds
                    if isinstance(guild_id, int) and not isinstance(guild_id, bool)
                )
            improvement_history = data.get("self_improvement_history")
            if isinstance(improvement_history, list):
                restored_history = self._restore_self_improvement_history(
                    improvement_history
                )
                self._self_improvement_history = restored_history
                if len(restored_history) != len(improvement_history):
                    self._state_needs_cleanup = True
            sessions = data.get("sessions")
            if isinstance(sessions, dict):
                state_now = time.time()
                for key, value in sessions.items():
                    if str(key).startswith("__"):
                        self._state_needs_cleanup = True
                        continue
                    if not isinstance(value, dict):
                        continue
                    thread_id = value.get("thread_id")
                    personality_name = value.get("personality_name")
                    personality_selected = value.get("personality_selected")
                    self_improvement_summary = value.get("self_improvement_summary")
                    instruction_fingerprint = value.get("instruction_fingerprint")
                    tool_policy = value.get("tool_policy")
                    mood = self._restore_mood_state(
                        value.get("mood"), restored_at=state_now
                    )
                    attention = self._restore_attention_state(value.get("attention"))
                    workspace = _restore_workspace_state(
                        value.get("workspace"), restored_at=state_now
                    )
                    commitments = restore_commitments(
                        value.get("commitments"), restored_at=state_now
                    )
                    mode = value.get("mode")
                    last_activity_at = value.get("last_activity_at")
                    if (
                        isinstance(last_activity_at, (int, float))
                        and not isinstance(last_activity_at, bool)
                        and last_activity_at > 0
                    ):
                        saved_last_activity_at: float | None = float(last_activity_at)
                    elif thread_id:
                        # State written before retention support gets a full
                        # retention window instead of being deleted on upgrade.
                        saved_last_activity_at = state_now
                    else:
                        saved_last_activity_at = None
                    if isinstance(tool_policy, bool):
                        saved_tool_policy: bool | None = tool_policy
                    else:
                        saved_tool_policy = None
                    saved_mode = (
                        str(mode)
                        if isinstance(mode, str) and mode in {TEXT_MODE, VOICE_MODE}
                        else DEFAULT_MODE
                    )
                    saved_self_improvement_summary = (
                        self._bound_self_improvement_summary(self_improvement_summary)
                        if isinstance(self_improvement_summary, str)
                        else None
                    )
                    has_non_mood_state = bool(
                        thread_id
                        or personality_name
                        or personality_selected is True
                        or saved_self_improvement_summary
                        or saved_tool_policy is not None
                        or saved_mode != DEFAULT_MODE
                        or attention is not None
                        or workspace is not None
                        or bool(commitments)
                    )
                    if has_non_mood_state or (
                        mood is not None
                        and (thread_id or saved_last_activity_at is not None)
                    ):
                        self._sessions[str(key)] = _Session(
                            key=str(key),
                            mode=saved_mode,
                            thread_id=str(thread_id) if thread_id else None,
                            personality_name=(
                                str(personality_name) if personality_name else None
                            ),
                            personality_selected=(
                                personality_selected
                                if isinstance(personality_selected, bool)
                                else bool(personality_name)
                            ),
                            pending_self_improvement_summary=(
                                saved_self_improvement_summary
                            ),
                            instruction_fingerprint=(
                                str(instruction_fingerprint)
                                if instruction_fingerprint
                                else None
                            ),
                            tool_policy=saved_tool_policy,
                            mood=mood,
                            attention=attention,
                            workspace=workspace,
                            commitments=commitments,
                            archived=bool(value.get("archived"))
                            if thread_id
                            else False,
                            last_activity_at=saved_last_activity_at,
                        )
                    elif mood is not None:
                        self._state_needs_cleanup = True
            aliases = data.get("session_aliases")
            if isinstance(aliases, dict):
                self._session_aliases.update(
                    {
                        str(source): str(target)
                        for source, target in aliases.items()
                        if source and target and source != target
                    }
                )
            message_ledger = data.get("message_ledger")
            if isinstance(message_ledger, dict):
                now = time.time()
                for message_id, value in message_ledger.items():
                    if not isinstance(value, dict):
                        continue
                    updated_at = value.get("updated_at")
                    if not isinstance(updated_at, (int, float)):
                        continue
                    if now - updated_at <= MESSAGE_LEDGER_RETRY_AFTER:
                        self._message_ledger[str(message_id)] = {
                            "status": str(value.get("status") or "processing"),
                            "updated_at": updated_at,
                        }
            discord_threads = data.get("discord_threads")
            if isinstance(discord_threads, list):
                self._discord_threads.update(
                    thread_id
                    for thread_id in discord_threads
                    if isinstance(thread_id, int) and not isinstance(thread_id, bool)
                )
            checkpoints = data.get("channel_checkpoints")
            if isinstance(checkpoints, dict):
                for channel_id, message_id in checkpoints.items():
                    if str(channel_id).isdigit() and isinstance(message_id, int):
                        self._channel_checkpoints[int(channel_id)] = message_id
            self._restore_usage_state(data.get("theia_usage"))

    def _quarantine_state(self, reason: str) -> None:
        """Preserve an unreadable state file before allowing recovery writes."""
        quarantine = self._state_path.with_name(
            f"{self._state_path.name}.corrupt-{time.time_ns()}"
        )
        try:
            self._state_path.replace(quarantine)
        except OSError as exc:
            self._state_recovery_blocked = True
            logger.error(
                "Could not quarantine corrupt Theia state (reason=%s, error=%s); "
                "the original file will not be overwritten",
                reason,
                type(exc).__name__,
            )
            return
        with contextlib.suppress(OSError):
            quarantine.chmod(0o600)
        self._state_recovery_blocked = False
        logger.warning(
            "Preserved corrupt Theia state as %s (reason=%s)",
            quarantine.name,
            reason,
        )

    def _persist_state(self) -> None:
        if self._state_recovery_blocked:
            self._state_dirty = True
            logger.error(
                "Theia state remains dirty because its unreadable file is being "
                "preserved"
            )
            return
        data = {
            "model": self._model,
            "authenticated_users": sorted(self._authenticated_users),
            "authenticated_guilds": sorted(self._authenticated_guilds),
            "personality_scopes": {
                key: dict(record) for key, record in self._personality_scopes.items()
            },
            "self_improvement_history": self._serialize_self_improvement_history(),
            "sessions": {
                key: {
                    "mode": session.mode,
                    "thread_id": session.thread_id,
                    "personality_name": session.personality_name,
                    "personality_selected": session.personality_selected,
                    "self_improvement_summary": session.pending_self_improvement_summary,
                    "instruction_fingerprint": session.instruction_fingerprint,
                    "tool_policy": session.tool_policy,
                    "mood": self._serialize_mood_state(session.mood),
                    "attention": self._serialize_attention_state(session.attention),
                    "workspace": _serialize_workspace_state(session.workspace),
                    "commitments": serialize_commitments(session.commitments),
                    "archived": session.archived,
                    "last_activity_at": session.last_activity_at,
                }
                for key, session in self._sessions.items()
                if not session.key.startswith("__")
                and (
                    session.mode != DEFAULT_MODE
                    or session.thread_id
                    or session.personality_name
                    or session.personality_selected
                    or session.pending_self_improvement_summary
                    or session.tool_policy is not None
                    or session.attention is not None
                    or (
                        session.workspace is not None
                        and bool(session.workspace.entries)
                    )
                    or bool(session.commitments)
                    or (
                        session.mood is not None
                        and session.last_activity_at is not None
                    )
                )
            },
            "session_aliases": dict(self._session_aliases),
            "message_ledger": dict(
                sorted(
                    self._message_ledger.items(),
                    key=lambda item: float(item[1].get("updated_at", 0)),
                    reverse=True,
                )[:MESSAGE_LEDGER_LIMIT]
            ),
            "discord_threads": sorted(self._discord_threads),
            "channel_checkpoints": dict(
                sorted(
                    self._channel_checkpoints.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:CHANNEL_CHECKPOINT_LIMIT]
            ),
            "theia_usage": self._serialize_usage_state(),
        }
        temporary = self._state_path.with_suffix(".tmp")
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temporary.chmod(0o600)
            temporary.replace(self._state_path)
        except OSError as exc:
            self._state_dirty = True
            logger.error(
                "Could not persist Theia state; the previous file was retained "
                "(error=%s)",
                type(exc).__name__,
            )
            with contextlib.suppress(OSError):
                temporary.unlink()
        else:
            self._state_dirty = False

    @staticmethod
    def _serialize_mood_state(mood: _MoodState | None) -> dict[str, Any] | None:
        """Serialize temporary mood separately from memories and other learned data."""
        if mood is None:
            return None
        return {
            "profile_key": mood.profile_key,
            "baseline_traits": mood.baseline_traits,
            "baseline_cause": mood.baseline_cause,
            "traits": mood.traits,
            "label": mood.label,
            "strength": max(0.0, min(1.0, mood.strength)),
            "causes": list(mood.causes[:_MOOD_MAX_CAUSES]),
            "updated_at": mood.updated_at,
            "transient": mood.transient and mood.label != "neutral",
            "last_event_signature": mood.last_event_signature,
        }

    def _canonical_session_key(self, key: str) -> str:
        current = key
        visited: set[str] = set()
        while current in self._session_aliases and current not in visited:
            visited.add(current)
            current = self._session_aliases[current]
        return current

    def _session(self, key: str) -> _Session:
        canonical_key = self._canonical_session_key(key)
        session = self._sessions.get(canonical_key)
        if session is None:
            session = _Session(key=canonical_key)
            self._sessions[canonical_key] = session
        if session.lock is None:
            session.lock = asyncio.Lock()
        return session

    def select_session(
        self,
        session_key: str,
        *,
        channel: Any | None = None,
        user: Any | None = None,
        event: str = "session_selected",
        preserve_route: bool = False,
    ) -> _Session:
        """Select an existing session for live operator state.

        The selection is a pointer to the normal ``_Session`` store, not a
        second session object. It is runtime-only so restored sessions remain
        invisible until the harness explicitly resumes or uses them again.
        """
        canonical_key = self._canonical_session_key(session_key)
        session = self._session(canonical_key)
        if canonical_key.startswith("__"):
            return session

        self._lighthouse_active_session_key = canonical_key
        session.lighthouse_status = "active"
        session.lighthouse_reason = None
        if channel is not None:
            guild = getattr(channel, "guild", None)
            session.lighthouse_is_guild = guild is not None
            session.lighthouse_channel_name = (
                _safe_intermediate_text(getattr(channel, "name", None), 60) or None
            )
            session.lighthouse_user_name = None
            if guild is None and user is not None:
                session.lighthouse_user_name = (
                    _safe_intermediate_text(
                        getattr(user, "display_name", None)
                        or getattr(user, "name", None),
                        60,
                    )
                    or None
                )
        elif not preserve_route:
            # A direct resume may not have a Discord channel object. Infer only
            # the conversation kind from the opaque key; never display the key.
            session.lighthouse_is_guild = (
                False
                if canonical_key.startswith("guild:0:")
                else True
                if canonical_key.startswith("guild:")
                else None
            )
            session.lighthouse_channel_name = None
            session.lighthouse_user_name = None
        self._record_runtime_event(event)
        return session

    def _mark_lighthouse_session_degraded(self, session: _Session, reason: str) -> None:
        """Expose a bounded lifecycle failure without retaining raw errors."""
        if session.key.startswith("__"):
            return
        self._lighthouse_active_session_key = session.key
        session.lighthouse_status = "degraded"
        session.lighthouse_reason = _safe_intermediate_text(reason, 96) or "unavailable"
        self._record_runtime_event("session_degraded", session.lighthouse_reason)

    def _clear_lighthouse_session(self, session: _Session) -> None:
        """Remove one session from the live dashboard selection."""
        if self._lighthouse_active_session_key == session.key:
            self._lighthouse_active_session_key = None
        session.lighthouse_status = "inactive"
        session.lighthouse_reason = None
        session.lighthouse_is_guild = None
        session.lighthouse_channel_name = None
        session.lighthouse_user_name = None
        self._record_runtime_event("session_reset")

    def rebind_session(self, old_key: str, new_key: str) -> bool:
        """Keep a turn's Codex session available after moving to a Discord thread."""
        old_canonical = self._canonical_session_key(old_key)
        new_canonical = self._canonical_session_key(new_key)
        if old_canonical == new_canonical:
            return True
        session = self._sessions.get(old_canonical)
        if session is None:
            return False
        existing = self._sessions.get(new_canonical)
        if existing is not None and existing is not session:
            logger.warning(
                "Could not rebind a Discord session because the target is active"
            )
            return False
        self._sessions.pop(old_canonical, None)
        session.key = new_canonical
        self._sessions[new_canonical] = session
        if self._lighthouse_active_session_key == old_canonical:
            self._lighthouse_active_session_key = new_canonical
        for alias, target in tuple(self._session_aliases.items()):
            if self._canonical_session_key(target) == old_canonical:
                self._session_aliases[alias] = new_canonical
        self._session_aliases[old_canonical] = new_canonical
        self._persist_state()
        logger.info("Rebound Codex session to a newly created Discord thread")
        return True

    @staticmethod
    def _validated_thread_id(thread_id: str) -> str:
        value = thread_id.strip()
        if not value:
            raise CodexAppServerError("A Codex thread id is required.")
        return value

    def _set_loaded_thread_ids(self, thread_ids: Iterable[str]) -> None:
        self._loaded_thread_ids = set(thread_ids)
        for session in self._sessions.values():
            if session.thread_id:
                session.loaded = session.thread_id in self._loaded_thread_ids

    def _set_thread_loaded(self, thread_id: str, loaded: bool) -> None:
        if loaded:
            self._loaded_thread_ids.add(thread_id)
        else:
            self._loaded_thread_ids.discard(thread_id)
        for session in self._sessions.values():
            if session.thread_id == thread_id:
                session.loaded = loaded

    def _set_thread_archived(self, thread_id: str, archived: bool) -> None:
        for session in self._sessions.values():
            if session.thread_id != thread_id:
                continue
            session.archived = archived
            if archived:
                session.loaded = False
                self._loaded_thread_ids.discard(thread_id)

    def _forget_thread(self, thread_id: str) -> None:
        self._loaded_thread_ids.discard(thread_id)
        affected_keys = {
            key
            for key, session in self._sessions.items()
            if session.thread_id == thread_id
        }
        for session in self._sessions.values():
            if session.thread_id != thread_id:
                continue
            self._clear_lighthouse_session(session)
            session.thread_id = None
            session.loaded = False
            session.turn_id = None
            session.instruction_fingerprint = None
            session.tool_policy = None
            session.archived = False
            session.last_activity_at = None
        for alias in tuple(self._session_aliases):
            if (
                alias in affected_keys
                or self._canonical_session_key(alias) in affected_keys
            ):
                self._session_aliases.pop(alias, None)
        for key, pending in tuple(self._pending_approvals.items()):
            if pending.thread_id != thread_id:
                continue
            self._pending_approvals.pop(key, None)
            if not pending.future.done():
                pending.future.set_result(
                    self._approval_result(pending.kind, pending.params, approved=False)
                )
        self._persist_state()

    def _forget_session(self, session_key: str) -> None:
        """Remove a session record and any aliases that point to it."""
        canonical_key = self._canonical_session_key(session_key)
        session = self._sessions.pop(canonical_key, None)
        if session is not None:
            self._clear_lighthouse_session(session)
        for alias in tuple(self._session_aliases):
            if (
                alias == canonical_key
                or self._canonical_session_key(alias) == canonical_key
            ):
                self._session_aliases.pop(alias, None)

    def has_session(self, key: str) -> bool:
        """Return whether ``key`` has an associated Codex thread."""
        session = self._sessions.get(key)
        return bool(session and session.thread_id)

    def is_participating_thread(self, thread_id: int) -> bool:
        """Return whether Theia has previously participated in a Discord thread."""
        return thread_id in self._discord_threads

    def mark_thread_participating(self, thread_id: int) -> None:
        """Persist a Discord thread as eligible for mention-free follow-ups."""
        if thread_id not in self._discord_threads:
            self._discord_threads.add(thread_id)
            self._persist_state()

    def channel_checkpoint(self, channel_id: int) -> int | None:
        """Return the newest Discord message checkpoint for a channel."""
        return self._channel_checkpoints.get(channel_id)

    def channel_checkpoints(self) -> tuple[int, ...]:
        """Return channel ids with persisted gateway backfill checkpoints."""
        return tuple(self._channel_checkpoints)

    def checkpoint_channel(self, channel_id: int, message_id: int) -> None:
        """Advance and persist a channel checkpoint without moving it backwards."""
        previous = self._channel_checkpoints.get(channel_id)
        if previous is not None and previous >= message_id:
            return
        self._channel_checkpoints[channel_id] = message_id
        if len(self._channel_checkpoints) > CHANNEL_CHECKPOINT_LIMIT:
            oldest = sorted(
                self._channel_checkpoints.items(), key=lambda item: item[1]
            )[: len(self._channel_checkpoints) - CHANNEL_CHECKPOINT_LIMIT]
            for channel, _ in oldest:
                self._channel_checkpoints.pop(channel, None)
        self._persist_state()

    def claim_message(self, message_id: str | int) -> bool:
        """Claim a Discord event, suppressing duplicate gateway deliveries."""
        key = str(message_id)
        now = time.time()
        previous = self._message_ledger.get(key)
        if previous is not None:
            updated_at = previous.get("updated_at", 0)
            if (
                isinstance(updated_at, (int, float))
                and now - updated_at < MESSAGE_LEDGER_RETRY_AFTER
            ):
                return False
        self._message_ledger[key] = {"status": "processing", "updated_at": now}
        self._persist_state()
        return True

    def complete_message(self, message_id: str | int) -> None:
        """Mark a claimed Discord message as delivered successfully."""
        key = str(message_id)
        if key not in self._message_ledger:
            return
        self._message_ledger[key] = {"status": "completed", "updated_at": time.time()}
        self._persist_state()
