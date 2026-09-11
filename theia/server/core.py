"""Codex App Server transport, session state, and Discord-facing orchestration."""

import asyncio
import contextlib
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import discord

from ..core import (
    ADAPTIVE_REASONING_ENV,
    APPROVAL_LEVEL_ENV,
    APPROVAL_LEVELS,
    DEFAULT_CODEX_MODEL,
    DEFAULT_APPROVAL_LEVEL,
    DEFAULT_NIGHTLY_RECAP_TIMEOUT,
    DEFAULT_SELF_IMPROVEMENT,
    DEFAULT_SELF_IMPROVEMENT_TIMEOUT,
    CodexAppServerError,
    _configured_paths,
    _env_bool,
    _env_float,
    _error_message,
    _is_always_admin_user,
    _codex_logger,
    _PendingApproval,
    _Session,
    _skill_entries,
    _truncate,
    _TurnState,
    SELF_IMPROVEMENT_ENV,
    SELF_IMPROVEMENT_TIMEOUT_ENV,
    NIGHTLY_RECAP_TIMEOUT_ENV,
)
from ..personality import PersonalityStore
from ..audio import OpenAICompatibleAudio
from .policy import (
    CODEX_STDIO_LIMIT_ENV,
    CODEX_MEMORY_BREACH_SAMPLES_ENV,
    CODEX_MEMORY_CHECK_INTERVAL_ENV,
    CODEX_MEMORY_RESTART_GRACE_ENV,
    CODEX_MEMORY_WATCHDOG_ENV,
    CODEX_MAX_RSS_MB_ENV,
    DEFAULT_ATTACHMENT_CACHE_LIMIT_BYTES,
    DEFAULT_ATTACHMENT_CACHE_MAX_AGE,
    DEFAULT_CODEX_MEMORY_BREACH_SAMPLES,
    DEFAULT_CODEX_MEMORY_CHECK_INTERVAL,
    DEFAULT_CODEX_MEMORY_RESTART_GRACE,
    DEFAULT_CODEX_MEMORY_WATCHDOG,
    DEFAULT_CODEX_MAX_RSS_MB,
    DEFAULT_CODEX_STDIO_LIMIT,
    MAX_CODEX_STDIO_LIMIT,
    MIN_CODEX_STDIO_LIMIT,
    _CODEX_CHILD_SECRET_ENV_NAMES,
)
from .transport import CodexTransportMixin
from .state import CodexStateMixin
from .notifications import CodexNotificationMixin
from .personality_state import CodexPersonalityStateMixin
from .conversation import CodexConversationMixin
from .lifecycle import CodexLifecycleMixin
from .requests import CodexRequestMixin
from .realtime import CodexRealtimeMixin
from .self_improvement import CodexSelfImprovementMixin
from .workers import CodexWorkerMixin

logger = _codex_logger()


class CodexAppServer(  # pylint: disable=too-many-ancestors
    CodexStateMixin,
    CodexPersonalityStateMixin,
    CodexRealtimeMixin,
    CodexConversationMixin,
    CodexLifecycleMixin,
    CodexRequestMixin,
    CodexSelfImprovementMixin,
    CodexWorkerMixin,
    CodexTransportMixin,
    CodexNotificationMixin,
):
    """Own the local Codex process and map Discord sessions to Codex threads.

    The server keeps Discord-specific authorization, session metadata, memory
    roots, and personality selection beside the JSON-RPC transport. Persisted
    state is intentionally private to Theia so a Discord deployment does not
    alter a user's global Codex runtime.
    """

    @staticmethod
    def _build_codex_environment() -> dict[str, str]:
        """Build a child environment without Theia or provider credentials."""
        excluded = _CODEX_CHILD_SECRET_ENV_NAMES
        environment = {
            name: value
            for name, value in os.environ.items()
            if name.upper() not in excluded
        }
        return environment

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None  # pylint: disable=no-member
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._memory_watchdog_task: asyncio.Task[None] | None = None
        self._memory_recovery_task: asyncio.Task[None] | None = None
        self._memory_recovery_active = False
        self._memory_breach_count = 0
        self._server_tasks: set[asyncio.Task[Any]] = set()
        self._write_lock = asyncio.Lock()
        self._models_lock = asyncio.Lock()
        self._next_request_id = 1
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._turns: dict[str, _TurnState] = {}
        self._realtime_sessions: dict[str, Any] = {}
        self._realtime_feature_enabled = False
        self._realtime_model = os.getenv("THEIA_REALTIME_MODEL", "").strip()
        self._realtime_voice = os.getenv("THEIA_REALTIME_VOICE", "").strip()
        self._models: tuple[dict[str, Any], ...] = ()
        self._models_loaded_at = 0.0
        self._provider_capabilities: dict[str, Any] | None = None
        self._provider_capabilities_key: tuple[str | None, str | None] | None = None
        self._frontend_customizer: Any | None = None
        self._view_registrar: Callable[[Any, Any], Awaitable[None]] | None = None
        self._loaded_thread_ids: set[str] = set()
        self._model: str | None = DEFAULT_CODEX_MODEL
        self._login_id: str | None = None
        self._login_channel: discord.abc.Messageable | None = None
        self._login_user_id: int | None = None
        self._login_guild_id: int | None = None
        self._login_sender: Callable[..., Awaitable[Any]] | None = None
        self._stderr_tail: list[str] = []
        self._request_timeout = _env_float("CODEX_REQUEST_TIMEOUT", 60)
        self._turn_timeout = _env_float("CODEX_TURN_TIMEOUT", 1800)
        self._assessment_timeout = _env_float("CODEX_ASSESSMENT_TIMEOUT", 60)
        self._adaptive_reasoning = _env_bool(ADAPTIVE_REASONING_ENV, True)
        self._self_improvement_enabled = _env_bool(
            SELF_IMPROVEMENT_ENV, DEFAULT_SELF_IMPROVEMENT
        )
        self._self_improvement_timeout = max(
            5.0,
            _env_float(
                SELF_IMPROVEMENT_TIMEOUT_ENV,
                DEFAULT_SELF_IMPROVEMENT_TIMEOUT,
            ),
        )
        self._nightly_recap_timeout = max(
            5.0,
            _env_float(NIGHTLY_RECAP_TIMEOUT_ENV, DEFAULT_NIGHTLY_RECAP_TIMEOUT),
        )
        self._self_improvement_lock = asyncio.Lock()
        configured_approval_level = (
            os.getenv(APPROVAL_LEVEL_ENV, DEFAULT_APPROVAL_LEVEL).strip().casefold()
        )
        if configured_approval_level not in APPROVAL_LEVELS:
            logger.warning("Ignoring unsupported approval level; using high instead")
            configured_approval_level = DEFAULT_APPROVAL_LEVEL
        self._approval_level = configured_approval_level
        try:
            configured_stdio_limit = int(
                os.getenv(CODEX_STDIO_LIMIT_ENV, str(DEFAULT_CODEX_STDIO_LIMIT))
            )
        except ValueError:
            configured_stdio_limit = DEFAULT_CODEX_STDIO_LIMIT
        self._stdio_limit = max(
            MIN_CODEX_STDIO_LIMIT,
            min(MAX_CODEX_STDIO_LIMIT, configured_stdio_limit),
        )
        self._memory_watchdog_enabled = _env_bool(
            CODEX_MEMORY_WATCHDOG_ENV,
            DEFAULT_CODEX_MEMORY_WATCHDOG,
        )
        self._memory_watchdog_limit = (
            max(
                0.0,
                _env_float(CODEX_MAX_RSS_MB_ENV, DEFAULT_CODEX_MAX_RSS_MB),
            )
            * 1024
            * 1024
        )
        self._memory_watchdog_interval = max(
            5.0,
            _env_float(
                CODEX_MEMORY_CHECK_INTERVAL_ENV,
                DEFAULT_CODEX_MEMORY_CHECK_INTERVAL,
            ),
        )
        self._memory_restart_grace = max(
            1.0,
            _env_float(
                CODEX_MEMORY_RESTART_GRACE_ENV,
                DEFAULT_CODEX_MEMORY_RESTART_GRACE,
            ),
        )
        try:
            configured_breach_samples = int(
                os.getenv(
                    CODEX_MEMORY_BREACH_SAMPLES_ENV,
                    str(DEFAULT_CODEX_MEMORY_BREACH_SAMPLES),
                )
            )
        except ValueError:
            configured_breach_samples = DEFAULT_CODEX_MEMORY_BREACH_SAMPLES
        self._memory_breach_samples = max(1, configured_breach_samples)
        try:
            attachment_cache_limit = int(
                os.getenv(
                    "THEIA_ATTACHMENT_CACHE_LIMIT_BYTES",
                    str(DEFAULT_ATTACHMENT_CACHE_LIMIT_BYTES),
                )
            )
        except ValueError:
            attachment_cache_limit = DEFAULT_ATTACHMENT_CACHE_LIMIT_BYTES
        self._attachment_cache_limit = max(0, attachment_cache_limit)
        self._attachment_cache_max_age = max(
            3600.0,
            _env_float(
                "THEIA_ATTACHMENT_CACHE_MAX_AGE",
                DEFAULT_ATTACHMENT_CACHE_MAX_AGE,
            ),
        )
        self._cwd = str(Path(os.getenv("CODEX_CWD") or os.getcwd()).resolve())
        self._global_codex_home = (
            Path(os.getenv("CODEX_HOME") or (Path.home() / ".codex"))
            .expanduser()
            .resolve()
        )
        legacy_home = (
            Path(
                os.getenv("CODEX_DISCORD_HOME")
                or (Path.home() / ".codexdiscord" / "codex")
            )
            .expanduser()
            .resolve()
        )
        self._codex_home = (
            Path(
                os.getenv("THEIA_HOME")
                or os.getenv("CODEX_DISCORD_HOME")
                or (Path.home() / ".theia")
            )
            .expanduser()
            .resolve()
        )
        self._hermes_home = (
            Path(os.getenv("HERMES_HOME") or (Path.home() / ".hermes"))
            .expanduser()
            .resolve()
        )
        self._legacy_codex_home = (
            legacy_home if legacy_home != self._codex_home else None
        )
        self._codex_environment = self._build_codex_environment()
        self._codex_environment["CODEX_HOME"] = str(self._codex_home)
        self._personalities = PersonalityStore(self._codex_home)
        self._audio = OpenAICompatibleAudio.from_environment()
        self._hermes_memory_root = self._codex_home / "memories" / "hermes"
        self._attachment_root = self._codex_home / "attachments"
        self._generated_image_root = self._codex_home / "generated_images"
        self._memory_roots = _configured_paths(
            "CODEX_MEMORY_ROOTS",
            (
                self._codex_home / "memories",
                self._hermes_memory_root,
                self._global_codex_home / "memories",
                self._hermes_home / "memories",
            ),
        )
        self._skill_roots = _configured_paths(
            "CODEX_SKILL_ROOTS",
            (
                self._codex_home / "skills",
                self._global_codex_home / "skills",
                self._hermes_home / "skills",
                Path(self._cwd) / ".agents" / "skills",
                Path(self._cwd) / ".codex" / "skills",
            ),
        )
        workspace_skill_roots = tuple(
            path
            for path in self._skill_roots
            if self._codex_home == self._global_codex_home
            or path != self._global_codex_home / "skills"
        )
        self._shared_workspace_roots = tuple(
            dict.fromkeys(
                (
                    Path(self._cwd),
                    self._attachment_root,
                    *self._memory_roots,
                    *workspace_skill_roots,
                )
            )
        )
        # Codex stores native image-generation artifacts here. Keep this as a
        # delivery-only root rather than exposing the private directory to
        # ordinary tool authorization.
        self._image_artifact_roots = tuple(
            dict.fromkeys((*self._shared_workspace_roots, self._generated_image_root))
        )
        self._safe_workspace_roots = tuple(
            dict.fromkeys(
                (
                    self._attachment_root,
                    *_configured_paths("THEIA_SAFE_WORKSPACE_ROOTS", ()),
                )
            )
        )
        self._state_path = Path(
            os.getenv("THEIA_STATE")
            or os.getenv("CODEX_DISCORD_STATE")
            or (self._codex_home / "sessions.json")
        ).expanduser()
        legacy_state = (
            Path(
                os.getenv("CODEX_DISCORD_STATE")
                or (Path.home() / ".codexdiscord" / "sessions.json")
            )
            .expanduser()
            .resolve()
        )
        self._legacy_state_path = (
            legacy_state if legacy_state != self._state_path.resolve() else None
        )
        self._sessions: dict[str, _Session] = {}
        self._personality_scopes: dict[str, dict[str, Any]] = {}
        self._session_aliases: dict[str, str] = {}
        self._authenticated_users: set[int] = set()
        self._authenticated_guilds: set[int] = set()
        self._pending_approvals: dict[str, _PendingApproval] = {}
        self._message_ledger: dict[str, dict[str, Any]] = {}
        self._discord_threads: set[int] = set()
        self._channel_checkpoints: dict[int, int] = {}
        self._usage_threads: dict[str, dict[str, int]] = {}
        self._usage_daily: dict[str, int] = {}
        self._usage_tracked_since: float | None = None
        self._usage_longest_running_turn_sec = 0.0
        self._state_dirty = False
        self._state_recovery_blocked = False
        self._state_needs_cleanup = False
        self._skills_cache: tuple[dict[str, Any], ...] = ()
        self._skills_loaded_at = 0.0
        self._skills_lock = asyncio.Lock()
        self._skills_refresh_task: asyncio.Task[Any] | None = None
        self._rate_limits: dict[str, Any] | None = None
        self.account: dict[str, Any] | None = None
        self.requires_openai_auth = True
        self._auth_imported = False
        self._migrate_legacy_state()
        self._load_state()
        if self._state_needs_cleanup:
            self._persist_state()
            self._state_needs_cleanup = False
        logger.debug(
            "Codex layer initialized (adaptive_reasoning=%s, approval_level=%s, "
            "self_improvement=%s, memory_roots=%d, skill_roots=%d, "
            "transcription=%s, tts=%s)",
            self._adaptive_reasoning,
            self._approval_level,
            self._self_improvement_enabled,
            len(self._memory_roots),
            len(self._skill_roots),
            self._audio.transcription.enabled,
            self._audio.tts.enabled,
        )

    @staticmethod
    def _approval_policy(allow_tools: bool) -> str:
        if not allow_tools:
            return "never"
        return os.getenv("CODEX_APPROVAL_POLICY", "on-request")

    @staticmethod
    def _sandbox(allow_tools: bool) -> str:
        if not allow_tools:
            return "read-only"
        return os.getenv("CODEX_SANDBOX", "workspace-write")

    async def _wait_for_turn(
        self,
        session_key: str,
        session: _Session,
        state: _TurnState,
        turn_id: str,
        *,
        timeout: float | None = None,
    ) -> str:
        started_at = time.monotonic()
        try:
            wait_timeout = self._turn_timeout if timeout is None else timeout
            if wait_timeout:
                await asyncio.wait_for(asyncio.shield(state.done), timeout=wait_timeout)
            else:
                await state.done
            completed = state.completed or {}
            if completed.get("status") != "completed":
                error = completed.get("error") or {}
                message = (
                    _error_message(error)
                    or str(completed.get("status") or "")
                    or "unknown error"
                )
                logger.warning(
                    "Codex turn failed (status=%s, error_type=%s, duration_ms=%.1f)",
                    completed.get("status"),
                    type(error).__name__,
                    (time.monotonic() - started_at) * 1000,
                )
                raise CodexAppServerError(f"Codex turn failed: {message}")
            fallback = ""
            if state.last_agent_message_id:
                fallback = str(
                    state.agent_messages.get(state.last_agent_message_id, {}).get(
                        "text", ""
                    )
                )
            result = (
                state.final_text
                or fallback
                or ("Codex completed the request without a text response.")
            )
            logger.info(
                "Codex turn completed (items=%d, duration_ms=%.1f)",
                len(state.items),
                (time.monotonic() - started_at) * 1000,
            )
            return result
        except asyncio.TimeoutError as exc:
            logger.warning(
                "Codex turn timed out; interrupting it (duration_ms=%.1f)",
                (time.monotonic() - started_at) * 1000,
            )
            with contextlib.suppress(CodexAppServerError):
                await self.interrupt(session_key)
            raise CodexAppServerError(
                "Codex turn timed out and was interrupted."
            ) from exc
        finally:
            if state.event_tasks:
                await asyncio.gather(*state.event_tasks, return_exceptions=True)
            self._record_usage_turn_duration(time.monotonic() - started_at, session)
            if session.thread_id:
                self._clear_pending_for_turn(session.thread_id, turn_id)
            session.turn_id = None
            self._turns.pop(turn_id, None)

    async def interrupt(self, session_key: str) -> bool:
        """Interrupt the active turn for a session, if one is running."""
        session = self._session(session_key)
        if not session.thread_id or not session.turn_id:
            logger.debug("Ignored Codex interrupt because no turn is active")
            return False
        self._clear_pending_for_turn(session.thread_id, session.turn_id)
        await self._request(
            "turn/interrupt",
            {"threadId": session.thread_id, "turnId": session.turn_id},
        )
        logger.info("Codex turn interrupt requested")
        return True

    async def undo(self, session_key: str) -> None:
        """Remove the most recent completed Codex turn for a session."""
        session = self._session(session_key)
        assert session.lock is not None
        async with session.lock:
            await self._prepare_session_for_activity(session)
            if not session.thread_id:
                raise CodexAppServerError(
                    "There is no previous Codex response to undo."
                )
            await self._ensure_thread(session)
            if not session.thread_id:
                raise CodexAppServerError(
                    "There is no previous Codex response to undo."
                )
            await self.rollback_thread(session.thread_id)
            session.last_activity_at = time.time()
            self._persist_state()
        logger.info("Rolled back the most recent Codex turn")

    def resolve_approval(
        self,
        user_id: int,
        approved: bool,
        channel: Any | None = None,
        *,
        current_user: Any | None = None,
    ) -> bool:
        """Resolve the newest matching approval for the requesting user and channel."""
        channel_id = getattr(channel, "id", None)
        if not self._has_current_server_admin_access(
            channel, user_id, current_user=current_user
        ):
            logger.info("Ignored approval response after administrator access changed")
            for key, pending in tuple(self._pending_approvals.items()):
                if (
                    pending.user_id == user_id
                    and pending.channel_id == channel_id
                    and not pending.future.done()
                ):
                    self._pending_approvals.pop(key, None)
                    pending.future.set_result(
                        self._approval_result(
                            pending.kind, pending.params, approved=False
                        )
                    )
            return False
        candidates = [
            pending
            for pending in self._pending_approvals.values()
            if pending.user_id == user_id
            and not pending.future.done()
            and pending.channel_id == channel_id
        ]
        if not candidates:
            logger.debug("Ignored approval response because no matching request exists")
            return False
        pending = candidates[-1]
        self._pending_approvals.pop(pending.key, None)
        if pending.kind == "permissions":
            result = {
                "permissions": pending.params.get("permissions") if approved else {},
                "scope": "turn",
            }
        else:
            result = {"decision": "accept" if approved else "decline"}
        pending.future.set_result(result)
        logger.info("Codex approval request resolved (approved=%s)", approved)
        return True

    @staticmethod
    def _has_current_server_admin_access(
        channel: Any | None,
        user_id: int | None,
        *,
        current_user: Any | None = None,
    ) -> bool:
        """Return whether the current guild member still has administrator access."""
        if user_id is None:
            return False
        if current_user is not None and getattr(current_user, "id", None) != user_id:
            return False
        if _is_always_admin_user(user_id):
            return True
        guild = getattr(channel, "guild", None)
        if guild is None:
            return False
        if current_user is not None:
            permissions = getattr(current_user, "guild_permissions", None)
            return bool(permissions and getattr(permissions, "administrator", False))
        get_member = getattr(guild, "get_member", None)
        if not callable(get_member):
            return False
        member = get_member(user_id)
        permissions = getattr(member, "guild_permissions", None)
        return bool(permissions and getattr(permissions, "administrator", False))

    @staticmethod
    def _has_turn_server_admin_access(
        channel: Any | None,
        user_id: int | None,
        request_user: Any | None,
    ) -> bool:
        """Check a turn against current guild data, with a cache-miss fallback."""
        guild = getattr(channel, "guild", None)
        get_member = getattr(guild, "get_member", None)
        if callable(get_member) and get_member(user_id) is not None:
            return CodexAppServer._has_current_server_admin_access(channel, user_id)
        return CodexAppServer._has_current_server_admin_access(
            channel, user_id, current_user=request_user
        )

    def _clear_pending_for_turn(
        self, thread_id: str, turn_id: str, *, approved: bool = False
    ) -> None:
        for key, pending in tuple(self._pending_approvals.items()):
            if pending.thread_id != thread_id or pending.turn_id != turn_id:
                continue
            self._pending_approvals.pop(key, None)
            if not pending.future.done():
                if pending.kind == "permissions":
                    result = {
                        "permissions": pending.params.get("permissions")
                        if approved
                        else {},
                        "scope": "turn",
                    }
                else:
                    result = {"decision": "accept" if approved else "decline"}
                pending.future.set_result(result)

    def _clear_all_pending(self) -> None:
        for pending in tuple(self._pending_approvals.values()):
            if not pending.future.done():
                pending.future.set_result(
                    self._approval_result(pending.kind, pending.params, approved=False)
                )
        self._pending_approvals.clear()

    async def steer(self, session_key: str, prompt: str) -> None:
        """Send a follow-up instruction to the active Codex turn."""
        session = self._session(session_key)
        if not session.thread_id or not session.turn_id:
            raise CodexAppServerError("There is no active Codex turn to steer.")
        await self._request(
            "turn/steer",
            {
                "threadId": session.thread_id,
                "expectedTurnId": session.turn_id,
                "input": [{"type": "text", "text": prompt}],
            },
        )

    async def new_session(self, session_key: str) -> None:
        """Discard the current Codex thread while retaining session preferences."""
        session = self._session(session_key)
        if session.turn_id:
            raise CodexAppServerError(
                "Stop the active turn before starting a new session."
            )
        session.thread_id = None
        session.loaded = False
        session.archived = False
        session.last_activity_at = None
        session.instruction_fingerprint = None
        session.tool_policy = None
        self._persist_state()

    async def resume_session(self, session_key: str, thread_id: str) -> None:
        """Resume a persisted Codex thread and bind it to a Discord session."""
        params: dict[str, Any] = {
            "threadId": thread_id,
            "runtimeWorkspaceRoots": [
                str(path) for path in self._shared_workspace_roots
            ],
        }
        if self._model is not None:
            params["model"] = self._model
        await self._request("thread/resume", params)
        session = self._session(session_key)
        session.thread_id = thread_id
        session.loaded = True
        self._set_thread_loaded(thread_id, True)
        self._persist_state()

    async def fork_session(self, session_key: str) -> str:
        """Fork the session's current Codex thread and return the new thread id."""
        session = self._session(session_key)
        await self._ensure_thread(session)
        params: dict[str, Any] = {"threadId": session.thread_id}
        if self._model is not None:
            params["model"] = self._model
        result = await self._request("thread/fork", params)
        thread_id = (result.get("thread") or {}).get("id")
        if not thread_id:
            raise CodexAppServerError("Codex did not return the forked thread id.")
        session.thread_id = thread_id
        session.loaded = True
        self._set_thread_loaded(thread_id, True)
        self._persist_state()
        return thread_id

    async def compact(self, session_key: str) -> None:
        """Request context compaction for the session's current Codex thread."""
        session = self._session(session_key)
        await self._ensure_thread(session)
        await self._request("thread/compact/start", {"threadId": session.thread_id})

    async def archive(self, session_key: str) -> None:
        """Archive the session's Codex thread and update local retention state."""
        session = self._session(session_key)
        await self._ensure_thread(session)
        thread_id = session.thread_id
        if thread_id is None:
            raise CodexAppServerError("The Codex session has no thread id.")
        try:
            await self._request("thread/archive", {"threadId": thread_id})
        except CodexAppServerError as exc:
            if "no rollout found" not in str(exc).casefold():
                raise
        else:
            self._set_thread_archived(thread_id, True)
            self._set_thread_loaded(thread_id, False)
            self._persist_state()

    async def read_thread(self, session_key: str) -> dict[str, Any]:
        """Read the session's thread without unnecessarily resuming it."""
        session = self._session(session_key)
        await self._ensure_thread(session)
        params: dict[str, Any] = {
            "threadId": session.thread_id,
            "includeTurns": False,
        }
        try:
            return await self._request("thread/read", params)
        except CodexAppServerError as exc:
            # Keep compatibility with older app-server builds that predate
            # thread/read, while preferring the non-resuming API above.
            message = str(exc).casefold()
            if (
                "unsupported" not in message
                and "method not found" not in message
                and "unknown method" not in message
            ):
                raise
            resume_params: dict[str, Any] = {
                "threadId": session.thread_id,
                "runtimeWorkspaceRoots": [
                    str(path) for path in self._shared_workspace_roots
                ],
            }
            if self._model is not None:
                resume_params["model"] = self._model
            return await self._request("thread/resume", resume_params)

    async def list_threads(self) -> dict[str, Any]:
        """List recent Codex threads visible to the configured runtime."""
        await self._ensure_running()
        return await self._request(
            "thread/list",
            {
                "limit": 20,
                "sortKey": "updated_at",
                "sortDirection": "desc",
                "sourceKinds": ["appServer", "cli", "vscode"],
            },
        )

    async def goal(
        self, session_key: str, action: str, objective: str | None = None
    ) -> dict[str, Any]:
        """Get, set, or clear the Codex goal associated with a session thread."""
        session = self._session(session_key)
        await self._ensure_thread(session)
        if action == "get":
            return await self._request(
                "thread/goal/get", {"threadId": session.thread_id}
            )
        if action == "clear":
            return await self._request(
                "thread/goal/clear", {"threadId": session.thread_id}
            )
        if not objective:
            raise CodexAppServerError("A goal objective is required for `set`.")
        return await self._request(
            "thread/goal/set",
            {"threadId": session.thread_id, "objective": objective, "status": "active"},
        )

    async def skills(self, *, force_reload: bool = False) -> dict[str, Any]:
        """Return the cached skill catalog, optionally forcing a protocol refresh."""
        await self._ensure_running()
        async with self._skills_lock:
            if (
                not force_reload
                and self._skills_cache
                and time.monotonic() - self._skills_loaded_at < 60
            ):
                return {"data": list(self._skills_cache)}
            result = await self._request(
                "skills/list", {"cwds": [self._cwd], "forceReload": force_reload}
            )
            entries = _skill_entries(result)
            self._skills_cache = tuple(entries)
            self._skills_loaded_at = time.monotonic()
            return result

    async def refresh_skills(self, *, force: bool = False) -> dict[str, Any]:
        """Refresh the skill catalog using the public convenience API."""
        return await self.skills(force_reload=force)

    def skill_names(self) -> tuple[tuple[str, str], ...]:
        """Return enabled skill names and their user-facing display labels."""
        values: list[tuple[str, str]] = []
        for skill in self._skills_cache:
            if skill.get("enabled") is False:
                continue
            name = str(skill.get("name") or "").strip()
            if not name:
                continue
            interface = skill.get("interface") or {}
            display = str(interface.get("displayName") or name)
            values.append((name, display))
        return tuple(values)

    async def _refresh_skills_after_change(self) -> None:
        with contextlib.suppress(CodexAppServerError):
            await self.refresh_skills(force=True)

    async def apps(self, session_key: str) -> dict[str, Any]:
        """List apps available to the session's current Codex thread."""
        session = self._session(session_key)
        await self._ensure_thread(session)
        return await self._request(
            "app/list",
            {"threadId": session.thread_id, "limit": 50, "forceRefetch": False},
        )

    async def review(
        self,
        session_key: str,
        instructions: str | None = None,
        *,
        channel: discord.abc.Messageable | None = None,
        user_id: int | None = None,
        on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Run a Codex review against the session's current workspace or instructions."""
        session = self._session(session_key)
        assert session.lock is not None
        async with session.lock:
            return await self._review_locked(
                session_key,
                session,
                instructions,
                channel=channel,
                user_id=user_id,
                on_event=on_event,
            )

    async def _review_locked(
        self,
        session_key: str,
        session: _Session,
        instructions: str | None,
        *,
        channel: discord.abc.Messageable | None,
        user_id: int | None,
        on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None,
    ) -> dict[str, Any]:
        await self._ensure_thread(session)
        target = (
            {"type": "custom", "instructions": instructions}
            if instructions
            else {"type": "uncommittedChanges"}
        )
        result = await self._request(
            "review/start",
            {"threadId": session.thread_id, "target": target, "delivery": "inline"},
        )
        turn = result.get("turn") or {}
        turn_id = turn.get("id")
        if not turn_id:
            return result
        state = self._turns.setdefault(
            str(turn_id),
            _TurnState(
                thread_id=session.thread_id,
                session=session,
                channel=channel,
                user_id=user_id,
                on_event=on_event,
            ),
        )
        state.thread_id = session.thread_id
        state.session = session
        state.channel = channel
        state.user_id = user_id
        state.on_event = on_event
        session.turn_id = str(turn_id)
        result = dict(result)
        result["text"] = await self._wait_for_turn(
            session_key, session, state, str(turn_id)
        )
        return result

    def status(self, session_key: str) -> dict[str, Any]:
        """Return the compact runtime status used by Discord command responses."""
        session = self._session(session_key)
        return {
            "thread_id": session.thread_id,
            "turn_id": session.turn_id,
            "model": self._model or DEFAULT_CODEX_MODEL,
            "logged_in": self.account is not None or not self.requires_openai_auth,
        }

    def debug_state(self, session_key: str) -> dict[str, Any]:
        """Return sanitized, read-only runtime diagnostics for an administrator."""
        session = self._session(session_key)
        mood = self.mood_state(session_key)
        internal_workers: dict[str, int] = {}
        active_turns = 0
        for state in self._turns.values():
            state_session = state.session
            if state_session is None:
                continue
            if state_session.key.startswith("__"):
                worker_name = state_session.key.removeprefix("__").split(":", 1)[0]
                internal_workers[worker_name] = internal_workers.get(worker_name, 0) + 1
            else:
                active_turns += 1
        process = self._process
        process_running = (
            process is not None
            and process.returncode is None
            and self._reader_task is not None
            and not self._reader_task.done()
        )
        usage = self.theia_usage().get("summary", {})
        return {
            "runtime": {
                "process": "running" if process_running else "stopped",
                "exit_code": process.returncode if process is not None else None,
                "authenticated": self.account is not None
                or not self.requires_openai_auth,
                "state_dirty": self._state_dirty,
                "state_recovery_blocked": self._state_recovery_blocked,
            },
            "configuration": {
                "model": self._model or DEFAULT_CODEX_MODEL,
                "approval_level": self._approval_level,
                "adaptive_reasoning": self._adaptive_reasoning,
                "self_improvement": self._self_improvement_enabled,
            },
            "session": {
                "mode": session.mode,
                "personality": self.active_personality(session_key) or "None",
                "thread_id": _truncate(session.thread_id, 80)
                if session.thread_id
                else None,
                "turn_id": _truncate(session.turn_id, 80) if session.turn_id else None,
                "loaded": session.loaded,
                "archived": session.archived,
                "mood": mood,
            },
            "counts": {
                "sessions": sum(
                    not item.key.startswith("__") for item in self._sessions.values()
                ),
                "loaded_threads": len(self._loaded_thread_ids),
                "active_turns": active_turns,
                "internal_workers": internal_workers,
                "pending_protocol_requests": len(self._pending),
                "pending_approvals": len(self._pending_approvals),
                "background_tasks": sum(not task.done() for task in self._server_tasks),
            },
            "usage": {
                "cumulative_tokens": usage.get("totalCumulativeTokens", 0),
                "longest_turn_seconds": usage.get("longestRunningTurnSec", 0.0),
            },
        }
