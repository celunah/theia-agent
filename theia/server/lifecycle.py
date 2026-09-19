"""Process lifecycle, authentication, and Codex account operations."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import subprocess
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

from ..core import (
    AGENT_DISPLAY_NAME,
    AGENT_NAME,
    THEIA_VERSION,
    CodexAppServerError,
    CodexTransientRestartError,
    _codex_logger,
    _TurnState,
)
from ..identifiers import new_unique_token
from .codex_update import CodexUpdateResult
from .policy import MAX_CODEX_MEMORY_RESTART_BACKOFF
from .secure_credentials import SecureCredentialLifecycleMixin

logger = _codex_logger()

_CODEX_DATABASE_SUFFIXES = (
    ".sqlite",
    ".sqlite-shm",
    ".sqlite-wal",
    ".sqlite-journal",
)
_SQLITE_STARTUP_ERROR_MARKERS = (
    "failed to initialize sqlite state runtime",
    "failed to initialize state runtime",
    "database is locked",
    "database is busy",
    "sqlite error",
)


class CodexLifecycleMixin(SecureCredentialLifecycleMixin):
    """Control App Server startup, health checks, recovery, and shutdown."""

    if TYPE_CHECKING:
        _model: str | None
        _approval_level: str
        _adaptive_reasoning: bool
        _self_improvement_enabled: bool
        _provider_capabilities: dict[str, Any] | None
        _provider_capabilities_key: tuple[str | None, str | None] | None
        _login_user_id: int | None
        _lifecycle_lock: asyncio.Lock
        _memory_watchdog_enabled: bool
        _memory_watchdog_limit: float
        _memory_watchdog_interval: float
        _memory_restart_grace: float
        _memory_breach_samples: int
        _memory_restart_backoff: float
        _memory_restart_backoff_until: float
        _memory_restart_streak: int
        _memory_watchdog_task: asyncio.Task[None] | None
        _memory_recovery_task: asyncio.Task[None] | None
        _memory_recovery_active: bool
        _memory_breach_count: int
        _codex_updater: Any
        _codex_update_skip_once: bool
        _heartbeat_last_success_at: float | None
        _heartbeat_last_attempt_at: float | None
        _heartbeat_latency_ms: float | None
        _stderr_tail: list[str]
        _codex_home: Path

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    async def start(self) -> None:
        """Launch and initialize Codex while serializing lifecycle changes."""
        async with self._lifecycle_lock:
            if getattr(self, "_startup_blocked", False):
                reason = (
                    getattr(self, "_startup_reason", None) or "Theia could not start."
                )
                self._record_runtime_event("codex_start_failed")
                raise CodexAppServerError(f"FATAL: {reason}")
            try:
                await self._start_locked()
            except Exception:
                self._record_runtime_event("codex_start_failed")
                self.record_startup_failure(
                    "Theia could not start the Codex App Server."
                )
                raise

    async def _start_locked(self, *, allow_database_repair: bool = True) -> None:
        """Launch and initialize the project-local Codex App Server process."""
        if getattr(self, "_secure_credentials_enabled", False) and not getattr(
            self, "_credentials_ready", False
        ):
            raise CodexAppServerError("FATAL: The credential vault is locked.")
        if self._process is not None and self._process.returncode is None:
            if self._reader_task is not None and not self._reader_task.done():
                logger.debug("Codex App Server is already running")
                return
            await self._close_locked()

        logger.info("Starting Codex App Server")
        self._record_runtime_event("codex_starting")

        for session in self._sessions.values():
            session.loaded = False
            session.turn_id = None
        self._loaded_thread_ids.clear()

        try:
            self._codex_home.mkdir(parents=True, exist_ok=True)
            self._codex_home.chmod(0o700)
        except OSError as exc:
            raise CodexAppServerError(
                "The private Codex home could not be initialized."
            ) from exc
        self._auth_imported = False
        self._migrate_legacy_home()
        self._ensure_web_search_config()
        self._auth_imported = self._import_global_auth()
        update_result = await self._maybe_update_codex()
        executable = self._codex_executable()
        if executable is None:
            raise CodexAppServerError(
                "The Codex CLI is not installed. Run the project bootstrap command "
                "or install Codex CLI on PATH."
            )

        self._codex_version = self.codex_cli_version()
        self._stderr_tail = []

        try:
            self._process = await asyncio.create_subprocess_exec(
                executable,
                "app-server",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._codex_environment,
                limit=self._stdio_limit,
            )
        except OSError as exc:
            logger.error(
                "Codex App Server process could not be launched (error=%s)",
                type(exc).__name__,
            )
            if await self._retry_after_codex_update(update_result):
                return
            raise CodexAppServerError(
                "The Codex App Server could not be started."
            ) from None
        self._reader_task = asyncio.create_task(self._read_output())
        self._stderr_task = asyncio.create_task(self._read_stderr())

        try:
            await self._request(
                "initialize",
                {
                    "clientInfo": {
                        "name": AGENT_NAME.casefold(),
                        "title": AGENT_DISPLAY_NAME,
                        "version": THEIA_VERSION,
                    },
                    "capabilities": {"experimentalApi": True},
                },
            )
            await self._send({"method": "initialized", "params": {}})
            await self._refresh_realtime_capability()
            await self._configure_shared_roots()
            await self.refresh_account()
            if (
                self.account is None
                and self.requires_openai_auth
                and not self._auth_imported
                and self._import_global_auth(force=True)
            ):
                self._auth_imported = True
                await self.refresh_account()
            await self.refresh_skills(force=True)
            try:
                await self.loaded_threads()
            except CodexAppServerError as exc:
                logger.debug(
                    "Codex loaded-thread discovery is unavailable (error=%s)",
                    type(exc).__name__,
                )
            if not self._memory_recovery_active:
                self._start_memory_watchdog()
            self._mark_startup_ready()
            logger.info("Codex App Server is ready")
            self._record_runtime_event("codex_connected")
        except BaseException as exc:
            logger.error(
                "Codex App Server failed during startup (error=%s)",
                type(exc).__name__,
            )
            self._record_runtime_event("codex_start_failed")
            await self._close_locked()
            if allow_database_repair and self._has_sqlite_startup_failure():
                backup = self._backup_codex_databases()
                if backup is not None:
                    logger.warning(
                        "Codex App Server SQLite startup failed; backed up "
                        "%d database files and retrying",
                        len(tuple(backup.iterdir())),
                    )
                    await self._start_locked(allow_database_repair=False)
                    return
            if await self._retry_after_codex_update(update_result):
                return
            raise

    def _has_sqlite_startup_failure(self) -> bool:
        """Return whether the child reported a repairable SQLite startup error."""
        return any(
            marker in line.casefold()
            for line in self._stderr_tail
            for marker in _SQLITE_STARTUP_ERROR_MARKERS
        )

    def _backup_codex_databases(self) -> Path | None:
        """Move Codex SQLite artifacts into a private, reversible backup."""
        try:
            database_files = tuple(
                sorted(
                    path
                    for path in self._codex_home.iterdir()
                    if path.is_file() and path.name.endswith(_CODEX_DATABASE_SUFFIXES)
                )
            )
        except OSError as exc:
            logger.error(
                "Could not inspect Codex database files for repair (error=%s)",
                type(exc).__name__,
            )
            return None
        if not database_files:
            return None

        backup = self._codex_home / f"codex-database-repair-{new_unique_token()}"
        moved: list[tuple[Path, Path]] = []
        try:
            backup.mkdir(mode=0o700)
            for source in database_files:
                target = backup / source.name
                source.replace(target)
                moved.append((source, target))
        except OSError as exc:
            for source, target in reversed(moved):
                with contextlib.suppress(OSError):
                    target.replace(source)
            with contextlib.suppress(OSError):
                backup.rmdir()
            logger.error(
                "Could not back up Codex database files for repair (error=%s)",
                type(exc).__name__,
            )
            return None
        return backup

    async def _maybe_update_codex(self) -> CodexUpdateResult:
        """Stage a configured Codex update before launching the child process."""
        if self._codex_update_skip_once:
            self._codex_update_skip_once = False
            return self._codex_updater_result("skipped")
        if os.getenv("THEIA_CODEX_CLI", "").strip():
            logger.debug("Codex CLI auto-update skipped for an explicit executable")
            return self._codex_updater_result("skipped")
        result = await self._codex_updater.maybe_update()
        if result.status == "updated":
            logger.info("Codex CLI updated (version=%s)", result.version or "unknown")
        elif result.status == "failed":
            logger.warning("Codex CLI auto-update failed; using the current CLI")
        return result

    async def _retry_after_codex_update(self, result: CodexUpdateResult) -> bool:
        """Retry startup with the previous CLI when a fresh candidate fails."""
        if not result.activated or not self._codex_updater.rollback(result.install_dir):
            return False
        self._codex_update_skip_once = True
        logger.warning(
            "Codex App Server startup failed after an update; retrying the previous CLI"
        )
        await self._start_locked()
        return True

    def _codex_updater_result(self, status: str) -> CodexUpdateResult:
        """Create a no-op result for a skipped update attempt."""
        return CodexUpdateResult(status)

    def _codex_executable(self) -> str | None:
        """Find Theia's bundled CLI before accepting a system installation."""
        configured = os.getenv("THEIA_CODEX_CLI")
        if configured:
            configured_path = Path(configured).expanduser()
            if configured_path.is_file():
                logger.debug("Using explicitly configured Codex CLI")
                return str(configured_path.resolve())
            configured_executable = shutil.which(configured)
            if configured_executable is not None:
                logger.debug("Using explicitly configured Codex CLI from PATH")
                return configured_executable
            logger.warning("Configured Codex CLI is unavailable; continuing search")

        managed = self._codex_updater.active_executable()
        if managed is not None:
            logger.debug("Using Theia-managed Codex CLI")
            return managed

        project_root = Path(__file__).resolve().parent.parent
        roots = tuple(dict.fromkeys((project_root, Path(self._cwd))))
        for root in roots:
            local_bin = root / "node_modules" / ".bin"
            for name in ("codex.exe", "codex.cmd", "codex"):
                candidate = local_bin / name
                if candidate.is_file():
                    logger.debug("Using project-local Codex CLI")
                    return str(candidate)

        executable = (
            shutil.which("codex.exe")
            or shutil.which("codex.cmd")
            or shutil.which("codex")
        )
        if executable is not None:
            logger.debug("Using Codex CLI from PATH")
        return executable

    def codex_cli_version(self) -> str | None:
        """Read the version of the Codex CLI selected for this runtime."""
        executable = self._codex_executable()
        if executable is None:
            return None
        try:
            result = subprocess.run(
                [executable, "--version"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                env=self._codex_environment,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        output = f"{result.stdout}\n{result.stderr}"
        match = re.search(r"(?<![0-9])([0-9]+\.[0-9]+\.[0-9]+)(?![0-9])", output)
        return match.group(1) if match else None

    def _import_global_auth(self, *, force: bool = False) -> bool:
        """Bootstrap the private home from existing global Codex auth once."""
        if getattr(self, "_secure_credentials_enabled", False):
            return False
        if self._codex_home == self._global_codex_home:
            return False
        source = self._global_codex_home / "auth.json"
        target = self._codex_home / "auth.json"
        try:
            if target.is_symlink() or (target.exists() and not force):
                return False
            if not source.is_file():
                return False
        except OSError:
            return False

        temporary = target.with_name(f".{target.name}.tmp")
        try:
            shutil.copyfile(source, temporary)
            temporary.chmod(0o600)
            temporary.replace(target)
            logger.info("Copied existing Codex login into the private runtime")
            return True
        except OSError:
            with contextlib.suppress(OSError):
                temporary.unlink()
            return False

    async def _ensure_running(self) -> None:
        self.touch_secure_credentials()
        while self._memory_recovery_active:
            recovery = self._memory_recovery_task
            if recovery is not None and recovery is not asyncio.current_task():
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(recovery)
            else:
                await asyncio.sleep(0.05)
        process = self._process
        if (
            process is None
            or process.returncode is not None
            or (self._reader_task is not None and self._reader_task.done())
        ):
            await self.start()

    async def close(self) -> None:
        """Stop Codex and cancel any memory recovery in progress."""
        current = asyncio.current_task()
        for task in (
            self._memory_watchdog_task,
            self._memory_recovery_task,
            getattr(self, "_credential_watchdog_task", None),
        ):
            if task is None or task is current or task.done():
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        async with self._lifecycle_lock:
            await self._close_locked()
            if getattr(self, "_secure_credentials_enabled", False):
                self._credential_watchdog_task = None
                self._credential_vault.lock(reason="Vault closed")
                self._credential_environment.clear()
                self._credentials_ready = False
                self._clear_provider_credentials()
                if self._vault_auth_written:
                    with contextlib.suppress(OSError):
                        (self._codex_home / "auth.json").unlink()
                    self._vault_auth_written = False

    async def _close_locked(self) -> None:
        """Stop the Codex process and resolve pending interaction state safely."""
        was_running = self._process is not None
        watchdog = self._memory_watchdog_task
        if watchdog is not None and watchdog is not asyncio.current_task():
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog
        self._memory_watchdog_task = None
        self._memory_breach_count = 0
        if self._skills_refresh_task is not None:
            self._skills_refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._skills_refresh_task
            self._skills_refresh_task = None
        self._clear_all_pending()
        await self._close_realtime_sessions()
        server_tasks = tuple(
            task
            for task in self._server_tasks
            if task is not asyncio.current_task() and not task.done()
        )
        for task in server_tasks:
            task.cancel()
        if server_tasks:
            await asyncio.gather(*server_tasks, return_exceptions=True)
        self._server_tasks.clear()
        process = self._process
        reader_task = self._reader_task
        stderr_task = self._stderr_task
        self._process = None
        self._reader_task = None
        self._stderr_task = None

        if process is not None:
            if process.stdin is not None and not process.stdin.is_closing():
                process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(process.wait(), timeout=5)

        for task in (reader_task, stderr_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if was_running:
            logger.info("Codex App Server stopped")
            self._record_runtime_event("codex_stopped")

    async def heartbeat(self, *, timeout: float = 1.5) -> dict[str, Any]:
        """Check the live App Server transport without starting a model turn."""
        attempted_at = time.time()
        self._heartbeat_last_attempt_at = attempted_at
        started_at = time.monotonic()
        process = self._process
        reader = self._reader_task
        if (
            process is None
            or process.returncode is not None
            or reader is None
            or reader.done()
        ):
            self._heartbeat_consecutive_failures = min(
                1000, self._heartbeat_consecutive_failures + 1
            )
            self._heartbeat_latency_ms = None
            return self.heartbeat_snapshot()
        try:
            # account/read is an App Server transport probe. Its response is
            # deliberately discarded so the heartbeat cannot alter account,
            # session, thread, usage, or prompt state.
            await self._request(
                "account/read",
                {"refreshToken": False},
                timeout=max(0.1, min(5.0, timeout)),
            )
        except Exception as exc:  # noqa: BLE001 - heartbeat is best effort
            self._heartbeat_consecutive_failures = min(
                1000, self._heartbeat_consecutive_failures + 1
            )
            self._heartbeat_latency_ms = None
            logger.debug("Codex heartbeat failed (error=%s)", type(exc).__name__)
            return self.heartbeat_snapshot()
        self._heartbeat_last_success_at = time.time()
        self._heartbeat_latency_ms = max(
            0.0, min(60_000.0, (time.monotonic() - started_at) * 1000.0)
        )
        self._heartbeat_consecutive_failures = 0
        return self.heartbeat_snapshot()

    def _start_memory_watchdog(self) -> None:
        """Start RSS monitoring for the complete Codex child process tree."""
        if not self._memory_watchdog_enabled or self._memory_watchdog_limit <= 0:
            return
        task = self._memory_watchdog_task
        if task is None or task.done():
            self._memory_breach_count = 0
            self._memory_watchdog_task = asyncio.create_task(
                self._memory_watchdog_loop()
            )

    def _codex_process_rss(self) -> int | None:
        """Return RSS for the launcher and every Codex descendant."""
        process = self._process
        if process is None or process.returncode is not None:
            return None
        try:
            root = psutil.Process(process.pid)
            processes = (root, *root.children(recursive=True))
        except psutil.Error:
            return None
        total = 0
        for candidate in processes:
            try:
                total += candidate.memory_info().rss
            except psutil.Error:
                continue
        return total

    async def _memory_watchdog_loop(self) -> None:
        """Restart Codex after a sustained process-tree RSS limit breach."""
        try:
            while True:
                await asyncio.sleep(self._memory_watchdog_interval)
                rss = self._codex_process_rss()
                if rss is None or rss < self._memory_watchdog_limit:
                    self._memory_breach_count = 0
                    self._memory_restart_streak = 0
                    self._memory_restart_backoff_until = 0.0
                    continue
                if time.monotonic() < self._memory_restart_backoff_until:
                    self._memory_breach_count = 0
                    continue
                self._memory_breach_count += 1
                if self._memory_breach_count < self._memory_breach_samples:
                    continue
                if self._memory_recovery_task is None:
                    self._memory_recovery_task = asyncio.create_task(
                        self._recover_memory_pressure(rss)
                    )
                    self._memory_recovery_task.add_done_callback(
                        self._memory_recovery_done
                    )
                return
        except Exception as exc:  # noqa: BLE001 - monitoring must not kill Theia
            logger.warning(
                "Codex memory watchdog stopped (error=%s)", type(exc).__name__
            )

    async def _recover_memory_pressure(self, rss: int) -> None:
        """Interrupt current turns, then replace an over-sized Codex process."""
        self._memory_recovery_active = True
        started = False
        backoff = min(
            self._memory_restart_backoff * (2 ** min(self._memory_restart_streak, 8)),
            MAX_CODEX_MEMORY_RESTART_BACKOFF,
        )
        try:
            async with self._lifecycle_lock:
                if self._process is None or self._process.returncode is not None:
                    return
                logger.warning(
                    "Codex RSS exceeded the watchdog limit; restarting "
                    "(rss_mb=%.1f, limit_mb=%.1f)",
                    rss / (1024 * 1024),
                    self._memory_watchdog_limit / (1024 * 1024),
                )
                await self._interrupt_active_turns()
                await self._close_locked()
                await self._start_locked()
                started = True
                self._memory_restart_streak += 1
                self._memory_restart_backoff_until = time.monotonic() + backoff
                logger.info("Codex App Server recovered after memory pressure")
                self._record_runtime_event("codex_recovered")
        except Exception as exc:  # noqa: BLE001 - recovery is best effort
            logger.error(
                "Codex App Server memory recovery failed (error=%s)",
                type(exc).__name__,
            )
            self.record_startup_failure(
                "Theia could not restart the Codex App Server after memory pressure."
            )
        finally:
            self._memory_recovery_active = False
            if started or (
                self._process is not None and self._process.returncode is None
            ):
                self._memory_recovery_task = None
                self._start_memory_watchdog()

    def _memory_recovery_done(self, task: asyncio.Task[None]) -> None:
        """Clear the recovery handle and consume unexpected task failures."""
        if self._memory_recovery_task is task:
            self._memory_recovery_task = None
        if task.cancelled():
            return
        try:
            error = task.exception()
        except asyncio.InvalidStateError:
            return
        if error is not None:
            logger.error(
                "Codex App Server memory recovery task failed (error=%s)",
                type(error).__name__,
            )

    async def _interrupt_active_turns(self) -> None:
        """Ask every active turn to stop, bounded by the recovery grace period."""
        active: list[tuple[_TurnState, str, str]] = []
        for state in tuple(self._turns.values()):
            session = state.session
            thread_id = state.thread_id or (session.thread_id if session else None)
            turn_id = session.turn_id if session else None
            if state.done.done() or not thread_id or not turn_id:
                continue
            self._clear_pending_for_turn(thread_id, turn_id)
            active.append((state, thread_id, turn_id))
        if not active:
            return

        request_timeout = max(0.5, min(5.0, self._memory_restart_grace))
        requests = [
            self._request(
                "turn/interrupt",
                {"threadId": thread_id, "turnId": turn_id},
                timeout=request_timeout,
            )
            for _, thread_id, turn_id in active
        ]
        results = await asyncio.gather(*requests, return_exceptions=True)
        failures = sum(isinstance(result, BaseException) for result in results)
        if failures:
            logger.warning(
                "Some Codex turns could not be interrupted before memory recovery "
                "(failed=%d, active=%d)",
                failures,
                len(active),
            )
        deadline = time.monotonic() + self._memory_restart_grace
        while time.monotonic() < deadline and any(
            not state.done.done() for state, _, _ in active
        ):
            await asyncio.sleep(0.1)
        failure = CodexTransientRestartError(
            "Codex restarted because its memory usage exceeded the configured limit."
        )
        for state, _, _ in active:
            if not state.done.done():
                state.done.set_exception(failure)

    async def refresh_account(self) -> dict[str, Any]:
        """Refresh and cache the Codex account authentication state."""
        result = await self._request("account/read", {"refreshToken": False})
        self.account = result.get("account")
        self.requires_openai_auth = bool(result.get("requiresOpenaiAuth", False))
        self._persist_vault_auth()
        logger.debug(
            "Codex account state refreshed (authenticated=%s, auth_required=%s)",
            self.account is not None,
            self.requires_openai_auth,
        )
        return result

    async def account_details(self) -> dict[str, Any]:
        """Fetch the current Codex account details for frontend status views."""
        await self._ensure_running()
        return await self.refresh_account()

    def is_authenticated(self, user_id: int, guild_id: int | None = None) -> bool:
        """Return whether a user or their server has completed Theia login."""
        return user_id in self._authenticated_users or (
            guild_id is not None and guild_id in self._authenticated_guilds
        )

    def mark_authenticated(self, user_id: int, *, guild_id: int | None = None) -> None:
        """Persist user access and optionally grant access across one server."""
        changed = user_id not in self._authenticated_users
        self._authenticated_users.add(user_id)
        if guild_id is not None:
            changed = guild_id not in self._authenticated_guilds or changed
            self._authenticated_guilds.add(guild_id)
        if changed:
            self._persist_state()

    def mark_server_authenticated(self, guild_id: int) -> None:
        """Persist server-wide authorization for subsequent Discord requests."""
        if guild_id in self._authenticated_guilds:
            return
        self._authenticated_guilds.add(guild_id)
        self._persist_state()

    def clear_authenticated_users(self) -> None:
        """Clear all persisted user and server login grants."""
        if self._authenticated_users or self._authenticated_guilds:
            self._authenticated_users.clear()
            self._authenticated_guilds.clear()
            self._persist_state()

    async def _configure_shared_roots(self) -> None:
        roots = [str(path) for path in self._skill_roots]
        if not roots:
            logger.debug("No additional Codex skill roots configured")
            return
        try:
            await self._request("skills/extraRoots/set", {"extraRoots": roots})
        except CodexAppServerError as exc:
            # Older app-server versions discover their configured roots without
            # this optional request. The normal skills/list call still works.
            logger.debug(
                "Codex extra skill roots request unsupported or unavailable (error=%s)",
                type(exc).__name__,
            )
            return
        logger.debug("Configured %d additional Codex skill roots", len(roots))

    async def available_models(
        self, *, force: bool = False
    ) -> tuple[dict[str, Any], ...]:
        """Return the cached model catalog, refreshing it when requested or stale."""
        await self._ensure_running()
        if (
            not force
            and self._models
            and time.monotonic() - self._models_loaded_at < 60
        ):
            logger.debug("Using cached Codex model capabilities")
            return self._models
        async with self._models_lock:
            if (
                not force
                and self._models
                and time.monotonic() - self._models_loaded_at < 60
            ):
                logger.debug("Using cached Codex model capabilities")
                return self._models
            result = await self._request(
                "model/list",
                {"limit": 100, "includeHidden": False},
            )
            data = result.get("data", [])
            self._models = tuple(
                model for model in data if isinstance(model, dict) and model.get("id")
            )
            self._models_loaded_at = time.monotonic()
            logger.debug("Loaded %d Codex model capabilities", len(self._models))
            return self._models

    async def set_model(self, model: str) -> None:
        """Validate and persist the model used for new Codex turns."""
        models = await self.available_models(force=True)
        if not any(item.get("id") == model for item in models):
            raise CodexAppServerError(
                f"Model `{model}` is not available for this account."
            )
        changed = self._model != model
        self._model = model
        self._persist_state()
        logger.info("Codex model selection updated (changed=%s)", changed)
        self._record_runtime_event("model_changed")

    def model_name(self) -> str | None:
        """Return the configured Codex model."""
        return self._model

    async def begin_login(
        self,
        channel: Any,
        user_id: int,
        *,
        guild_id: int | None = None,
        grant_server: bool = False,
        on_complete_send: Callable[..., Awaitable[Any]] | None = None,
    ) -> dict[str, Any]:
        """Start or reuse device-code login and record the requested access scope."""
        await self._ensure_running()
        await self.refresh_account()
        if self.account is not None or not self.requires_openai_auth:
            imported = self._auth_imported
            self._auth_imported = False
            self.mark_authenticated(
                user_id,
                guild_id=guild_id if grant_server else None,
            )
            logger.info(
                "Codex login reused (server_access_granted=%s)",
                grant_server and guild_id is not None,
            )
            return {"login_imported": True} if imported else {"login_cached": True}
        if self._login_id is not None:
            logger.info("Codex login is already in progress")
            return {"login_in_progress": True}
        self._auth_imported = False
        result = await self._request(
            "account/login/start",
            {"type": "chatgptDeviceCode"},
        )
        self._login_id = result.get("loginId")
        self._login_channel = channel
        self._login_user_id = user_id
        self._login_guild_id = guild_id if grant_server else None
        self._login_sender = on_complete_send
        logger.info("Codex login flow started")
        return result

    async def usage(self, *, date_value: str | None = None) -> dict[str, Any]:
        """Return token activity recorded from Theia-owned conversation threads."""
        return self.theia_usage(date_value=date_value)

    async def credits(self) -> dict[str, Any]:
        """Return account rate limits, or an empty result when login is required."""
        await self._ensure_running()
        await self.refresh_account()
        if self.account is None and self.requires_openai_auth:
            logger.info("Codex credits requested without an authenticated account")
            return {}
        logger.debug("Reading Codex account rate limits")
        result = await self._request("account/rateLimits/read", None)
        rate_limits = result.get("rateLimits")
        self._rate_limits = rate_limits if isinstance(rate_limits, dict) else None
        return result

    async def provider_capabilities(
        self,
        *,
        model: str | None = None,
        model_provider: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Read the Codex provider capability snapshot.

        Current app-server versions return process/account-level capability
        flags such as ``namespaceTools``, ``imageGeneration``, and
        ``webSearch``.  The optional model fields are sent for versions that
        use model/provider-specific capability resolution.
        """
        await self._ensure_running()
        cache_key = (model or self._model, model_provider)
        if (
            not force
            and self._provider_capabilities is not None
            and self._provider_capabilities_key == cache_key
        ):
            return dict(self._provider_capabilities)

        params: dict[str, Any] = {}
        if model:
            params["model"] = model
        if model_provider:
            params["modelProvider"] = model_provider
        result = await self._request("modelProvider/capabilities/read", params)
        self._provider_capabilities = dict(result)
        self._provider_capabilities_key = cache_key
        logger.debug(
            "Loaded Codex provider capabilities (capabilities=%d)",
            len(result),
        )
        return result

    async def list_thread_turns(
        self,
        thread_id: str,
        *,
        cursor: str | None = None,
        limit: int = 50,
        sort_direction: str = "desc",
        items_view: str = "summary",
    ) -> dict[str, Any]:
        """Page through persisted turns without resuming the thread."""
        await self._ensure_running()
        thread_id = self._validated_thread_id(thread_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise CodexAppServerError("The thread history page size must be positive.")
        if sort_direction not in {"asc", "desc"}:
            raise CodexAppServerError("Thread history sort direction is invalid.")
        if items_view not in {"notLoaded", "summary", "full"}:
            raise CodexAppServerError("Thread history item view is invalid.")
        params: dict[str, Any] = {
            "threadId": thread_id,
            "limit": min(limit, 100),
            "sortDirection": sort_direction,
            "itemsView": items_view,
        }
        if cursor:
            params["cursor"] = cursor
        return await self._request("thread/turns/list", params)

    async def list_thread_items(
        self,
        thread_id: str,
        *,
        cursor: str | None = None,
        limit: int = 100,
        sort_direction: str = "desc",
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Page through persisted items, optionally limited to one turn."""
        await self._ensure_running()
        thread_id = self._validated_thread_id(thread_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise CodexAppServerError("The thread item page size must be positive.")
        if sort_direction not in {"asc", "desc"}:
            raise CodexAppServerError("Thread item sort direction is invalid.")
        params: dict[str, Any] = {
            "threadId": thread_id,
            "limit": min(limit, 100),
            "sortDirection": sort_direction,
        }
        if cursor:
            params["cursor"] = cursor
        if turn_id:
            params["turnId"] = self._validated_thread_id(turn_id)
        return await self._request("thread/items/list", params)

    async def loaded_threads(self) -> dict[str, Any]:
        """Return and locally record the threads loaded by this app-server."""
        await self._ensure_running()
        result = await self._request("thread/loaded/list", {})
        values = result.get("data", [])
        if isinstance(values, dict):
            values = values.get("threadIds", [])
        loaded = (
            {value for value in values if isinstance(value, str) and value.strip()}
            if isinstance(values, list)
            else set()
        )
        self._set_loaded_thread_ids(loaded)
        logger.debug("Codex reported loaded threads (count=%d)", len(loaded))
        return result

    async def set_thread_name(self, thread_id: str, name: str) -> dict[str, Any]:
        """Set the user-facing name of a persisted Codex thread."""
        await self._ensure_running()
        thread_id = self._validated_thread_id(thread_id)
        name = name.strip()
        if not name:
            raise CodexAppServerError("A thread name is required.")
        return await self._request(
            "thread/name/set",
            {"threadId": thread_id, "name": name[:100]},
        )

    async def rollback_thread(
        self, thread_id: str, num_turns: int = 1
    ) -> dict[str, Any]:
        """Remove the most recent persisted turns from a Codex thread."""
        await self._ensure_running()
        thread_id = self._validated_thread_id(thread_id)
        if (
            isinstance(num_turns, bool)
            or not isinstance(num_turns, int)
            or num_turns < 1
        ):
            raise CodexAppServerError(
                "The number of turns to roll back must be positive."
            )
        for session in self._sessions.values():
            if session.thread_id == thread_id and session.turn_id:
                raise CodexAppServerError(
                    "Stop the active Codex turn before rolling back the thread."
                )
        result = await self._request(
            "thread/rollback",
            {"threadId": thread_id, "numTurns": num_turns},
        )
        self._set_thread_loaded(thread_id, True)
        return result

    async def delete_thread(self, thread_id: str) -> dict[str, Any]:
        """Permanently delete a persisted Codex thread."""
        await self._ensure_running()
        thread_id = self._validated_thread_id(thread_id)
        for session in self._sessions.values():
            if session.thread_id == thread_id and session.turn_id:
                raise CodexAppServerError(
                    "Stop the active Codex turn before deleting the thread."
                )
        result = await self._request("thread/delete", {"threadId": thread_id})
        self._forget_thread(thread_id)
        return result

    async def unarchive_thread(self, thread_id: str) -> dict[str, Any]:
        """Restore an archived Codex thread to the active session store."""
        await self._ensure_running()
        thread_id = self._validated_thread_id(thread_id)
        result = await self._request("thread/unarchive", {"threadId": thread_id})
        self._set_thread_archived(thread_id, False)
        self._set_thread_loaded(thread_id, False)
        self._persist_state()
        return result
