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
    _codex_logger,
    _TurnState,
)

logger = _codex_logger()


class CodexLifecycleMixin:
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
        _memory_watchdog_task: asyncio.Task[None] | None
        _memory_recovery_task: asyncio.Task[None] | None
        _memory_recovery_active: bool
        _memory_breach_count: int

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    async def start(self) -> None:
        """Launch and initialize Codex while serializing lifecycle changes."""
        async with self._lifecycle_lock:
            await self._start_locked()

    async def _start_locked(self) -> None:
        """Launch and initialize the project-local Codex App Server process."""
        if self._process is not None and self._process.returncode is None:
            if self._reader_task is not None and not self._reader_task.done():
                logger.debug("Codex App Server is already running")
                return
            await self._close_locked()

        logger.info("Starting Codex App Server")

        for session in self._sessions.values():
            session.loaded = False
            session.turn_id = None
        self._loaded_thread_ids.clear()

        executable = self._codex_executable()
        if executable is None:
            raise CodexAppServerError(
                "The Codex CLI is not installed. Run the project bootstrap command "
                "or install Codex CLI on PATH."
            )
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
            logger.info("Codex App Server is ready")
        except BaseException as exc:
            logger.error(
                "Codex App Server failed during startup (error=%s)",
                type(exc).__name__,
            )
            await self._close_locked()
            raise

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
        if self._memory_recovery_active:
            raise CodexAppServerError(
                "Codex is restarting after exceeding its memory limit; "
                "try again shortly."
            )
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
        for task in (self._memory_watchdog_task, self._memory_recovery_task):
            if task is None or task is current or task.done():
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        async with self._lifecycle_lock:
            await self._close_locked()

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
        for task in self._server_tasks:
            task.cancel()
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
                logger.info("Codex App Server recovered after memory pressure")
        except Exception as exc:  # noqa: BLE001 - recovery is best effort
            logger.error(
                "Codex App Server memory recovery failed (error=%s)",
                type(exc).__name__,
            )
        finally:
            self._memory_recovery_active = False
            if started:
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
        failure = CodexAppServerError(
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

    async def usage(self) -> dict[str, Any]:
        """Return token activity recorded from Theia-owned conversation threads."""
        return self.theia_usage()

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
