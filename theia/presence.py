"""Dynamic Discord presence for recent activity and long-running tool turns."""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import discord

from .core import (
    _codex_logger,
    _env_bool,
    _env_float,
    _is_tool_item,
    _safe_log_label,
    _truncate,
)

logger = _codex_logger()

PRESENCE_IDLE_AFTER_ENV = "THEIA_PRESENCE_IDLE_AFTER"
PRESENCE_LONG_TASK_AFTER_ENV = "THEIA_PRESENCE_LONG_TASK_AFTER"
PRESENCE_UPDATE_INTERVAL_ENV = "THEIA_PRESENCE_UPDATE_INTERVAL"
DEFAULT_PRESENCE_IDLE_AFTER = 15 * 60
DEFAULT_PRESENCE_LONG_TASK_AFTER = 60
DEFAULT_PRESENCE_UPDATE_INTERVAL = 15
RICH_PRESENCE_ENABLED_ENV = "THEIA_RICH_PRESENCE_ENABLED"
RICH_PRESENCE_ACTIVE_DEBOUNCE_ENV = "THEIA_RICH_PRESENCE_ACTIVE_DEBOUNCE"
RICH_PRESENCE_IDLE_INTERVAL_ENV = "THEIA_RICH_PRESENCE_IDLE_INTERVAL"
RICH_PRESENCE_RECENT_IDLE_INTERVAL_ENV = "THEIA_RICH_PRESENCE_RECENT_IDLE_INTERVAL"
RICH_PRESENCE_CONTEXT_MAX_AGE_ENV = "THEIA_RICH_PRESENCE_CONTEXT_MAX_AGE"
RICH_PRESENCE_TIMEOUT_ENV = "THEIA_RICH_PRESENCE_TIMEOUT"
DEFAULT_RICH_PRESENCE_ENABLED = True
DEFAULT_RICH_PRESENCE_ACTIVE_DEBOUNCE = 3.0
DEFAULT_RICH_PRESENCE_IDLE_INTERVAL = 15 * 60
DEFAULT_RICH_PRESENCE_RECENT_IDLE_INTERVAL = 10 * 60
DEFAULT_RICH_PRESENCE_CONTEXT_MAX_AGE = 30 * 60
DEFAULT_RICH_PRESENCE_TIMEOUT = 8.0
MAX_RICH_PRESENCE_NAME = 128


@dataclass
class _ActiveRequest:
    started_at: float
    tool_started: bool = False


class PresenceManager:
    """Keep Discord presence aligned with meaningful bot activity.

    A request does not become DND merely because it is active. It must first
    produce a real Codex tool event and remain active past the long-task
    threshold. Context compaction is deliberately ignored.
    """

    def __init__(
        self,
        change_presence: Callable[..., Awaitable[None]],
        *,
        idle_after: float | None = None,
        long_task_after: float | None = None,
        update_interval: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._change_presence = change_presence
        self.idle_after = max(
            1.0,
            idle_after
            if idle_after is not None
            else _env_float(PRESENCE_IDLE_AFTER_ENV, DEFAULT_PRESENCE_IDLE_AFTER),
        )
        self.long_task_after = max(
            1.0,
            long_task_after
            if long_task_after is not None
            else _env_float(
                PRESENCE_LONG_TASK_AFTER_ENV, DEFAULT_PRESENCE_LONG_TASK_AFTER
            ),
        )
        self.update_interval = max(
            1.0,
            update_interval
            if update_interval is not None
            else _env_float(
                PRESENCE_UPDATE_INTERVAL_ENV, DEFAULT_PRESENCE_UPDATE_INTERVAL
            ),
        )
        self._clock = clock
        self._last_interaction: float | None = None
        self._active_requests: dict[str, _ActiveRequest] = {}
        self._current_status: discord.Status | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._closed = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the background loop that refreshes Discord presence state."""
        if self._loop_task is None or self._loop_task.done():
            self._closed = False
            self._loop_task = asyncio.create_task(self._run())
        await self.refresh()

    async def on_ready(self) -> None:
        """Re-send the state after the gateway reconnects."""
        self._current_status = None
        await self.refresh()

    async def close(self) -> None:
        """Stop presence updates and mark the manager closed."""
        self._closed = True
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None

    async def touch(self) -> None:
        """Record a user interaction and immediately prefer online presence."""
        self._last_interaction = self._clock()
        await self.refresh()

    async def begin_request(self, request_id: str) -> None:
        """Track a request so long-running tool work can drive DND presence."""
        self._active_requests[request_id] = _ActiveRequest(self._clock())
        await self.refresh()

    async def observe_event(
        self, request_id: str, event: str, payload: dict[str, Any]
    ) -> None:
        """Record only real tool activity; compaction never qualifies as a task."""
        if event == "compacted":
            return
        active = self._active_requests.get(request_id)
        if active is None:
            return
        if event == "tool_activity" or (
            event == "item_started" and _is_tool_item(payload)
        ):
            active.tool_started = True
            await self.refresh()

    async def finish_request(self, request_id: str) -> None:
        """Remove a request from activity tracking and refresh the status."""
        self._active_requests.pop(request_id, None)
        await self.refresh()

    async def refresh(self) -> None:
        """Recalculate and publish presence from recent and active requests."""
        async with self._lock:
            now = self._clock()
            long_task = any(
                request.tool_started
                and now - request.started_at >= self.long_task_after
                for request in self._active_requests.values()
            )
            if long_task:
                status = discord.Status.dnd
            elif (
                self._last_interaction is not None
                and now - self._last_interaction < self.idle_after
            ):
                status = discord.Status.online
            else:
                status = discord.Status.idle
            if status == self._current_status:
                return
            try:
                await self._change_presence(status=status)
            except Exception as exc:  # noqa: BLE001 - presence must not affect chat
                logger.debug(
                    "Could not update Discord presence (status=%s, error=%s)",
                    _safe_log_label(status.name),
                    type(exc).__name__,
                )
                return
            self._current_status = status
            logger.debug("Discord presence updated (status=%s)", status.name)

    async def _run(self) -> None:
        while not self._closed:
            await asyncio.sleep(self.update_interval)
            await self.refresh()

    @property
    def current_status(self) -> discord.Status | None:
        """Return the status last published by this manager."""
        return self._current_status


@dataclass(frozen=True)
class _RichActivity:
    activity_type: str
    text: str


@dataclass
class _RichContext:
    session_key: str
    guild_id: int | None
    prompt: str
    channel_context: str | None
    last_seen: float
    recent: deque[str] = field(default_factory=lambda: deque(maxlen=4))


@dataclass
class _RichTask:
    request_id: str
    session_key: str
    guild_id: int | None
    prompt: str
    channel_context: str | None
    phase: str
    phase_key: str
    sequence: int
    generation: int = 0
    generation_task: asyncio.Task[Any] | None = None
    activity: _RichActivity | None = None


class RichPresenceManager:
    """Generate safe global activity text without changing Discord status."""

    _ACTIVITY_TYPES = frozenset(
        {"playing", "streaming", "listening", "watching", "competing", "none"}
    )
    _TOOL_PHASES: ClassVar[dict[str, str]] = {
        "commandexecution": "running a command",
        "filechange": "updating files",
        "mcptoolcall": "using an integration",
        "websearch": "searching the web",
        "imagegeneration": "creating an image",
        "computertoolcall": "using a computer tool",
        "local_shell": "running a local task",
    }

    def __init__(
        self,
        change_presence: Callable[..., Awaitable[None]],
        generate_presence: Callable[..., Awaitable[dict[str, str] | None]],
        *,
        enabled: bool | None = None,
        active_debounce: float | None = None,
        idle_interval: float | None = None,
        recent_idle_interval: float | None = None,
        context_max_age: float | None = None,
        timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._change_presence = change_presence
        self._generate_presence = generate_presence
        self.enabled = (
            enabled
            if enabled is not None
            else _env_bool(RICH_PRESENCE_ENABLED_ENV, DEFAULT_RICH_PRESENCE_ENABLED)
        )
        self.active_debounce = max(
            0.0,
            active_debounce
            if active_debounce is not None
            else _env_float(
                RICH_PRESENCE_ACTIVE_DEBOUNCE_ENV,
                DEFAULT_RICH_PRESENCE_ACTIVE_DEBOUNCE,
            ),
        )
        self.idle_interval = max(
            1.0,
            idle_interval
            if idle_interval is not None
            else _env_float(
                RICH_PRESENCE_IDLE_INTERVAL_ENV,
                DEFAULT_RICH_PRESENCE_IDLE_INTERVAL,
            ),
        )
        self.recent_idle_interval = max(
            1.0,
            recent_idle_interval
            if recent_idle_interval is not None
            else _env_float(
                RICH_PRESENCE_RECENT_IDLE_INTERVAL_ENV,
                DEFAULT_RICH_PRESENCE_RECENT_IDLE_INTERVAL,
            ),
        )
        self.context_max_age = max(
            1.0,
            context_max_age
            if context_max_age is not None
            else _env_float(
                RICH_PRESENCE_CONTEXT_MAX_AGE_ENV,
                DEFAULT_RICH_PRESENCE_CONTEXT_MAX_AGE,
            ),
        )
        self.timeout = max(
            1.0,
            timeout
            if timeout is not None
            else _env_float(RICH_PRESENCE_TIMEOUT_ENV, DEFAULT_RICH_PRESENCE_TIMEOUT),
        )
        self._clock = clock
        self._contexts: dict[str, _RichContext] = {}
        self._active_tasks: dict[str, _RichTask] = {}
        self._sequence = 0
        self._current_spec: _RichActivity | None = None
        self._current_activity: discord.BaseActivity | None = None
        self._current_source: str | None = None
        self._last_idle_spec: _RichActivity | None = None
        self._last_idle_source: str | None = None
        self._published_key: tuple[str, str] | None = None
        self._idle_generation_task: asyncio.Task[Any] | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def current_activity(self) -> discord.BaseActivity | None:
        """Return the latest activity object requested from Discord."""
        return self._current_activity

    async def start(self) -> None:
        """Start periodic idle activity generation."""
        if not self.enabled:
            return
        if self._loop_task is None or self._loop_task.done():
            self._closed = False
            self._loop_task = asyncio.create_task(self._run())
        await self.refresh_idle()

    async def on_ready(self) -> None:
        """Republish the current activity after a Discord reconnect."""
        if not self.enabled:
            return
        async with self._lock:
            spec = self._current_spec
            source = self._current_source
        if spec is not None and source is not None:
            await self._publish(spec, source=source, force=True)

    async def close(self) -> None:
        """Stop idle generation and cancel all pending status requests."""
        self._closed = True
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None
        async with self._lock:
            tasks = [
                task.generation_task
                for task in self._active_tasks.values()
                if task.generation_task is not None
            ]
            if self._idle_generation_task is not None:
                tasks.append(self._idle_generation_task)
            self._idle_generation_task = None
        await self._cancel_tasks(tasks)

    async def begin_task(
        self,
        request_id: str,
        *,
        session_key: str,
        guild_id: int | None,
        prompt: str,
        channel_context: str | None,
    ) -> None:
        """Track one task and debounce its first phase summary."""
        if not self.enabled:
            return
        idle_task: asyncio.Task[Any] | None = None
        async with self._lock:
            now = self._clock()
            context = self._contexts.get(session_key)
            if context is None:
                context = _RichContext(
                    session_key,
                    guild_id,
                    _truncate(prompt, 1200),
                    _truncate(channel_context, 1800) if channel_context else None,
                    now,
                )
                self._contexts[session_key] = context
            else:
                context.guild_id = guild_id
                context.prompt = _truncate(prompt, 1200)
                context.channel_context = (
                    _truncate(channel_context, 1800) if channel_context else None
                )
                context.last_seen = now
            self._sequence += 1
            task = _RichTask(
                request_id=request_id,
                session_key=session_key,
                guild_id=guild_id,
                prompt=context.prompt,
                channel_context=context.channel_context,
                phase="starting the task",
                phase_key="start",
                sequence=self._sequence,
            )
            self._active_tasks[request_id] = task
            task.generation = 1
            task.generation_task = asyncio.create_task(
                self._generate_active(request_id, task.generation)
            )
            idle_task = self._idle_generation_task
            self._idle_generation_task = None
        await self._cancel_tasks((idle_task,))

    async def observe_event(
        self, request_id: str, event: str, payload: dict[str, Any]
    ) -> None:
        """Schedule a new summary only when a task enters a new material phase."""
        if not self.enabled:
            return
        phase = self._phase_for_event(event, payload)
        if phase is None:
            return
        old_task: asyncio.Task[Any] | None = None
        async with self._lock:
            task = self._active_tasks.get(request_id)
            if task is None or task.phase_key == phase[0]:
                return
            task.phase_key, task.phase = phase
            task.generation += 1
            old_task = task.generation_task
            task.generation_task = asyncio.create_task(
                self._generate_active(request_id, task.generation)
            )
        await self._cancel_tasks((old_task,))

    async def finish_task(
        self, request_id: str, *, response: str | None = None
    ) -> None:
        """Stop task status generation and retain only bounded idle context."""
        if not self.enabled:
            return
        generation_task: asyncio.Task[Any] | None = None
        next_spec: _RichActivity | None = None
        next_source: str | None = None
        publish_next = False
        async with self._lock:
            task = self._active_tasks.pop(request_id, None)
            if task is None:
                return
            generation_task = task.generation_task
            context = self._contexts[task.session_key]
            context.recent.append(self._completed_context(task, response))
            context.last_seen = self._clock()
            if not self._active_tasks:
                publish_next = True
                next_spec = self._last_idle_spec
                next_source = self._last_idle_source or f"idle:{task.session_key}"
            elif self._current_source == request_id:
                next_task = self._latest_task_locked()
                if next_task is not None and next_task.activity is not None:
                    publish_next = True
                    next_spec = next_task.activity
                    next_source = next_task.request_id
                # Keep the completed task's activity until the next active
                # task produces its replacement rather than showing a blank.
        await self._cancel_tasks((generation_task,))
        if publish_next and next_source is not None:
            await self._publish(next_spec, source=next_source)

    async def refresh_idle(self) -> None:
        """Schedule one idle line when no task is currently active."""
        if not self.enabled:
            return
        async with self._lock:
            if self._active_tasks:
                return
            if (
                self._idle_generation_task is not None
                and not self._idle_generation_task.done()
            ):
                return
            context = self._latest_context_locked()
            self._idle_generation_task = asyncio.create_task(
                self._generate_idle(context)
            )

    async def _run(self) -> None:
        while not self._closed:
            await asyncio.sleep(await self._idle_delay())
            try:
                await self.refresh_idle()
            except Exception as exc:  # noqa: BLE001 - presence must not affect chat
                logger.debug(
                    "Could not schedule idle Rich Presence (error=%s)",
                    type(exc).__name__,
                )

    async def _idle_delay(self) -> float:
        async with self._lock:
            if any(
                self._clock() - context.last_seen < self.context_max_age
                for context in self._contexts.values()
            ):
                return self.recent_idle_interval
        return self.idle_interval

    async def _generate_active(self, request_id: str, generation: int) -> None:
        try:
            await asyncio.sleep(self.active_debounce)
            async with self._lock:
                task = self._active_tasks.get(request_id)
                if task is None or task.generation != generation:
                    return
                prompt = self._active_prompt(task)
                session_key = task.session_key
            result = await asyncio.wait_for(
                self._generate_presence(
                    prompt,
                    session_key=session_key,
                    timeout=self.timeout,
                ),
                timeout=self.timeout,
            )
            spec = self._activity_from_result(result)
            if spec is None:
                return
            async with self._lock:
                task = self._active_tasks.get(request_id)
                if task is None or task.generation != generation:
                    return
                task.activity = spec
            await self._publish(spec, source=request_id)
        except Exception as exc:  # noqa: BLE001 - presence must not affect chat
            logger.debug(
                "Could not generate active Rich Presence (error=%s)",
                type(exc).__name__,
            )

    async def _generate_idle(self, context: _RichContext | None) -> None:
        try:
            if context is None:
                session_key = None
                recent = None
            else:
                session_key = context.session_key
                recent = "\n\n".join(context.recent)
            prompt = self._idle_prompt(recent)
            result = await asyncio.wait_for(
                self._generate_presence(
                    prompt,
                    session_key=session_key,
                    timeout=self.timeout,
                ),
                timeout=self.timeout,
            )
            spec = self._activity_from_result(result)
            if spec is not None:
                source = f"idle:{session_key or 'none'}"
                async with self._lock:
                    self._last_idle_spec = spec
                    self._last_idle_source = source
                await self._publish(spec, source=source)
        except Exception as exc:  # noqa: BLE001 - presence must not affect chat
            logger.debug(
                "Could not generate idle Rich Presence (error=%s)",
                type(exc).__name__,
            )

    async def _publish(
        self,
        spec: _RichActivity | None,
        *,
        source: str,
        force: bool = False,
    ) -> None:
        async with self._lock:
            if source in self._active_tasks:
                latest = self._latest_task_locked()
                if latest is None or latest.request_id != source:
                    return
                if (
                    spec is None
                    and self._current_source in self._active_tasks
                    and self._current_source != source
                ):
                    return
            elif source.startswith("idle:"):
                if self._active_tasks:
                    return
            else:
                return

            key = None if spec is None else (spec.activity_type, spec.text)
            if not force and key == self._published_key:
                self._current_spec = spec
                self._current_source = source
                return
            activity = self._discord_activity(spec)
            try:
                await self._change_presence(activity=activity)
            except Exception as exc:  # noqa: BLE001 - presence must not affect chat
                logger.debug(
                    "Could not update Rich Presence (error=%s)",
                    type(exc).__name__,
                )
                return
            self._current_spec = spec
            self._current_activity = activity
            self._current_source = source
            self._published_key = key
            logger.debug(
                "Rich Presence updated (source=%s, activity_type=%s)",
                _safe_log_label("active" if source in self._active_tasks else "idle"),
                _safe_log_label(spec.activity_type if spec else "none"),
            )

    async def _cancel_tasks(self, tasks: Iterable[asyncio.Task[Any] | None]) -> None:
        pending = tuple(task for task in tasks if task is not None and not task.done())
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def _latest_task_locked(self) -> _RichTask | None:
        return max(
            self._active_tasks.values(),
            key=lambda task: task.sequence,
            default=None,
        )

    def _latest_context_locked(self) -> _RichContext | None:
        return max(
            self._contexts.values(), key=lambda context: context.last_seen, default=None
        )

    @classmethod
    def _phase_for_event(
        cls, event: str, payload: dict[str, Any]
    ) -> tuple[str, str] | None:
        if event == "tool_activity":
            return "tool:generic", "using a tool"
        if event == "verified_change":
            return "verified_change", "checking changes"
        if event == "thread_opening":
            return "thread_opening", "organizing the conversation"
        if event not in {"item_started", "item_completed"}:
            return None
        item_type = str(payload.get("type") or "").casefold()
        if _is_tool_item(payload):
            phase = cls._TOOL_PHASES.get(item_type, "working with tools")
            return f"tool:{item_type}", phase
        if event == "item_completed" and item_type == "agentmessage":
            if payload.get("phase") == "final_answer":
                return None
            return "agent_message", "preparing a response"
        return None

    @staticmethod
    def _completed_context(task: _RichTask, response: str | None) -> str:
        parts = [f"Completed task category: {task.phase}"]
        if task.prompt:
            parts.append(f"Request context:\n{_truncate(task.prompt, 900)}")
        if task.channel_context:
            parts.append(
                f"Recent exchange context:\n{_truncate(task.channel_context, 1200)}"
            )
        if response:
            parts.append(f"Response context:\n{_truncate(response, 900)}")
        return "\n".join(parts)

    @staticmethod
    def _active_prompt(task: _RichTask) -> str:
        context = task.channel_context or "No recent channel context was supplied."
        return (
            "Create a concise Discord Rich Presence for the current task phase. "
            "Use the phase and bounded context only to choose a generic, useful "
            "description. The activity is globally visible, so do not reveal any "
            "user, guild, channel, request, message, file, path, URL, or other "
            "private detail. Do not answer the task. Return the requested JSON.\n\n"
            f"<phase>{task.phase}</phase>\n"
            f"<task_context>{_truncate(task.prompt, 900)}</task_context>\n"
            f"<recent_context>{_truncate(context, 1200)}</recent_context>"
        )

    @staticmethod
    def _idle_prompt(recent: str | None) -> str:
        context = recent or "There is no recent task or message context."
        return (
            "Create one concise generic idle Discord Rich Presence line for Theia. "
            "Use the selected personality's style and the bounded recent context "
            "only to choose an appropriate mood or broad activity. The activity is "
            "globally visible, so never mention users, guilds, channels, requests, "
            "message text, files, paths, URLs, identifiers, or private subjects. "
            "Do not answer or continue any task. Return the requested JSON.\n\n"
            f"<recent_context>{_truncate(context, 3000)}</recent_context>"
        )

    @classmethod
    def _activity_from_result(cls, result: Any) -> _RichActivity | None:
        if not isinstance(result, dict):
            return None
        activity_type = str(result.get("activity_type") or "").casefold()
        if activity_type not in cls._ACTIVITY_TYPES:
            return None
        text = re.sub(r"<@!?\d+>|https?://\S+", "", str(result.get("text") or ""))
        text = re.sub(r"\s+", " ", text).strip(" `\"'")
        text = re.sub(r"\.{3,}$", "", text).strip()
        text = text[:MAX_RICH_PRESENCE_NAME].rstrip()
        if not text:
            return None
        return _RichActivity(activity_type, text)

    @staticmethod
    def _discord_activity(spec: _RichActivity | None) -> discord.BaseActivity | None:
        if spec is None:
            return None
        if spec.activity_type == "none":
            return discord.CustomActivity(spec.text)
        activity_type = discord.ActivityType.unknown
        if spec.activity_type == "playing":
            activity_type = discord.ActivityType.playing
        elif spec.activity_type == "streaming":
            activity_type = discord.ActivityType.streaming
        elif spec.activity_type == "listening":
            activity_type = discord.ActivityType.listening
        elif spec.activity_type == "watching":
            activity_type = discord.ActivityType.watching
        elif spec.activity_type == "competing":
            activity_type = discord.ActivityType.competing
        return discord.Activity(type=activity_type, name=spec.text)
