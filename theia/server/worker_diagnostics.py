"""Small, non-persistent observability helpers for internal Codex workers."""

from __future__ import annotations

import asyncio
import contextvars
import re
import time
from dataclasses import dataclass
from typing import Any

from ..core import _Session, _TurnDiagnostics

_LOW_SIGNAL_TEXT = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "ok",
        "okay",
        "k",
        "yes",
        "no",
        "sure",
        "got it",
        "thanks",
        "thank you",
        "👍",
        "👌",
    }
)
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
_ACTIVE_WORKER: contextvars.ContextVar[WorkerRun | None] = contextvars.ContextVar(
    "theia_active_worker", default=None
)


def is_low_signal_message(value: Any) -> bool:
    """Identify short acknowledgements that do not need semantic workers."""
    if not isinstance(value, str):
        return True
    normalized = re.sub(r"\s+", " ", value).strip().casefold()
    if not normalized or normalized in _LOW_SIGNAL_TEXT:
        return True
    words = _WORD_RE.findall(normalized)
    return len(words) <= 1 and len(normalized) <= 24


def diagnostics_for_session(
    server: Any, session_key: str | None
) -> _TurnDiagnostics | None:
    """Find the originating normal-turn diagnostics without creating state."""
    if not isinstance(session_key, str) or not session_key:
        return None
    sessions = getattr(server, "_sessions", {})
    session = sessions.get(session_key)
    if not isinstance(session, _Session):
        return None
    diagnostics = session.turn_diagnostics
    return diagnostics if isinstance(diagnostics, _TurnDiagnostics) else None


def current_worker() -> WorkerRun | None:
    """Return the worker scope active in this asyncio task, if any."""
    return _ACTIVE_WORKER.get()


def record_internal_request() -> None:
    """Attribute one protocol request to the current internal worker."""
    worker = current_worker()
    if worker is not None:
        worker.request()


def record_current_worker_timeout() -> None:
    """Mark a timeout raised by a protocol request or turn wait."""
    worker = current_worker()
    if worker is not None:
        worker.timeout()


def record_current_worker_failure() -> None:
    """Mark a worker failure that its caller converted into a safe result."""
    worker = current_worker()
    if worker is not None:
        worker.failed()


@dataclass
class WorkerRun:
    """One scoped internal worker execution with no content-bearing state."""

    diagnostics: _TurnDiagnostics | None
    worker: str
    started_at: float
    token: contextvars.Token[WorkerRun | None]
    terminal: str | None = None

    def request(self) -> None:
        """Count one internal App Server request for this worker scope."""
        if self.diagnostics is not None:
            self.diagnostics.record_internal_request()

    def timeout(self) -> None:
        """Record the first timeout terminal state for this worker scope."""
        if self.terminal is None:
            self.terminal = "timeout"
            if self.diagnostics is not None:
                self.diagnostics.record_timeout()

    def cancelled(self) -> None:
        """Record the first cancellation terminal state for this worker scope."""
        if self.terminal is None:
            self.terminal = "cancelled"
            if self.diagnostics is not None:
                self.diagnostics.record_cancellation()

    def failed(self) -> None:
        """Record the first failure terminal state for this worker scope."""
        if self.terminal is None:
            self.terminal = "failed"
            if self.diagnostics is not None:
                self.diagnostics.record_failure()

    def finish(self) -> None:
        """Record elapsed time and restore the previous worker context."""
        if self.diagnostics is not None:
            self.diagnostics.record_worker_duration(
                self.worker, time.monotonic() - self.started_at
            )
        _ACTIVE_WORKER.reset(self.token)


def begin_worker(diagnostics: _TurnDiagnostics | None, worker: str) -> WorkerRun:
    """Start a worker scope used for safe request accounting and timing."""
    token = _ACTIVE_WORKER.set(None)
    run = WorkerRun(
        diagnostics=diagnostics,
        worker=worker,
        started_at=time.monotonic(),
        token=token,
    )
    _ACTIVE_WORKER.set(run)
    return run


def record_worker_exception(run: WorkerRun, error: BaseException) -> None:
    """Classify an already-caught worker exception without storing its text."""
    if isinstance(error, asyncio.CancelledError):
        run.cancelled()
    elif (
        isinstance(error, asyncio.TimeoutError) or "timed out" in str(error).casefold()
    ):
        run.timeout()
    else:
        run.failed()


async def run_worker(
    diagnostics: _TurnDiagnostics | None,
    worker: str,
    operation: Any,
) -> Any:
    """Run one awaitable operation with bounded timing and failure accounting."""
    run = begin_worker(diagnostics, worker)
    try:
        return await operation
    except BaseException as exc:
        record_worker_exception(run, exc)
        raise
    finally:
        run.finish()
