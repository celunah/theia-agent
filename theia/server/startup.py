"""Startup health state shared by Codex and the Lighthouse dashboard."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from ..core import _codex_logger, _safe_intermediate_text

logger = _codex_logger()


class CodexStartupMixin:
    """Keep startup failures visible without retaining unsafe exception text."""

    def record_startup_failure(self, reason: str, *, block: bool = True) -> None:
        """Expose a safe fatal startup state without retaining the exception."""
        safe_reason = _safe_intermediate_text(reason, 120) or "Theia could not start."
        was_blocked = getattr(self, "_startup_blocked", False)
        if block:
            self._startup_blocked = True
        if getattr(self, "_startup_status", "starting") != "degraded" or (
            block and not was_blocked
        ):
            self._startup_status = "degraded"
            self._startup_reason = safe_reason
        current_reason = getattr(self, "_startup_reason", None) or safe_reason
        record_event = cast(
            Callable[[str, str], None] | None,
            getattr(self, "_record_runtime_event", None),
        )
        if record_event is not None:
            record_event("fatal", current_reason)
        logger.critical("FATAL: %s", current_reason)

    def _mark_startup_ready(self) -> None:
        """Mark startup healthy unless another startup component already failed."""
        if getattr(self, "_startup_status", "starting") == "starting":
            self._startup_status = "ready"
            self._startup_reason = None

    def startup_snapshot(self) -> dict[str, Any]:
        """Return bounded startup health for Lighthouse and diagnostics."""
        status = getattr(self, "_startup_status", "starting")
        if status not in {"starting", "ready", "degraded"}:
            status = "degraded"
        return {
            "status": status,
            "severity": "FATAL" if status == "degraded" else "INFO",
            "reason": (
                _safe_intermediate_text(getattr(self, "_startup_reason", None), 120)
                if status == "degraded"
                else None
            ),
        }
