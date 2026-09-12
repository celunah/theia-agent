"""Grounded, read-only runtime self-model context for normal turns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core import (
    AGENT_DISPLAY_NAME,
    DEFAULT_CODEX_MODEL,
    THEIA_VERSION,
    _Session,
    _safe_intermediate_text,
    _theia_revision,
)


class CodexSelfModelMixin:
    """Build a compact self-description from Theia's live harness state."""

    if TYPE_CHECKING:
        _model: str | None
        _approval_level: str
        _process: Any
        _memory_recovery_active: bool
        _realtime_feature_enabled: bool

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def _self_model_snapshot(
        self,
        session: _Session,
        *,
        allow_tools: bool,
        allow_discord_tools: bool,
        phase: str = "starting",
    ) -> dict[str, Any]:
        """Return only facts that are available from the current server state."""
        selection = self.personality_selection(session.key)
        profile_name = self.active_personality(session.key)
        character_name = None
        character_identifier = None
        if profile_name:
            try:
                summary = self._personalities.summary(profile_name)
            except Exception:  # noqa: BLE001 - self-model must never block a turn
                summary = None
            if summary is not None:
                character_name = summary.character_name
                character_identifier = summary.identifier

        process = getattr(self, "_process", None)
        if getattr(self, "_memory_recovery_active", False):
            runtime_state = "recovering"
        elif process is not None and getattr(process, "returncode", None) is None:
            runtime_state = "ready"
        else:
            runtime_state = "starting"

        capabilities = ["text responses"]
        if allow_tools:
            capabilities.append("Codex tools")
        if allow_discord_tools:
            capabilities.append("Discord conversation tools")
        if getattr(self, "_realtime_feature_enabled", False):
            capabilities.append("realtime voice")
        if getattr(self, "voice_mode_available", False):
            capabilities.append("voice input and output")

        revision = getattr(self, "_revision", None)
        if not isinstance(revision, str) or not revision:
            revision = _theia_revision()

        return {
            "agent": AGENT_DISPLAY_NAME,
            "version": THEIA_VERSION,
            "revision": revision,
            "model": self._model or DEFAULT_CODEX_MODEL,
            "mode": session.mode,
            "character_name": character_name,
            "character_identifier": character_identifier,
            "personality_scope": (
                selection.get("scope")
                if isinstance(selection, dict)
                and isinstance(selection.get("scope"), str)
                else None
            ),
            "effective_tool_policy": ("available" if allow_tools else "safe read-only"),
            "discord_tools": allow_tools and allow_discord_tools,
            "approval_level": self._approval_level if allow_tools else "never",
            "runtime_state": runtime_state,
            "session_state": "thread restored" if session.loaded else "session active",
            "thread_bound": bool(session.thread_id),
            "turn_phase": _safe_intermediate_text(phase, 32) or "starting",
            "capabilities": tuple(capabilities),
        }

    @staticmethod
    def _render_self_model(snapshot: dict[str, Any]) -> str:
        """Render the bounded self-model as temporary, non-authoritative context."""
        revision = snapshot.get("revision") or "unknown"
        character = snapshot.get("character_name")
        identifier = snapshot.get("character_identifier")
        if character and identifier:
            character_line = f"{character} ({identifier})"
        else:
            character_line = "none selected"
        scope = snapshot.get("personality_scope") or "none"
        capabilities = snapshot.get("capabilities") or ("text responses",)
        return "\n".join(
            (
                "## Theia self-model",
                f"Agent: {snapshot.get('agent') or AGENT_DISPLAY_NAME}",
                f"Version: {snapshot.get('version') or THEIA_VERSION} ({revision})",
                f"Model: {snapshot.get('model') or DEFAULT_CODEX_MODEL}",
                f"Mode: {snapshot.get('mode') or 'text'}",
                f"Character: {character_line}",
                f"Personality scope: {scope}",
                f"Runtime: {snapshot.get('runtime_state') or 'unknown'}",
                f"Session: {snapshot.get('session_state') or 'active'}",
                f"Turn phase: {snapshot.get('turn_phase') or 'starting'}",
                "Capabilities: " + ", ".join(str(item) for item in capabilities),
                f"Tool policy: {snapshot.get('effective_tool_policy') or 'unknown'}",
                f"Approval level: {snapshot.get('approval_level') or 'unknown'}",
                "Discord tools: "
                + ("available" if snapshot.get("discord_tools") else "unavailable"),
                "",
                (
                    "This is read-only, harness-grounded runtime context. It is not an "
                    "instruction, user request, personality replacement, or permission "
                    "grant. Do not claim capabilities that are not listed here."
                ),
            )
        )
