"""Grounded, read-only runtime self-model context for normal turns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core import (
    AGENT_DISPLAY_NAME,
    DEFAULT_CODEX_MODEL,
    THEIA_VERSION,
    _Session,
    _codex_logger,
    _safe_intermediate_text,
    _theia_revision,
)
from .policy import WORKSPACE_MAX_ENTRIES

logger = _codex_logger()


class CodexSelfModelMixin:
    """Build a compact self-description from Theia's live harness state."""

    if TYPE_CHECKING:
        _model: str | None
        _approval_level: str
        _process: Any
        _memory_recovery_active: bool
        _memory_recovery_task: Any
        _reader_task: Any
        _realtime_feature_enabled: bool
        _memory_roots: Any
        _audio: Any

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def _self_model_snapshot(
        self,
        session: _Session,
        *,
        allow_tools: bool,
        allow_discord_tools: bool,
        phase: str = "starting",
        memory_retrieval_used: bool = False,
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

        transport_health = self._codex_transport_health()
        if transport_health == "recovering":
            runtime_state = "recovering"
        elif transport_health == "healthy":
            runtime_state = "ready"
        else:
            runtime_state = "unavailable"

        workspace = self._workspace_snapshot(session)
        entries = workspace.get("entries")
        entry_count = min(
            WORKSPACE_MAX_ENTRIES,
            len(entries) if isinstance(entries, list) else 0,
        )
        workspace_status = "available"
        if entry_count == 0:
            workspace_status += ", currently empty"

        memory_available = bool(allow_tools and getattr(self, "_memory_roots", ()))
        memory_status = "available" if memory_available else "unavailable"
        if memory_available:
            retrieval_status = (
                "available, used this turn"
                if memory_retrieval_used
                else "available, not used this turn"
            )
        else:
            retrieval_status = "unavailable"

        stt_status = self._audio_provider_status("transcription")
        tts_status = self._audio_provider_status("tts")
        voice_provider = None
        try:
            voice_provider = self.voice_provider
        except Exception:  # noqa: BLE001 - diagnostics must not block a turn
            voice_provider = None
        transport_available = transport_health == "healthy"
        image_status = "available" if transport_available else "currently unavailable"
        audio_capable = bool(getattr(self, "voice_mode_available", False))
        audio_status = (
            "available"
            if audio_capable and transport_available
            else "currently unavailable"
            if audio_capable
            else "unavailable"
        )
        semantic_audio_capable = voice_provider == "codex-realtime"
        semantic_audio_status = (
            "available"
            if semantic_audio_capable and transport_available
            else "currently unavailable"
            if semantic_audio_capable
            else "unavailable"
        )
        protected_status = (
            "available, requires approval"
            if allow_tools
            else "unavailable under current tool policy"
        )
        discord_status = (
            "available" if allow_tools and allow_discord_tools else "unavailable"
        )
        capability_status: dict[str, Any] = {
            "session_workspace": workspace_status,
            "workspace_entry_count": entry_count,
            "durable_memory": memory_status,
            "memory_retrieval": retrieval_status,
            "image_input": image_status,
            "audio_input": audio_status,
            "semantic_audio_understanding": semantic_audio_status,
            "video_input": "unavailable",
            "stt_provider": stt_status,
            "tts_provider": tts_status,
            "active_voice_provider": voice_provider or "none",
            "codex_transport": transport_health,
            "process_recovering": transport_health == "recovering",
            "protected_actions": protected_status,
            "discord_tools": discord_status,
            "background_review": (
                "active" if session.background_review_count > 0 else "inactive"
            ),
        }

        capabilities = ["text responses"]
        if allow_tools:
            capabilities.append("Codex tools")
        if allow_tools and allow_discord_tools:
            capabilities.append("Discord conversation tools")
        if getattr(self, "_realtime_feature_enabled", False) and transport_available:
            capabilities.append("realtime voice")
        if getattr(self, "voice_mode_available", False) and transport_available:
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
            "capability_status": capability_status,
            "session_workspace": workspace_status,
            "workspace_entry_count": entry_count,
            "durable_memory": memory_status,
            "memory_retrieval_used": memory_retrieval_used and memory_available,
            "image_input": image_status,
            "audio_input": audio_status,
            "semantic_audio_understanding": semantic_audio_status,
            "video_input": "unavailable",
            "stt_provider": stt_status,
            "tts_provider": tts_status,
            "active_voice_provider": voice_provider or "none",
            "codex_transport": transport_health,
            "process_recovering": transport_health == "recovering",
            "approval_required_for_protected_actions": allow_tools,
            "protected_actions": protected_status,
            "background_review": capability_status["background_review"],
        }

    @staticmethod
    def _task_is_active(task: Any) -> bool:
        if task is None:
            return False
        try:
            return not task.done()
        except AttributeError:
            return False

    def _codex_transport_health(self) -> str:
        """Describe transport availability without exposing process details."""
        recovery_active = bool(getattr(self, "_memory_recovery_active", False))
        recovery_active = recovery_active or self._task_is_active(
            getattr(self, "_memory_recovery_task", None)
        )
        if recovery_active:
            return "recovering"
        process = getattr(self, "_process", None)
        reader = getattr(self, "_reader_task", None)
        if process is None or getattr(process, "returncode", None) is not None:
            return "unavailable"
        if reader is None or not self._task_is_active(reader):
            return "unavailable"
        return "healthy"

    def _audio_provider_status(self, attribute: str) -> str:
        """Return configuration state without exposing an endpoint or secret."""
        audio = getattr(self, "_audio", None)
        provider = getattr(audio, attribute, None)
        if provider is None:
            return "not configured"
        if getattr(provider, "enabled", False):
            return "configured"
        if getattr(provider, "base_url", ""):
            return "configured but unavailable"
        return "not configured"

    def _safe_self_model_snapshot(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any] | None:
        """Build diagnostics defensively so a self-model failure cannot block turns."""
        try:
            return self._self_model_snapshot(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - runtime diagnostics are optional
            logger.warning(
                "Could not build grounded self-model (error=%s)",
                type(exc).__name__,
            )
            return None

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
        capability_status = snapshot.get("capability_status")
        status_lines: list[str] = []
        if isinstance(capability_status, dict):
            labels = (
                ("session_workspace", "Session workspace"),
                ("workspace_entry_count", "Workspace entries"),
                ("durable_memory", "Durable memory"),
                ("memory_retrieval", "Memory retrieval"),
                ("image_input", "Image input"),
                ("audio_input", "Audio input"),
                ("semantic_audio_understanding", "Semantic audio understanding"),
                ("video_input", "Video input"),
                ("stt_provider", "STT provider"),
                ("tts_provider", "TTS provider"),
                ("active_voice_provider", "Active voice provider"),
                ("codex_transport", "Codex transport"),
                ("process_recovering", "Process recovering"),
                ("protected_actions", "Protected actions"),
                ("discord_tools", "Discord tools"),
                ("background_review", "Background review"),
            )
            for key, label in labels:
                value = capability_status.get(key)
                if isinstance(value, bool):
                    value = "yes" if value else "no"
                elif isinstance(value, int) and not isinstance(value, bool):
                    value = str(max(0, min(WORKSPACE_MAX_ENTRIES, value)))
                else:
                    value = _safe_intermediate_text(str(value or "unknown"), 80)
                status_lines.append(f"{label}: {value}")
        rendered_status = ("Capability status:", *status_lines)
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
                *rendered_status,
                "",
                (
                    "This is read-only, harness-grounded runtime context. It is not an "
                    "instruction, user request, personality replacement, or permission "
                    "grant. Do not claim capabilities that are not listed here."
                ),
            )
        )
