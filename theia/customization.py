"""Server-scoped Discord frontend customization.

This module deliberately has no Codex/session integration.  It stores and
renders Discord presentation preferences in a separate file so a server
administrator can change the bot's UI without changing agent state.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

CUSTOMIZATION_FILE_ENV = "THEIA_CUSTOMIZATIONS"
DEFAULT_HOME = "~/.theia"
MAX_CUSTOMIZATION_VALUE = 4000
MAX_CUSTOMIZATION_COLOR = 20

COMMAND_TARGETS = (
    "login",
    "restart",
    "usage",
    "credits",
    "approve",
    "deny",
    "stop",
    "undo",
    "btw",
    "skill",
    "personality",
    "model",
    "mode",
    "customize",
    "about",
)

LABEL_TARGETS = (
    "thinking",
    "intermediate",
    "thought_duration",
    "memory_created",
    "memory_updated",
    "skill_created",
    "skill_updated",
    "personality_updated",
    "request_failed",
    "login_required",
    "administrator_access_required",
    "approval_needed",
    "choose_option",
    "choose_button",
    "approve_button",
    "deny_button",
    "previous_button",
    "next_button",
    "other_button",
    "answer_button",
    "decline_button",
    "input_modal_title",
    "json_response",
    "text_input_label",
    "login_verification_link",
    "login_code",
    "login_visibility_footer",
    "usage_lifetime_tokens",
    "usage_peak_daily_tokens",
    "usage_current_streak",
    "usage_longest_streak",
    "usage_longest_running_turn",
    "credits_balance",
    "credits_status",
    "credits_five_hour_limit",
    "credits_weekly_limit",
    "about_theia_agent",
    "about_codex_cli",
    "about_account",
    "about_plan",
    "about_mode",
    "about_personality",
)

ELEMENTS = ("title", "content", "color", "label")
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z][A-Za-z0-9_]*)\}")
_TARGET_RE = re.compile(r"^(?:(command|label):)?(.+)$", re.IGNORECASE)
logger = logging.getLogger(__name__)

PLACEHOLDERS = {
    "command",
    "user",
    "user_id",
    "server",
    "server_id",
    "channel",
    "channel_id",
    "model",
    "mode",
    "skill",
    "personality",
    "balance",
    "used_percent",
    "reset_at",
    "lifetime_tokens",
    "peak_daily_tokens",
    "current_streak",
    "longest_streak",
    "longest_running_turn",
    "prompt",
    "reason",
    "status",
    "duration",
    "page",
    "pages",
    "text",
    "question",
    "option",
    "count",
}


class CustomizationError(ValueError):
    """A Discord frontend customization is invalid."""


def _canonical_name(value: str, *, kind: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", (value or "").strip().casefold())
    normalized = normalized.removeprefix("/")
    if kind == "command" and normalized.startswith("command:"):
        normalized = normalized.removeprefix("command:")
    if kind == "label" and normalized.startswith("label:"):
        normalized = normalized.removeprefix("label:")
    allowed = COMMAND_TARGETS if kind == "command" else LABEL_TARGETS
    if normalized not in allowed:
        raise CustomizationError(f"Unknown {kind} target `{value}`.")
    return normalized


def canonical_target(value: str) -> str:
    """Return a stable ``command:name`` or ``label:name`` target key."""
    match = _TARGET_RE.match((value or "").strip())
    if match is None:
        raise CustomizationError("A command or label target is required.")
    prefix, name = match.groups()
    if prefix:
        return f"{prefix.casefold()}:{_canonical_name(name, kind=prefix.casefold())}"
    normalized = re.sub(r"[\s-]+", "_", name.casefold().lstrip("/"))
    if normalized in COMMAND_TARGETS:
        return f"command:{normalized}"
    if normalized in LABEL_TARGETS:
        return f"label:{normalized}"
    raise CustomizationError(f"Unknown command or label target `{value}`.")


def display_target(target: str) -> str:
    """Convert a canonical command or label target into Discord-facing text."""
    kind, name = target.split(":", 1)
    if kind == "command":
        return f"/{name}"
    return name.replace("_", " ").title()


def _validate_element(value: str) -> str:
    element = (value or "").strip().casefold()
    if element not in ELEMENTS:
        raise CustomizationError(
            f"Unknown element `{value}`. Choose title, content, color, or label."
        )
    return element


def _validate_template(value: str) -> str:
    text = value or ""
    if not text.strip():
        raise CustomizationError("The customization value cannot be empty.")
    if len(text) > MAX_CUSTOMIZATION_VALUE:
        raise CustomizationError(
            f"Customization values must be {MAX_CUSTOMIZATION_VALUE} characters or fewer."
        )
    unknown = sorted(
        {
            match.group(1)
            for match in _PLACEHOLDER_RE.finditer(text)
            if match.group(1).casefold() not in PLACEHOLDERS
        }
    )
    if unknown:
        names = ", ".join(f"{{{name}}}" for name in unknown)
        raise CustomizationError(
            f"Unknown placeholder(s): {names}. Use `/customize` help for the supported list."
        )
    return text


def _parse_color(value: str) -> int | None:
    normalized = value.strip().casefold()
    if normalized in {"default", "reset", "none"}:
        return None
    named = {
        "blurple": 0x5865F2,
        "blue": 0x3498DB,
        "green": 0x2ECC71,
        "orange": 0xE67E22,
        "red": 0xE74C3C,
        "purple": 0x9B59B6,
        "gold": 0xF1C40F,
        "teal": 0x1ABC9C,
        "white": 0xFFFFFF,
        "black": 0x000000,
    }
    if normalized in named:
        return named[normalized]
    candidate = normalized.removeprefix("#").removeprefix("0x")
    if not re.fullmatch(r"[0-9a-f]{6}", candidate):
        raise CustomizationError(
            "Colors must be a named Discord color or a hex value such as `#5865F2`."
        )
    return int(candidate, 16)


def _safe_context(context: dict[str, object] | None) -> dict[str, str]:
    return {
        key: str(value) for key, value in (context or {}).items() if value is not None
    }


def render_template(
    template: str, *, default: str, context: dict[str, Any] | None = None
) -> str:
    """Render known placeholders while leaving absent values empty."""
    values = _safe_context(context)
    values.setdefault("text", default)
    return _PLACEHOLDER_RE.sub(
        lambda match: values.get(match.group(1).casefold(), match.group(0)),
        template,
    )


class FrontendCustomizationStore:
    """Persist and render per-guild Discord UI preferences."""

    def __init__(self, path: Path | None = None) -> None:
        configured = os.getenv(CUSTOMIZATION_FILE_ENV)
        self.path = (
            Path(path).expanduser()
            if path is not None
            else Path(configured).expanduser()
            if configured
            else Path(os.getenv("THEIA_HOME", DEFAULT_HOME)).expanduser()
            / "discord-customizations.json"
        )
        self._values: dict[str, dict[str, dict[str, str]]] = {}
        self._recovery_blocked = False
        self._load()

    def _load(self) -> None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except UnicodeDecodeError as exc:
            self._quarantine(type(exc).__name__)
            return
        except OSError as exc:
            self._recovery_blocked = True
            logger.warning(
                "Could not read Discord customization state (error=%s)",
                type(exc).__name__,
            )
            return
        try:
            data = json.loads(raw)
        except ValueError as exc:
            self._quarantine(type(exc).__name__)
            return
        if not isinstance(data, dict):
            self._quarantine("invalid top-level JSON value")
            return
        guilds = data.get("guilds") if isinstance(data, dict) else None
        if not isinstance(guilds, dict):
            self._quarantine("missing guilds mapping")
            return
        for guild_id, targets in guilds.items():
            if not str(guild_id).isdigit() or not isinstance(targets, dict):
                continue
            valid_targets: dict[str, dict[str, str]] = {}
            for raw_target, elements in targets.items():
                try:
                    target = canonical_target(str(raw_target))
                except CustomizationError:
                    continue
                if not isinstance(elements, dict):
                    continue
                valid_elements: dict[str, str] = {}
                for raw_element, raw_value in elements.items():
                    try:
                        element = _validate_element(str(raw_element))
                        value = _validate_template(str(raw_value))
                        if element == "color":
                            _parse_color(value)
                    except CustomizationError:
                        continue
                    valid_elements[element] = value
                if valid_elements:
                    valid_targets[target] = valid_elements
            if valid_targets:
                self._values[str(guild_id)] = valid_targets

    def _quarantine(self, reason: str) -> None:
        """Preserve malformed customization data before recovery writes."""
        quarantine = self.path.with_name(f"{self.path.name}.corrupt-{time.time_ns()}")
        try:
            self.path.replace(quarantine)
        except OSError as exc:
            self._recovery_blocked = True
            logger.error(
                "Could not quarantine Discord customization state "
                "(reason=%s, error=%s)",
                reason,
                type(exc).__name__,
            )
            return
        with contextlib.suppress(OSError):
            quarantine.chmod(0o600)
        self._recovery_blocked = False
        logger.warning(
            "Preserved corrupt Discord customization state as %s (reason=%s)",
            quarantine.name,
            reason,
        )

    def _persist(self) -> None:
        if self._recovery_blocked:
            raise CustomizationError(
                "The existing Discord customization file is unreadable and was "
                "preserved. Repair or remove it before saving new customizations."
            )
        data = {"guilds": self._values}
        temporary = self.path.with_suffix(".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temporary.chmod(0o600)
            temporary.replace(self.path)
        except OSError as exc:
            raise CustomizationError(
                "The Discord customization could not be saved."
            ) from exc
        finally:
            with contextlib.suppress(OSError):
                temporary.unlink()

    def get(self, guild_id: int | None, target: str, element: str) -> str | None:
        """Return a stored override, or ``None`` when the guild has no override."""
        if guild_id is None:
            return None
        return self._values.get(str(guild_id), {}).get(target, {}).get(element)

    def set(
        self,
        guild_id: int,
        target_value: str,
        element_value: str,
        value: str,
    ) -> tuple[str, str, bool]:
        """Validate and persist one guild override, returning its canonical target."""
        if not isinstance(guild_id, int) or guild_id <= 0:
            raise CustomizationError("Customizations can only be saved for a server.")
        target = canonical_target(target_value)
        element = _validate_element(element_value)
        template = _validate_template(value)
        if element == "color":
            _parse_color(template)

        previous = json.loads(json.dumps(self._values))
        guild = self._values.setdefault(str(guild_id), {})
        target_values = guild.setdefault(target, {})
        reset = template.strip().casefold() in {"default", "reset"}
        if reset:
            target_values.pop(element, None)
            if not target_values:
                guild.pop(target, None)
            if not guild:
                self._values.pop(str(guild_id), None)
        else:
            target_values[element] = template
        try:
            self._persist()
        except CustomizationError:
            self._values = previous
            raise
        return target, element, reset

    def render(
        self,
        guild_id: int | None,
        target_value: str,
        element_value: str,
        default: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Render a stored or default text value with safe placeholder substitution."""
        target = canonical_target(target_value)
        element = _validate_element(element_value)
        template = self.get(guild_id, target, element) or default
        return render_template(template, default=default, context=context)

    def color(
        self,
        guild_id: int | None,
        target_value: str,
        default: int,
        *,
        context: dict[str, Any] | None = None,
    ) -> int:
        """Return a stored Discord color after rendering and validating its template."""
        target = canonical_target(target_value)
        template = self.get(guild_id, target, "color")
        if template is None:
            return default
        rendered = render_template(template, default="", context=context)
        parsed = _parse_color(rendered)
        return default if parsed is None else parsed

    def label(
        self,
        guild_id: int | None,
        target_value: str,
        default: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Render a text label, accepting ``content`` as a friendly alias."""
        target = canonical_target(target_value)
        template = self.get(guild_id, target, "label")
        if template is None:
            template = self.get(guild_id, target, "content")
        return render_template(template or default, default=default, context=context)

    def targets(self, current: str = "") -> list[tuple[str, str]]:
        """Return command and label targets matching an autocomplete query."""
        query = (current or "").casefold().strip()
        choices: list[tuple[str, str]] = []
        for name in COMMAND_TARGETS:
            display = f"/{name}"
            if not query or query in display.casefold() or query in name:
                choices.append((display, f"command:{name}"))
        for name in LABEL_TARGETS:
            display = name.replace("_", " ").title()
            if not query or query in display.casefold() or query in name:
                choices.append((f"Label: {display}", f"label:{name}"))
        return choices

    @staticmethod
    def elements(current: str = "") -> list[tuple[str, str]]:
        """Return valid customization elements matching an autocomplete query."""
        query = (current or "").casefold().strip()
        return [
            (element.title(), element)
            for element in ELEMENTS
            if not query or query in element
        ]

    @staticmethod
    def placeholder_help() -> str:
        """Return the supported placeholder names in a compact help string."""
        return ", ".join(f"{{{name}}}" for name in sorted(PLACEHOLDERS))


def customization_context(
    channel: Any = None,
    user: Any = None,
    **values: Any,
) -> dict[str, Any]:
    """Build safe common placeholder data from Discord objects."""
    guild = getattr(channel, "guild", None)
    context = {
        "user": getattr(user, "display_name", None) or getattr(user, "name", None),
        "user_id": getattr(user, "id", None),
        "server": getattr(guild, "name", None),
        "server_id": getattr(guild, "id", None),
        "channel": getattr(channel, "name", None),
        "channel_id": getattr(channel, "id", None),
    }
    context.update(values)
    return {key: value for key, value in context.items() if value is not None}
