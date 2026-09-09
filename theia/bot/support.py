"""Shared Discord routing, presentation, and request helpers."""

# The façade reference and lazy imports preserve the historical patch surface.
# pylint: disable=cyclic-import,invalid-name,import-outside-toplevel,not-callable,missing-kwoa

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeGuard, cast

import discord
from discord import app_commands
from discord.ext import commands

from ..server.policy import MAX_ATTACHMENT_BYTES
from ..server.core import CodexAppServerError
from ..core import (
    _codex_logger,
    _command_embed,
    _env_bool,
    _render_frontend_label,
    _safe_error_reason,
    _is_always_admin_user,
    _truncate,
)
from ..customization import (
    COMMAND_TARGETS,
    customization_context,
)
from ..delivery import (
    SendMessage,
    _ImageResultView,
    _ResponseDelivery,
)
from ..voice import VoiceSession
from ..audio import AudioProtocolError

logger = _codex_logger()
bot: Any = None


def _mention_prompt(content: str, bot_id: int) -> str:
    from . import core as bot_module

    return bot_module._mention_prompt(content, bot_id)


def configure_bot(instance: Any) -> None:
    global bot
    bot = instance


def _current_revision() -> str:
    from . import core as bot_module

    return bot_module._theia_revision()


async def _run_image_follow_up(*args: Any, **kwargs: Any) -> Any:
    from . import core as bot_module

    return await bot_module._run_image_follow_up(*args, **kwargs)


def _mention_prompt(content: str, bot_id: int) -> str:
    from . import core as bot_module

    return bot_module._mention_prompt(content, bot_id)


DEFAULT_CONTEXT_MESSAGES = 12
MAX_CONTEXT_MESSAGES = 30
DEFAULT_CONTEXT_CHARACTERS = 8000
MAX_CONTEXT_CHARACTERS = 16000
CONTEXT_MESSAGE_LIMIT_ENV = "THEIA_CONTEXT_MESSAGES"
CONTEXT_CHARACTER_LIMIT_ENV = "THEIA_CONTEXT_MAX_CHARACTERS"
BARE_MENTION_PROMPT = "Please respond to the recent conversation context."
DEBUG_REFRESH_INTERVAL = 2.0
DEBUG_VIEW_TIMEOUT = 15 * 60
PERSISTENT_VIEW_FILE = "discord-views.json"
PERSISTENT_VIEW_VERSION = 1
PERSISTENT_VIEW_LIMIT = 256
STALE_INTERACTION_FALLBACK_DELAY = 2.0


class _PersistentViewStore:
    """Persist enough Discord view state to re-register it after a restart."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return
        if not isinstance(data, dict) or data.get("version") != PERSISTENT_VIEW_VERSION:
            return
        records = data.get("views")
        if not isinstance(records, list):
            return
        for record in records:
            if not isinstance(record, dict):
                continue
            token = record.get("token")
            message_id = record.get("message_id")
            if not isinstance(token, str) or not token:
                continue
            if not isinstance(message_id, int) or message_id <= 0:
                continue
            if not isinstance(record.get("kind"), str):
                continue
            if not isinstance(record.get("state"), dict):
                continue
            self.records[token] = dict(record)

    def _persist(self) -> None:
        records = sorted(
            self.records.values(),
            key=lambda record: float(record.get("expires_at", 0)),
            reverse=True,
        )[:PERSISTENT_VIEW_LIMIT]
        data = {"version": PERSISTENT_VIEW_VERSION, "views": records}
        temporary = self.path.with_suffix(".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temporary.chmod(0o600)
            temporary.replace(self.path)
        except OSError as exc:
            logger.debug(
                "Could not persist Discord view state (error=%s)", type(exc).__name__
            )
            with contextlib.suppress(OSError):
                temporary.unlink()

    def register(self, view: Any, message: Any) -> None:
        token = getattr(view, "persistence_token", None)
        message_id = getattr(message, "id", None)
        if not isinstance(token, str) or not token:
            return
        if not isinstance(message_id, int) or message_id <= 0:
            return
        try:
            state = view.persistence_data()
            json.dumps(state)
        except (AttributeError, TypeError, ValueError):
            return
        timeout = view.persistence_timeout()
        expires_at = time.time() + timeout if timeout is not None else None
        self.records[token] = {
            "token": token,
            "message_id": message_id,
            "kind": str(getattr(view, "persistence_kind", "")),
            "expires_at": expires_at,
            "state": state,
        }
        self._persist()
        view.set_persistence_callbacks(
            on_state_change=self.update,
            on_stop=self.remove,
        )

    async def update(self, view: Any) -> None:
        token = getattr(view, "persistence_token", None)
        if token not in self.records:
            return
        try:
            state = view.persistence_data()
            json.dumps(state)
        except (AttributeError, TypeError, ValueError):
            return
        self.records[token]["state"] = state
        self._persist()

    def remove(self, view: Any) -> None:
        token = getattr(view, "persistence_token", None)
        if token in self.records:
            self.records.pop(token, None)
            self._persist()

    def restore(
        self,
        bot_instance: commands.Bot,
        factory: Callable[[dict[str, Any]], Any | None],
    ) -> None:
        now = time.time()
        changed = False
        for token, record in tuple(self.records.items()):
            expires_at = record.get("expires_at")
            if expires_at is not None:
                try:
                    if float(expires_at) <= now:
                        self.records.pop(token, None)
                        changed = True
                        continue
                except (TypeError, ValueError):
                    self.records.pop(token, None)
                    changed = True
                    continue
            try:
                view = factory(record)
            except Exception as exc:  # noqa: BLE001 - corrupt view state is disposable
                logger.debug(
                    "Could not restore a Discord view (error=%s)",
                    type(exc).__name__,
                )
                view = None
            if view is None:
                self.records.pop(token, None)
                changed = True
                continue
            try:
                bot_instance.add_view(view, message_id=record["message_id"])
            except (TypeError, ValueError):
                self.records.pop(token, None)
                changed = True
                continue
            view.set_persistence_callbacks(
                on_state_change=self.update,
                on_stop=self.remove,
            )
        if changed:
            self._persist()


class _GeneratedImageAttachment:
    """Present a validated local image as an input attachment for a new turn."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.filename = path.name
        self.content_type = {
            ".gif": "image/gif",
            ".jpeg": "image/jpeg",
            ".jpg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(path.suffix.casefold(), "image/png")
        self.url = ""
        try:
            self.size = path.stat().st_size
        except OSError:
            self.size = 0

    async def read(self) -> bytes:
        """Read the generated image only while it remains a regular file."""
        if self.path.is_symlink() or not self.path.is_file():
            raise OSError("generated image is no longer available")
        if self.path.stat().st_size > MAX_ATTACHMENT_BYTES:
            raise OSError("generated image is too large")
        return self.path.read_bytes()


def _frontend_embed(
    target: str,
    title: str,
    description: str,
    *,
    channel: Any | None = None,
    user: discord.abc.User | None = None,
    color: discord.Color | None = None,
    context: dict[str, Any] | None = None,
) -> discord.Embed:
    """Render a command embed using server-only Discord preferences."""
    values = customization_context(channel, user, command=target)
    if context:
        values.update(context)
    return _command_embed(
        title,
        description,
        color=color,
        target=target,
        guild_id=_guild_id(channel),
        customizer=getattr(globals().get("bot"), "customizations", None),
        context=values,
    )


def _frontend_label(
    target: str,
    default: str,
    *,
    channel: Any | None = None,
    user: discord.abc.User | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    """Render a server-scoped label for a Discord embed or control."""
    values = customization_context(channel, user)
    if context:
        values.update(context)
    return _render_frontend_label(
        getattr(globals().get("bot"), "customizations", None),
        _guild_id(channel),
        target,
        default,
        context=values,
    )


def _env_bool_any(names: Iterable[str], default: bool = False) -> bool:
    for name in names:
        if os.getenv(name) is not None:
            return _env_bool(name, default)
    return default


def _configured_ids(*names: str) -> set[int]:
    values: set[int] = set()
    for name in names:
        for raw in (item.strip() for item in (os.getenv(name) or "").split(",")):
            if raw.isdigit():
                values.add(int(raw))
    return values


def _is_server_admin(user: discord.abc.User, channel: Any | None) -> bool:
    if _is_always_admin_user(getattr(user, "id", None)):
        return True
    if getattr(channel, "guild", None) is None:
        return False
    permissions = getattr(user, "guild_permissions", None)
    return bool(permissions and getattr(permissions, "administrator", False))


def _interaction_install_flag(interaction: Any, name: str) -> bool | None:
    checker = getattr(interaction, name, None)
    if not callable(checker):
        return None
    try:
        return bool(checker())
    except (AttributeError, TypeError):
        return None


def _is_guild_install(interaction: Any) -> bool:
    """Return whether this interaction was authorized by a guild install."""
    flag = _interaction_install_flag(interaction, "is_guild_integration")
    if flag is not None:
        return flag
    return getattr(interaction, "guild", None) is not None


def _is_user_only_install(interaction: Any) -> bool:
    """Return whether this interaction comes only from a user installation."""
    user_flag = _interaction_install_flag(interaction, "is_user_integration")
    guild_flag = _interaction_install_flag(interaction, "is_guild_integration")
    if user_flag is None and guild_flag is None:
        return False
    return user_flag is True and guild_flag is not True


def _interaction_allows_tools(interaction: Any) -> bool:
    """Apply the existing tool boundary to both guild and user installations."""
    if _is_user_only_install(interaction):
        return _is_always_admin_user(getattr(interaction.user, "id", None))
    return _is_server_admin(interaction.user, interaction.channel)


def _interaction_can_manage_server(interaction: discord.Interaction) -> bool:
    """Require a guild install for server admins, while honoring trusted users."""
    if _is_always_admin_user(getattr(interaction.user, "id", None)):
        return True
    return _is_guild_install(interaction) and _is_server_admin(
        interaction.user, interaction.channel
    )


def _user_installable_command(command: Any) -> Any:
    """Expose a command to guild and account installs in every Discord context."""
    command = app_commands.allowed_contexts(
        guilds=True,
        dms=True,
        private_channels=True,
    )(command)
    return app_commands.allowed_installs(guilds=True, users=True)(command)


def _voice_session_allows_tools(session: VoiceSession) -> bool:
    """Re-check the voice session owner's current guild permissions."""
    guild = getattr(session.text_channel, "guild", None)
    if guild is None:
        get_guild = getattr(bot, "get_guild", None)
        guild = get_guild(session.guild_id) if callable(get_guild) else None
    get_member = getattr(guild, "get_member", None)
    member = get_member(session.user_id) if callable(get_member) else None
    return member is not None and _is_server_admin(
        cast(discord.abc.User, member), session.text_channel
    )


def _channel_id(channel: Any | None) -> int | None:
    value = getattr(channel, "id", None)
    return value if isinstance(value, int) else None


def _guild_id(channel: Any | None) -> int | None:
    value = getattr(getattr(channel, "guild", None), "id", None)
    return value if isinstance(value, int) else None


def _is_thread(channel: Any | None) -> TypeGuard[discord.Thread]:
    return isinstance(channel, discord.Thread)


def _thread_has_bot(channel: Any | None) -> bool:
    if not _is_thread(channel):
        return False
    if (
        getattr(bot, "_participating_threads", set())
        and channel.id in bot._participating_threads
    ):
        return True
    if getattr(bot, "codex", None) is not None and bot.codex.is_participating_thread(
        channel.id
    ):
        return True
    if getattr(channel, "me", None) is not None:
        return True
    bot_user = getattr(bot, "user", None)
    return bool(
        bot_user
        and any(
            getattr(member, "id", None) == bot_user.id
            for member in getattr(channel, "members", ())
        )
    )


def _free_response_channel(channel: Any | None) -> bool:
    channel_id = _channel_id(channel)
    return channel_id is not None and channel_id in _configured_ids(
        "THEIA_FREE_RESPONSE_CHANNELS", "DISCORD_FREE_RESPONSE_CHANNELS"
    )


def _message_has_mention(message: discord.Message) -> bool:
    bot_user = getattr(bot, "user", None)
    return bool(
        bot_user
        and bot_user.id in {getattr(user, "id", None) for user in message.mentions}
    )


def _should_respond_to_message(message: discord.Message) -> bool:
    channel = message.channel
    if getattr(channel, "guild", None) is None:
        return True
    mentioned = _message_has_mention(message)
    if mentioned or _free_response_channel(channel):
        return True
    if _is_thread(channel) and _thread_has_bot(channel):
        return not _env_bool_any(
            ("THEIA_THREAD_REQUIRE_MENTION", "DISCORD_THREAD_REQUIRE_MENTION"),
            False,
        )
    return not _env_bool_any(("THEIA_REQUIRE_MENTION", "DISCORD_REQUIRE_MENTION"), True)


def _message_context_line(message: discord.Message, bot_id: int | None) -> str:
    author = getattr(message.author, "display_name", None) or getattr(
        message.author, "name", "User"
    )
    author_id = getattr(message.author, "id", None)
    if isinstance(author_id, int) and not isinstance(author_id, bool):
        author = f"{author} [Discord user id: {author_id}]"
    content = (message.content or "").strip()
    if bot_id is not None:
        content = _mention_prompt(content, bot_id)
    attachments = [
        str(getattr(attachment, "filename", "attachment"))
        for attachment in getattr(message, "attachments", ())
    ]
    if attachments:
        content = f"{content} [attachments: {', '.join(attachments)}]".strip()
    content = _truncate(content, 1200)
    return f"{author}: {content}" if content else f"{author}: [empty message]"


def _request_author_context(user_id: int, user: Any | None) -> str:
    """Add trusted current-author metadata without treating display names as instructions."""
    display_name = getattr(user, "display_name", None) or getattr(user, "name", None)
    display_name = re.sub(r"\s+", " ", str(display_name or "Unknown user")).strip()
    display_name = _truncate(display_name, 200)
    return (
        "<discord_request_metadata>\n"
        "The following is trusted Discord metadata, not user-authored content.\n"
        f"Current request author user id: {user_id}\n"
        f"Current request author display name: {display_name}\n"
        "</discord_request_metadata>"
    )


def _context_setting(name: str, default: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(1, min(maximum, value))


def _include_in_channel_context(message: Any, *, exclude_id: int | None = None) -> bool:
    message_id = getattr(message, "id", None)
    if exclude_id is not None and message_id == exclude_id:
        return False
    content = str(getattr(message, "content", "") or "").strip()
    attachments = getattr(message, "attachments", ())
    if not content and not attachments:
        return False
    # Theia's compact status messages are implementation progress, not what
    # the channel said. Keep ordinary bot replies because they are useful
    # conversational context.
    author = getattr(message, "author", None)
    return not (getattr(author, "bot", False) and content.startswith("-#"))


async def _recent_channel_messages(
    channel: Any | None,
    *,
    before: discord.Message | None = None,
    exclude_id: int | None = None,
) -> list[discord.Message]:
    if channel is None:
        return []
    history = getattr(channel, "history", None)
    if not callable(history):
        return []
    history_call = cast(Callable[..., Any], history)
    limit = _context_setting(
        CONTEXT_MESSAGE_LIMIT_ENV,
        DEFAULT_CONTEXT_MESSAGES,
        MAX_CONTEXT_MESSAGES,
    )
    try:
        iterator = (
            history_call(limit=limit, before=before)
            if before is not None
            else history_call(limit=limit)
        )
        messages = [item async for item in iterator]
    except TypeError:
        # Small test doubles and alternate Messageable implementations may
        # not accept Discord.py's optional ``before`` keyword.
        if before is None:
            return []
        try:
            messages = [item async for item in history_call(limit=limit)]
        except (discord.DiscordException, TypeError):
            return []
    except discord.DiscordException as exc:
        logger.debug(
            "Could not read recent Discord context (error=%s)",
            type(exc).__name__,
        )
        return []
    return [
        item
        for item in reversed(messages)
        if _include_in_channel_context(item, exclude_id=exclude_id)
    ]


def _render_channel_context(messages: Iterable[str]) -> str | None:
    lines = [item for item in messages if item]
    if not lines:
        return None
    character_limit = _context_setting(
        CONTEXT_CHARACTER_LIMIT_ENV,
        DEFAULT_CONTEXT_CHARACTERS,
        MAX_CONTEXT_CHARACTERS,
    )
    selected: list[str] = []
    used = 0
    for line in reversed(lines):
        separator = 1 if selected else 0
        if used + separator + len(line) > character_limit:
            break
        selected.append(line)
        used += separator + len(line)
    selected.reverse()
    if not selected:
        selected = [_truncate(lines[-1], character_limit)]
    return (
        "Recent messages from this Discord channel, ordered from oldest to "
        "newest:\n" + "\n".join(selected)
    )


async def _channel_context(
    channel: Any | None,
    *,
    before: discord.Message | None = None,
    exclude_id: int | None = None,
    extra: Iterable[discord.Message] = (),
) -> str | None:
    recent = await _recent_channel_messages(
        channel, before=before, exclude_id=exclude_id
    )
    seen = {
        message_id
        for message_id in (getattr(item, "id", None) for item in recent)
        if message_id is not None
    }
    for item in extra:
        message_id = getattr(item, "id", None)
        if message_id in seen or not _include_in_channel_context(
            item, exclude_id=exclude_id
        ):
            continue
        recent.insert(0, item)
        if message_id is not None:
            seen.add(message_id)
    bot_id = getattr(bot.user, "id", None)
    lines = (_message_context_line(item, bot_id) for item in recent)
    return _render_channel_context(lines)


async def _message_context(message: Any) -> str | None:
    """Collect bounded reply and recent-channel context for a request."""
    reference = getattr(message, "reference", None)
    resolved = getattr(reference, "resolved", None) if reference else None
    extra = (resolved,) if isinstance(resolved, discord.Message) else ()
    return await _channel_context(
        message.channel,
        before=message,
        exclude_id=getattr(message, "id", None),
        extra=extra,
    )


def session_key(channel: Any | None, user_id: int) -> str:
    """Build the persisted session key for one Discord user and channel."""
    channel_id = getattr(channel, "id", 0)
    guild = getattr(channel, "guild", None)
    guild_id = getattr(guild, "id", 0)
    suffix = "shared" if _env_bool("CODEX_SHARED_SESSIONS") else str(user_id)
    return f"guild:{guild_id}:channel:{channel_id}:user:{suffix}"


@asynccontextmanager
async def _typing_indicator(channel: Any | None):
    if channel is None or not hasattr(channel, "typing"):
        yield
        return
    typing = channel.typing()
    try:
        await typing.__aenter__()  # pylint: disable=unnecessary-dunder-call
    except (discord.DiscordException, AttributeError):
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(discord.DiscordException, AttributeError):
            await typing.__aexit__(None, None, None)


async def handle_login(
    channel: Any,
    send: SendMessage,
    *,
    user_id: int,
    guild_id: int | None = None,
    grant_server: bool = False,
    ephemeral: bool = False,
    on_complete_send: SendMessage | None = None,
) -> None:
    """Run the Codex login flow and deliver a safe Discord status embed."""
    await bot.presence.touch()
    try:
        result = await bot.codex.begin_login(
            channel,
            user_id,
            guild_id=guild_id,
            grant_server=grant_server,
            on_complete_send=on_complete_send,
        )
    except CodexAppServerError:
        await send(
            embed=_frontend_embed(
                "command:login",
                "Login unavailable",
                "Codex could not start authentication. Please try `/login` again.",
                channel=channel,
                context={"user_id": user_id},
                color=discord.Color.red(),
            ),
            ephemeral=ephemeral,
        )
        return
    except OSError as exc:
        logger.error(
            "Codex login command failed (error=%s)",
            type(exc).__name__,
        )
        await send(
            embed=_frontend_embed(
                "command:login",
                "Login unavailable",
                _safe_error_reason(exc),
                channel=channel,
                context={"user_id": user_id},
                color=discord.Color.red(),
            ),
            ephemeral=ephemeral,
        )
        return
    if result.get("login_imported") or result.get("login_cached"):
        # Keep the Discord-user grant in sync even when Codex authenticated
        # before this bot process started.
        bot.codex.mark_authenticated(
            user_id,
            guild_id=guild_id if grant_server else None,
        )
        embed = _frontend_embed(
            "command:login",
            (
                "Cached authentication imported"
                if result.get("login_imported")
                else "Already logged in"
            ),
            (
                "Your cached Codex login is active. Everyone in this server can "
                "now use `/btw` or `/skill`."
                if grant_server and guild_id is not None
                else "Your cached Codex login is active. You can use `/btw` or `/skill`."
            ),
            channel=channel,
            context={"user_id": user_id},
            color=discord.Color.green(),
        )
    elif result.get("login_in_progress"):
        embed = _frontend_embed(
            "command:login",
            "Login in progress",
            "A Codex login is already in progress. Complete it before trying again.",
            channel=channel,
            context={"user_id": user_id},
            color=discord.Color.orange(),
        )
    else:
        embed = _frontend_embed(
            "command:login",
            "Device code required",
            "Open the verification link and enter the displayed code.",
            channel=channel,
            context={"user_id": user_id},
            color=discord.Color.blurple(),
        )
        url = result.get("verificationUrl") or result.get("verification_url")
        code = result.get("userCode") or result.get("user_code")
        if url:
            embed.add_field(
                name=_frontend_label(
                    "label:login_verification_link",
                    "Verification link",
                    channel=channel,
                ),
                value=str(url),
                inline=False,
            )
        if code:
            embed.add_field(
                name=_frontend_label("label:login_code", "Code", channel=channel),
                value=str(code),
                inline=True,
            )
        embed.set_footer(
            text=_frontend_label(
                "label:login_visibility_footer",
                "This authentication message is visible only to you.",
                channel=channel,
            )
        )
    await send(embed=embed, ephemeral=ephemeral)


def _interaction_request_sender(interaction: discord.Interaction) -> SendMessage:
    """Use the deferred interaction response before consuming webhook followups."""
    original_available = True

    async def send(**kwargs: Any) -> Any:
        nonlocal original_available
        if original_available:
            original_available = False
            edit_original = getattr(interaction, "edit_original_response", None)
            if callable(edit_original):
                original_kwargs = dict(kwargs)
                # The visibility of a deferred response is fixed at defer time.
                original_kwargs.pop("ephemeral", None)
                try:
                    return await cast(
                        Coroutine[Any, Any, Any], edit_original(**original_kwargs)
                    )
                except discord.DiscordException:
                    pass
        return await interaction.followup.send(**kwargs)

    return send


async def handle_request(
    send: SendMessage,
    prompt: str,
    *,
    channel: Any | None,
    user_id: int,
    user: Any | None = None,
    attachments: Iterable[Any] = (),
    allow_tools: bool = True,
    context: str | None = None,
    request_id: str | int | None = None,
    speak_text: Callable[[str], Awaitable[None]] | None = None,
    use_webhook_thread: bool = False,
    thread_source: discord.Message | None = None,
    interaction_sender: SendMessage | None = None,
    allow_discord_tools: bool = True,
    image_message: Any | None = None,
    image_view: _ImageResultView | None = None,
    existing_image_paths: Iterable[Path] = (),
    **kwargs: Any,
) -> None:
    """Route one Discord request through Codex and stream its user-facing result."""
    if request_id is not None and not bot.codex.claim_message(request_id):
        logger.info("Ignored duplicate Discord request")
        return
    delivery = _ResponseDelivery(
        send,
        kwargs,
        owner_id=user_id,
        channel=channel,
        speak_text=speak_text,
        customizer=bot.customizations,
        guild_id=_guild_id(channel),
        context=customization_context(channel, user=None, user_id=user_id),
        image_path_resolver=bot.codex.image_artifact_path,
        on_view_created=bot.register_view,
        image_message=image_message,
        image_view=image_view,
        existing_image_paths=existing_image_paths,
    )
    request_session_key = session_key(channel, user_id)
    response_for_presence: str | None = None
    recap_started_at = bot.recaps.now()

    def on_channel_change(new_channel: Any) -> None:
        delivery.channel = new_channel
        if use_webhook_thread:
            # Interaction follow-ups must keep using the webhook sender while
            # targeting the newly-created thread explicitly.
            delivery.kwargs["thread"] = new_channel
        else:
            delivery.send = new_channel.send
            delivery.kwargs.pop("reference", None)
            delivery.kwargs.pop("thread", None)
        bot.codex.rebind_session(
            request_session_key,
            session_key(new_channel, user_id),
        )
        if _is_thread(new_channel):
            bot._participating_threads.add(new_channel.id)
            bot.codex.mark_thread_participating(new_channel.id)

    presence_request_id = f"request:{id(delivery)}"
    await bot.presence.touch()
    await bot.presence.begin_request(presence_request_id)
    with contextlib.suppress(Exception):
        await bot.rich_presence.begin_task(
            presence_request_id,
            session_key=request_session_key,
            guild_id=_guild_id(channel),
            prompt=prompt,
            channel_context=context,
        )
    error_reason: str | None = None

    async def on_codex_event(event: str, payload: dict[str, Any]) -> None:
        try:
            await delivery.on_event(event, payload)
        finally:
            await bot.presence.observe_event(presence_request_id, event, payload)
            with contextlib.suppress(Exception):
                await bot.rich_presence.observe_event(
                    presence_request_id, event, payload
                )

    prompt_parts = [_request_author_context(user_id, user)]
    recap_context = bot.recaps.context_for(
        user_id=user_id,
        guild_id=_guild_id(channel),
    )
    if recap_context:
        prompt_parts.append(
            "<theia_nightly_recaps>\n" + recap_context + "\n</theia_nightly_recaps>"
        )
    if context:
        prompt_parts.append("<discord_context>\n" + context + "\n</discord_context>")
    prompt_parts.append(prompt)
    effective_prompt = "\n\n".join(prompt_parts)
    try:
        async with _typing_indicator(channel):
            await delivery.start()
            failed = False
            try:
                response = await bot.codex.ask(
                    effective_prompt,
                    session_key=session_key(channel, user_id),
                    channel=channel,
                    user_id=user_id,
                    user=user,
                    attachments=attachments,
                    allow_tools=allow_tools,
                    thread_source=thread_source,
                    user_prompt=prompt,
                    on_channel_change=on_channel_change,
                    on_event=on_codex_event,
                    interaction_sender=interaction_sender,
                    allow_discord_tools=allow_discord_tools,
                )
            except CodexAppServerError as exc:
                failed = True
                error_reason = str(exc)
                response = "Codex could not complete this request."
            except Exception as exc:  # noqa: BLE001 - never leave a Discord request silent
                failed = True
                error_reason = str(exc)
                response = "Codex could not complete this request."
            response_for_presence = response
            speech = ()
            if not failed and speak_text is None:
                try:
                    speech = await bot.codex.synthesize_response(response)
                except AudioProtocolError as exc:
                    logger.warning(
                        "Optional TTS response failed (error=%s)",
                        type(exc).__name__,
                    )
            elif not failed and speak_text is not None:
                # Voice-mode responses are spoken through the active Discord
                # voice session instead of being duplicated as TTS files on
                # the text response. Keep the full final answer in text too.
                with contextlib.suppress(Exception):
                    await speak_text(response)
            await delivery.finalize(
                response,
                failed=failed,
                error_reason=error_reason if failed else None,
                speech=speech,
                image_paths=delivery.image_paths,
                on_image_action=lambda image_interaction, action_prompt, paths, view: (
                    _run_image_follow_up(
                        image_interaction,
                        action_prompt,
                        paths,
                        channel=delivery.channel,
                        image_view=view,
                        image_message=view.message,
                    )
                ),
            )
            with contextlib.suppress(Exception):
                bot.recaps.record_exchange(
                    user_id=user_id,
                    user_name=getattr(user, "display_name", None)
                    or getattr(user, "name", None),
                    guild_id=_guild_id(channel),
                    channel_id=_channel_id(channel),
                    session_key=session_key(channel, user_id),
                    prompt=prompt,
                    context=context,
                    response=response,
                    completed=not failed,
                    occurred_at=recap_started_at,
                    request_id=request_id,
                )
    finally:
        with contextlib.suppress(Exception):
            await bot.rich_presence.finish_task(
                presence_request_id,
                response=response_for_presence,
            )
        await bot.presence.finish_request(presence_request_id)
        if request_id is not None:
            bot.codex.complete_message(request_id)


async def _require_login(interaction: discord.Interaction) -> bool:
    await bot.presence.touch()
    guild_id = getattr(interaction.guild, "id", None)
    auth_guild_id = guild_id if _is_guild_install(interaction) else None
    if bot.codex.is_authenticated(interaction.user.id, auth_guild_id):
        return True
    # Upgrade an administrator who authenticated before server-scoped grants
    # were introduced. This also lets an already-authenticated admin opt a
    # server in without needing to repeat the device-code flow.
    if (
        auth_guild_id is not None
        and _interaction_can_manage_server(interaction)
        and bot.codex.is_authenticated(interaction.user.id)
    ):
        bot.codex.mark_server_authenticated(auth_guild_id)
        logger.info("Granted cached Codex access to a server")
        return True
    from .embeds import _login_required_embed

    embed = _login_required_embed(channel=interaction.channel, user=interaction.user)
    if interaction.response.is_done():
        await interaction.followup.send(embed=embed, ephemeral=True)
    else:
        await interaction.response.send_message(embed=embed, ephemeral=True)
    return False


async def _require_server_admin(
    interaction: discord.Interaction,
    *,
    message: str = "Only server administrators can approve or deny tool actions.",
) -> bool:
    if _interaction_can_manage_server(interaction):
        return True
    embed = _frontend_embed(
        "label:administrator_access_required",
        "Administrator access required",
        message,
        channel=interaction.channel,
        user=interaction.user,
        color=discord.Color.orange(),
    )
    if interaction.response.is_done():
        await interaction.followup.send(embed=embed, ephemeral=True)
    else:
        await interaction.response.send_message(embed=embed, ephemeral=True)
    return False


async def _send_command_failure(
    interaction: discord.Interaction, title: str, exc: BaseException
) -> None:
    logger.error(
        "Discord command failed: %s (error=%s)",
        title,
        type(exc).__name__,
    )
    command = title.removesuffix(" unavailable").strip().casefold()
    command = {"voice": "mode"}.get(command, command)
    target = (
        f"command:{command}" if command in COMMAND_TARGETS else "label:request_failed"
    )
    embed = _frontend_embed(
        target,
        title,
        _safe_error_reason(exc),
        channel=interaction.channel,
        user=interaction.user,
        color=discord.Color.orange(),
    )
    if interaction.response.is_done():
        await interaction.followup.send(embed=embed, ephemeral=True)
    else:
        await interaction.response.send_message(embed=embed, ephemeral=True)
