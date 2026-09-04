import asyncio
import contextlib
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from typing import Any, TypeGuard, cast

import discord
from discord import app_commands
from discord.ext import commands

from .app_server import CodexAppServer, CodexAppServerError
from .core import (
    DEFAULT_MODE,
    TEXT_MODE,
    VOICE_MODE,
    _codex_logger,
    _command_embed,
    _env_bool,
    _safe_error_reason,
    _subtext,
    _truncate,
)
from .customization import (
    COMMAND_TARGETS,
    CustomizationError,
    FrontendCustomizationStore,
    customization_context,
    display_target,
)
from .delivery import (
    SendMessage,
    _reaction_paginators,
    _ResponseDelivery,
)
from .presence import PresenceManager
from .voice import VoiceModeError, VoiceModeManager, VoiceSession
from .audio import AudioProtocolError

logger = _codex_logger()

DEFAULT_CONTEXT_MESSAGES = 12
MAX_CONTEXT_MESSAGES = 30
DEFAULT_CONTEXT_CHARACTERS = 8000
MAX_CONTEXT_CHARACTERS = 16000
CONTEXT_MESSAGE_LIMIT_ENV = "THEIA_CONTEXT_MESSAGES"
CONTEXT_CHARACTER_LIMIT_ENV = "THEIA_CONTEXT_MAX_CHARACTERS"


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
    if getattr(channel, "guild", None) is None:
        return False
    permissions = getattr(user, "guild_permissions", None)
    return bool(permissions and getattr(permissions, "administrator", False))


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
) -> None:
    await bot.presence.touch()
    try:
        result = await bot.codex.begin_login(
            channel,
            user_id,
            guild_id=guild_id,
            grant_server=grant_server,
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
    if result.get("login_cached"):
        # Keep the Discord-user grant in sync even when Codex authenticated
        # before this bot process started.
        bot.codex.mark_authenticated(
            user_id,
            guild_id=guild_id if grant_server else None,
        )
        embed = _frontend_embed(
            "command:login",
            "Login ready",
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
            "Log in to Codex",
            "Open the verification link and enter the displayed code.",
            channel=channel,
            context={"user_id": user_id},
            color=discord.Color.blurple(),
        )
        url = result.get("verificationUrl") or result.get("verification_url")
        code = result.get("userCode") or result.get("user_code")
        if url:
            embed.add_field(name="Verification link", value=str(url), inline=False)
        if code:
            embed.add_field(name="Code", value=str(code), inline=True)
        embed.set_footer(text="This authentication message is visible only to you.")
    await send(embed=embed, ephemeral=ephemeral)


async def handle_request(
    send: SendMessage,
    prompt: str,
    *,
    channel: Any | None,
    user_id: int,
    attachments: Iterable[discord.Attachment] = (),
    allow_tools: bool = True,
    context: str | None = None,
    request_id: str | int | None = None,
    speak_text: Callable[[str], Awaitable[None]] | None = None,
    use_webhook_thread: bool = False,
    thread_source: discord.Message | None = None,
    **kwargs: Any,
) -> None:
    if request_id is not None and not bot.codex.claim_message(request_id):
        logger.info("Ignored duplicate Discord request")
        return
    delivery = _ResponseDelivery(
        send,
        kwargs,
        owner_id=user_id,
        speak_text=speak_text,
        customizer=bot.customizations,
        guild_id=_guild_id(channel),
        context=customization_context(channel, user=None, user_id=user_id),
    )
    request_session_key = session_key(channel, user_id)

    def on_channel_change(new_channel: Any) -> None:
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
    error_reason: str | None = None

    async def on_codex_event(event: str, payload: dict[str, Any]) -> None:
        try:
            await delivery.on_event(event, payload)
        finally:
            await bot.presence.observe_event(presence_request_id, event, payload)

    effective_prompt = prompt
    if context:
        effective_prompt = (
            "<discord_context>\n" + context + "\n</discord_context>\n\n" + prompt
        )
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
                    attachments=attachments,
                    allow_tools=allow_tools,
                    thread_source=thread_source,
                    user_prompt=prompt,
                    on_channel_change=on_channel_change,
                    on_event=on_codex_event,
                )
            except CodexAppServerError as exc:
                failed = True
                error_reason = str(exc)
                response = "Codex could not complete this request."
            except Exception as exc:  # noqa: BLE001 - never leave a Discord request silent
                failed = True
                error_reason = str(exc)
                response = "Codex could not complete this request."
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
            )
    finally:
        await bot.presence.finish_request(presence_request_id)
        if request_id is not None:
            bot.codex.complete_message(request_id)


def _format_count(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "Unavailable"
    return f"{value:,}"


def _format_percent(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:g}% used"
    return "Unavailable"


def _format_reset(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(value))
    return "Unavailable"


def _usage_embed(
    result: dict[str, Any],
    *,
    channel: Any | None = None,
    user: discord.abc.User | None = None,
) -> discord.Embed:
    summary = result.get("summary") if isinstance(result, dict) else None
    if not isinstance(summary, dict):
        return _frontend_embed(
            "command:usage",
            "Usage unavailable",
            "Usage data is currently unavailable. Please try again later.",
            channel=channel,
            user=user,
            color=discord.Color.orange(),
        )
    if not any(value is not None for value in summary.values()):
        return _frontend_embed(
            "command:usage",
            "Usage unavailable",
            "Usage data is currently unavailable. Please try again later.",
            channel=channel,
            user=user,
            color=discord.Color.orange(),
        )
    embed = _frontend_embed(
        "command:usage",
        "Usage",
        "Account usage reported by Codex.",
        channel=channel,
        user=user,
        context={
            "lifetime_tokens": _format_count(summary.get("lifetimeTokens")),
            "peak_daily_tokens": _format_count(summary.get("peakDailyTokens")),
            "current_streak": _format_count(summary.get("currentStreakDays")),
            "longest_streak": _format_count(summary.get("longestStreakDays")),
            "longest_running_turn": _format_count(summary.get("longestRunningTurnSec")),
        },
    )
    fields = (
        ("Lifetime tokens", _format_count(summary.get("lifetimeTokens"))),
        ("Peak daily tokens", _format_count(summary.get("peakDailyTokens"))),
        ("Current streak", _format_count(summary.get("currentStreakDays"))),
        ("Longest streak", _format_count(summary.get("longestStreakDays"))),
        (
            "Longest running turn",
            f"{_format_count(summary.get('longestRunningTurnSec'))} seconds"
            if isinstance(summary.get("longestRunningTurnSec"), (int, float))
            else "Unavailable",
        ),
    )
    for name, value in fields:
        embed.add_field(name=name, value=value, inline=True)
    return embed


def _credits_embed(
    result: dict[str, Any],
    *,
    channel: Any | None = None,
    user: discord.abc.User | None = None,
) -> discord.Embed:
    snapshot = result.get("rateLimits") if isinstance(result, dict) else None
    if not isinstance(snapshot, dict):
        return _frontend_embed(
            "command:credits",
            "Credits unavailable",
            "Credits data is currently unavailable. Please try again later.",
            channel=channel,
            user=user,
            color=discord.Color.orange(),
        )
    credit_details = snapshot.get("credits")
    if not isinstance(credit_details, dict):
        return _frontend_embed(
            "command:credits",
            "Credits unavailable",
            "Credits data is currently unavailable. Please try again later.",
            channel=channel,
            user=user,
            color=discord.Color.orange(),
        )
    balance = credit_details.get("balance")
    balance_text = str(balance) if balance is not None else "Unavailable"
    if credit_details.get("unlimited") is True:
        balance_text = "Unlimited"
    balance_available = balance is not None or credit_details.get("unlimited") is True
    embed = _frontend_embed(
        "command:credits",
        "Credits" if balance_available else "Credits unavailable",
        "Current Codex credit information."
        if balance_available
        else "Credits data is currently unavailable. Please try again later.",
        channel=channel,
        user=user,
        context={"balance": balance_text},
        color=discord.Color.blurple() if balance_available else discord.Color.orange(),
    )
    embed.add_field(name="Balance", value=balance_text, inline=True)
    embed.add_field(
        name="Status",
        value=(
            "Unlimited"
            if credit_details.get("unlimited")
            else "Metered"
            if balance is not None
            else "Unavailable"
        ),
        inline=True,
    )
    primary = snapshot.get("primary") or {}
    secondary = snapshot.get("secondary") or {}
    embed.add_field(
        name="5-hour limit",
        value=(
            f"{_format_percent(primary.get('usedPercent'))}\n"
            f"Resets {_format_reset(primary.get('resetsAt'))}"
        ),
        inline=False,
    )
    embed.add_field(
        name="Weekly limit",
        value=(
            f"{_format_percent(secondary.get('usedPercent'))}\n"
            f"Resets {_format_reset(secondary.get('resetsAt'))}"
        ),
        inline=False,
    )
    return embed


def _login_required_embed(
    *,
    channel: Any | None = None,
    user: discord.abc.User | None = None,
) -> discord.Embed:
    return _frontend_embed(
        "label:login_required",
        "Login required",
        "Please use `/login` before starting or controlling a Codex request.",
        channel=channel,
        user=user,
        color=discord.Color.orange(),
    )


async def _require_login(interaction: discord.Interaction) -> bool:
    await bot.presence.touch()
    guild_id = getattr(interaction.guild, "id", None)
    if bot.codex.is_authenticated(interaction.user.id, guild_id):
        return True
    # Upgrade an administrator who authenticated before server-scoped grants
    # were introduced. This also lets an already-authenticated admin opt a
    # server in without needing to repeat the device-code flow.
    if (
        guild_id is not None
        and _is_server_admin(interaction.user, interaction.channel)
        and bot.codex.is_authenticated(interaction.user.id)
    ):
        bot.codex.mark_server_authenticated(guild_id)
        logger.info("Granted cached Codex access to a server")
        return True
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
    if _is_server_admin(interaction.user, interaction.channel):
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


intents = discord.Intents.default()
intents.message_content = True


class TheiaBot(commands.Bot):
    def __init__(self) -> None:
        super().__init__(command_prefix=(), intents=intents, help_command=None)
        self.customizations = FrontendCustomizationStore()
        self.codex = CodexAppServer()
        self.codex.set_frontend_customizer(self.customizations)
        self._participating_threads: set[int] = set()
        self._known_channels: dict[int, Any] = {}
        self._restart_task: asyncio.Task[None] | None = None
        self._retention_task: asyncio.Task[None] | None = None
        self.presence = PresenceManager(self._change_presence_when_ready)
        self.voice = VoiceModeManager(
            transcribe=self.codex.transcribe_audio,
            synthesize=self.codex.synthesize_response,
        )

    async def _change_presence_when_ready(self, **kwargs: Any) -> None:
        """Defer presence changes until Discord has established the gateway."""
        if not self.is_ready():
            return
        await self.change_presence(**kwargs)

    async def setup_hook(self) -> None:
        await super().setup_hook()
        await self.codex.start()
        await self.tree.sync()
        await self.presence.start()
        self._retention_task = asyncio.create_task(self._retention_loop())

    async def close(self) -> None:
        if self._retention_task is not None:
            self._retention_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._retention_task
            self._retention_task = None
        await self.presence.close()
        await self.voice.close()
        await self.codex.close()
        await super().close()

    async def on_ready(self) -> None:
        await self.presence.on_ready()

    async def _retention_loop(self) -> None:
        while True:
            try:
                await self.codex.enforce_retention()
            except Exception as exc:  # noqa: BLE001 - janitor must stay alive
                logger.warning(
                    "Codex session retention check failed (error=%s)",
                    type(exc).__name__,
                )
            await asyncio.sleep(60 * 60)

    async def backfill_after_resume(self) -> None:
        limit_text = os.getenv("THEIA_BACKFILL_LIMIT") or os.getenv(
            "DISCORD_BACKFILL_LIMIT", "20"
        )
        try:
            limit = max(0, min(100, int(limit_text)))
        except ValueError:
            limit = 20
        if limit == 0:
            return
        for channel_id in self.codex.channel_checkpoints():
            channel = self.get_channel(channel_id)
            if channel is not None:
                self._known_channels[channel_id] = channel
        for channel_id, channel in tuple(self._known_channels.items()):
            checkpoint = self.codex.channel_checkpoint(channel_id)
            if checkpoint is None:
                continue
            history = getattr(channel, "history", None)
            if not callable(history):
                continue
            history_call = cast(Callable[..., Any], history)
            try:
                missed = [
                    item
                    async for item in history_call(
                        limit=limit, after=discord.Object(id=checkpoint)
                    )
                ]
            except discord.DiscordException as exc:
                logger.info(
                    "Could not backfill a Discord channel after reconnect (error=%s)",
                    type(exc).__name__,
                )
                continue
            for item in reversed(missed):
                await on_message(item)


# Kept for callers that imported the pre-Theia harness name.
CodexBot = TheiaBot


bot = TheiaBot()


async def _restart_in_place(*, delay: float = 0.5) -> None:
    """Gracefully close the bot and replace this process with the same command."""
    await asyncio.sleep(delay)
    logger.info("Restarting Theia process in place")
    try:
        await bot.close()
    except Exception:
        logger.exception("Theia shutdown raised during in-place restart")

    executable = sys.executable or "python"
    argv = [executable, *sys.argv]
    try:
        os.execv(executable, argv)
    except Exception:
        logger.exception("Theia in-place process replacement failed")


def _voice_speak_callback(
    session_key_value: str,
) -> Callable[[str], Awaitable[None]] | None:
    if bot.codex.mode(session_key_value) == VOICE_MODE and bot.voice.has_session(
        session_key_value
    ):

        async def speak(text: str) -> None:
            await bot.voice.speak_text(session_key_value, text)

        return speak
    return None


async def _handle_voice_transcript(
    session: VoiceSession, speaker: str, transcript: str
) -> None:
    with contextlib.suppress(discord.DiscordException):
        await session.text_channel.send(
            content=_subtext(f"{speaker}: {transcript}"),
            allowed_mentions=discord.AllowedMentions.none(),
        )
    await bot.presence.touch()
    prompt = f"[Voice input from {speaker}]\n{transcript}"
    active_turn = bot.codex.status(session.session_key).get("turn_id")
    if active_turn:
        try:
            await bot.codex.steer(session.session_key, prompt)
            return
        except CodexAppServerError as exc:
            if "no active codex turn" not in str(exc).casefold():
                with contextlib.suppress(discord.DiscordException):
                    await session.text_channel.send(
                        content=_subtext(
                            "I could not steer the active request: "
                            + _safe_error_reason(exc)
                        ),
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                return
    context = await _channel_context(session.text_channel)
    await handle_request(
        session.text_channel.send,
        prompt,
        channel=session.text_channel,
        user_id=session.user_id,
        allow_tools=session.allow_tools,
        context=context,
        speak_text=_voice_speak_callback(session.session_key),
    )


@bot.tree.command(name="login", description="Authenticate this Discord user with Codex")
async def codex_login(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    channel = interaction.channel or interaction.user
    guild_id = getattr(interaction.guild, "id", None)
    grant_server = guild_id is not None and _is_server_admin(
        interaction.user, interaction.channel
    )
    await handle_login(
        channel,
        interaction.followup.send,
        user_id=interaction.user.id,
        guild_id=guild_id,
        grant_server=grant_server,
        ephemeral=True,
    )


@bot.tree.command(name="restart", description="Restart the Discord bot in place")
async def codex_restart(interaction: discord.Interaction) -> None:
    if not await _require_server_admin(
        interaction,
        message="Only server administrators can restart the Discord bot.",
    ):
        return

    existing = bot._restart_task
    if existing is not None and not existing.done():
        await interaction.response.send_message(
            embed=_frontend_embed(
                "command:restart",
                "Restart already scheduled",
                "The Discord bot is already preparing to restart.",
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.orange(),
            ),
            ephemeral=True,
        )
        return

    await interaction.response.send_message(
        embed=_frontend_embed(
            "command:restart",
            "Restarting Theia",
            "The bot will reconnect in place shortly. Persisted Codex sessions "
            "and frontend settings will be reused.",
            channel=interaction.channel,
            user=interaction.user,
            color=discord.Color.blurple(),
        ),
        ephemeral=True,
    )
    bot._restart_task = asyncio.create_task(_restart_in_place())


@bot.tree.command(name="usage", description="Show Codex account usage")
async def codex_usage(interaction: discord.Interaction) -> None:
    if not await _require_login(interaction):
        return
    await interaction.response.defer(ephemeral=True)
    try:
        result = await bot.codex.usage()
        await interaction.followup.send(
            embed=_usage_embed(
                result, channel=interaction.channel, user=interaction.user
            ),
            ephemeral=True,
        )
    except (CodexAppServerError, OSError, discord.DiscordException) as exc:
        await _send_command_failure(interaction, "Usage unavailable", exc)


@bot.tree.command(name="credits", description="Show Codex credits and limits")
async def codex_credits(interaction: discord.Interaction) -> None:
    if not await _require_login(interaction):
        return
    await interaction.response.defer(ephemeral=True)
    try:
        result = await bot.codex.credits()
        await interaction.followup.send(
            embed=_credits_embed(
                result, channel=interaction.channel, user=interaction.user
            ),
            ephemeral=True,
        )
    except (CodexAppServerError, OSError, discord.DiscordException) as exc:
        await _send_command_failure(interaction, "Credits unavailable", exc)


@bot.tree.command(name="mode", description="Choose text or voice interaction mode")
@app_commands.describe(mode="The interaction mode to use")
@app_commands.choices(
    mode=[
        app_commands.Choice(name="text", value=TEXT_MODE),
        app_commands.Choice(name="voice", value=VOICE_MODE),
    ]
)
async def codex_mode(
    interaction: discord.Interaction, mode: app_commands.Choice[str]
) -> None:
    if not await _require_login(interaction):
        return
    await interaction.response.defer(ephemeral=True)
    selected = mode.value if isinstance(mode, app_commands.Choice) else str(mode)
    key = session_key(interaction.channel, interaction.user.id)
    if selected == VOICE_MODE:
        if not bot.codex.voice_mode_available or not bot.voice.available:
            reason = (
                "Voice mode requires configured STT_BASE_URL and TTS_BASE_URL."
                if not bot.codex.voice_mode_available
                else "Voice receive support is unavailable in this installation."
            )
            await interaction.followup.send(
                embed=_frontend_embed(
                    "command:mode",
                    "Voice unavailable",
                    reason,
                    channel=interaction.channel,
                    user=interaction.user,
                    color=discord.Color.orange(),
                ),
                ephemeral=True,
            )
            return
        voice_state = getattr(interaction.user, "voice", None)
        voice_channel = getattr(voice_state, "channel", None)
        if voice_channel is None or interaction.channel is None:
            await interaction.followup.send(
                embed=_frontend_embed(
                    "command:mode",
                    "Voice unavailable",
                    "Join a voice channel before selecting voice mode.",
                    channel=interaction.channel,
                    user=interaction.user,
                    color=discord.Color.orange(),
                ),
                ephemeral=True,
            )
            return
        try:
            await bot.codex.set_mode(key, VOICE_MODE)
            await bot.voice.start(
                session_key=key,
                user_id=interaction.user.id,
                voice_channel=voice_channel,
                text_channel=interaction.channel,
                allow_tools=_is_server_admin(interaction.user, interaction.channel),
                on_transcript=_handle_voice_transcript,
            )
        except (CodexAppServerError, VoiceModeError) as exc:
            with contextlib.suppress(Exception):
                await bot.codex.set_mode(key, DEFAULT_MODE)
            await _send_command_failure(interaction, "Voice unavailable", exc)
            return
        await interaction.followup.send(
            embed=_frontend_embed(
                "command:mode",
                "Voice mode enabled",
                "Listening in your voice channel. Text messages in this channel "
                "remain available, and responses will be spoken back.",
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.green(),
            ),
            ephemeral=True,
        )
        return

    await bot.voice.stop(key)
    try:
        await bot.codex.set_mode(key, TEXT_MODE)
    except CodexAppServerError as exc:
        await _send_command_failure(interaction, "Mode unavailable", exc)
        return
    await interaction.followup.send(
        embed=_frontend_embed(
            "command:mode",
            "Text mode enabled",
            "Voice listening is disabled for this Discord session.",
            channel=interaction.channel,
            user=interaction.user,
            color=discord.Color.green(),
        ),
        ephemeral=True,
    )


async def model_autocomplete(
    interaction: Any,  # pylint: disable=unused-argument
    current: str,
) -> list[app_commands.Choice[str]]:
    try:
        models = await bot.codex.available_models()
    except (CodexAppServerError, OSError) as exc:
        logger.debug(
            "Could not load models for autocomplete (error=%s)",
            type(exc).__name__,
        )
        return []
    query = current.casefold().strip()
    choices: list[app_commands.Choice[str]] = []
    for model in models:
        model_id = str(model.get("id") or "").strip()
        if not model_id:
            continue
        display = str(model.get("name") or model_id).strip()
        if (
            query
            and query not in model_id.casefold()
            and query not in display.casefold()
        ):
            continue
        label = display if display == model_id else f"{display} ({model_id})"
        choices.append(app_commands.Choice(name=label[:100], value=model_id))
    return choices[:25]


@bot.tree.command(name="model", description="Select the Codex model for this bot")
@app_commands.describe(model="The Codex model to use")
@app_commands.autocomplete(model=model_autocomplete)
async def codex_model(interaction: discord.Interaction, model: str) -> None:
    if not await _require_login(interaction):
        return
    await interaction.response.defer(ephemeral=True)
    try:
        await bot.codex.set_model(model)
    except (CodexAppServerError, OSError) as exc:
        await _send_command_failure(interaction, "Model unavailable", exc)
        return
    await interaction.followup.send(
        embed=_frontend_embed(
            "command:model",
            "Model selected",
            f"Codex will use `{model}` for new requests.",
            channel=interaction.channel,
            user=interaction.user,
            context={"model": model},
            color=discord.Color.green(),
        ),
        ephemeral=True,
    )


async def personality_autocomplete(
    interaction: Any,  # pylint: disable=unused-argument
    current: str,
) -> list[app_commands.Choice[str]]:
    query = current.casefold().strip()
    choices: list[app_commands.Choice[str]] = []
    if not query or "none".startswith(query):
        choices.append(
            app_commands.Choice(name="none (clear personality)", value="none")
        )
    for name in bot.codex.personality_names():
        if query and query not in name.casefold():
            continue
        choices.append(app_commands.Choice(name=name[:100], value=name))
    return choices[:25]


@bot.tree.command(name="personality", description="Manage Codex personality profiles")
@app_commands.describe(
    file="A Markdown or plain-text personality prompt",
    name="The profile name, or `none` to clear the active personality",
)
@app_commands.autocomplete(name=personality_autocomplete)
async def codex_personality(
    interaction: discord.Interaction,
    file: discord.Attachment | None = None,
    name: str | None = None,
) -> None:
    await bot.presence.touch()
    await interaction.response.defer(ephemeral=True)
    if file is None and name is None:
        await interaction.followup.send(
            embed=_frontend_embed(
                "command:personality",
                "Personality",
                "Use `/personality file:<markdown-or-text> name:<name>` to upload "
                "and activate a profile. Use `/personality name:<name>` to switch "
                "profiles, or `/personality name:none` to clear the active profile. "
                "A file must be paired with a name.",
                channel=interaction.channel,
                user=interaction.user,
            ),
            ephemeral=True,
        )
        return
    try:
        selected = await bot.codex.configure_personality(
            session_key(interaction.channel, interaction.user.id),
            name=name,
            attachment=file,
        )
    except CodexAppServerError as exc:
        await interaction.followup.send(
            embed=_frontend_embed(
                "command:personality",
                "Personality unavailable",
                _safe_error_reason(exc),
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.orange(),
            ),
            ephemeral=True,
        )
        return
    if selected is None:
        description = "The active Codex personality has been cleared."
        title = "Personality cleared"
    elif file is not None:
        description = f"Personality `{selected}` was uploaded and is now active."
        title = "Personality uploaded"
    else:
        description = f"Personality `{selected}` is now active."
        title = "Personality selected"
    await interaction.followup.send(
        embed=_frontend_embed(
            "command:personality",
            title,
            description,
            channel=interaction.channel,
            user=interaction.user,
            context={"personality": selected or "none"},
            color=discord.Color.green(),
        ),
        ephemeral=True,
    )


@bot.tree.command(name="approve", description="Approve the active Codex request")
async def codex_approve(interaction: discord.Interaction) -> None:
    if not await _require_login(interaction):
        return
    if not await _require_server_admin(interaction):
        return
    await interaction.response.defer(ephemeral=True)
    active = bot.codex.resolve_approval(interaction.user.id, True, interaction.channel)
    await interaction.followup.send(
        embed=_frontend_embed(
            "command:approve",
            "Approved" if active else "No pending approval",
            "The active request was approved."
            if active
            else "There is no pending approval request active.",
            channel=interaction.channel,
            user=interaction.user,
            color=discord.Color.green() if active else discord.Color.orange(),
        ),
        ephemeral=True,
    )


@bot.tree.command(name="deny", description="Deny the active Codex request")
async def codex_deny(interaction: discord.Interaction) -> None:
    if not await _require_login(interaction):
        return
    if not await _require_server_admin(interaction):
        return
    await interaction.response.defer(ephemeral=True)
    active = bot.codex.resolve_approval(interaction.user.id, False, interaction.channel)
    await interaction.followup.send(
        embed=_frontend_embed(
            "command:deny",
            "Denied" if active else "No pending approval",
            "The active request was denied."
            if active
            else "There is no pending approval request active.",
            channel=interaction.channel,
            user=interaction.user,
            color=discord.Color.red() if active else discord.Color.orange(),
        ),
        ephemeral=True,
    )


@bot.tree.command(name="stop", description="Stop your active Codex request")
async def codex_stop(interaction: discord.Interaction) -> None:
    if not await _require_login(interaction):
        return
    await interaction.response.defer()
    try:
        stopped = await bot.codex.interrupt(
            session_key(interaction.channel, interaction.user.id)
        )
        await interaction.followup.send(
            embed=_frontend_embed(
                "command:stop",
                "Stopped" if stopped else "No active request",
                "The active Codex request was stopped."
                if stopped
                else "There is no active Codex request.",
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.orange() if not stopped else discord.Color.green(),
            )
        )
    except (CodexAppServerError, OSError, discord.DiscordException) as exc:
        await _send_command_failure(interaction, "Stop unavailable", exc)


@bot.tree.command(name="undo", description="Undo your last Codex response")
async def codex_undo(interaction: discord.Interaction) -> None:
    if not await _require_login(interaction):
        return
    await interaction.response.defer(ephemeral=True)
    try:
        await bot.codex.undo(session_key(interaction.channel, interaction.user.id))
    except (CodexAppServerError, OSError) as exc:
        await _send_command_failure(interaction, "Undo unavailable", exc)
        return
    await interaction.followup.send(
        embed=_frontend_embed(
            "command:undo",
            "Last response undone",
            "The most recent Codex turn was removed from this conversation.",
            channel=interaction.channel,
            user=interaction.user,
            color=discord.Color.green(),
        ),
        ephemeral=True,
    )


@bot.tree.command(name="btw", description="Send a request to Codex")
@app_commands.describe(
    prompt="The request to send to Codex",
    file="An optional file to include with the request",
)
async def codex_btw(
    interaction: discord.Interaction,
    prompt: str,
    file: discord.Attachment | None = None,
) -> None:
    if not await _require_login(interaction):
        return
    await interaction.response.defer()
    source_channel = interaction.channel
    response_channel = await _maybe_create_response_thread(source_channel, prompt)
    if response_channel is None:
        response_channel = source_channel
    if _is_thread(response_channel):
        await _name_new_response_thread(response_channel, prompt)
        bot._participating_threads.add(response_channel.id)
        bot.codex.mark_thread_participating(response_channel.id)
    key = session_key(response_channel, interaction.user.id)
    context = await _channel_context(interaction.channel)
    send_kwargs: dict[str, Any] = {}
    if response_channel is not source_channel and _is_thread(response_channel):
        # Webhook follow-ups can target the newly-created thread while keeping
        # the interaction acknowledgement valid.
        send_kwargs["thread"] = response_channel
    await handle_request(
        interaction.followup.send,
        prompt,
        channel=response_channel,
        user_id=interaction.user.id,
        attachments=(file,) if file is not None else (),
        allow_tools=_is_server_admin(interaction.user, response_channel),
        context=context,
        request_id=f"interaction:{interaction.id}",
        speak_text=_voice_speak_callback(key),
        use_webhook_thread=True,
        **send_kwargs,
    )


async def skill_autocomplete(
    interaction: discord.Interaction,  # pylint: disable=unused-argument
    current: str,
) -> list[app_commands.Choice[str]]:
    try:
        if not bot.codex.skill_names():
            await bot.codex.refresh_skills(force=True)
    except (CodexAppServerError, OSError):
        return []
    query = current.casefold()
    choices: list[app_commands.Choice[str]] = []
    for name, display in bot.codex.skill_names():
        if query not in name.casefold() and query not in display.casefold():
            continue
        choices.append(app_commands.Choice(name=display[:100], value=name))
    return choices[:25]


@bot.tree.command(name="skill", description="Invoke an available Codex skill")
@app_commands.describe(skill_name="The skill to invoke")
@app_commands.autocomplete(skill_name=skill_autocomplete)
async def codex_skill(interaction: discord.Interaction, skill_name: str) -> None:
    if not await _require_login(interaction):
        return
    try:
        if not bot.codex.skill_names():
            await bot.codex.refresh_skills(force=True)
    except (CodexAppServerError, OSError) as exc:
        await interaction.response.send_message(
            embed=_frontend_embed(
                "command:skill",
                "Skill unavailable",
                _safe_error_reason(exc),
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.orange(),
            ),
            ephemeral=True,
        )
        return
    known = {name.casefold(): name for name, _ in bot.codex.skill_names()}
    canonical = known.get(skill_name.casefold())
    if canonical is None:
        await interaction.response.send_message(
            embed=_frontend_embed(
                "command:skill",
                "Skill unavailable",
                "That skill is not available to this Codex session.",
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.orange(),
            ),
            ephemeral=True,
        )
        return
    await interaction.response.defer()
    key = session_key(interaction.channel, interaction.user.id)
    context = await _channel_context(interaction.channel)
    await handle_request(
        interaction.followup.send,
        f"${canonical}",
        channel=interaction.channel,
        user_id=interaction.user.id,
        allow_tools=_is_server_admin(interaction.user, interaction.channel),
        context=context,
        request_id=f"interaction:{interaction.id}",
        speak_text=_voice_speak_callback(key),
    )


async def customization_target_autocomplete(
    interaction: Any,  # pylint: disable=unused-argument
    current: str,
) -> list[app_commands.Choice[str]]:
    return [
        app_commands.Choice(name=display[:100], value=value)
        for display, value in bot.customizations.targets(current)[:25]
    ]


async def customization_element_autocomplete(
    interaction: Any,  # pylint: disable=unused-argument
    current: str,
) -> list[app_commands.Choice[str]]:
    return [
        app_commands.Choice(name=display, value=value)
        for display, value in bot.customizations.elements(current)
    ]


@bot.tree.command(name="customize", description="Customize the Discord frontend")
@app_commands.describe(
    target="A command such as /usage, or a frontend label such as Thinking",
    element="The title, content, color, or label to customize",
    value="The value or template; use `default` to reset it",
)
@app_commands.autocomplete(
    target=customization_target_autocomplete,
    element=customization_element_autocomplete,
)
async def codex_customize(
    interaction: discord.Interaction,
    target: str | None = None,
    element: str | None = None,
    value: str | None = None,
) -> None:
    await bot.presence.touch()
    if getattr(interaction.guild, "id", None) is None:
        await interaction.response.send_message(
            embed=_frontend_embed(
                "command:customize",
                "Server only",
                "Frontend customization is available only inside a Discord server.",
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.orange(),
            ),
            ephemeral=True,
        )
        return
    if not await _require_server_admin(
        interaction,
        message="Only server administrators can customize the Discord frontend.",
    ):
        return
    if target is None and element is None and value is None:
        await interaction.response.send_message(
            embed=_frontend_embed(
                "command:customize",
                "Customize the Discord frontend",
                (
                    "Use `/customize target:<command-or-label> "
                    "element:<title|content|color|label> value:<value>`.\n\n"
                    "Targets can be commands such as `/usage` or labels such as "
                    "`Thinking`. Values support Markdown and placeholders: "
                    f"{bot.customizations.placeholder_help()}.\n\n"
                    "Use `default` as the value to reset a customization."
                ),
                channel=interaction.channel,
                user=interaction.user,
            ),
            ephemeral=True,
        )
        return
    if not target or not element or value is None:
        await interaction.response.send_message(
            embed=_frontend_embed(
                "command:customize",
                "Customization incomplete",
                "Provide target, element, and value together.",
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.orange(),
            ),
            ephemeral=True,
        )
        return
    if interaction.guild is None:
        await _send_command_failure(
            interaction,
            "Customization unavailable",
            RuntimeError("Customization requires a server."),
        )
        return
    guild_id = interaction.guild.id
    try:
        canonical, selected_element, reset = bot.customizations.set(
            guild_id, target, element, value
        )
    except CustomizationError as exc:
        await interaction.response.send_message(
            embed=_frontend_embed(
                "command:customize",
                "Customization unavailable",
                str(exc),
                channel=interaction.channel,
                user=interaction.user,
                color=discord.Color.orange(),
            ),
            ephemeral=True,
        )
        return
    target_name = display_target(canonical)
    description = (
        f"Reset the {selected_element} customization for {target_name}."
        if reset
        else f"Updated the {selected_element} customization for {target_name}."
    )
    await interaction.response.send_message(
        embed=_frontend_embed(
            "command:customize",
            "Customization reset" if reset else "Customization updated",
            description,
            channel=interaction.channel,
            user=interaction.user,
            color=discord.Color.green(),
        ),
        ephemeral=True,
    )


def _mention_prompt(content: str, bot_id: int) -> str:
    mention = re.compile(rf"<@!?{re.escape(str(bot_id))}>")
    return mention.sub("", content or "").strip()


def _thread_name(prompt: str) -> str:
    """Build a compact Discord thread name from its first real request."""
    summary = re.sub(r"\s+", " ", prompt).strip()
    return _truncate(f"Codex: {summary}", 100)


_THREAD_NOUN = (
    r"(?:(?:a|an|the)\s+)?"
    r"(?:(?:new|separate|dedicated|discord|private|public|discussion|response)\s+)*"
    r"thread\b"
)
_THREAD_REQUEST_PATTERN = re.compile(
    rf"(?:\b(?:create|make|start|open|begin|spawn)\s+"
    rf"(?:(?:me|us)\s+)?{_THREAD_NOUN})"
    rf"|(?:\b(?:make|turn|convert)\s+(?:this|it|the conversation)\s+"
    rf"(?:into|to|as|a)\s+{_THREAD_NOUN})"
    rf"|(?:\b(?:put|move|continue|take|reply|respond)\s+"
    rf"(?:this conversation|our conversation|the conversation|this|it)?\s*"
    rf"(?:in|into|to)\s+{_THREAD_NOUN})"
    r"|(?:\bthread\s+(?:this|it|the conversation)\b)",
    re.IGNORECASE,
)
_THREAD_REQUEST_NEGATION_PATTERN = re.compile(
    rf"(?:\b(?:don't|dont|do not|never|avoid)\s+"
    rf"(?:(?:create|make|start|open|begin)\s+)?{_THREAD_NOUN})"
    rf"|(?:\b(?:no|without)\s+{_THREAD_NOUN})"
    r"|(?:\bwithout\s+(?:creating|making|starting|opening)\s+"
    rf"{_THREAD_NOUN})",
    re.IGNORECASE,
)
_THREAD_INTENT_QUESTION_PATTERN = re.compile(
    r"\b(?:explain|describe|define|show|teach|tell)\b.{0,60}\b"
    r"(?:how to|how do i|how can i|what is|what are)\b.{0,40}\bthread\b"
    r"|\b(?:how do i|how can i|what is|what are|when|why)\b"
    r".{0,80}\bthread\b",
    re.IGNORECASE,
)
_THREAD_SEMANTIC_PATTERN = re.compile(
    r"\b(?:separate|split|branch)\s+(?:this|it|the conversation)\b"
    r"|\b(?:keep|continue|move|take|put)\b.{0,50}\b"
    r"(?:separate|apart|in its own space|in a dedicated space|"
    r"in a separate discussion|in a separate conversation)\b"
    r"|\b(?:give|make)\b.{0,50}\b(?:its own|a dedicated|a separate)\s+"
    r"(?:space|discussion|conversation|thread)\b",
    re.IGNORECASE,
)


def _user_requested_thread(prompt: str) -> bool:
    """Return whether the user's message has a high-confidence thread intent."""
    normalized = re.sub(r"\s+", " ", prompt or "").strip()
    if not normalized:
        return False
    if _THREAD_REQUEST_NEGATION_PATTERN.search(normalized):
        return False
    # Questions about the Discord feature are not requests to create one.
    if _THREAD_INTENT_QUESTION_PATTERN.search(normalized):
        return False
    if _THREAD_REQUEST_PATTERN.search(normalized):
        return True
    return bool(
        re.search(
            rf"\b(?:i|we)\s+(?:want|need|would like)\s+"
            rf"(?:you\s+to\s+)?{_THREAD_NOUN}",
            normalized,
            re.IGNORECASE,
        )
        or _THREAD_SEMANTIC_PATTERN.search(normalized)
    )


async def _maybe_create_response_thread(
    source: Any | None,
    prompt: str,
) -> Any | None:
    """Create an opt-in response thread, retaining the source channel on failure."""
    original_channel = getattr(source, "channel", None) or source
    if original_channel is None:
        return None
    if not (
        getattr(original_channel, "guild", None) is not None
        and not _is_thread(original_channel)
        and _env_bool_any(("THEIA_AUTO_THREAD", "DISCORD_AUTO_THREAD"), False)
        and _user_requested_thread(prompt)
    ):
        return original_channel

    # Message.create_thread creates a thread anchored to the request, which is
    # the Discord API used by the previous auto-thread path.
    create_thread = getattr(source, "create_thread", None)
    if not callable(create_thread):
        create_thread = getattr(original_channel, "create_thread", None)
    if not callable(create_thread):
        logger.info(
            "Discord response threads unavailable; continuing in source channel"
        )
        return original_channel
    try:
        create_thread_async = cast(Callable[..., Awaitable[Any]], create_thread)
        response_channel = await create_thread_async(
            name=_thread_name(prompt),
            auto_archive_duration=1440,
        )
    except (discord.DiscordException, TypeError, RuntimeError) as exc:
        logger.info(
            "Could not create a Discord response thread; continuing in source "
            "channel (error=%s)",
            type(exc).__name__,
        )
        return original_channel
    if response_channel is None or not callable(
        getattr(response_channel, "send", None)
    ):
        logger.info(
            "Discord response thread was not created; continuing in source channel"
        )
        return original_channel
    return response_channel


async def _name_new_response_thread(channel: Any, prompt: str) -> None:
    """Name an existing Discord thread when the bot first joins it."""
    if not _is_thread(channel) or bot.codex.is_participating_thread(channel.id):
        return
    edit = getattr(channel, "edit", None)
    if not callable(edit):
        return
    try:
        edit_async = cast(Callable[..., Awaitable[Any]], edit)
        await edit_async(name=_thread_name(prompt))
    except discord.DiscordException as exc:
        # Naming is cosmetic and must never prevent the actual response.
        logger.info(
            "Could not name a Discord response thread (error=%s)",
            type(exc).__name__,
        )


@bot.event
async def on_message(message: discord.Message) -> None:
    if bot.user is None:
        return
    if message.author.id == bot.user.id:
        return
    channel_id = _channel_id(message.channel)
    if channel_id is not None and isinstance(message.id, int):
        bot._known_channels[channel_id] = message.channel
        bot.codex.checkpoint_channel(channel_id, message.id)
    if message.author.bot and not _env_bool_any(
        ("THEIA_ALLOW_BOTS", "DISCORD_ALLOW_BOTS"), False
    ):
        return
    if not _should_respond_to_message(message):
        return
    await bot.presence.touch()
    mentioned = _message_has_mention(message)
    prompt = (
        _mention_prompt(message.content, bot.user.id)
        if mentioned
        else (message.content or "").strip()
    )
    if not prompt and not message.attachments:
        return
    if not prompt:
        prompt = "Please process the attached file(s)."
    guild_id = getattr(getattr(message.channel, "guild", None), "id", None)
    authenticated = bot.codex.is_authenticated(message.author.id, guild_id)
    if (
        not authenticated
        and guild_id is not None
        and _is_server_admin(message.author, message.channel)
        and bot.codex.is_authenticated(message.author.id)
    ):
        bot.codex.mark_server_authenticated(guild_id)
        authenticated = True
        logger.info("Granted cached Codex access to a server")
    if not authenticated:
        await message.channel.send(
            content=_subtext(
                "Login required. Please use `/login` before starting or "
                "controlling a Codex request."
            ),
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return
    response_channel = await _maybe_create_response_thread(
        message,
        prompt,
    )
    if response_channel is None:
        response_channel = message.channel
    if response_channel is None:
        return
    if _is_thread(response_channel):
        await _name_new_response_thread(response_channel, prompt)
        bot._participating_threads.add(response_channel.id)
        bot.codex.mark_thread_participating(response_channel.id)
    context = await _message_context(message)
    key = session_key(response_channel, message.author.id)
    send_kwargs: dict[str, Any] = {"mention_author": False}
    if response_channel is message.channel:
        send_kwargs["reference"] = message
    await handle_request(
        response_channel.send,
        prompt,
        channel=response_channel,
        user_id=message.author.id,
        attachments=message.attachments,
        allow_tools=_is_server_admin(message.author, response_channel),
        context=context,
        request_id=f"message:{message.id}",
        thread_source=message,
        speak_text=_voice_speak_callback(key),
        **send_kwargs,
    )


@bot.event
async def on_resumed() -> None:
    await bot.backfill_after_resume()


@bot.event
async def on_reaction_add(reaction: discord.Reaction, user: discord.abc.User) -> None:
    if user.bot:
        return
    paginator = _reaction_paginators.get(reaction.message.id)
    if paginator is not None:
        await paginator.handle_reaction(reaction, user)
