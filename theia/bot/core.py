"""Discord gateway, command, and message-routing integration for Theia."""

# Compatibility imports intentionally preserve the historical public surface.
# pylint: disable=cyclic-import,unused-import

import asyncio
import contextlib
import os  # noqa: F401 - compatibility patch point used by restart tests
import re
import subprocess
import sys  # noqa: F401 - compatibility patch point used by restart tests
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import discord
from discord import app_commands

from ..server.core import CodexAppServerError
from ..core import (
    DEFAULT_MODE,
    DEFAULT_PERSONALITY_SCOPE,
    PERSONALITY_SCOPES,
    TEXT_MODE,
    VOICE_MODE,
    _codex_logger,
    _safe_error_reason,
    _subtext,
    _theia_revision,  # noqa: F401 - retained patch/import compatibility
)
from ..customization import (
    CustomizationError,
    display_target,
)
from ..delivery import (
    _ImageResultView,
    _reaction_paginators,
)
from ..voice import VoiceModeError
from ..server.threads import thread_name, user_requested_thread
from ..ui import (
    _DebugView,
    _PromptModal,
)
from .support import (
    BARE_MENTION_PROMPT,
    DEBUG_VIEW_TIMEOUT,
    STALE_INTERACTION_FALLBACK_DELAY,  # noqa: F401 - compatibility export
    _GeneratedImageAttachment,
    _channel_context,
    _channel_id,
    _env_bool_any,
    _frontend_embed,
    _interaction_allows_tools,
    _interaction_request_sender,
    _is_guild_install,
    _is_server_admin,
    _is_thread,
    _is_user_only_install,
    _message_context,
    _message_has_mention,
    _require_login,
    _require_server_admin,
    _send_command_failure,
    _should_respond_to_message,
    _typing_indicator,  # noqa: F401 - compatibility export used by main.py
    _PersistentViewStore,  # noqa: F401 - compatibility export used by tests
    _user_installable_command,
    _voice_session_allows_tools,  # noqa: F401 - compatibility used by voice helper
    configure_bot,
    handle_login,
    handle_request,
    session_key,
)
from .runtime import TheiaBot
from .embeds import (
    _about_embed,
    _credits_embed,
    _debug_embed,
    _login_required_embed,  # noqa: F401 - compatibility export used by main.py
    _personality_summary_embed,
    _usage_embed,
)
from .voice import (
    _handle_voice_transcript,
    _restart_in_place,
    _voice_speak_callback,
)

logger = _codex_logger()

# Kept for callers that imported the pre-Theia harness name.
CodexBot = TheiaBot


bot = TheiaBot()
configure_bot(bot)


@_user_installable_command
@bot.tree.command(name="login", description="Authenticate this Discord user with Codex")
async def codex_login(interaction: discord.Interaction) -> None:
    """Authenticate the invoking Discord user, optionally authorizing their server."""
    await interaction.response.defer(ephemeral=True)
    channel = interaction.channel or interaction.user
    guild_id = getattr(interaction.guild, "id", None)
    grant_server = (
        _is_guild_install(interaction)
        and guild_id is not None
        and _is_server_admin(interaction.user, interaction.channel)
    )
    complete_sender = (
        _interaction_request_sender(interaction)
        if _is_user_only_install(interaction)
        else None
    )
    await handle_login(
        channel,
        interaction.followup.send,
        user_id=interaction.user.id,
        guild_id=guild_id,
        grant_server=grant_server,
        ephemeral=True,
        on_complete_send=complete_sender,
    )


@_user_installable_command
@bot.tree.command(name="restart", description="Restart the Discord bot in place")
async def codex_restart(interaction: discord.Interaction) -> None:
    """Schedule an administrator-only in-place bot restart."""
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
    )
    bot._restart_task = asyncio.create_task(_restart_in_place())


@_user_installable_command
@bot.tree.command(name="usage", description="Show local conversation usage")
async def codex_usage(interaction: discord.Interaction) -> None:
    """Display Theia's locally tracked conversation usage privately."""
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


@_user_installable_command
@bot.tree.command(name="credits", description="Show Codex credits and limits")
async def codex_credits(interaction: discord.Interaction) -> None:
    """Display the authenticated Codex account's rate limits privately."""
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


@_user_installable_command
@bot.tree.command(name="about", description="Show Codex and session details")
async def codex_about(interaction: discord.Interaction) -> None:
    """Display the current Theia, Codex, account, and session details privately."""
    await bot.presence.touch()
    await interaction.response.defer(ephemeral=True)
    account_result: dict[str, Any] = {}
    try:
        account_result = await bot.codex.account_details()
    except (CodexAppServerError, OSError):
        logger.debug("Could not fetch Codex account details for About")
    try:
        cli_version = bot.codex.codex_cli_version()
    except (OSError, subprocess.SubprocessError):
        cli_version = None
    key = session_key(interaction.channel, interaction.user.id)
    account = account_result.get("account")
    await interaction.followup.send(
        embed=_about_embed(
            account=account if isinstance(account, dict) else None,
            cli_version=cli_version,
            mode=bot.codex.mode(key),
            personality=bot.codex.active_personality(key),
            channel=interaction.channel,
            user=interaction.user,
        ),
        ephemeral=True,
    )


@_user_installable_command
@bot.tree.command(name="debug", description="Show live runtime diagnostics")
async def codex_debug(interaction: discord.Interaction) -> None:
    """Show sanitized live runtime diagnostics to the invoking administrator."""
    if not await _require_server_admin(
        interaction,
        message="Only Theia administrators can view debug state.",
    ):
        return
    key = session_key(interaction.channel, interaction.user.id)
    view = _DebugView(
        interaction.user.id,
        channel=interaction.channel,
        customizer=bot.customizations,
        timeout=DEBUG_VIEW_TIMEOUT,
    )
    await interaction.response.send_message(
        embed=_debug_embed(
            bot.codex.debug_state(key),
            channel=interaction.channel,
            user=interaction.user,
        ),
        view=view,
        ephemeral=True,
    )
    try:
        message = await interaction.original_response()
    except (discord.DiscordException, AttributeError) as exc:
        logger.debug(
            "Could not attach live debug refresh (error=%s)", type(exc).__name__
        )
        return
    if message is not None:
        await bot.register_view(view, message)
        bot.schedule_debug_refresh(
            message,
            view,
            session_key_value=key,
            channel=interaction.channel,
            user=interaction.user,
        )


@_user_installable_command
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
    """Switch the current Discord session between text and optional voice mode."""
    if not await _require_login(interaction):
        return
    await interaction.response.defer(ephemeral=True)
    selected = mode.value if isinstance(mode, app_commands.Choice) else str(mode)
    key = session_key(interaction.channel, interaction.user.id)
    if selected == VOICE_MODE:
        if _is_user_only_install(interaction):
            await interaction.followup.send(
                embed=_frontend_embed(
                    "command:mode",
                    "Voice unavailable",
                    "Voice mode requires Theia to be installed in the server.",
                    channel=interaction.channel,
                    user=interaction.user,
                    color=discord.Color.orange(),
                ),
                ephemeral=True,
            )
            return
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
    """Offer account-backed Codex model choices for the slash command."""
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


@_user_installable_command
@bot.tree.command(name="model", description="Select the Codex model for this bot")
@app_commands.describe(model="The Codex model to use")
@app_commands.autocomplete(model=model_autocomplete)
async def codex_model(interaction: discord.Interaction, model: str) -> None:
    """Select the Codex model used for new requests in this installation."""
    if not await _require_login(interaction):
        return
    await interaction.response.defer()
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
    )


async def personality_autocomplete(
    interaction: Any,  # pylint: disable=unused-argument
    current: str,
) -> list[app_commands.Choice[str]]:
    """Offer stored personality profiles and the option to clear one."""
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


@_user_installable_command
@bot.tree.command(name="personality", description="Manage Codex personality profiles")
@app_commands.describe(
    file="A Markdown or plain-text personality prompt",
    name="The profile name, or `none` to clear the active personality",
    scope="Who should use this personality: me, server, or everyone",
)
@app_commands.choices(
    scope=[app_commands.Choice(name=scope, value=scope) for scope in PERSONALITY_SCOPES]
)
@app_commands.autocomplete(name=personality_autocomplete)
async def codex_personality(
    interaction: discord.Interaction,
    file: discord.Attachment | None = None,
    name: str | None = None,
    scope: app_commands.Choice[str] | None = None,
) -> None:
    """Upload, select, or clear a personality at the requested scope."""
    await bot.presence.touch()
    await interaction.response.defer(ephemeral=True)
    selected_scope = (
        scope.value
        if isinstance(scope, app_commands.Choice)
        else str(scope or DEFAULT_PERSONALITY_SCOPE)
    )
    if file is None and name is None:
        key = session_key(interaction.channel, interaction.user.id)
        try:
            summary = await bot.codex.personality_summary(key)
            mood = bot.codex.mood_state(key)
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
        await interaction.followup.send(
            embed=_personality_summary_embed(
                summary,
                mood,
                bot.rich_presence.current_line,
                channel=interaction.channel,
                user=interaction.user,
            ),
            ephemeral=True,
        )
        return
    if selected_scope != "me" and not await _require_server_admin(
        interaction,
        message="Only server administrators can change server or everyone personalities.",
    ):
        return
    try:
        selected = await bot.codex.configure_personality(
            session_key(interaction.channel, interaction.user.id),
            name=name,
            attachment=file,
            scope=selected_scope,
            actor_user_id=interaction.user.id,
            guild_id=getattr(getattr(interaction, "guild", None), "id", None),
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
        description = (
            f"The active Codex personality has been cleared for `{selected_scope}`."
        )
        title = "Personality cleared"
    elif file is not None:
        description = (
            f"Personality `{selected}` was uploaded and is now active for "
            f"`{selected_scope}`."
        )
        title = "Personality uploaded"
    else:
        description = f"Personality `{selected}` is now active for `{selected_scope}`."
        title = "Personality selected"
    await interaction.followup.send(
        embed=_frontend_embed(
            "command:personality",
            title,
            description,
            channel=interaction.channel,
            user=interaction.user,
            context={
                "personality": selected or "none",
                "personality_scope": selected_scope,
            },
            color=discord.Color.green(),
        ),
        ephemeral=True,
    )


@_user_installable_command
@bot.tree.command(name="approve", description="Approve the active Codex request")
async def codex_approve(interaction: discord.Interaction) -> None:
    """Approve the invoking administrator's pending Codex request."""
    if not await _require_login(interaction):
        return
    if not await _require_server_admin(interaction):
        return
    await interaction.response.defer(ephemeral=True)
    active = bot.codex.resolve_approval(
        interaction.user.id,
        True,
        interaction.channel,
        current_user=interaction.user,
    )
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


@_user_installable_command
@bot.tree.command(name="deny", description="Deny the active Codex request")
async def codex_deny(interaction: discord.Interaction) -> None:
    """Deny the invoking administrator's pending Codex request."""
    if not await _require_login(interaction):
        return
    if not await _require_server_admin(interaction):
        return
    await interaction.response.defer(ephemeral=True)
    active = bot.codex.resolve_approval(
        interaction.user.id,
        False,
        interaction.channel,
        current_user=interaction.user,
    )
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


@_user_installable_command
@bot.tree.command(name="stop", description="Stop your active Codex request")
async def codex_stop(interaction: discord.Interaction) -> None:
    """Interrupt the invoking user's active Codex request."""
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


@_user_installable_command
@bot.tree.command(name="undo", description="Undo your last Codex response")
async def codex_undo(interaction: discord.Interaction) -> None:
    """Roll back the most recent completed Codex turn for this session."""
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


@_user_installable_command
@bot.tree.command(name="btw", description="Send a request to Codex")
@app_commands.describe(
    prompt="The request to send to Codex, or leave blank to open the request modal",
    file="An optional file to include with the request",
)
async def codex_btw(
    interaction: discord.Interaction,
    prompt: str | None = None,
    file: discord.Attachment | None = None,
) -> None:
    """Send a prompt and optional attachment through the current Discord session."""
    if not await _require_login(interaction):
        return
    prompt_value = (prompt or "").strip()
    if not prompt_value:

        async def submit_modal_prompt(
            modal_interaction: discord.Interaction,
            modal_prompt: str,
        ) -> None:
            if not await _require_login(modal_interaction):
                return
            await modal_interaction.response.defer()
            bot.schedule_request(
                _run_btw_request(modal_interaction, modal_prompt, file)
            )

        await interaction.response.send_modal(
            _PromptModal(
                interaction.user.id,
                on_submit=submit_modal_prompt,
                channel=interaction.channel,
                customizer=bot.customizations,
                title="Send a request",
                placeholder="Tell Codex what to do.",
            )
        )
        return
    await interaction.response.defer()
    bot.schedule_request(_run_btw_request(interaction, prompt_value, file))


async def _run_btw_request(
    interaction: discord.Interaction,
    prompt: str,
    file: discord.Attachment | None,
) -> None:
    """Prepare and run a slash-command request outside Discord's callback task."""
    try:
        source_channel = interaction.channel
        user_only = _is_user_only_install(interaction)
        response_channel = (
            source_channel
            if user_only
            else await _maybe_create_response_thread(source_channel, prompt)
        )
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
        request_sender = (
            _interaction_request_sender(interaction)
            if user_only
            else interaction.followup.send
        )
        await handle_request(
            request_sender,
            prompt,
            channel=response_channel,
            user_id=interaction.user.id,
            user=interaction.user,
            attachments=(file,) if file is not None else (),
            allow_tools=_interaction_allows_tools(interaction),
            context=context,
            request_id=f"interaction:{interaction.id}",
            speak_text=_voice_speak_callback(key),
            use_webhook_thread=True,
            interaction_sender=request_sender if user_only else None,
            allow_discord_tools=not user_only,
            **send_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - a deferred interaction must resolve
        logger.error(
            "Background /btw request failed (error=%s)",
            type(exc).__name__,
        )
        with contextlib.suppress(Exception):
            await _send_command_failure(interaction, "Request unavailable", exc)


async def _run_image_follow_up(
    interaction: discord.Interaction,
    prompt: str,
    image_paths: tuple[Path, ...],
    *,
    channel: Any | None,
    image_view: _ImageResultView,
    image_message: Any | None,
) -> None:
    """Acknowledge an image control and run its follow-up in the same session."""
    if not await _require_login(interaction):
        return
    await interaction.response.defer()
    bot.schedule_request(
        _process_image_follow_up(
            interaction,
            prompt,
            image_paths,
            channel=channel,
            image_view=image_view,
            image_message=image_message,
        )
    )


async def _process_image_follow_up(
    interaction: discord.Interaction,
    prompt: str,
    image_paths: tuple[Path, ...],
    *,
    channel: Any | None,
    image_view: _ImageResultView,
    image_message: Any | None,
) -> None:
    """Run one image action after its Discord interaction is acknowledged."""
    request_channel = channel or interaction.channel
    user_only = _is_user_only_install(interaction)
    request_sender = interaction.followup.send
    if image_message is None:
        image_message = getattr(image_view, "message", None)
    if image_message is None:
        message_id = getattr(image_view, "message_id", None)
        fetch_message = getattr(request_channel, "fetch_message", None)
        if isinstance(message_id, int) and callable(fetch_message):
            with contextlib.suppress(discord.DiscordException, TypeError):
                fetch = cast(Callable[[int], Awaitable[Any]], fetch_message)
                image_message = await fetch(message_id)
    request_prompt = prompt.strip()
    if not request_prompt:
        request_prompt = "Continue working with the attached generated image."
    key = session_key(request_channel, interaction.user.id)
    try:
        await handle_request(
            request_sender,
            request_prompt,
            channel=request_channel,
            user_id=interaction.user.id,
            user=interaction.user,
            attachments=tuple(_GeneratedImageAttachment(path) for path in image_paths),
            allow_tools=_interaction_allows_tools(interaction),
            context=await _channel_context(request_channel),
            request_id=f"image:{interaction.id}",
            speak_text=_voice_speak_callback(key),
            use_webhook_thread=True,
            interaction_sender=(
                _interaction_request_sender(interaction) if user_only else None
            ),
            allow_discord_tools=not user_only,
            image_message=image_message,
            image_view=image_view,
            existing_image_paths=image_paths,
        )
    except Exception as exc:  # noqa: BLE001 - an image action must not go silent
        logger.error(
            "Background image action failed (error=%s)",
            type(exc).__name__,
        )
        with contextlib.suppress(Exception):
            await _send_command_failure(interaction, "Image action unavailable", exc)


async def skill_autocomplete(
    interaction: discord.Interaction,  # pylint: disable=unused-argument
    current: str,
) -> list[app_commands.Choice[str]]:
    """Offer enabled Codex skills matching the user's autocomplete query."""
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


@_user_installable_command
@bot.tree.command(name="skill", description="Invoke an available Codex skill")
@app_commands.describe(skill_name="The skill to invoke")
@app_commands.autocomplete(skill_name=skill_autocomplete)
async def codex_skill(interaction: discord.Interaction, skill_name: str) -> None:
    """Invoke one enabled Codex skill as a normal session request."""
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
    bot.schedule_request(
        _run_skill_request(
            interaction,
            canonical,
            key,
        )
    )


async def _run_skill_request(
    interaction: discord.Interaction,
    skill_name: str,
    session_key_value: str,
) -> None:
    """Run a skill invocation outside the slash-command callback task."""
    try:
        context = await _channel_context(interaction.channel)
        user_only = _is_user_only_install(interaction)
        request_sender = (
            _interaction_request_sender(interaction)
            if user_only
            else interaction.followup.send
        )
        await handle_request(
            request_sender,
            f"${skill_name}",
            channel=interaction.channel,
            user_id=interaction.user.id,
            user=interaction.user,
            allow_tools=_interaction_allows_tools(interaction),
            context=context,
            request_id=f"interaction:{interaction.id}",
            speak_text=_voice_speak_callback(session_key_value),
            interaction_sender=request_sender if user_only else None,
            allow_discord_tools=not user_only,
        )
    except Exception as exc:  # noqa: BLE001 - a deferred interaction must resolve
        logger.error(
            "Background /skill request failed (error=%s)",
            type(exc).__name__,
        )
        with contextlib.suppress(Exception):
            await _send_command_failure(interaction, "Skill unavailable", exc)


async def customization_target_autocomplete(
    interaction: Any,  # pylint: disable=unused-argument
    current: str,
) -> list[app_commands.Choice[str]]:
    """Offer command and frontend-label targets for administrator customization."""
    return [
        app_commands.Choice(name=display[:100], value=value)
        for display, value in bot.customizations.targets(current)[:25]
    ]


async def customization_element_autocomplete(
    interaction: Any,  # pylint: disable=unused-argument
    current: str,
) -> list[app_commands.Choice[str]]:
    """Offer valid presentation elements for administrator customization."""
    return [
        app_commands.Choice(name=display, value=value)
        for display, value in bot.customizations.elements(current)
    ]


@_user_installable_command
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
    """Read or update server-scoped Discord presentation preferences."""
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
    )


def _mention_prompt(content: str, bot_id: int) -> str:
    mention = re.compile(rf"<@!?{re.escape(str(bot_id))}>")
    return mention.sub("", content or "").strip()


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
        and _env_bool_any(("THEIA_AUTO_THREAD", "DISCORD_AUTO_THREAD"), True)
        and user_requested_thread(prompt)
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
            name=thread_name(prompt),
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
        await edit_async(name=thread_name(prompt))
    except discord.DiscordException as exc:
        # Naming is cosmetic and must never prevent the actual response.
        logger.info(
            "Could not name a Discord response thread (error=%s)",
            type(exc).__name__,
        )


@bot.event
async def on_message(message: discord.Message) -> None:
    """Filter Discord messages, recover access, and route eligible requests."""
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
    if not prompt:
        if mentioned:
            prompt = BARE_MENTION_PROMPT
        elif message.attachments:
            prompt = "Please process the attached file(s)."
        else:
            return
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
    bot.schedule_request(_run_message_request(message, prompt))


async def _run_message_request(message: discord.Message, prompt: str) -> None:
    """Prepare and run a message request outside the gateway event callback."""
    response_channel = await _maybe_create_response_thread(message, prompt)
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
        user=message.author,
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
    """Backfill bounded channel history after a Discord gateway resume."""
    await bot.backfill_after_resume()


@bot.event
async def on_reaction_add(reaction: discord.Reaction, user: discord.abc.User) -> None:
    """Route pagination reactions to the response view that owns the message."""
    if user.bot:
        return
    paginator = _reaction_paginators.get(reaction.message.id)
    if paginator is not None:
        await paginator.handle_reaction(reaction, user)
