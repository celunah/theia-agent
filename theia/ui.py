"""Discord approval and structured-input views used by Codex interactions."""

import json
import secrets
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, cast

import discord

from .core import (
    _command_embed,
    _render_frontend_label,
    _safe_intermediate_text,
    _subtext,
    _truncate,
)

PromptSubmit = Callable[[discord.Interaction, str], Awaitable[None]]
StateChange = Callable[[Any], Awaitable[None]]
StateStop = Callable[[Any], None]
RESTART_RECOVERY_MESSAGE = (
    "This request was interrupted when Theia restarted. Please start it again."
)


def _component_custom_id(kind: str, token: str, action: str) -> str:
    """Build a bounded component ID that can be restored after a restart."""
    return f"theia:{kind}:{token}:{action}"


def _modal_custom_id(kind: str, user_id: int | None) -> str:
    """Build a namespaced modal ID for stale-interaction recovery."""
    owner = str(user_id) if user_id is not None else "0"
    return f"theia:modal:{kind}:{owner}:{secrets.token_hex(6)}"


class _PersistentViewMixin:
    """Provide IDs and lifecycle hooks for views restored after a restart."""

    def _init_persistence(
        self,
        kind: str,
        token: str | None,
        *,
        recovered: bool,
    ) -> None:
        self.persistence_kind = kind
        self.persistence_token = token or secrets.token_hex(8)
        self.recovered = recovered
        self._on_state_change: StateChange | None = None
        self._on_state_stop: StateStop | None = None

    def _custom_id(self, action: str) -> str:
        return _component_custom_id(
            self.persistence_kind,
            self.persistence_token,
            action,
        )

    def set_persistence_callbacks(
        self,
        *,
        on_state_change: StateChange | None = None,
        on_stop: StateStop | None = None,
    ) -> None:
        self._on_state_change = on_state_change
        self._on_state_stop = on_stop

    async def _notify_state_change(self) -> None:
        if self._on_state_change is not None:
            await self._on_state_change(self)

    def stop(self) -> None:
        already_finished = bool(cast(Any, self).is_finished())
        super().stop()  # type: ignore[misc]
        if not already_finished and self._on_state_stop is not None:
            self._on_state_stop(self)

    def persistence_data(self) -> dict[str, Any]:
        """Return JSON-safe state used to restore this view."""
        return {}

    def persistence_timeout(self) -> float | None:
        """Return the normal lifetime used by the persistence registry."""
        timeout = getattr(self, "timeout", None)
        return float(timeout) if timeout is not None else None


async def _check_interaction_owner(
    interaction: discord.Interaction, user_id: int | None
) -> bool:
    """Allow only the Discord user who owns an outstanding interaction."""
    if user_id is not None and interaction.user.id != user_id:
        await interaction.response.send_message(
            "Only the user who started this request can answer it.",
            ephemeral=True,
        )
        return False
    return True


class _DecisionView(_PersistentViewMixin, discord.ui.View):
    def __init__(
        self,
        user_id: int | None,
        choices: Iterable[tuple[str, str, discord.ButtonStyle]],
        *,
        timeout: float = 300,
        on_decision: Callable[[str, discord.abc.User], Awaitable[None]] | None = None,
        token: str | None = None,
        recovered: bool = False,
    ) -> None:
        self._init_persistence("decision", token, recovered=recovered)
        super().__init__(timeout=None if recovered else timeout)
        self.user_id = user_id
        self.value: str | None = None
        self.on_decision = on_decision
        self.choices = tuple(choices)
        for index, (label, value, style) in enumerate(self.choices):
            button = discord.ui.Button(
                label=label,
                style=style,
                custom_id=self._custom_id(f"decision-{index}"),
            )

            async def callback(
                interaction: discord.Interaction,
                *,
                decision: str = value,
            ) -> None:
                if not await self.interaction_check(interaction):
                    return
                self.value = decision
                for child in self.children:
                    if isinstance(child, discord.ui.Button):
                        child.disabled = True
                if self.recovered:
                    await interaction.response.edit_message(
                        content=RESTART_RECOVERY_MESSAGE,
                        embed=None,
                        view=self,
                    )
                else:
                    await interaction.response.edit_message(view=self)
                if self.on_decision is not None:
                    await self.on_decision(decision, interaction.user)
                await self._notify_state_change()
                self.stop()

            button.callback = callback
            self.add_item(button)

    def persistence_data(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "choices": [
                {"label": label, "value": value, "style": style.value}
                for label, value, style in self.choices
            ],
        }

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Allow a decision only from the user who owns the pending request."""
        return await _check_interaction_owner(interaction, self.user_id)

    async def on_timeout(self) -> None:
        """Disable decision controls when the approval window expires."""
        self.value = None
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
        self.stop()


class _DebugView(_PersistentViewMixin, discord.ui.View):
    """Owner-only control for a live administrator diagnostics message."""

    def __init__(
        self,
        user_id: int | None,
        *,
        channel: Any | None = None,
        customizer: Any | None = None,
        timeout: float = 900,
        guild_id: int | None = None,
        token: str | None = None,
        recovered: bool = False,
    ) -> None:
        self._init_persistence("debug", token, recovered=recovered)
        super().__init__(timeout=None if recovered else timeout)
        self.user_id = user_id
        self.guild_id = (
            guild_id
            if guild_id is not None
            else getattr(getattr(channel, "guild", None), "id", None)
        )
        self.customizer = customizer
        stop = discord.ui.Button(
            label=_render_frontend_label(
                customizer,
                self.guild_id,
                "label:debug_stop_updates",
                "Stop live updates",
            ),
            style=discord.ButtonStyle.secondary,
            custom_id=self._custom_id("stop"),
        )

        async def stop_callback(interaction: discord.Interaction) -> None:
            if not await self.interaction_check(interaction):
                return
            for child in self.children:
                if isinstance(child, discord.ui.Button):
                    child.disabled = True
            self.stop()
            if self.recovered:
                await interaction.response.edit_message(
                    content="Theia restarted, so live debug updates have ended.",
                    embed=None,
                    view=self,
                )
            else:
                await interaction.response.edit_message(view=self)

        stop.callback = stop_callback
        self.add_item(stop)

    def persistence_data(self) -> dict[str, Any]:
        return {"user_id": self.user_id, "guild_id": self.guild_id}

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Allow only the administrator who opened the diagnostic view to stop it."""
        return await _check_interaction_owner(interaction, self.user_id)

    async def on_timeout(self) -> None:
        """Stop refreshing after the bounded live-view lifetime."""
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
        self.stop()


class _PromptModal(discord.ui.Modal):
    """Collect one normal-language request for a follow-up interaction."""

    def __init__(
        self,
        user_id: int | None,
        *,
        on_submit: PromptSubmit,
        channel: Any | None = None,
        customizer: Any | None = None,
        title: str = "Follow up",
        placeholder: str = "Tell Codex what to do next.",
    ) -> None:
        guild_id = getattr(getattr(channel, "guild", None), "id", None)
        modal_title = _render_frontend_label(
            customizer,
            guild_id,
            "label:input_modal_title",
            title,
        )
        super().__init__(
            title=_truncate(modal_title, 45),
            custom_id=_modal_custom_id("prompt", user_id),
        )
        self.user_id = user_id
        self.on_prompt_submit = on_submit
        self.prompt = discord.ui.TextInput(
            label=_render_frontend_label(
                customizer,
                guild_id,
                "label:text_input_label",
                "Request",
            ),
            placeholder=_truncate(placeholder, 100),
            style=discord.TextStyle.paragraph,
            required=True,
            max_length=4000,
        )
        self.add_item(self.prompt)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        """Pass the owner's submitted request to the owning interaction flow."""
        if not await _check_interaction_owner(interaction, self.user_id):
            return
        await self.on_prompt_submit(interaction, str(self.prompt))


class _JsonModal(discord.ui.Modal):
    def __init__(
        self,
        view: "_FormView",
        user_id: int | None,
        *,
        title: str,
        prompt: str,
    ) -> None:
        modal_title = _render_frontend_label(
            view.customizer,
            view.guild_id,
            "label:input_modal_title",
            title,
        )
        super().__init__(
            title=_truncate(modal_title, 45),
            custom_id=_modal_custom_id("json", user_id),
        )
        self.view = view
        self.user_id = user_id
        self.value = discord.ui.TextInput(
            label=_render_frontend_label(
                view.customizer,
                view.guild_id,
                "label:json_response",
                "JSON response",
            ),
            placeholder=prompt[:100],
            style=discord.TextStyle.paragraph,
            required=True,
            max_length=4000,
        )
        self.add_item(self.value)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        """Parse the owner's JSON answer and resolve or decline the form."""
        if not await _check_interaction_owner(interaction, self.user_id):
            return
        try:
            parsed = json.loads(str(self.value))
        except json.JSONDecodeError:
            await interaction.response.send_message(
                "That is not valid JSON. The request was declined.",
                ephemeral=True,
            )
            self.view.value = None
        else:
            self.view.value = parsed
            await interaction.response.defer(ephemeral=True)
        self.view.stop()


class _FormView(_PersistentViewMixin, discord.ui.View):
    def __init__(
        self,
        user_id: int | None,
        *,
        prompt: str,
        channel: discord.abc.Messageable | None = None,
        customizer: Any | None = None,
        timeout: float = 300,
        guild_id: int | None = None,
        token: str | None = None,
        recovered: bool = False,
    ) -> None:
        self._init_persistence("form", token, recovered=recovered)
        super().__init__(timeout=None if recovered else timeout)
        self.user_id = user_id
        self.prompt = prompt
        self.guild_id = (
            guild_id
            if guild_id is not None
            else getattr(getattr(channel, "guild", None), "id", None)
        )
        self.customizer = customizer
        self.value: Any = None
        answer = discord.ui.Button(
            label=_render_frontend_label(
                customizer,
                self.guild_id,
                "label:answer_button",
                "Answer",
            ),
            style=discord.ButtonStyle.primary,
            custom_id=self._custom_id("answer"),
        )
        decline = discord.ui.Button(
            label=_render_frontend_label(
                customizer,
                self.guild_id,
                "label:decline_button",
                "Decline",
            ),
            style=discord.ButtonStyle.secondary,
            custom_id=self._custom_id("decline"),
        )

        async def answer_callback(interaction: discord.Interaction) -> None:
            if await self.interaction_check(interaction):
                if self.recovered:
                    await interaction.response.edit_message(
                        content=RESTART_RECOVERY_MESSAGE,
                        embed=None,
                        view=self,
                    )
                    self.stop()
                    return
                await interaction.response.send_modal(
                    _JsonModal(
                        self,
                        self.user_id,
                        title="Codex input",
                        prompt=self.prompt,
                    )
                )

        async def decline_callback(interaction: discord.Interaction) -> None:
            if await self.interaction_check(interaction):
                self.value = None
                if self.recovered:
                    await interaction.response.edit_message(
                        content=RESTART_RECOVERY_MESSAGE,
                        embed=None,
                        view=self,
                    )
                else:
                    await interaction.response.edit_message(view=self)
                self.stop()

        answer.callback = answer_callback
        decline.callback = decline_callback
        self.add_item(answer)
        self.add_item(decline)

    def persistence_data(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "prompt": self.prompt,
            "guild_id": self.guild_id,
        }

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Allow form actions only from the user who owns the pending request."""
        return await _check_interaction_owner(interaction, self.user_id)

    async def on_timeout(self) -> None:
        """Stop the form when its Discord interaction window expires."""
        self.stop()


class _UserInputView(_PersistentViewMixin, discord.ui.View):
    def __init__(
        self,
        user_id: int | None,
        questions: list[dict[str, Any]],
        *,
        channel: discord.abc.Messageable | None = None,
        customizer: Any | None = None,
        timeout: float = 300,
        guild_id: int | None = None,
        token: str | None = None,
        recovered: bool = False,
        question_index: int = 0,
        answers: dict[str, dict[str, list[str]]] | None = None,
    ) -> None:
        self._init_persistence("user-input", token, recovered=recovered)
        super().__init__(timeout=None if recovered else timeout)
        self.user_id = user_id
        self.questions = questions
        self.guild_id = (
            guild_id
            if guild_id is not None
            else getattr(getattr(channel, "guild", None), "id", None)
        )
        self.customizer = customizer
        self.value: dict[str, Any] | None = None
        self.question_index = max(0, min(question_index, max(0, len(questions) - 1)))
        self._answers = dict(answers or {})
        self._build_question_items()

    @property
    def current_question(self) -> dict[str, Any]:
        """Return the unanswered question currently shown by the input view."""
        if not self.questions:
            return {}
        return self.questions[self.question_index]

    def _question_prompt(self) -> str:
        question = self.current_question
        header = str(question.get("header") or question.get("id") or "Question")
        prompt = str(question.get("question") or "Please provide an answer.")
        return (
            _safe_intermediate_text(f"**{header}:** {prompt}", 1800)
            or "Codex needs your input."
        )

    def message_kwargs(self, *, for_edit: bool = False) -> dict[str, Any]:
        """Render the current question for the initial send or next step."""
        question = self.current_question
        message: dict[str, Any] = {"view": self}
        options = question.get("options") or []
        if options:
            if for_edit:
                message["content"] = None
            message["embed"] = _command_embed(
                "Choose an option",
                self._question_prompt(),
                color=discord.Color.blurple(),
                target="label:choose_option",
                guild_id=self.guild_id,
                customizer=self.customizer,
                context={"question": self._question_prompt()},
            )
        else:
            if for_edit:
                message["embed"] = None
            message["content"] = _subtext(self._question_prompt())
        return message

    def _build_question_items(self) -> None:
        self.clear_items()
        question = self.current_question
        options = [
            option
            for option in (question.get("options") or [])
            if isinstance(option, dict)
        ]
        for option in options[:4]:
            option_label = option.get("label")
            label = (
                str(option_label)
                if option_label
                else _render_frontend_label(
                    self.customizer,
                    self.guild_id,
                    "label:choose_button",
                    "Choose",
                )
            )
            label = _truncate(label, 80)
            button = discord.ui.Button(
                label=label,
                style=discord.ButtonStyle.primary,
                custom_id=self._custom_id(
                    f"option-{self.question_index}-{len(self.children)}"
                ),
            )

            async def callback(
                interaction: discord.Interaction,
                *,
                answer: str = str(option.get("label") or ""),
            ) -> None:
                if await self.interaction_check(interaction):
                    if self.recovered:
                        await interaction.response.edit_message(
                            content=RESTART_RECOVERY_MESSAGE,
                            embed=None,
                            view=self,
                        )
                        self.stop()
                        return
                    complete = self._record_answer(answer)
                    if complete:
                        await interaction.response.edit_message(view=self)
                        self.stop()
                    else:
                        await interaction.response.edit_message(
                            **self.message_kwargs(for_edit=True)
                        )
                    await self._notify_state_change()

            button.callback = callback
            self.add_item(button)

        if question.get("isOther") or not options:
            other = discord.ui.Button(
                label=_render_frontend_label(
                    self.customizer,
                    self.guild_id,
                    "label:other_button" if options else "label:answer_button",
                    "Other" if options else "Answer",
                ),
                style=discord.ButtonStyle.secondary,
                custom_id=self._custom_id(f"other-{self.question_index}"),
            )

            async def other_callback(interaction: discord.Interaction) -> None:
                if await self.interaction_check(interaction):
                    if self.recovered:
                        await interaction.response.edit_message(
                            content=RESTART_RECOVERY_MESSAGE,
                            embed=None,
                            view=self,
                        )
                        self.stop()
                        return
                    await interaction.response.send_modal(
                        _TextModal(self, self.user_id, question),
                    )

            other.callback = other_callback
            self.add_item(other)

    def persistence_data(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "questions": self.questions,
            "guild_id": self.guild_id,
            "question_index": self.question_index,
            "answers": self._answers,
        }

    def _record_answer(self, answer: str) -> bool:
        question_id = str(self.current_question.get("id") or self.question_index)
        self._answers[question_id] = {"answers": [answer]}
        if self.question_index + 1 < len(self.questions):
            self.question_index += 1
            self._build_question_items()
            return False
        self.value = {"answers": dict(self._answers)}
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
        return True

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Allow answers only from the user who owns the pending request."""
        return await _check_interaction_owner(interaction, self.user_id)

    async def on_timeout(self) -> None:
        """Stop the question sequence when its Discord interaction window expires."""
        self.stop()


class _TextModal(discord.ui.Modal):
    def __init__(
        self,
        view: _UserInputView,
        user_id: int | None,
        question: dict[str, Any],
    ) -> None:
        title = str(question.get("header") or "Codex input")
        modal_title = _render_frontend_label(
            view.customizer,
            view.guild_id,
            "label:input_modal_title",
            title,
        )
        super().__init__(
            title=_truncate(modal_title, 45),
            custom_id=_modal_custom_id("text", user_id),
        )
        self.view = view
        self.user_id = user_id
        self.question = question
        input_label = str(question.get("header") or "Answer")
        self.answer = discord.ui.TextInput(
            label=_render_frontend_label(
                view.customizer,
                view.guild_id,
                "label:text_input_label",
                input_label,
            ),
            placeholder=_truncate(question.get("question") or "Answer", 100),
            style=discord.TextStyle.paragraph,
            required=True,
            max_length=4000,
        )
        self.add_item(self.answer)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        """Record the owner's text answer and advance or complete the sequence."""
        if not await _check_interaction_owner(interaction, self.user_id):
            return
        complete = self.view._record_answer(str(self.answer))
        if complete:
            await interaction.response.edit_message(view=self.view)
            self.view.stop()
        else:
            await interaction.response.edit_message(
                **self.view.message_kwargs(for_edit=True)
            )
        await self.view._notify_state_change()
