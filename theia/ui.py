import json
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

import discord

from .core import (
    _command_embed,
    _render_frontend_label,
    _safe_intermediate_text,
    _subtext,
    _truncate,
)


async def _check_interaction_owner(
    interaction: discord.Interaction, user_id: int | None
) -> bool:
    if user_id is not None and interaction.user.id != user_id:
        await interaction.response.send_message(
            "Only the user who started this request can answer it.",
            ephemeral=True,
        )
        return False
    return True


class _DecisionView(discord.ui.View):
    def __init__(
        self,
        user_id: int | None,
        choices: Iterable[tuple[str, str, discord.ButtonStyle]],
        *,
        timeout: float = 300,
        on_decision: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self.user_id = user_id
        self.value: str | None = None
        self.on_decision = on_decision
        for label, value, style in choices:
            button = discord.ui.Button(label=label, style=style)

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
                await interaction.response.edit_message(view=self)
                if self.on_decision is not None:
                    await self.on_decision(decision)
                self.stop()

            button.callback = callback
            self.add_item(button)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return await _check_interaction_owner(interaction, self.user_id)

    async def on_timeout(self) -> None:
        self.value = None
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
        self.stop()


class _JsonModal(discord.ui.Modal):
    def __init__(
        self,
        view: "_FormView",
        user_id: int | None,
        *,
        title: str,
        prompt: str,
    ) -> None:
        super().__init__(title=_truncate(title, 45))
        self.view = view
        self.user_id = user_id
        self.value = discord.ui.TextInput(
            label="JSON response",
            placeholder=prompt[:100],
            style=discord.TextStyle.paragraph,
            required=True,
            max_length=4000,
        )
        self.add_item(self.value)

    async def on_submit(self, interaction: discord.Interaction) -> None:
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


class _FormView(discord.ui.View):
    def __init__(
        self,
        user_id: int | None,
        *,
        prompt: str,
        channel: discord.abc.Messageable | None = None,
        customizer: Any | None = None,
        timeout: float = 300,
    ) -> None:
        super().__init__(timeout=timeout)
        self.user_id = user_id
        self.prompt = prompt
        self.guild_id = getattr(getattr(channel, "guild", None), "id", None)
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
        )
        decline = discord.ui.Button(
            label=_render_frontend_label(
                customizer,
                self.guild_id,
                "label:decline_button",
                "Decline",
            ),
            style=discord.ButtonStyle.secondary,
        )

        async def answer_callback(interaction: discord.Interaction) -> None:
            if await self.interaction_check(interaction):
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
                await interaction.response.edit_message(view=self)
                self.stop()

        answer.callback = answer_callback
        decline.callback = decline_callback
        self.add_item(answer)
        self.add_item(decline)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return await _check_interaction_owner(interaction, self.user_id)

    async def on_timeout(self) -> None:
        self.stop()


class _UserInputView(discord.ui.View):
    def __init__(
        self,
        user_id: int | None,
        questions: list[dict[str, Any]],
        *,
        channel: discord.abc.Messageable | None = None,
        customizer: Any | None = None,
        timeout: float = 300,
    ) -> None:
        super().__init__(timeout=timeout)
        self.user_id = user_id
        self.questions = questions
        self.guild_id = getattr(getattr(channel, "guild", None), "id", None)
        self.customizer = customizer
        self.value: dict[str, Any] | None = None
        self.question_index = 0
        self._answers: dict[str, dict[str, list[str]]] = {}
        self._build_question_items()

    @property
    def current_question(self) -> dict[str, Any]:
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
            label = _truncate(option.get("label") or "Choose", 80)
            button = discord.ui.Button(label=label, style=discord.ButtonStyle.primary)

            async def callback(
                interaction: discord.Interaction,
                *,
                answer: str = str(option.get("label") or ""),
            ) -> None:
                if await self.interaction_check(interaction):
                    complete = self._record_answer(answer)
                    if complete:
                        await interaction.response.edit_message(view=self)
                        self.stop()
                    else:
                        await interaction.response.edit_message(
                            **self.message_kwargs(for_edit=True)
                        )

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
            )

            async def other_callback(interaction: discord.Interaction) -> None:
                if await self.interaction_check(interaction):
                    await interaction.response.send_modal(
                        _TextModal(self, self.user_id, question),
                    )

            other.callback = other_callback
            self.add_item(other)

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
        return await _check_interaction_owner(interaction, self.user_id)

    async def on_timeout(self) -> None:
        self.stop()


class _TextModal(discord.ui.Modal):
    def __init__(
        self,
        view: _UserInputView,
        user_id: int | None,
        question: dict[str, Any],
    ) -> None:
        super().__init__(title=_truncate(question.get("header") or "Codex input", 45))
        self.view = view
        self.user_id = user_id
        self.question = question
        self.answer = discord.ui.TextInput(
            label=_truncate(question.get("header") or "Answer", 45),
            placeholder=_truncate(question.get("question") or "Answer", 100),
            style=discord.TextStyle.paragraph,
            required=True,
            max_length=4000,
        )
        self.add_item(self.answer)

    async def on_submit(self, interaction: discord.Interaction) -> None:
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
