import json
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

import discord

from .core import _truncate


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
        if self.user_id is not None and interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "Only the user who started this request can answer it.",
                ephemeral=True,
            )
            return False
        return True

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
        if self.user_id is not None and interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "Only the user who started this request can answer it.",
                ephemeral=True,
            )
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
        self, user_id: int | None, *, prompt: str, timeout: float = 300
    ) -> None:
        super().__init__(timeout=timeout)
        self.user_id = user_id
        self.prompt = prompt
        self.value: Any = None
        answer = discord.ui.Button(label="Answer", style=discord.ButtonStyle.primary)
        decline = discord.ui.Button(
            label="Decline", style=discord.ButtonStyle.secondary
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
        if self.user_id is not None and interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "Only the user who started this request can answer it.",
                ephemeral=True,
            )
            return False
        return True

    async def on_timeout(self) -> None:
        self.stop()


class _UserInputView(discord.ui.View):
    def __init__(
        self,
        user_id: int | None,
        questions: list[dict[str, Any]],
        *,
        timeout: float = 300,
    ) -> None:
        super().__init__(timeout=timeout)
        self.user_id = user_id
        self.questions = questions
        self.value: dict[str, Any] | None = None
        first = questions[0] if questions else {}
        options = first.get("options") or []
        for option in options[:4]:
            label = _truncate(option.get("label") or "Choose", 80)
            button = discord.ui.Button(label=label, style=discord.ButtonStyle.primary)

            async def callback(
                interaction: discord.Interaction,
                *,
                answer: str = str(option.get("label") or ""),
            ) -> None:
                if await self.interaction_check(interaction):
                    self._set_answer(answer)
                    await interaction.response.edit_message(view=self)
                    self.stop()

            button.callback = callback
            self.add_item(button)

        if first.get("isOther") or not options:
            other = discord.ui.Button(
                label="Other", style=discord.ButtonStyle.secondary
            )

            async def other_callback(interaction: discord.Interaction) -> None:
                if await self.interaction_check(interaction):
                    await interaction.response.send_modal(
                        _TextModal(self, self.user_id, first),
                    )

            other.callback = other_callback
            self.add_item(other)

        if len(questions) > 1:
            all_answers = discord.ui.Button(
                label="Answer all (JSON)",
                style=discord.ButtonStyle.secondary,
            )

            async def all_callback(interaction: discord.Interaction) -> None:
                if await self.interaction_check(interaction):
                    ids = ", ".join(str(question.get("id")) for question in questions)
                    await interaction.response.send_modal(
                        _JsonModal(
                            _FormViewProxy(self),
                            self.user_id,
                            title="Codex questions",
                            prompt=f"JSON object using these ids: {ids}",
                        )
                    )

            all_answers.callback = all_callback
            self.add_item(all_answers)

    def _set_answer(self, answer: str) -> None:
        answers: dict[str, dict[str, list[str]]] = {}
        if self.questions:
            answers[str(self.questions[0].get("id"))] = {"answers": [answer]}
        for question in self.questions[1:]:
            answers[str(question.get("id"))] = {"answers": []}
        self.value = {"answers": answers}

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if self.user_id is not None and interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "Only the user who started this request can answer it.",
                ephemeral=True,
            )
            return False
        return True

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
        if self.user_id is not None and interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "Only the user who started this request can answer it.",
                ephemeral=True,
            )
            return
        self.view._set_answer(str(self.answer))
        await interaction.response.defer(ephemeral=True)
        self.view.stop()


class _FormViewProxy(_FormView):
    """Adapts the JSON modal's result back into a request_user_input view."""

    def __init__(self, target: _UserInputView) -> None:
        self.target = target

    @property
    def value(self) -> Any:
        return self.target.value

    @value.setter
    def value(self, value: Any) -> None:
        if value is None:
            self.target.value = None
            return
        answers: dict[str, dict[str, list[str]]] = {}
        if isinstance(value, dict):
            for question in self.target.questions:
                question_id = str(question.get("id"))
                raw = value.get(question_id, "")
                values = raw if isinstance(raw, list) else [raw]
                answers[question_id] = {"answers": [str(item) for item in values]}
        self.target.value = {"answers": answers}

    def stop(self) -> None:
        self.target.stop()
