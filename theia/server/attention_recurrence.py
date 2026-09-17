"""Deterministic validation helpers for conversational recurrence candidates."""

from __future__ import annotations

import re
from typing import Any

from ..core import _ConversationContext
from .policy import (
    ATTENTION_OPEN_LOOP_LIMIT,
    ATTENTION_RECENT_EXCHANGE_LIMIT,
)

MINIMUM_REPETITION_CONFIDENCE = 0.80
RECURRENCE_SIGNATURE_LIMIT = 32
RECURRENCE_RELATIONSHIP_TYPES = frozenset({"same_topic", "related_topic", "return"})
RECURRENCE_MINIMUM_MESSAGE_CHARACTERS = 32
RECURRENCE_MINIMUM_MESSAGE_WORDS = 5
RECURRENCE_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "again",
        "also",
        "an",
        "and",
        "are",
        "back",
        "be",
        "because",
        "but",
        "can",
        "could",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "of",
        "on",
        "or",
        "should",
        "so",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "why",
        "with",
        "would",
        "you",
    }
)
RECURRENCE_TRIVIAL_MESSAGES = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "ok",
        "okay",
        "yes",
        "no",
        "thanks",
        "thank you",
        "sounds good",
        "got it",
        "go ahead",
        "let's do it",
    }
)


def recurrence_tokens(value: str) -> set[str]:
    """Extract meaningful bounded tokens used to compare conversation topics."""
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{2,}", value)
        if token.casefold() not in RECURRENCE_STOP_WORDS
    }


def context_material(context: _ConversationContext) -> str:
    """Flatten bounded conversation evidence into text for overlap checks."""
    return " ".join(
        (
            context.title,
            context.summary,
            *context.recent_exchanges[-ATTENTION_RECENT_EXCHANGE_LIMIT:],
            *context.open_loops[:ATTENTION_OPEN_LOOP_LIMIT],
        )
    )


def is_substantial_message(text: str) -> bool:
    """Reject greetings and short acknowledgements as recurrence candidates."""
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    if normalized in RECURRENCE_TRIVIAL_MESSAGES:
        return False
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text)
    return (
        len(text.strip()) >= RECURRENCE_MINIMUM_MESSAGE_CHARACTERS
        and len(words) >= RECURRENCE_MINIMUM_MESSAGE_WORDS
    )


def has_meaningful_overlap(
    text: str,
    matched_material: str,
    matched_title: str,
    current_title: str,
) -> bool:
    """Require repeated content or title terms before linking two conversations."""
    current_tokens = recurrence_tokens(text)
    if len(current_tokens) < 3:
        return False
    matched_tokens = recurrence_tokens(matched_material)
    if len(current_tokens & matched_tokens) >= 2:
        return True
    title_overlap = recurrence_tokens(matched_title) & recurrence_tokens(current_title)
    return len(title_overlap) >= 2


def same_bounded_text(first: Any, second: Any, limit: int) -> bool:
    """Compare values after the same bounded whitespace normalization."""

    def normalize(value: Any) -> str:
        """Normalize one value without retaining unbounded text."""
        return re.sub(r"\s+", " ", str(value or "")).strip()[:limit].casefold()

    return normalize(first) == normalize(second)
