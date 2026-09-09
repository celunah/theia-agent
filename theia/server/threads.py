"""Thread naming and explicit Discord thread-intent detection."""

import re

from ..core import _truncate


def thread_name(prompt: str) -> str:
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


def user_requested_thread(prompt: str) -> bool:
    """Return whether the user's message has a high-confidence thread intent."""
    normalized = re.sub(r"\s+", " ", prompt or "").strip()
    if not normalized:
        return False
    if _THREAD_REQUEST_NEGATION_PATTERN.search(normalized):
        return False
    # Questions about the Discord feature are not requests to create a thread.
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
