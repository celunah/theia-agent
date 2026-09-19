"""Collision-resistant identifiers for short-lived Theia state."""

from uuid import uuid4


def new_unique_token() -> str:
    """Return a process-independent token that is safe for internal IDs."""
    return uuid4().hex
