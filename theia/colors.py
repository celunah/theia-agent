"""Theia's shared default presentation palette."""

from __future__ import annotations

import discord


THEIA_COLORS: dict[str, str] = {
    "INFO": "#A5BAFF",
    "WARNING": "#C0E68C",
    "ERROR": "#C07178",
    "FATAL": "#C07178",
    # Generated from the existing Celune-derived palette for states that need
    # a distinct semantic color in the Lighthouse View.
    "ACTIVE": "#8CCFA3",
    "FATAL_DARK": "#9A5A60",
    "CONNECTED": "#A5BAFF",
    "HEALTHY": "#A5BAFF",
    "DEGRADED": "#C0E68C",
    # Keep secondary and unavailable values muted without losing contrast on
    # the dark terminal background used by the Lighthouse View.
    "DISABLED": "#8A86A0",
}


def color_value(name: str) -> int:
    """Return a palette color as a Discord-compatible integer."""
    value = THEIA_COLORS.get(name.upper(), THEIA_COLORS["INFO"])
    return int(value[1:], 16)


def discord_color(name: str) -> discord.Color:
    """Build a Discord color from a shared semantic palette entry."""
    return discord.Color(color_value(name))


def ansi_color(name: str, *, emphasis: bool = False) -> str:
    """Return a true-color ANSI prefix for terminal presentation."""
    value = color_value(name)
    red = (value >> 16) & 0xFF
    green = (value >> 8) & 0xFF
    blue = value & 0xFF
    prefix = f"\x1b[38;2;{red};{green};{blue}m"
    if emphasis:
        prefix = f"\x1b[1;7m{prefix}"
    return prefix


def rich_style(name: str, *, emphasis: bool = False) -> str:
    """Return a Rich style expression for a palette entry."""
    style = THEIA_COLORS.get(name.upper(), THEIA_COLORS["INFO"])
    return f"{'bold reverse ' if emphasis else ''}{style}"
