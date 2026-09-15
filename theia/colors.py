"""Theia's shared default presentation palette."""

from __future__ import annotations

import discord


THEIA_COLORS: dict[str, str] = {
    # Celune CCS accent-500 with Theia's branding transform: R * 0.8.
    "INFO": "#A5BAFF",
    "CONNECTED": "#A5BAFF",
    "HEALTHY": "#A5BAFF",
    # Celune CCS semantic colors with HSL(S * 0.8, L * 0.9).
    "ACTIVE": "#92D886",  # green-500
    "WARNING": "#DFD477",  # yellow-500
    "DEGRADED": "#DFD477",
    "ERROR": "#DD6167",  # red-500
    "FATAL": "#DD6167",
    "FATAL_DARK": "#9C5256",  # red-700
    "DISABLED": "#8B78BC",  # faded-500 / sleep
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
