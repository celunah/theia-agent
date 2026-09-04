# SPDX-License-Identifier: Apache-2.0
"""Run a command while turning warning-like output into CI annotations."""

from __future__ import annotations

import contextlib
import io
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable
from typing import Any, Optional, cast


if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


_WARNING_LINE = re.compile(
    r"(?:\b\w*warn(?:ing)?\b|\bdeprecated\b|\bW\d{4}\b)",
    re.IGNORECASE,
)
_CELTEST_STATUS_LINE = re.compile(r"^\s*(?:⚙️|✅|❌|⚠️)\s")
_CELTEST_SUMMARY_LINE = re.compile(
    r"^\s*passed\s+\d+/\d+\s+time\s+\S+\s+warnings\s+\d+\s*$",
    re.IGNORECASE,
)


def _is_warning_line(line: str) -> bool:
    """Return whether one command-output line describes a warning."""
    if _CELTEST_STATUS_LINE.match(line) or _CELTEST_SUMMARY_LINE.match(line):
        return False
    return bool(_WARNING_LINE.search(line))


def _annotation_message(line: str) -> str:
    """Escape one output line for a GitHub Actions warning command."""
    return line.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _stop_process_tree(process: subprocess.Popen[str]) -> None:
    """Stop the wrapped command and descendants after Ctrl+C."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        getpgid = cast(
            Callable[[int], int] | None,
            getattr(os, "getpgid", None),
        )
        killpg = cast(
            Callable[[int, int], None] | None,
            getattr(os, "killpg", None),
        )
        if callable(getpgid) and callable(killpg):
            getpgid_func = getpgid
            killpg_func = killpg
            with contextlib.suppress(OSError):
                # These APIs are POSIX-only and are looked up dynamically so
                # this helper remains importable on Windows.
                # pylint: disable=not-callable
                killpg_func(
                    getpgid_func(process.pid),
                    getattr(signal, "SIGKILL", signal.SIGTERM),
                )
                # pylint: enable=not-callable
    with contextlib.suppress(OSError):
        process.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=2)


def main(arguments: Optional[list[str]] = None) -> int:
    """Run the requested command and preserve its exit status.

    Args:
        arguments: Command arguments excluding the Python executable and script path.

    Returns:
        int: The wrapped command's exit status, or ``2`` when no command is supplied.
    """
    command = list(sys.argv[1:] if arguments is None else arguments)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        print("Usage: ci_warnings.py -- command [args ...]", file=sys.stderr)
        return 2

    popen_kwargs: dict[str, Any] = {}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(  # pylint: disable=R1732
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **popen_kwargs,
    )
    assert process.stdout is not None

    try:
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            print(line, flush=True)
            if _is_warning_line(line) and not line.startswith("::warning::"):
                print(f"::warning::{_annotation_message(line)}", flush=True)
    except KeyboardInterrupt:
        _stop_process_tree(process)
        return 130
    finally:
        process.stdout.close()

    return process.wait()


if __name__ == "__main__":
    sys.exit(main())
