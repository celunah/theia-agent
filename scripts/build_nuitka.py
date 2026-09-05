"""Build an optional one-file Nuitka executable for Theia Agent."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENTRY_POINT = PROJECT_ROOT / "main.py"
_REVISION_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)


def _git_revision() -> str:
    """Return the short Git revision used to build the executable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short=7", "HEAD"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    revision = result.stdout.strip()
    return revision if _REVISION_RE.fullmatch(revision) else "unknown"


def build_executable(output_dir: Path, executable_name: str) -> int:
    """Build ``main.py`` as a one-file executable and return its exit code."""
    if importlib.util.find_spec("nuitka") is None:
        print(
            "Nuitka is not installed. Run "
            "`uv run --with nuitka python scripts/build_nuitka.py`.",
            file=sys.stderr,
        )
        return 2

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    revision_source = output_dir / f".{executable_name}.build-revision"
    revision_source.write_text(_git_revision() + "\n", encoding="ascii")
    try:
        command = [
            sys.executable,
            "-m",
            "nuitka",
            "--onefile",
            f"--output-dir={output_dir}",
            f"--output-filename={executable_name}",
            f"--include-data-files={revision_source}=theia/build-revision.txt",
            str(ENTRY_POINT),
        ]
        print(f"Building {executable_name} in {output_dir}")
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        return completed.returncode
    finally:
        with contextlib.suppress(OSError):
            revision_source.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse builder options and invoke Nuitka."""
    parser = argparse.ArgumentParser(
        description="Build Theia Agent as an optional Nuitka one-file executable."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "dist",
        help="Directory for the generated executable (default: dist).",
    )
    parser.add_argument(
        "--name",
        default="theia",
        help="Executable name without a platform suffix (default: theia).",
    )
    args = parser.parse_args(argv)
    return build_executable(args.output_dir, args.name)


if __name__ == "__main__":
    raise SystemExit(main())
