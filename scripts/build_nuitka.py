"""Build an optional one-file Nuitka executable for Theia Agent."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENTRY_POINT = PROJECT_ROOT / "main.py"


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
    command = [
        sys.executable,
        "-m",
        "nuitka",
        "--onefile",
        f"--output-dir={output_dir}",
        f"--output-filename={executable_name}",
        str(ENTRY_POINT),
    ]
    print(f"Building {executable_name} in {output_dir}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


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
