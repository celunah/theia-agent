"""Install Theia's Python and local Codex CLI dependencies."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(command: list[str]) -> None:
    print(f"Running {command[0]} setup step")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    """Install the locked Python and project-local Codex CLI dependencies."""
    uv = shutil.which("uv")
    if uv is None:
        print("uv is required to install the Python dependencies.", file=sys.stderr)
        return 1

    npm = shutil.which("npm")
    if npm is None:
        print("npm is required to install the local Codex CLI.", file=sys.stderr)
        return 1

    _run([uv, "sync"])
    _run([npm, "ci", "--no-audit", "--no-fund"])
    print("Theia dependencies are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
