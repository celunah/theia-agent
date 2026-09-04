# SPDX-License-Identifier: Apache-2.0
"""Run the TypeScript test-harness quality gate on every supported host."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    """Find npm across Windows and POSIX and run the harness checks."""
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm is None:
        print("npm is required to check the TypeScript test harness.", file=sys.stderr)
        return 1
    return subprocess.run(
        [npm, "run", "check:test-harness"],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
