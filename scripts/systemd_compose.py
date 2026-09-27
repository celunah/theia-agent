"""Start the systemd-managed Compose service with optional vault recovery."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    """Run Compose and use /dev/null when no recovery file is configured."""
    recovery_path = os.getenv("THEIA_VAULT_PASSPHRASE_FILE", "")
    candidate = Path(recovery_path).expanduser() if recovery_path else None
    if candidate is None or candidate.is_symlink() or not candidate.is_file():
        os.environ["THEIA_VAULT_PASSPHRASE_FILE"] = "/dev/null"

    arguments = [
        "/usr/bin/docker",
        "compose",
        "-f",
        "compose.yaml",
        "-f",
        "compose.systemd.yaml",
        *sys.argv[1:],
    ]
    os.execv(arguments[0], arguments)


if __name__ == "__main__":
    main()
