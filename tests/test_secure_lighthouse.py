"""Security-sensitive Lighthouse rendering tests."""

from __future__ import annotations

import unittest

from theia.server.lighthouse_render import render_lighthouse


class SecureLighthouseTests(unittest.TestCase):
    def test_locked_lighthouse_redacts_runtime_data(self) -> None:
        snapshot = {
            "version": "2.0.0",
            "action": "Processing request",
            "model": "gpt-5.6-luna",
            "character": {"name": "Cel"},
            "session": "Server conversation · #general",
            "runtime": {"rss_bytes": 7 * 1024**3},
            "vault": {
                "status": "locked",
                "reason": "Vault unlock required",
                "input_hint": "Type the vault passphrase in the terminal and press Enter.",
                "events": (
                    {
                        "timestamp": 0,
                        "severity": "INFO",
                        "detail": "Passphrase required",
                    },
                ),
            },
        }

        rendered = render_lighthouse(snapshot)

        self.assertIn("Status       Locked", rendered)
        self.assertIn("Reason       Vault unlock required", rendered)
        self.assertIn("Passphrase required", rendered)
        self.assertIn("Input", rendered)
        self.assertNotIn("GPT-5.6 Luna", rendered)
        self.assertNotIn("Cel", rendered)
        self.assertNotIn("Server conversation", rendered)
        self.assertNotIn("rss_bytes", rendered)
