# SPDX-License-Identifier: Apache-2.0
"""Tests for the portable CI helper scripts."""

# pylint: disable=protected-access

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from unittest import mock

from scripts import run_ci


ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT / "scripts" / "ci_warnings.py"
SPEC = importlib.util.spec_from_file_location("ci_warnings_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CI_WARNINGS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CI_WARNINGS
SPEC.loader.exec_module(CI_WARNINGS)


class TestRunCi:
    """Verify CI process cleanup behavior."""

    def test_windows_start_keeps_console_interrupts_on_runner(self) -> None:
        """Verify the child is not isolated from the runner's console group."""
        process = mock.Mock()

        with (
            mock.patch.object(run_ci.os, "name", "nt"),
            mock.patch.object(
                run_ci.subprocess, "Popen", return_value=process
            ) as popen,
            mock.patch.object(run_ci, "_attach_windows_job"),
        ):
            assert run_ci._start_process() is process

        popen.assert_called_once_with(run_ci.POE_COMMAND, text=True)

    def test_windows_cleanup_kills_tree_after_wrapper_exits(self) -> None:
        """Verify fallback cleanup still runs when the wrapper has exited."""
        process = mock.Mock()
        process.pid = 1234
        process.poll.return_value = 0

        with (
            mock.patch.object(run_ci.os, "name", "nt"),
            mock.patch.object(run_ci, "_close_windows_job", return_value=False),
            mock.patch.object(run_ci.subprocess, "run") as taskkill,
            mock.patch.object(run_ci.subprocess, "CREATE_NO_WINDOW", 0, create=True),
        ):
            run_ci.stop_process_tree(process)

        taskkill.assert_called_once_with(
            ["taskkill", "/PID", "1234", "/T", "/F"],
            stdout=run_ci.subprocess.DEVNULL,
            stderr=run_ci.subprocess.DEVNULL,
            check=False,
            creationflags=0,
        )
        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=run_ci.GRACE_PERIOD)

    def test_posix_cleanup_uses_immediate_process_group_kill(self) -> None:
        """Verify cleanup does not let Poe advance to another task."""
        process = mock.Mock()
        process.pid = 1234

        with (
            mock.patch.object(run_ci.os, "name", "posix"),
            mock.patch.object(
                run_ci.os, "getpgid", return_value=5678, create=True
            ) as getpgid,
            mock.patch.object(run_ci.os, "killpg", create=True) as killpg,
        ):
            run_ci.stop_process_tree(process)

        getpgid.assert_called_once_with(1234)
        sigkill = getattr(run_ci.signal, "SIGKILL", run_ci.signal.SIGTERM)
        killpg.assert_called_once_with(5678, sigkill)
        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=run_ci.GRACE_PERIOD)

    def test_wait_for_process_polls_until_child_finishes(self) -> None:
        """Verify the runner does not block signal handling in one long wait."""
        process = mock.Mock()
        process.args = ["uv", "run", "poe", "ci"]
        process.wait.side_effect = [
            run_ci.subprocess.TimeoutExpired(process.args, 0.1),
            0,
        ]

        with mock.patch.object(run_ci.time, "monotonic", side_effect=(0.0, 0.0, 0.2)):
            exit_code = run_ci._wait_for_process(process, 600.0)

        assert exit_code == 0
        assert process.wait.call_count == 2


class TestCIWarnings:
    """Verify warning output is recognized and safely annotated."""

    def test_warning_lines_include_common_tool_formats(self) -> None:
        """Recognize prose, deprecation, and Pylint warning formats."""
        assert CI_WARNINGS._is_warning_line("WARN type inference degraded")
        assert CI_WARNINGS._is_warning_line("DeprecationWarning: old API")
        assert CI_WARNINGS._is_warning_line("theia/core.py:12: W0611")
        assert not CI_WARNINGS._is_warning_line("All checks passed")
        assert not CI_WARNINGS._is_warning_line(
            "⚙️ verify warning-only results do not masquerade as a clean pass"
        )
        assert not CI_WARNINGS._is_warning_line("⚠️ warnings")
        assert not CI_WARNINGS._is_warning_line("passed 2/3 time 0:01 warnings 1")

    def test_annotation_messages_escape_github_command_delimiters(self) -> None:
        """Escape percent and line delimiters before emitting annotations."""
        assert CI_WARNINGS._annotation_message("50%\r\nwarning") == (
            "50%25%0D%0Awarning"
        )

    def test_main_preserves_failure_status_while_annotating_output(self) -> None:
        """Preserve wrapped command failures instead of masking them as warnings."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = CI_WARNINGS.main(
                [
                    "--",
                    sys.executable,
                    "-c",
                    "print('warning'); raise SystemExit(7)",
                ]
            )

        assert status == 7
        assert "::warning::warning" in output.getvalue()

    def test_main_does_not_annotate_celtest_status_descriptions(self) -> None:
        """Do not annotate Celtest descriptions containing warning terminology."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = CI_WARNINGS.main(
                [
                    "--",
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "sys.stdout.buffer.write("
                        "('\\u2699\\ufe0f verify warning-only results\\n'"
                        "'passed 2/3 time 0:01 warnings 1\\n').encode()"
                        ")"
                    ),
                ]
            )

        assert status == 0
        assert "⚙️ verify warning-only results" in output.getvalue()
        assert "::warning::" not in output.getvalue()
