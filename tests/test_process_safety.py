from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from entroly.process_safety import run_bounded_process


def test_successful_process_is_captured_without_shell() -> None:
    result = run_bounded_process(
        [sys.executable, "-c", "print('ok')"],
        timeout=5,
    )

    assert result.succeeded
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"
    assert result.stderr == ""
    assert not result.timed_out


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), "bad"])
def test_invalid_timeout_fails_before_launch(timeout) -> None:
    with pytest.raises(ValueError, match="timeout"):
        run_bounded_process([sys.executable, "-c", "pass"], timeout=timeout)


def test_missing_executable_is_structured_failure() -> None:
    result = run_bounded_process(
        ["entroly-command-that-does-not-exist-9f47"],
        timeout=1,
    )

    assert not result.succeeded
    assert result.returncode is None
    assert result.execution_error
    assert not result.timed_out


def test_output_is_bounded_and_reports_truncation() -> None:
    result = run_bounded_process(
        [sys.executable, "-c", "print('x' * 100000)"],
        timeout=5,
        max_output_bytes=1024,
        preserve_stdout_tail=True,
    )

    assert result.succeeded
    assert result.stdout_truncated
    assert "output truncated" in result.stdout
    assert len(result.stdout.encode("utf-8")) < 2048


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group integration test")
def test_timeout_kills_descendant_before_it_can_escape(tmp_path: Path) -> None:
    marker = tmp_path / "descendant-survived.txt"
    child_code = (
        "import pathlib,time; "
        "time.sleep(1.2); "
        f"pathlib.Path({str(marker)!r}).write_text('survived')"
    )
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "time.sleep(10)"
    )

    result = run_bounded_process(
        [sys.executable, "-c", parent_code],
        timeout=0.2,
    )

    assert result.timed_out
    assert not result.succeeded
    time.sleep(1.5)
    assert not marker.exists(), "a timed-out grandchild survived the process-tree kill"
