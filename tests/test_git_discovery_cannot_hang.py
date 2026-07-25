"""Git-based file discovery must never hang the caller.

Production failure this pins: the incremental watcher thread hung permanently
inside `git ls-files` while holding the index mutation lock, so all index
maintenance stalled indefinitely. `subprocess.run(capture_output=True,
timeout=...)` was not sufficient — the timeout bounds the wait, but
`communicate()` then joins the stdout/stderr reader threads, which only exit
when the pipes close. A child holding a pipe open blocks that join forever.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from entroly.auto_index import _git_env, _git_ls_files, _run_git


def test_run_git_returns_promptly_when_the_command_never_exits(tmp_path):
    # Stand in for a git that blocks (credential prompt, pager, index.lock).
    slow = [sys.executable, "-c", "import time; time.sleep(120)"]

    start = time.perf_counter()
    result = _run_git(slow, str(tmp_path), timeout=2)
    elapsed = time.perf_counter() - start

    assert result is None, "a timed-out git call must not return partial output"
    assert elapsed < 30, (
        f"_run_git blocked for {elapsed:.1f}s on a hanging child — the caller "
        "can be stuck holding the index mutation lock"
    )


def test_run_git_kills_the_child_rather_than_leaking_it(tmp_path):
    slow = [sys.executable, "-c", "import time; time.sleep(120)"]
    _run_git(slow, str(tmp_path), timeout=1)
    # If the child were leaked, it would still hold its pipes. Re-running must
    # still be prompt (no accumulation of stuck children).
    start = time.perf_counter()
    _run_git(slow, str(tmp_path), timeout=1)
    assert time.perf_counter() - start < 30


def test_run_git_survives_a_child_that_writes_then_hangs(tmp_path):
    # The nastiest shape: output is produced, then the process keeps the pipe
    # open. A naive reader waits for EOF forever.
    code = "import sys,time; sys.stdout.write('a.py\\n'); sys.stdout.flush(); time.sleep(120)"
    start = time.perf_counter()
    result = _run_git([sys.executable, "-c", code], str(tmp_path), timeout=2)
    elapsed = time.perf_counter() - start

    assert result is None
    assert elapsed < 30, f"blocked {elapsed:.1f}s waiting for EOF on a held-open pipe"


def test_git_env_is_strictly_non_interactive():
    env = _git_env()
    assert env["GIT_TERMINAL_PROMPT"] == "0"   # never prompt for credentials
    assert env["GIT_OPTIONAL_LOCKS"] == "0"    # never block on index.lock
    assert env["GIT_PAGER"] == "cat"           # never open a pager
    assert "GIT_EDITOR" not in env


def test_missing_git_binary_degrades_to_empty_not_an_exception(tmp_path):
    assert _run_git(["definitely-not-a-real-binary-xyz"], str(tmp_path)) is None


def test_git_ls_files_returns_empty_when_git_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr("entroly.auto_index._run_git", lambda *a, **k: None)
    assert _git_ls_files(str(tmp_path)) == []


def test_git_ls_files_parses_real_output(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "entroly.auto_index._run_git", lambda *a, **k: "a.py\n\n  b/c.py  \n"
    )
    assert _git_ls_files(str(tmp_path)) == ["a.py", "b/c.py"]


@pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git not available",
)
def test_git_ls_files_still_works_on_a_real_repository(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    files = _git_ls_files(str(tmp_path))
    assert "tracked.py" in files, f"real discovery regressed: {files}"
