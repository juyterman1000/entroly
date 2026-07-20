"""Tests for the Experiment 0 task miner (benchmarks/agentic_task_miner.py).

Builds a synthetic git repo with one real bug-fix commit (source + test
changed together, fail-to-pass holds) and one noise commit, then asserts the
miner discovers exactly the fix and validates it end-to-end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agentic_task_miner import discover, validate  # noqa: E402


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True,
        text=True,
    )


@pytest.fixture()
def mined_repo(tmp_path):
    repo = tmp_path / "proj"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")

    # Commit 1: buggy source + unrelated passing test.
    (repo / "calc.py").write_text(
        "def add(a, b):\n    return a - b  # bug\n\n\ndef mul(a, b):\n    return a * b\n",
        encoding="utf-8",
    )
    (repo / "test_calc.py").write_text(
        "from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "initial: calc with mul test")

    # Commit 2 (the task): fix add AND add the covering test.
    (repo / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n",
        encoding="utf-8",
    )
    (repo / "test_calc.py").write_text(
        "from calc import add, mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fix: add() subtracted instead of adding")

    # Commit 3: docs-only noise — must not be discovered.
    (repo / "README.md").write_text("# calc\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "docs: readme")

    return repo


def test_discover_finds_only_the_fix_commit(mined_repo):
    candidates = discover(mined_repo, max_commits=50)
    assert len(candidates) == 2  # initial commit + fix both touch source+test
    subjects = [c.subject for c in candidates]
    assert any("fix: add()" in s for s in subjects)
    fix = next(c for c in candidates if "fix: add()" in c.subject)
    assert fix.source_files == ["calc.py"]
    assert fix.test_files == ["test_calc.py"]


def test_validate_proves_fail_to_pass_for_the_fix(mined_repo):
    candidates = discover(mined_repo, max_commits=50)
    fix = next(c for c in candidates if "fix: add()" in c.subject)

    task = validate(mined_repo, fix, timeout=120)
    assert task is not None
    assert task.status == "validated"
    assert task.test_files == ["test_calc.py"]
    assert task.pass_at_fix_s > 0
    assert task.fail_at_revert_s > 0


def test_validate_rejects_commit_without_oracle(mined_repo):
    # The initial commit has no parent to revert to — must be rejected,
    # not crash.
    candidates = discover(mined_repo, max_commits=50)
    initial = next(c for c in candidates if "initial" in c.subject)
    assert validate(mined_repo, initial, timeout=120) is None
