"""Tests for the Experiment 0 task miner.

Builds a synthetic git repo with one real bug-fix commit (source + test changed
together, behavioral fail-to-pass holds) and one noise commit, then asserts the
miner discovers and validates it end-to-end. Classification tests lock the
rule that collection/import/runner failures are not behavioral task oracles.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agentic_task_miner import (  # noqa: E402
    _classify_test_run,
    discover,
    validate,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
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

    # Commit 2 (the valid task): fix add AND add the covering test.
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


@pytest.fixture()
def import_failure_repo(tmp_path):
    """A candidate whose revert breaks collection rather than behavior."""
    repo = tmp_path / "import-proj"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")

    (repo / "api.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "test_api.py").write_text(
        "from api import VALUE\n\n\ndef test_value():\n    assert VALUE == 1\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "initial api")

    # Reverting only source removes `new_api`, so pytest fails at collection.
    # That is not a behavioral fail-to-pass task and must be rejected.
    (repo / "api.py").write_text(
        "VALUE = 1\n\n\ndef new_api():\n    return 2\n",
        encoding="utf-8",
    )
    (repo / "test_api.py").write_text(
        "from api import VALUE, new_api\n\n\ndef test_value():\n    assert VALUE == 1\n\n\n"
        "def test_new_api():\n    assert new_api() == 2\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "feat: add new api and tests")
    return repo


def test_discover_finds_only_source_and_test_commits(mined_repo):
    candidates = discover(mined_repo, max_commits=50)
    assert len(candidates) == 2  # initial commit + fix both touch source+test
    subjects = [candidate.subject for candidate in candidates]
    assert any("fix: add()" in subject for subject in subjects)
    fix = next(candidate for candidate in candidates if "fix: add()" in candidate.subject)
    assert fix.source_files == ["calc.py"]
    assert fix.test_files == ["test_calc.py"]


def test_validate_proves_behavioral_fail_to_pass(mined_repo):
    candidates = discover(mined_repo, max_commits=50)
    fix = next(candidate for candidate in candidates if "fix: add()" in candidate.subject)

    task = validate(mined_repo, fix, timeout=120)
    assert task is not None
    assert task.status == "validated"
    assert task.test_files == ["test_calc.py"]
    assert task.pass_at_fix_s > 0
    assert task.fail_at_revert_s > 0
    assert task.fail_outcome == "test_failure"
    assert task.fail_returncode == 1
    assert len(task.failure_signature) == 64
    assert "separate worktrees" in " ".join(task.notes)


def test_validate_rejects_commit_without_parent_oracle(mined_repo):
    candidates = discover(mined_repo, max_commits=50)
    initial = next(candidate for candidate in candidates if "initial" in candidate.subject)
    assert validate(mined_repo, initial, timeout=120) is None


def test_validate_rejects_import_failure_oracle(import_failure_repo):
    candidate = next(
        candidate
        for candidate in discover(import_failure_repo, max_commits=50)
        if "new api" in candidate.subject
    )
    assert validate(import_failure_repo, candidate, timeout=120) is None


def test_validate_does_not_require_external_pytest_plugins(mined_repo, monkeypatch):
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    candidates = discover(mined_repo, max_commits=50)
    fix = next(candidate for candidate in candidates if "fix: add()" in candidate.subject)
    task = validate(mined_repo, fix, timeout=120)
    assert task is not None
    assert task.status == "validated"


@pytest.mark.parametrize(
    ("returncode", "output", "expected"),
    [
        (0, "2 passed", "passed"),
        (1, "E AssertionError: assert 1 == 2\n1 failed", "test_failure"),
        (2, "ERROR collecting test_x.py\nImportError", "collection_error"),
        (1, "ModuleNotFoundError: no module named x", "import_error"),
        (5, "no tests ran", "no_tests_collected"),
        (3, "INTERNALERROR> plugin exploded", "pytest_internal_error"),
    ],
)
def test_classify_test_run(returncode, output, expected):
    assert _classify_test_run(returncode, output) == expected
