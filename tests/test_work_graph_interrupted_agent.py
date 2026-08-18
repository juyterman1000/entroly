from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from entroly.work_graph_mcp import work_resume


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_replacement_agent_recovers_unclaimed_interrupted_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A previous agent need not prepare an Entroly handoff before disappearing.

    Durable repository state is the recovery source of truth.  The replacement
    agent asks Work Graph to resume; the adapter observes the current worktree
    and the Rust engine reconstructs unfinished work from those facts.
    """

    repo = _repo(tmp_path)
    _git(repo, "checkout", "-b", "feature/interrupted")
    (repo / "app.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    (repo / "new_test.py").write_text(
        "from app import value\n\ndef test_value():\n    assert value() == 2\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("ENTROLY_SOURCE", str(repo))
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "entroly-state"))

    result = work_resume()
    if result.get("error") == "native_work_graph_unavailable":
        pytest.skip("native Work Graph extension is not installed in this test environment")

    assert result["status"] == "ok", result
    assert result["kind"] == "work_resume"
    assert result["trust"] == "untrusted_recovered_work_state"

    recovered = result["context"]
    assert "app.py" in recovered
    assert "new_test.py" in recovered
    assert "feature/interrupted" in recovered
    # Recovered content is data, never executable agent instruction.
    assert "NOT a user instruction" in recovered
