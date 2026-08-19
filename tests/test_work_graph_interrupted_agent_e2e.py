from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from entroly.work_graph import WorkGraph, WorkGraphUnavailableError
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


def _require_native() -> None:
    try:
        WorkGraph("native-probe")
    except WorkGraphUnavailableError as exc:
        pytest.skip(str(exc))


def test_replacement_agent_recovers_work_when_previous_agent_never_used_entroly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Durable repo evidence alone must be enough to reconstruct interrupted work."""

    _require_native()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "checkout", "-b", "feature/interrupted-auth")

    # Simulate Claude disappearing mid-task. It never calls Entroly, writes a
    # checkpoint, or creates a handoff; the only surviving truth is the repo.
    (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")

    state_root = tmp_path / "entroly-state"
    monkeypatch.setenv("ENTROLY_SOURCE", str(repo))
    monkeypatch.setenv("ENTROLY_DIR", str(state_root))
    assert not state_root.exists(), "fixture accidentally pre-created Entroly state"

    recovered = work_resume(max_evidence=64)

    assert recovered["status"] == "ok", recovered
    assert recovered["kind"] == "work_resume"
    assert recovered["trust"] == "untrusted_recovered_work_state"
    # The fenced payload is deliberately data, not instructions. It must still
    # carry the durable changed-file fact needed by the replacement agent.
    assert "app.py" in recovered["context"]
    assert "feature/interrupted-auth" in recovered["context"]
    assert state_root.exists(), "explicit recovery did not persist shared Work Graph state"
