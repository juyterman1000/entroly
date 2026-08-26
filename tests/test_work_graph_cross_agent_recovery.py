from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from entroly.work_graph import WorkGraph, WorkGraphUnavailableError
from entroly.work_graph_cli import main as work_main


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


def test_replacement_agent_recovers_unclaimed_interrupted_work_from_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No prior Entroly call is required from the agent that disappeared."""

    _require_native()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "auth.py").write_text("TOKEN_VERSION = 1\n", encoding="utf-8")
    _git(repo, "add", "auth.py")
    _git(repo, "commit", "-m", "baseline")
    _git(repo, "checkout", "-b", "feature/auth-refresh")

    # Simulate a previous agent disappearing mid-edit. There is deliberately no
    # claim, checkpoint or handoff written before this durable repository change.
    (repo / "auth.py").write_text("TOKEN_VERSION = 2\n", encoding="utf-8")

    state = tmp_path / "state"
    monkeypatch.setenv("ENTROLY_DIR", str(state))
    assert not state.exists()

    # Any replacement agent can invoke the same vendor-neutral recovery path.
    assert work_main(["--project", str(repo), "--json", "resume"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "ok"
    resume = payload["resume"]
    assert resume["selected_workstream"]["status"] == "in_progress"
    assert "auth.py" in resume["changed_paths"]
    assert resume["graph_revision"] >= 1
    assert resume["graph_commitment"]

    # The reconstructed graph is durable for a different process/agent to load.
    assert any(state.rglob("state.json"))
