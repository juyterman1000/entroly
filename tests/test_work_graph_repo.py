from __future__ import annotations

import subprocess
from pathlib import Path

from entroly.work_graph_repo import discover_repository_observation


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
    (repo / "app.py").write_text("print('one')\n")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_clean_repo_is_null_control(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    observation = discover_repository_observation(repo, observed_at_ms=1234)

    assert observation["observed_at_ms"] == 1234
    assert observation["branch"]["name"] == "main"
    assert observation["branch"]["default_branch"] == "main"
    assert observation["branch"]["ahead_by"] == 0
    assert observation["branch"]["behind_by"] == 0
    assert observation["changes"] == []
    assert observation["commits"] == []
    assert observation["task_hint"] is None


def test_feature_branch_observes_only_durable_work_facts(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "checkout", "-b", "feature/stream")
    (repo / "app.py").write_text("print('two')\n")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "change app")
    (repo / "app.py").write_text("print('three')\n")
    (repo / "new.txt").write_text("new\n")

    observation = discover_repository_observation(
        repo,
        agent_id="claude",
        session_id="session-1",
        task_hint={
            "task_id": "stream",
            "title": "Fix streaming",
            "trust": "observed",
            "source_kind": "user_statement",
            "source_ref": "user:task",
        },
        observed_at_ms=2000,
    )

    assert observation["repo_id"].startswith("git-root:")
    assert observation["agent_id"] == "claude"
    assert observation["session_id"] == "session-1"
    assert observation["task_hint"]["title"] == "Fix streaming"
    assert observation["branch"]["name"] == "feature/stream"
    assert observation["branch"]["base_ref"] == "main"
    assert observation["branch"]["ahead_by"] == 1
    assert len(observation["commits"]) == 1
    assert observation["commits"][0]["subject"] == "change app"
    kinds = {item["path"]: item["kind"] for item in observation["changes"]}
    assert kinds == {"app.py": "modified", "new.txt": "untracked"}


def test_rename_and_staged_state_are_preserved(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "mv", "app.py", "renamed.py")
    observation = discover_repository_observation(repo, observed_at_ms=3000)

    assert len(observation["changes"]) == 1
    change = observation["changes"][0]
    assert change["kind"] == "renamed"
    assert change["path"] == "renamed.py"
    assert change["old_path"] == "app.py"
    assert change["staged"] is True


def test_remote_credentials_never_affect_or_leak_repo_identity(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "remote", "add", "origin", "https://alice:secret-one@example.com/org/repo.git")
    first = discover_repository_observation(repo, observed_at_ms=4000)["repo_id"]
    _git(repo, "remote", "set-url", "origin", "https://bob:secret-two@example.com/org/repo.git")
    second = discover_repository_observation(repo, observed_at_ms=4001)["repo_id"]

    assert first == second
    assert "secret" not in first
    assert "alice" not in first
    assert "bob" not in second


def test_detached_head_is_observed_without_inventing_task(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "--detach", sha)
    observation = discover_repository_observation(repo, observed_at_ms=5000)

    assert observation["branch"]["detached"] is True
    assert observation["branch"]["name"] == ""
    assert observation["task_hint"] is None
