from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from entroly import work_graph_repo
from entroly.work_graph_repo import (
    RepositoryDiscoveryError,
    discover_repository_identity,
    discover_repository_observation,
)


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


def test_identity_lookup_survives_full_observer_change_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(work_graph_repo, "_MAX_CHANGES", 512)
    repo = _repo(tmp_path)
    for index in range(513):
        (repo / f"untracked-{index}.txt").write_text("x\n")

    identity = discover_repository_identity(repo)
    assert identity["repo_id"].startswith("git-root:")
    assert Path(identity["root"]) == repo.resolve()

    with pytest.raises(RepositoryDiscoveryError, match="partial Work Graph"):
        discover_repository_observation(repo)


def test_checkpoint_lookup_is_side_effect_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("ENTROLY_DIR", raising=False)

    observation = discover_repository_observation(repo, include_checkpoint=True, observed_at_ms=1)
    assert observation["task_hint"] is None
    assert not (fake_home / ".entroly").exists()

    project_hash = hashlib.sha256(str(repo.resolve()).encode()).hexdigest()[:12]
    existing = fake_home / ".entroly" / "checkpoints" / project_hash
    existing.mkdir(parents=True)
    observation = discover_repository_observation(repo, include_checkpoint=True, observed_at_ms=2)
    assert observation["task_hint"] is None
