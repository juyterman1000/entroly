from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from entroly.work_graph_repo import RepositoryDiscoveryError, discover_repository_observation


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
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "feature/interrupted")
    return repo


def test_clean_ahead_branch_recovers_complete_per_commit_paths(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "base.txt").write_text("changed\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "new.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "base.txt", "src/new.py")
    _git(repo, "commit", "-m", "feature one")

    (repo / "docs.md").write_text("docs\n", encoding="utf-8")
    _git(repo, "add", "docs.md")
    _git(repo, "commit", "-m", "feature two")

    observation = discover_repository_observation(
        repo,
        include_checkpoint=False,
        observed_at_ms=1,
    )
    assert observation["changes"] == []
    assert observation["branch"]["ahead_by"] == 2
    assert len(observation["commits"]) == 2

    by_subject = {commit["subject"]: commit for commit in observation["commits"]}
    assert by_subject["feature one"]["changed_paths"] == ["base.txt", "src/new.py"]
    assert by_subject["feature two"]["changed_paths"] == ["docs.md"]


def test_commit_path_expansion_fails_closed_instead_of_returning_partial_history(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    bulk = repo / "bulk"
    bulk.mkdir()
    for index in range(513):
        (bulk / f"f{index:03d}.txt").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "bulk")
    _git(repo, "commit", "-m", "large commit")

    with pytest.raises(RepositoryDiscoveryError, match="commit-path observation"):
        discover_repository_observation(
            repo,
            include_checkpoint=False,
            observed_at_ms=2,
        )
