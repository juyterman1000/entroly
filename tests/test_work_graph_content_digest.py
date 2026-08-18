from __future__ import annotations

import subprocess
from pathlib import Path

from entroly.work_graph_content_digest import enrich_worktree_content_digests


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
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "initial")
    return repo


def _observation(*changes: dict) -> dict:
    return {"repo_id": "repo:test", "changes": [dict(change) for change in changes]}


def test_same_bytes_produce_stable_digest_across_observations(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    a = _observation(path="app.py", kind="modified", staged=False, conflicted=False)
    b = _observation(path="app.py", kind="modified", staged=False, conflicted=False)

    enrich_worktree_content_digests(repo, a)
    enrich_worktree_content_digests(repo, b)

    assert a["changes"][0]["content_digest"].startswith("git-blob:")
    assert a["changes"][0]["content_digest"] == b["changes"][0]["content_digest"]


def test_digest_changes_when_worktree_bytes_change_without_status_change(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    observation = _observation(path="app.py", kind="modified", staged=False, conflicted=False)
    (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    enrich_worktree_content_digests(repo, observation)
    before = observation["changes"][0]["content_digest"]

    observation2 = _observation(path="app.py", kind="modified", staged=False, conflicted=False)
    (repo / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
    enrich_worktree_content_digests(repo, observation2)
    after = observation2["changes"][0]["content_digest"]

    assert before != after


def test_staged_and_conflicted_changes_remain_non_dedupeable(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    staged = _observation(path="app.py", kind="modified", staged=True, conflicted=False)
    conflicted = _observation(path="app.py", kind="unmerged", staged=False, conflicted=True)

    enrich_worktree_content_digests(repo, staged)
    enrich_worktree_content_digests(repo, conflicted)

    assert staged["changes"][0]["content_digest"] == ""
    assert conflicted["changes"][0]["content_digest"] == ""


def test_deleted_unstaged_path_has_stable_terminal_marker(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "app.py").unlink()
    observation = _observation(path="app.py", kind="deleted", staged=False, conflicted=False)
    enrich_worktree_content_digests(repo, observation)
    assert observation["changes"][0]["content_digest"] == "worktree:deleted"


def test_missing_file_hash_fails_closed_without_partial_digest(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    observation = _observation(path="missing.py", kind="modified", staged=False, conflicted=False)
    enrich_worktree_content_digests(repo, observation)
    assert observation["changes"][0]["content_digest"] == ""
