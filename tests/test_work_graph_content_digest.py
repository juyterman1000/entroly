from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

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


def test_symlink_never_fingerprints_target_outside_repository(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("outside-secret\n", encoding="utf-8")
    link = repo / "outside-link.txt"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    observation = _observation(
        path="outside-link.txt", kind="untracked", staged=False, conflicted=False
    )
    enrich_worktree_content_digests(repo, observation)
    assert observation["changes"][0]["content_digest"] == ""


def test_oversized_file_is_not_read_for_passive_dedupe(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    large = repo / "large.bin"
    with large.open("wb") as handle:
        handle.truncate(64 * 1024 * 1024 + 1)

    observation = _observation(path="large.bin", kind="untracked", staged=False, conflicted=False)
    enrich_worktree_content_digests(repo, observation)
    assert observation["changes"][0]["content_digest"] == ""


def test_aggregate_budget_refuses_partial_semantic_snapshot(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    changes = []
    # Sparse files make this fixture cheap: each file is individually below the
    # 64 MiB cap, but together they exceed the 128 MiB observation budget.
    for index in range(3):
        name = f"large-{index}.bin"
        with (repo / name).open("wb") as handle:
            handle.truncate(45 * 1024 * 1024)
        changes.append(
            {"path": name, "kind": "untracked", "staged": False, "conflicted": False}
        )

    observation = _observation(*changes)
    enrich_worktree_content_digests(repo, observation)

    assert [item["content_digest"] for item in observation["changes"]] == ["", "", ""]
