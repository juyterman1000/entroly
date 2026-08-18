from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from entroly.work_graph import WorkGraphUnavailableError
from entroly.work_graph_store import (
    WorkGraphLockTimeout,
    WorkGraphStateError,
    WorkGraphStore,
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
    _git(repo, "checkout", "-b", "feature/work")
    (repo / "app.py").write_text("print('two')\n")
    return repo


def _store(repo: Path, root: Path) -> WorkGraphStore:
    try:
        return WorkGraphStore.for_repository(
            repo,
            root=root,
            lock_timeout_seconds=0.05,
            stale_lock_seconds=1.0,
        )
    except WorkGraphUnavailableError as exc:
        pytest.skip(str(exc))


def test_store_roundtrip_is_atomic_private_and_integrity_checked(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    store = _store(repo, tmp_path / "state")
    graph = store.update_repository(
        repo,
        agent_id="claude",
        session_id="s1",
        task_hint={
            "task_id": "work",
            "title": "Finish work",
            "trust": "observed",
            "source_kind": "user_statement",
            "source_ref": "user:task",
        },
        observed_at_ms=1000,
    )
    loaded = store.load()
    assert loaded.graph_commitment == graph.graph_commitment
    assert loaded.unfinished() == graph.unfinished()
    assert not list(store.repo_dir.glob(".state-*.tmp"))
    if os.name == "posix":
        assert store.state_path.stat().st_mode & 0o777 == 0o600
        assert store.repo_dir.stat().st_mode & 0o777 == 0o700

    document = json.loads(store.state_path.read_text(encoding="utf-8"))
    document["graph_commitment"] = "0" * 64
    store.state_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(WorkGraphStateError, match="cannot load Work Graph state"):
        store.load()


def test_lock_contention_times_out_without_deleting_owner_lock(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    store = _store(repo, tmp_path / "state")
    lock = store.lock()
    lock.__enter__()
    try:
        with pytest.raises(WorkGraphLockTimeout):
            store.load()
        assert store.lock_path.is_file()
    finally:
        lock.__exit__(None, None, None)
    assert not store.lock_path.exists()


def test_abandoned_stale_lock_is_reclaimed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    store = _store(repo, tmp_path / "state")
    store.lock_path.write_text("stale-owner\n0\n", encoding="utf-8")
    old = time.time() - 5
    os.utime(store.lock_path, (old, old))
    with store.lock():
        owner = store.lock_path.read_text(encoding="utf-8").split("\n", 1)[0]
        assert owner != "stale-owner"
    assert not store.lock_path.exists()


def test_two_process_views_merge_against_latest_disk_state(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = _store(repo, tmp_path / "state")
    second = WorkGraphStore(
        first.repo_id,
        root=first.root,
        lock_timeout_seconds=0.05,
        stale_lock_seconds=1.0,
    )
    first.update_repository(repo, agent_id="claude", session_id="s1", observed_at_ms=1000)
    (repo / "second.txt").write_text("two\n")
    second.update_repository(repo, agent_id="codex", session_id="s2", observed_at_ms=2000)
    final = first.load()
    assert final.event_count >= 2
    assert final.summary()["event_count"] == final.event_count


def test_parallel_agent_claims_surface_advisory_overlap(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    store = _store(repo, tmp_path / "state")
    first, lease_a = store.claim_work(
        repo,
        agent_id="claude",
        task_title="Fix auth",
        task_id="auth",
        scope_paths=["src/auth"],
        observed_at_ms=3000,
    )
    second, lease_b = store.claim_work(
        repo,
        agent_id="codex",
        task_title="Add auth tests",
        task_id="auth-tests",
        scope_paths=["src/auth/token.py"],
        observed_at_ms=3100,
    )
    assert lease_a != lease_b
    assert second.event_count >= first.event_count
    report = store.coordination(now_ms=3200)
    assert report["active_leases"] == 2
    assert len(report["conflicts"]) == 1
    assert {report["conflicts"][0]["agent_a"], report["conflicts"][0]["agent_b"]} == {
        "claude",
        "codex",
    }


def test_nonfinite_lock_configuration_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(WorkGraphStateError, match="finite non-negative"):
        WorkGraphStore("repo:test", root=tmp_path / "nan", lock_timeout_seconds=float("nan"))
    with pytest.raises(WorkGraphStateError, match="finite non-negative"):
        WorkGraphStore("repo:test", root=tmp_path / "inf", stale_lock_seconds=float("inf"))


def test_symlink_store_root_is_rejected(tmp_path: Path) -> None:
    if os.name == "nt" or not hasattr(os, "symlink"):
        pytest.skip("symlink semantics differ on this platform")
    target = tmp_path / "real"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(WorkGraphStateError, match="unsafe Work Graph directory"):
        WorkGraphStore("repo:test", root=alias)
