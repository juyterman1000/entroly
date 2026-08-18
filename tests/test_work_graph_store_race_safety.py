from __future__ import annotations

import os
import socket
import time
from pathlib import Path

import pytest

from entroly.work_graph_store import WorkGraphStateError, WorkGraphStore


def _store(tmp_path: Path) -> WorkGraphStore:
    return WorkGraphStore(
        "repo:race-safety",
        root=tmp_path / "state",
        lock_timeout_seconds=0.01,
        stale_lock_seconds=1.0,
    )


def _age(path: Path, seconds: float = 5.0) -> None:
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_live_local_owner_is_not_reclaimed_only_because_lock_is_old(tmp_path: Path) -> None:
    store = _store(tmp_path)
    token = f"{socket.gethostname()}:{os.getpid()}:live-owner"
    store.lock_path.write_text(f"{token}\n0\n", encoding="utf-8")
    _age(store.lock_path)

    assert store._break_stale_lock() is False
    assert store.lock_path.exists()
    assert store._lock_token() == token


def test_dead_or_unparseable_old_owner_is_still_reclaimable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.lock_path.write_text("dead-owner\n0\n", encoding="utf-8")
    _age(store.lock_path)

    assert store._break_stale_lock() is True
    assert not store.lock_path.exists()


def test_oversized_lock_metadata_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.lock_path.write_bytes(b"x" * 5000)

    with pytest.raises(WorkGraphStateError, match="lock"):
        store._lock_token()


def test_symlinked_lock_is_never_followed(tmp_path: Path) -> None:
    if os.name == "nt" or not hasattr(os, "symlink"):
        pytest.skip("symlink semantics differ on this platform")
    store = _store(tmp_path)
    outside = tmp_path / "outside-lock"
    outside.write_text("outside\n", encoding="utf-8")
    store.lock_path.symlink_to(outside)

    with pytest.raises(WorkGraphStateError, match="unsafe Work Graph lock"):
        store._lock_token()
    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_symlinked_state_is_rejected_before_native_graph_loading(tmp_path: Path) -> None:
    if os.name == "nt" or not hasattr(os, "symlink"):
        pytest.skip("symlink semantics differ on this platform")
    store = _store(tmp_path)
    outside = tmp_path / "outside-state"
    outside.write_text("not-a-work-graph\n", encoding="utf-8")
    store.state_path.symlink_to(outside)

    with pytest.raises(WorkGraphStateError, match="unsafe Work Graph state"):
        store._load_unlocked()
    assert outside.read_text(encoding="utf-8") == "not-a-work-graph\n"
