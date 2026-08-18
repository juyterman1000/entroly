from __future__ import annotations

import os
import time
from pathlib import Path

from entroly.work_graph_store import WorkGraphStore


def test_old_foreign_host_lock_is_not_auto_reclaimed(tmp_path: Path) -> None:
    store = WorkGraphStore(
        "repo:foreign-lock",
        root=tmp_path / "state",
        lock_timeout_seconds=0.01,
        stale_lock_seconds=1.0,
    )
    token = f"definitely-other-host:{os.getpid()}:foreign-owner"
    store.lock_path.write_text(f"{token}\n0\n", encoding="utf-8")
    old = time.time() - 5
    os.utime(store.lock_path, (old, old))

    assert store._break_stale_lock() is False
    assert store.lock_path.exists()
    assert store._lock_token() == token
