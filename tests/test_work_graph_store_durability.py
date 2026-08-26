"""Scenarios R and Q of the section 19 dogfood gauntlet.

Section 19 asks for two things the suite did not assert:

* **R — crash during persistence.** "Terminate during temp-write/replace
  boundary where fixture allows; last committed state remains readable."
* **Q — multiprocessing contention.** "Concurrent local processes update state;
  atomicity/integrity preserved; **stale lock recovery works**."

`WorkGraphStore` implements both. `_save_unlocked` writes to a `mkstemp`
temporary in the same directory, fsyncs, then `os.replace`s onto the state path,
which is atomic on POSIX and on Windows for same-volume replaces. `lock()` calls
`_break_stale_lock`, which removes a lock whose mtime has not moved for
`stale_lock_seconds`.

Neither had a test. An unbreakable stale lock is an availability outage -- every
subsequent claim, resume and handoff blocks until a human deletes a dotfile --
and a persistence crash that leaves unreadable state is unrecoverable by
definition. Both are cheap to exercise, so they are exercised here.

The crash is simulated at the `os.replace` boundary rather than by killing a
process: the point of the atomic-replace design is that the old state survives
*whatever* happens before the replace commits, and raising there reproduces that
without depending on signal timing.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from entroly.work_graph import WorkGraph, WorkGraphUnavailableError
from entroly.work_graph_store import WorkGraphStore

REPO = "repo:durability"


def _skip_without_native() -> None:
    try:
        WorkGraph("native-probe")
    except WorkGraphUnavailableError as exc:  # pragma: no cover - environment
        pytest.skip(str(exc))


def _store(tmp_path: Path, **kwargs) -> WorkGraphStore:
    return WorkGraphStore(
        REPO,
        root=str(tmp_path / "state"),
        **kwargs,
    )


def _seed(store: WorkGraphStore, observed_at_ms: int) -> str:
    """Commit one durable event and return the resulting commitment."""
    graph = store.load()
    graph.apply_event(
        {
            "observed_at_ms": observed_at_ms,
            "source_kind": "repository_fact",
            "source_ref": "durability-fixture",
            "operations": [
                {
                    "op": "upsert_node",
                    "node": {
                        "node_id": f"file:{observed_at_ms:024d}",
                        "kind": "file",
                        "label": f"f{observed_at_ms}.py",
                        "trust": "observed",
                        "attributes": {"path": f"f{observed_at_ms}.py"},
                    },
                }
            ],
        }
    )
    saved = store.save(graph)
    return saved.graph_commitment


# ── Scenario R — crash during persistence ────────────────────────────────


def test_crash_at_the_replace_boundary_leaves_the_last_state_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The committed state must survive a failure before the replace lands."""
    _skip_without_native()
    store = _store(tmp_path)
    committed = _seed(store, 1_000)

    graph = store.load()
    assert graph.event_count == 1

    # Fail exactly at the temp-write / replace boundary.
    def boom(src, dst, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise OSError("simulated crash before replace committed")

    monkeypatch.setattr(os, "replace", boom)

    second = store.load()
    second.apply_event(
        {
            "observed_at_ms": 2_000,
            "source_kind": "repository_fact",
            "source_ref": "durability-fixture",
            "operations": [
                {
                    "op": "upsert_node",
                    "node": {
                        "node_id": "file:doomed",
                        "kind": "file",
                        "label": "doomed.py",
                        "trust": "observed",
                        "attributes": {"path": "doomed.py"},
                    },
                }
            ],
        }
    )
    with pytest.raises(OSError):
        store.save(second)

    monkeypatch.undo()

    recovered = store.load()
    assert recovered.event_count == 1, "the interrupted write was partially applied"
    assert recovered.graph_commitment == committed


def test_a_failed_write_leaves_no_temporary_debris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_save_unlocked` unlinks its temp file in a `finally`.

    Debris matters beyond tidiness: the temp files live in the state directory
    alongside `state.json`, so an accumulation is both a disk leak and a source
    of confusion for anyone inspecting the directory after an incident.
    """
    _skip_without_native()
    store = _store(tmp_path)
    _seed(store, 1_000)
    state_dir = store.state_path.parent

    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    graph = store.load()
    graph.apply_event(
        {
            "observed_at_ms": 3_000,
            "source_kind": "repository_fact",
            "source_ref": "durability-fixture",
            "operations": [
                {
                    "op": "upsert_node",
                    "node": {
                        "node_id": "file:debris",
                        "kind": "file",
                        "label": "debris.py",
                        "trust": "observed",
                        "attributes": {"path": "debris.py"},
                    },
                }
            ],
        }
    )
    with pytest.raises(OSError):
        store.save(graph)
    monkeypatch.undo()

    leftovers = list(state_dir.glob(".state-*.tmp"))
    assert leftovers == [], f"temp files survived a failed write: {leftovers}"


def test_state_is_still_writable_after_a_failed_write(tmp_path: Path, monkeypatch) -> None:
    """A crash must not wedge the store — the lock has to have been released."""
    _skip_without_native()
    store = _store(tmp_path)
    _seed(store, 1_000)

    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    graph = store.load()
    graph.apply_event(
        {
            "observed_at_ms": 4_000,
            "source_kind": "repository_fact",
            "source_ref": "durability-fixture",
            "operations": [
                {
                    "op": "upsert_node",
                    "node": {
                        "node_id": "file:wedge",
                        "kind": "file",
                        "label": "wedge.py",
                        "trust": "observed",
                        "attributes": {"path": "wedge.py"},
                    },
                }
            ],
        }
    )
    with pytest.raises(OSError):
        store.save(graph)
    monkeypatch.undo()

    # The next write must succeed rather than block on a lock never released.
    commitment = _seed(store, 5_000)
    assert store.load().event_count == 2
    assert commitment


# ── Scenario Q — stale lock recovery ─────────────────────────────────────


def test_a_stale_lock_is_broken_and_work_proceeds(tmp_path: Path) -> None:
    """A lock left behind by a dead process must not block the next agent."""
    _skip_without_native()
    store = _store(tmp_path, stale_lock_seconds=1.0, lock_timeout_seconds=15.0)
    _seed(store, 1_000)

    # Simulate a process that died holding the lock: the file exists and its
    # mtime is old enough that `_stale_lock` will see two identical readings.
    store.lock_path.write_text("dead-host:9999:deadbeef\n0.0\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(store.lock_path, (old, old))
    assert store.lock_path.exists()

    commitment = _seed(store, 2_000)

    assert commitment
    assert store.load().event_count == 2
    assert not store.lock_path.exists(), "the lock was not released after the operation"


def test_a_fresh_lock_is_respected_rather_than_broken(tmp_path: Path) -> None:
    """The counterpart, and the one that matters for correctness.

    Breaking a lock that is merely *held* rather than stale would defeat the
    mutual exclusion entirely, so a live lock must produce a timeout instead.
    """
    _skip_without_native()
    from entroly.work_graph_store import WorkGraphLockTimeout

    store = _store(tmp_path, stale_lock_seconds=3600.0, lock_timeout_seconds=0.5)
    _seed(store, 1_000)

    store.lock_path.write_text("live-host:1234:cafebabe\n0.0\n", encoding="utf-8")

    with pytest.raises(WorkGraphLockTimeout):
        store.load()

    assert store.lock_path.exists(), "a live lock must survive a timed-out acquirer"
    store.lock_path.unlink()


def test_breaking_a_stale_lock_does_not_disturb_committed_state(tmp_path: Path) -> None:
    """Recovery must be a lock operation only, never a state operation."""
    _skip_without_native()
    store = _store(tmp_path, stale_lock_seconds=1.0, lock_timeout_seconds=15.0)
    committed = _seed(store, 1_000)

    store.lock_path.write_text("dead:1:1\n0.0\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(store.lock_path, (old, old))

    recovered = store.load()

    assert recovered.event_count == 1
    assert recovered.graph_commitment == committed
