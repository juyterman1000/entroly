"""Checkpoint retention must bound disk growth across restarts.

Pruning previously covered only the current instance. Every restart mints a new
instance id, so the previous run's checkpoints became permanently unprunable
"peers". An observed dev machine held 127 checkpoints / 264 MB with
`own_checkpoints: 0` at ~40 MB per write.
"""

from __future__ import annotations

import time
from pathlib import Path

from entroly.checkpoint import CheckpointManager


def _write(dir_: Path, name: str, *, age_s: float = 0.0) -> Path:
    p = dir_ / name
    p.write_bytes(b"x" * 32)
    if age_s:
        old = time.time() - age_s
        import os

        os.utime(p, (old, old))
    return p


# instance_id is "<hosthash>_<pid>"; peers on the SAME host share the hosthash.
_HOST = "abc123def456"   # auto-generated ids are a 12-hex host hash


def _mgr(tmp_path: Path, **kw) -> CheckpointManager:
    return CheckpointManager(tmp_path, instance_id=f"{_HOST}_1", **kw)


def test_own_checkpoints_respect_the_retention_limit(tmp_path: Path):
    for i in range(15):
        _write(tmp_path, f"ckpt_{_HOST}_1_{i:03d}.json.gz", age_s=15 - i)
    _mgr(tmp_path, max_checkpoints=10)._prune_old_checkpoints()
    assert len(list(tmp_path.glob(f"ckpt_{_HOST}_1_*.json.gz"))) == 10


def test_a_peer_from_a_different_host_is_never_reaped(tmp_path: Path):
    # Shared volumes / NFS homes: pids live in independent namespaces, so a
    # local pid probe would judge a live remote peer dead.
    for i in range(4):
        _write(tmp_path, f"ckpt_otherhost_{_DEAD_PID}_{i:03d}.json.gz", age_s=90_000)
    _mgr(tmp_path, max_checkpoints=10, peer_retention_seconds=1)._prune_old_checkpoints()
    assert len(list(tmp_path.glob(f"ckpt_otherhost_{_DEAD_PID}_*.json.gz"))) == 4


def test_windows_access_denied_is_not_treated_as_a_dead_process():
    # OpenProcess returns NULL for both "no such pid" and "access denied"; only
    # the former means dead. PID 4 (System) is always running but not openable.
    import os as _os

    assert CheckpointManager._pid_is_alive(_os.getpid()) is True
    if _os.name == "nt":
        assert CheckpointManager._pid_is_alive(4) is True, (
            "an alive-but-protected process must never be reported dead"
        )
    assert CheckpointManager._pid_is_alive(_DEAD_PID) is False


# Real layout is ckpt_<hosthash>_<pid>_<counter>.json.gz; instance_id is
# "<hosthash>_<pid>", so the owning pid is the third underscore field.
def _make_dead_pid() -> int:
    """A pid that definitely belonged to a process and definitely exited.

    An out-of-range sentinel is NOT usable: os.kill raises OverflowError for a
    pid above INT_MAX, which _pid_is_alive fail-safes to "alive". Spawning and
    reaping a real process gives a pid that is dead on every platform.
    """
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


_DEAD_PID = _make_dead_pid()


def test_peer_reaping_is_off_by_default(tmp_path: Path):
    # Every restart mints a new instance_id, so the PREVIOUS run's checkpoints
    # are "peers" — and cross-instance reads are exactly how resume works.
    # Reaping peers by default would delete the resume history.
    for i in range(20):
        _write(tmp_path, f"ckpt_{_HOST}_{_DEAD_PID}_{i:03d}.json.gz", age_s=90_000)
    _mgr(tmp_path, max_checkpoints=10)._prune_old_checkpoints()
    assert len(list(tmp_path.glob(f"ckpt_{_HOST}_{_DEAD_PID}_*.json.gz"))) == 20, (
        "peer reaping must be opt-in; resume history must survive a restart"
    )


def test_opted_in_reaping_removes_peers_whose_process_is_gone(tmp_path: Path):
    for i in range(20):
        _write(tmp_path, f"ckpt_{_HOST}_{_DEAD_PID}_{i:03d}.json.gz", age_s=90_000)
    _mgr(tmp_path, max_checkpoints=10, peer_retention_seconds=3600)._prune_old_checkpoints()
    assert list(tmp_path.glob(f"ckpt_{_HOST}_{_DEAD_PID}_*.json.gz")) == []


def test_a_live_peer_is_never_reaped_even_when_stale(tmp_path: Path):
    # Age is not liveness: an idle-but-running instance stops writing, so its
    # checkpoints age. Deleting them would destroy a live process's state.
    import os as _os

    live = _os.getpid()
    for i in range(3):
        _write(tmp_path, f"ckpt_{_HOST}_{live}_{i:03d}.json.gz", age_s=90_000)
    _mgr(tmp_path, max_checkpoints=10, peer_retention_seconds=1)._prune_old_checkpoints()
    assert len(list(tmp_path.glob(f"ckpt_{_HOST}_{live}_*.json.gz"))) == 3, (
        "a running peer's checkpoints must never be deleted on age alone"
    )


def test_recent_peer_checkpoints_are_left_alone(tmp_path: Path):
    for i in range(5):
        _write(tmp_path, f"ckpt_{_HOST}_{_DEAD_PID}_{i:03d}.json.gz", age_s=10)
    _mgr(tmp_path, max_checkpoints=10, peer_retention_seconds=3600)._prune_old_checkpoints()
    assert len(list(tmp_path.glob(f"ckpt_{_HOST}_{_DEAD_PID}_*.json.gz"))) == 5


def test_unparseable_pid_is_kept_not_guessed(tmp_path: Path):
    _write(tmp_path, "ckpt_weirdname.json.gz", age_s=90_000)
    _mgr(tmp_path, max_checkpoints=10, peer_retention_seconds=1)._prune_old_checkpoints()
    assert (tmp_path / "ckpt_weirdname.json.gz").exists(), (
        "unknown ownership must never be deleted on a guess"
    )


def test_ttl_zero_and_malformed_disable_reaping_rather_than_wiping(
    monkeypatch, tmp_path: Path
):
    # "0" is the natural way to express "off". It must not mean "cutoff = now",
    # which would delete every peer checkpoint immediately.
    for raw in ("0", "-1", "not-a-number", ""):
        monkeypatch.setenv("ENTROLY_PEER_CHECKPOINT_TTL", raw)
        mgr = CheckpointManager(tmp_path, instance_id="me")
        assert mgr.peer_retention_seconds == 0.0, f"TTL={raw!r} must disable reaping"
        _write(tmp_path, f"ckpt_{_HOST}_{_DEAD_PID}_900.json.gz", age_s=90_000)
        mgr._prune_old_checkpoints()
        assert (tmp_path / f"ckpt_{_HOST}_{_DEAD_PID}_900.json.gz").exists()


def test_pruning_never_raises_on_an_unreadable_directory(tmp_path: Path):
    mgr = CheckpointManager(tmp_path / "missing", instance_id="me")
    mgr._prune_old_checkpoints()  # must not raise — checkpointing is best-effort


def test_total_checkpoints_are_bounded_across_restarts_by_default(tmp_path: Path):
    # THE reported bug: instance_id embeds the pid, so each restart orphans up
    # to max_checkpoints files that no later run will ever match. Per-instance
    # pruning alone left 127 checkpoints / 264 MB with own_checkpoints: 0.
    # The global cap must bound this with NO env var and NO liveness guessing.
    for restart in range(20):
        for i in range(5):
            _write(tmp_path, f"ckpt_{_HOST}_{9000 + restart}_{i:03d}.json.gz",
                   age_s=(20 - restart) * 100 + i)
    assert len(list(tmp_path.glob("ckpt_*.json.gz"))) == 100

    mgr = CheckpointManager(tmp_path, instance_id=f"{_HOST}_1",
                            max_checkpoints=10, max_total_checkpoints=40)
    mgr._prune_old_checkpoints()

    remaining = list(tmp_path.glob("ckpt_*.json.gz"))
    assert len(remaining) <= 40, (
        f"unbounded checkpoint growth is still unfixed: {len(remaining)} files"
    )


def test_global_cap_keeps_the_newest_because_resume_reads_the_newest(tmp_path: Path):
    for i in range(30):
        _write(tmp_path, f"ckpt_{_HOST}_{8000 + i}_000.json.gz", age_s=(30 - i) * 100)
    mgr = CheckpointManager(tmp_path, instance_id=f"{_HOST}_1",
                            max_total_checkpoints=10)
    mgr._prune_old_checkpoints()
    kept = sorted(p.stat().st_mtime for p in tmp_path.glob("ckpt_*.json.gz"))
    assert len(kept) == 10
    # The survivors must be the most recent — that is what load_latest reads.
    newest = _write(tmp_path, "probe.tmp")
    assert all(m <= newest.stat().st_mtime for m in kept)


def test_global_cap_never_drops_own_instance_checkpoints(tmp_path: Path):
    for i in range(30):
        _write(tmp_path, f"ckpt_{_HOST}_{7000 + i}_000.json.gz", age_s=1000)
    for i in range(3):
        _write(tmp_path, f"ckpt_{_HOST}_1_{i:03d}.json.gz", age_s=5000)  # older!
    mgr = CheckpointManager(tmp_path, instance_id=f"{_HOST}_1",
                            max_checkpoints=10, max_total_checkpoints=5)
    mgr._prune_old_checkpoints()
    assert len(list(tmp_path.glob(f"ckpt_{_HOST}_1_*.json.gz"))) == 3, (
        "the running instance's own state must survive the global cap"
    )


def test_global_cap_survives_a_peer_deleting_a_file_mid_scan(tmp_path: Path):
    # Now that each instance unlinks other instances' files, a peer removing a
    # path between our glob() and its stat() must not abort the whole retention
    # pass — on a multi-instance machine that contention is the steady state.
    for i in range(60):
        _write(tmp_path, f"ckpt_{_HOST}_{9000 + i}_170000_000.json.gz", age_s=(60 - i) * 10)
    mgr = CheckpointManager(tmp_path, instance_id=f"{_HOST}_1",
                            max_checkpoints=10, max_total_checkpoints=40)

    real_glob = type(mgr.checkpoint_dir).glob

    def racing(self, pattern):
        out = list(real_glob(self, pattern))
        if out:
            try:
                out[0].unlink()      # a peer got there first
            except OSError:
                pass
        return iter(out)

    type(mgr.checkpoint_dir).glob = racing
    try:
        mgr._prune_old_checkpoints()
    finally:
        type(mgr.checkpoint_dir).glob = real_glob

    remaining = len(list(tmp_path.glob("ckpt_*.json.gz")))
    assert remaining <= 40, (
        f"one concurrent unlink disabled retention: {remaining} files left"
    )


def test_readers_survive_a_peer_deleting_a_checkpoint_mid_scan(tmp_path: Path):
    # The global cap is the first code that makes one instance unlink ANOTHER
    # instance's files, so every reader that globs-then-stats can now race a
    # peer's retention pass. A vanished file must read as "gone", never raise
    # FileNotFoundError out of an MCP tool.
    for i in range(6):
        _write(tmp_path, f"ckpt_{_HOST}_{100 + i}_170000_000.json.gz")
    mgr = CheckpointManager(tmp_path, instance_id=f"{_HOST}_1")

    real_glob = type(mgr.checkpoint_dir).glob

    def racing(self, pattern):
        out = list(real_glob(self, pattern))
        if out:
            try:
                out[0].unlink()
            except OSError:
                pass
        return iter(out)

    type(mgr.checkpoint_dir).glob = racing
    raised: dict[str, str] = {}
    try:
        for name, call in (
            ("load_latest", lambda: mgr.load_latest()),
            ("_load_latest_for_instance", lambda: mgr._load_latest_for_instance()),
            ("find_relevant", lambda: mgr.find_relevant("refactor the proxy")),
            ("list_checkpoints", lambda: mgr.list_checkpoints()),
            ("merge_from_peers", lambda: mgr.merge_from_peers([])),
            ("stats", lambda: mgr.stats()),
        ):
            try:
                call()
            except Exception as exc:  # noqa: BLE001 — the point is to catch any
                raised[name] = f"{type(exc).__name__}: {exc}"
    finally:
        type(mgr.checkpoint_dir).glob = real_glob

    assert raised == {}, (
        f"a peer's prune made these readers raise instead of degrading: {raised}"
    )
    # Non-vacuity: the racing glob must actually have deleted something, or the
    # test would pass even with the guard reverted.
    assert len(list(tmp_path.glob("ckpt_*.json.gz"))) < 6, (
        "the race was never triggered — this test would pass without the fix"
    )


def test_global_cap_preserves_a_live_peers_latest_recovery_frontier(
    tmp_path: Path, monkeypatch
):
    """A busy writer must not erase a quieter live peer's only resume point."""
    import os

    live_pid = os.getpid()
    monkeypatch.setattr(
        CheckpointManager,
        "_pid_is_alive",
        staticmethod(lambda pid: pid == live_pid),
    )
    live_paths = [
        _write(
            tmp_path,
            f"ckpt_{_HOST}_{live_pid}_{i:03d}.json.gz",
            age_s=10_000 - i,
        )
        for i in range(3)
    ]
    expected_frontier = max(live_paths, key=lambda path: path.stat().st_mtime)

    # Newer abandoned history creates enough pressure to delete the live peer's
    # entire history under the previous globally-newest-only implementation.
    for i in range(30):
        _write(
            tmp_path,
            f"ckpt_{_HOST}_{_DEAD_PID}_{i:03d}.json.gz",
            age_s=100 - i,
        )

    mgr = CheckpointManager(
        tmp_path,
        instance_id=f"{_HOST}_1",
        max_checkpoints=10,
        max_total_checkpoints=5,
    )
    mgr._prune_old_checkpoints()

    live_remaining = list(tmp_path.glob(f"ckpt_{_HOST}_{live_pid}_*.json.gz"))
    assert live_remaining == [expected_frontier], (
        "the global cap erased or duplicated a live peer's protected frontier: "
        f"{live_remaining}"
    )
    assert len(list(tmp_path.glob("ckpt_*.json.gz"))) <= 5


def test_global_cap_is_soft_when_protected_frontiers_exceed_it(tmp_path: Path):
    """Recovery safety outranks a count cap for unclassifiable ownership."""
    for i in range(3):
        _write(tmp_path, f"ckpt_unknown-owner-{i}.json.gz", age_s=100 + i)
    mgr = CheckpointManager(
        tmp_path,
        instance_id=f"{_HOST}_1",
        max_total_checkpoints=1,
    )

    mgr._prune_old_checkpoints()

    assert len(list(tmp_path.glob("ckpt_*.json.gz"))) == 3, (
        "unknown ownership was treated as permission to destroy recovery state"
    )
