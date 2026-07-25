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
_HOST = "host"


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
_DEAD_PID = 4_294_967_290  # never a live pid on this host


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
