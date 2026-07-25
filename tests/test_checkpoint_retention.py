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


def _mgr(tmp_path: Path, **kw) -> CheckpointManager:
    return CheckpointManager(tmp_path, instance_id="me", **kw)


def test_own_checkpoints_respect_the_retention_limit(tmp_path: Path):
    for i in range(15):
        _write(tmp_path, f"ckpt_me_{i:03d}.json.gz", age_s=15 - i)
    _mgr(tmp_path, max_checkpoints=10)._prune_old_checkpoints()
    assert len(list(tmp_path.glob("ckpt_me_*.json.gz"))) == 10


def test_abandoned_peer_checkpoints_are_reaped(tmp_path: Path):
    # The regression: peers from dead instances accumulated forever.
    for i in range(20):
        _write(tmp_path, f"ckpt_peer{i}_000.json.gz", age_s=90_000)  # >24h old
    _mgr(tmp_path, max_checkpoints=10)._prune_old_checkpoints()
    assert list(tmp_path.glob("ckpt_peer*.json.gz")) == [], (
        "stale peer checkpoints still accumulate without bound"
    )


def test_recent_peer_checkpoints_are_left_alone(tmp_path: Path):
    # Another server may legitimately be running; do not delete its state.
    for i in range(5):
        _write(tmp_path, f"ckpt_live{i}_000.json.gz", age_s=10)
    _mgr(tmp_path, max_checkpoints=10)._prune_old_checkpoints()
    assert len(list(tmp_path.glob("ckpt_live*.json.gz"))) == 5, (
        "a live peer's checkpoints must not be deleted"
    )


def test_peer_ttl_is_configurable_via_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ENTROLY_PEER_CHECKPOINT_TTL", "1")
    _write(tmp_path, "ckpt_peer_000.json.gz", age_s=60)
    CheckpointManager(tmp_path, instance_id="me")._prune_old_checkpoints()
    assert list(tmp_path.glob("ckpt_peer_*.json.gz")) == []


def test_malformed_ttl_falls_back_to_the_default(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ENTROLY_PEER_CHECKPOINT_TTL", "not-a-number")
    mgr = CheckpointManager(tmp_path, instance_id="me")
    assert mgr.peer_retention_seconds == 24 * 3600


def test_pruning_never_raises_on_an_unreadable_directory(tmp_path: Path):
    mgr = CheckpointManager(tmp_path / "missing", instance_id="me")
    mgr._prune_old_checkpoints()  # must not raise — checkpointing is best-effort
