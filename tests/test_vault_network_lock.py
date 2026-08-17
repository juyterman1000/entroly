"""Ledger locking where advisory locks do not work.

`fcntl.flock` needs a working `lockd` to mean anything over NFS, and SMB is
worse. On such a mount it succeeds locally while serializing nothing, so two
machines sharing a vault would each believe they held the lock and interleave
their read-modify-write appends. These tests disable the advisory layer to
check the lock file alone carries the guarantee.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import entroly.vault_time as vault_time
from entroly.vault import BeliefArtifact, VaultConfig, VaultManager
from entroly.vault_time import BeliefLedger


@pytest.fixture()
def no_advisory_locks(monkeypatch):
    """Make every advisory lock attempt fail, as an NFS mount would."""

    monkeypatch.setattr(vault_time, "_advisory_lock", lambda handle: False)
    monkeypatch.setattr(vault_time, "_advisory_unlock", lambda handle: None)


def _write_many(vault: VaultManager, prefix: str, count: int) -> None:
    for index in range(count):
        vault.write_belief(
            BeliefArtifact(entity=f"{prefix}-{index}", title="t", body=f"b{index}",
                           sources=["a.py:1"])
        )


def test_appends_stay_serialized_without_advisory_locks(tmp_path, no_advisory_locks):
    """The lock file must serialize on its own.

    With no effective lock at all the same workload leaves 55 of 60 records and
    a broken chain; `O_CREAT | O_EXCL` is what closes that, and it is the one
    mutual-exclusion primitive NFSv3+ and SMB implement atomically.
    """

    vault = VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))
    threads = [
        threading.Thread(target=_write_many, args=(vault, prefix, 20))
        for prefix in ("a", "b", "c")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    log = tmp_path / "vault" / "ledger" / "beliefs.jsonl"
    records = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(records) == 60
    assert BeliefLedger(tmp_path / "vault").verify_chain()["status"] == "intact"


def test_an_abandoned_lock_is_broken_rather_than_blocking_forever(tmp_path, monkeypatch):
    """A lock file outlives the process that made it.

    A crash mid-append would otherwise block every later writer permanently.
    """

    monkeypatch.setattr(vault_time, "_LOCK_STALE_SECONDS", 0.2)
    monkeypatch.setattr(vault_time, "_LOCK_SETTLE_SECONDS", 0.05)

    vault = VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))
    vault.ensure_structure()
    ledger_dir = tmp_path / "vault" / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    stranded = ledger_dir / ".lock"
    stranded.write_text("somehost:999999:deadbeef\n0\n", encoding="utf-8")
    time.sleep(0.3)

    vault.write_belief(
        BeliefArtifact(entity="after-crash", title="t", body="b", sources=["a.py:1"])
    )

    assert vault.read_belief("after-crash") is not None
    assert BeliefLedger(tmp_path / "vault").verify_chain()["status"] == "intact"


def test_a_live_lock_is_not_mistaken_for_an_abandoned_one(tmp_path, monkeypatch):
    """Age alone is not proof of abandonment; a slow holder is not a dead one.

    The mtime is read twice across a settle interval, so a lock that is still
    being touched is left alone.
    """

    monkeypatch.setattr(vault_time, "_LOCK_STALE_SECONDS", 0.2)
    monkeypatch.setattr(vault_time, "_LOCK_SETTLE_SECONDS", 0.3)

    ledger_dir = tmp_path / "vault" / "ledger"
    ledger_dir.mkdir(parents=True)
    lock_path = ledger_dir / ".lock"
    lock_path.write_text("host:1:token\n0\n", encoding="utf-8")
    time.sleep(0.3)

    keep_touching = True

    def toucher() -> None:
        while keep_touching:
            lock_path.touch()
            time.sleep(0.05)

    worker = threading.Thread(target=toucher)
    worker.start()
    try:
        assert vault_time._lock_is_stale(lock_path) is False
    finally:
        keep_touching = False
        worker.join()


def test_release_never_removes_another_holders_lock(tmp_path):
    """A broken-as-stale lock can be re-taken while the original holder runs.

    Deleting unconditionally would then release a lock the caller does not
    hold, letting two writers proceed at once.
    """

    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir(parents=True)
    lock_path = ledger_dir / ".lock"

    assert vault_time._try_acquire(lock_path, "holder-A") is True
    # A takes over after A's lock was broken; A must not delete B's claim.
    lock_path.write_text("holder-B\n0\n", encoding="utf-8")

    vault_time._release(lock_path, "holder-A")

    assert lock_path.exists()
    assert lock_path.read_text(encoding="utf-8").startswith("holder-B")


def test_exclusive_create_admits_exactly_one_claimant(tmp_path):
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir(parents=True)
    lock_path = ledger_dir / ".lock"

    assert vault_time._try_acquire(lock_path, "first") is True
    assert vault_time._try_acquire(lock_path, "second") is False

    vault_time._release(lock_path, "first")
    assert vault_time._try_acquire(lock_path, "second") is True


def test_lock_tokens_are_unique_across_machines(tmp_path):
    """A pid alone repeats across hosts sharing a network vault."""

    first, second = vault_time._lock_token(), vault_time._lock_token()
    assert first != second
    assert len(first.split(":")) == 3
