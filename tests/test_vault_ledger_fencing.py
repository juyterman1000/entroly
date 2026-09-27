"""Write attribution on the belief ledger — who wrote a record, and fenced or not.

The lock in `vault_time` has three exits and the record could not tell them
apart. It acquires; or it breaks an abandoned lock and acquires; or it gives up
after `_LOCK_TIMEOUT_SECONDS` and appends anyway, because a ledger that refuses
to record is worse than one that can be raced. There is a fourth: a lock broken
as stale can be re-taken while the original holder is still inside its critical
section, and `_release` notices that only *after* the append has landed.

So an unserialized write produced a record indistinguishable from a serialized
one, and `verify_chain` reported the resulting fork as `prev_sha256 mismatch` --
the same words it uses for an edited record. Those need opposite responses: one
is a lock to fix, the other is an intrusion. These tests pin the difference.

Attribution is a diagnostic, not a seal. The chain is unkeyed, so write access
is still enough to forge a consistent history; `test_an_edited_record_is_not
_excused_as_a_concurrent_write` is the guard that the new classification never
turns a broken chain into an intact one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import entroly.vault_time as vault_time
from entroly.vault import BeliefArtifact, VaultConfig, VaultManager
from entroly.vault_time import BeliefLedger


def _records(vault_base: Path) -> list[dict[str, Any]]:
    log = vault_base / "ledger" / "beliefs.jsonl"
    return [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_one(vault_base: Path, entity: str = "e") -> None:
    vault = VaultManager(VaultConfig(base_path=str(vault_base)))
    vault.write_belief(
        BeliefArtifact(entity=entity, title="t", body=f"body-{entity}",
                       sources=["a.py:1"])
    )


def _hold_lock(vault_base: Path, token: str = "other:1:held") -> Path:
    """Leave a live lock file nobody will release."""
    ledger_dir = vault_base / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_dir / ".lock"
    lock_path.write_text(f"{token}\n0\n", encoding="utf-8")
    return lock_path


# ── Attribution on the record ────────────────────────────────────────


def test_a_serialized_append_records_the_authority_that_wrote_it(tmp_path):
    """A record must say which writer produced it, and that it was fenced."""

    base = tmp_path / "vault"
    _write_one(base)

    record = _records(base)[-1]
    assert record["serialized"] is True
    assert len(record["writer"]) == 16
    int(record["writer"], 16)  # a hex digest, not a raw token


def test_the_writer_digest_does_not_publish_the_hostname_or_pid(tmp_path):
    """The ledger is append-only and `redact` cannot un-say metadata.

    `_lock_token` is `hostname:pid:uuid`, so stamping it raw would put the
    writing machine's identity into a permanent record that redaction is
    documented as unable to remove.
    """

    base = tmp_path / "vault"
    _write_one(base)

    raw = (base / "ledger" / "beliefs.jsonl").read_text(encoding="utf-8")
    import socket

    assert socket.gethostname() not in raw
    assert ":" not in _records(base)[-1]["writer"]


def test_two_appends_from_one_process_share_a_writer_digest(tmp_path):
    """Attribution must identify the writer, not the individual append.

    `_lock_token` ends in a fresh uuid per acquisition, so digesting the whole
    token gave every record a different writer -- including consecutive writes
    from one process. That reads like two racing machines and cannot group
    records by who wrote them, which is the only question a fork diagnosis
    needs to ask. The uuid still belongs in the lock file, where it makes
    custody checks exact; it does not belong in the writer's identity.
    """

    base = tmp_path / "vault"
    _write_one(base, entity="first")
    _write_one(base, entity="second")

    first, second = _records(base)
    assert first["writer"] == second["writer"]
    assert first["writer"] != ""


def test_a_distinct_writer_gets_a_distinct_digest(tmp_path):
    """The grouping must still discriminate, or it groups everything."""

    same = vault_time._writer_digest("host-a:100:aaaa")
    again = vault_time._writer_digest("host-a:100:bbbb")
    other_pid = vault_time._writer_digest("host-a:101:aaaa")
    other_host = vault_time._writer_digest("host-b:100:aaaa")

    assert same == again  # same writer, different acquisition
    assert other_pid != same
    assert other_host != same


def test_an_append_that_never_took_the_lock_is_recorded_as_unserialized(
    tmp_path, monkeypatch
):
    """The timeout path appends anyway; the record must admit it.

    Previously this logged `appending unserialized` and wrote a record that
    looked exactly like a fenced one, so the only trace was a log line.
    """

    monkeypatch.setattr(vault_time, "_LOCK_TIMEOUT_SECONDS", 0.2)
    base = tmp_path / "vault"
    VaultManager(VaultConfig(base_path=str(base))).ensure_structure()
    _hold_lock(base)  # fresh, so never broken as stale

    _write_one(base)

    record = _records(base)[-1]
    assert record["serialized"] is False
    assert record["writer"] == ""


def test_a_takeover_while_the_lock_is_held_marks_the_write_unserialized(
    tmp_path, monkeypatch
):
    """A stale-break takeover must be caught before the append, not after.

    `_release` already refuses to delete another holder's lock, but that guard
    runs when the critical section exits -- the record is on disk by then.

    The advisory lock is disabled because that is the only world where this is
    reachable. Measured while writing this test: with the advisory lock held,
    Windows refuses both the clobber (`PermissionError`) and the unlink
    (`WinError 32`), so the OS prevents the takeover outright. On the NFS and
    SMB mounts `_try_acquire` was written for, `_advisory_lock` returns False,
    nothing holds the file open, and the takeover is fully available.
    """

    monkeypatch.setattr(vault_time, "_advisory_lock", lambda handle: False)
    monkeypatch.setattr(vault_time, "_advisory_unlock", lambda handle: None)

    base = tmp_path / "vault"
    VaultManager(VaultConfig(base_path=str(base))).ensure_structure()

    real_last_record = BeliefLedger._last_record
    seized: list[bool] = []

    def seize_then_read(self):
        # Runs inside the lock, before the record is built: exactly where a
        # writer that broke our lock as stale would have re-taken it.
        if not seized:
            (self._dir / ".lock").write_text("thief:2:taken\n0\n", encoding="utf-8")
            seized.append(True)
        return real_last_record(self)

    monkeypatch.setattr(BeliefLedger, "_last_record", seize_then_read)

    _write_one(base)

    assert seized, "the takeover was never injected"
    assert _records(base)[-1]["serialized"] is False


def test_a_redaction_tombstone_carries_attribution_too(tmp_path):
    """Redaction appends through the same path, so it is a write like any other."""

    base = tmp_path / "vault"
    _write_one(base, entity="secret")
    BeliefLedger(base).redact(entity="secret")

    tombstone = _records(base)[-1]
    assert tombstone["kind"] == "redaction"
    assert tombstone["serialized"] is True
    assert len(tombstone["writer"]) == 16


# ── What verification can now say ────────────────────────────────────


def _fork_the_chain(base: Path) -> None:
    """Append a second record claiming the same position, as a racing writer does.

    Two writers that read the same tail both chain onto it. The result is two
    records with the same `seq` and the same `prev_sha256`, each internally
    consistent, written under different authorities.
    """
    records = _records(base)
    rival = dict(records[-1])
    rival["entity"] = "rival"
    rival["writer"] = "f" * 16
    rival.pop("record_sha256")
    rival["record_sha256"] = vault_time._record_hash(rival)

    log = base / "ledger" / "beliefs.jsonl"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rival, sort_keys=True, ensure_ascii=False) + "\n")


def test_a_concurrent_fork_is_named_as_one_rather_than_as_tampering(tmp_path):
    """Same fork, different diagnosis -- a lock failure, not an intrusion."""

    base = tmp_path / "vault"
    _write_one(base, entity="first")
    _write_one(base, entity="second")
    _fork_the_chain(base)

    result = BeliefLedger(base).verify_chain()

    assert result["concurrent_write"] is True
    assert "concurrent" in result["reason"]


def test_a_concurrent_fork_still_leaves_the_chain_broken(tmp_path):
    """Classifying a break must never downgrade it to intact.

    The chain is unkeyed: write access is enough to fabricate a record that
    forks from a real ancestor under an invented authority. Naming the likely
    cause is a diagnostic; treating it as healthy would be a fail-open hole.
    """

    base = tmp_path / "vault"
    _write_one(base, entity="first")
    _fork_the_chain(base)

    assert BeliefLedger(base).verify_chain()["status"] == "broken"


def test_an_edited_record_is_not_excused_as_a_concurrent_write(tmp_path):
    """The guard on the new classification.

    A record whose own hash does not recompute is tampering however its
    `prev_sha256` reads, and must not borrow the gentler label.
    """

    base = tmp_path / "vault"
    _write_one(base, entity="first")
    _write_one(base, entity="second")

    log = base / "ledger" / "beliefs.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    edited = json.loads(lines[-1])
    edited["confidence"] = 0.99  # body changed, hash left alone
    lines[-1] = json.dumps(edited, sort_keys=True, ensure_ascii=False)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = BeliefLedger(base).verify_chain()

    assert result["status"] == "broken"
    assert result.get("concurrent_write") is not True
    assert "record_sha256" in result["reason"]


def test_a_record_forking_from_nothing_is_tampering_not_a_race(tmp_path):
    """A racing writer forks from an ancestor that exists; a forger may not.

    Dropping a record from the middle leaves its successor pointing at a hash
    no longer in the file. That is removal, not contention.
    """

    base = tmp_path / "vault"
    for entity in ("first", "second", "third"):
        _write_one(base, entity=entity)

    log = base / "ledger" / "beliefs.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    del lines[1]  # third still chains onto the deleted second
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = BeliefLedger(base).verify_chain()

    assert result["status"] == "broken"
    assert result.get("concurrent_write") is not True


# ── Compatibility with ledgers written before attribution ────────────


def test_records_written_before_attribution_still_verify(tmp_path):
    """Old records have no `writer` or `serialized`; the hash covers what is there.

    A ledger that stopped verifying after an upgrade would report every
    existing vault as tampered.
    """

    base = tmp_path / "vault"
    ledger_dir = base / "ledger"
    ledger_dir.mkdir(parents=True)

    legacy = {
        "schema": vault_time.LEDGER_SCHEMA,
        "seq": 1,
        "tx_time": "2026-01-01T00:00:00+00:00",
        "valid_time": "",
        "claim_id": "c1",
        "entity": "legacy",
        "status": "inferred",
        "confidence": 0.5,
        "sources": ["a.py:1"],
        "title": "legacy",
        "body_sha256": hashlib.sha256(b"x").hexdigest(),
        "backfilled": False,
        "prev_sha256": "",
    }
    legacy["record_sha256"] = vault_time._record_hash(legacy)
    (ledger_dir / "beliefs.jsonl").write_text(
        json.dumps(legacy, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    assert BeliefLedger(base).verify_chain()["status"] == "intact"


def test_appending_after_a_legacy_record_keeps_the_chain_intact(tmp_path):
    """Attribution starts mid-ledger without breaking the link across the seam."""

    test_records_written_before_attribution_still_verify(tmp_path)
    base = tmp_path / "vault"

    _write_one(base, entity="modern")

    records = _records(base)
    assert "writer" not in records[0]
    assert records[1]["serialized"] is True
    assert BeliefLedger(base).verify_chain()["status"] == "intact"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
