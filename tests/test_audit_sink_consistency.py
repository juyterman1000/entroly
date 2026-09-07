"""The audit log's two stores must describe the same history.

`append` writes a SQLite row and a JSONL line. They had different rules: the
database rejected a repeated `event_id` through its UNIQUE constraint while the
JSONL was an unconditional write. `query` reads the database and `verify_chain`
walks the JSONL, so the two surfaces answered differently and the inflated one
was the one certified intact -- measured at 2 rows against 3 chain lines, with
`verify_chain` reporting "Chain intact (3 records verified)".

Worse than the inflation was the anchor. `_prev_hash` advanced for the ignored
duplicate, and on restart was restored from the database, so the next record was
hashed from a link that no chain line carried. A duplicate followed by a restart
made `verify_chain` report "Chain broken" at a record nobody had touched. An
integrity check that cries wolf is one operators learn to ignore, so a false
alarm is not a lesser failure here than a missed one.

These tests pin prevention. The detection half -- what verification catches once
the stores disagree anyway -- lives in test_audit_chain_claim_honesty.py.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from entroly.governance.audit import GovernanceAuditLog


@pytest.fixture()
def log(tmp_path):
    return GovernanceAuditLog(audit_dir=tmp_path / "audit")


def _db_rows(log: GovernanceAuditLog) -> int:
    conn = sqlite3.connect(str(log.db_path))
    try:
        return int(conn.execute("SELECT COUNT(*) FROM governance_audit").fetchone()[0])
    finally:
        conn.close()


def _chain_lines(log: GovernanceAuditLog) -> list[dict]:
    if not log.jsonl_path.exists():
        return []
    return [
        json.loads(line)
        for line in log.jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_a_repeated_event_is_not_written_to_either_store_twice(log):
    """"Idempotent on event_id" has to mean both stores, or it means neither."""
    log.append(event_id="E1", event_type="permission.denied", payload={"n": 1})
    log.append(event_id="E1", event_type="permission.denied", payload={"n": 1})
    log.append(event_id="E2", event_type="permission.denied", payload={"n": 2})

    rows, lines = _db_rows(log), _chain_lines(log)
    assert (rows, len(lines)) == (2, 2), (
        f"two distinct events produced {rows} database rows and {len(lines)} "
        "chain lines; the stores disagree about what happened"
    )
    assert [r["event_id"] for r in lines] == ["E1", "E2"]


def test_a_repeated_event_does_not_advance_the_chain_anchor(log):
    """A link must not be spent on a record that was never stored.

    The anchor is what the next record hashes from. Advancing it for a dropped
    duplicate leaves the next genuine record chained to a hash that appears in
    no store, which reads as tampering later.
    """
    log.append(event_id="X1", event_type="t", payload={})
    anchor_after_first = log._prev_hash

    log.append(event_id="X1", event_type="t", payload={})
    assert log._prev_hash == anchor_after_first, (
        "the chain anchor moved for a duplicate that was never persisted"
    )

    # The next real record must still chain cleanly onto the stored one.
    log.append(event_id="X2", event_type="t", payload={})
    ok, detail = log.verify_chain()
    assert ok, f"the chain did not survive a duplicate append: {detail}"


def test_a_duplicate_then_a_restart_is_not_reported_as_tampering(tmp_path):
    """The regression that made the audit system accuse itself.

    Before the fix: the duplicate pushed the chain past the database, the
    restart re-anchored on the database, and the next record could not link to
    the preceding chain line -- "Chain broken at event R3", with nothing having
    been touched.
    """
    audit_dir = tmp_path / "audit"
    first = GovernanceAuditLog(audit_dir=audit_dir)
    first.append(event_id="R1", event_type="t", payload={})
    first.append(event_id="R2", event_type="t", payload={})
    first.append(event_id="R2", event_type="t", payload={})  # duplicate is last

    reopened = GovernanceAuditLog(audit_dir=audit_dir)  # a new process
    reopened.append(event_id="R3", event_type="t", payload={})

    ok, detail = reopened.verify_chain()
    assert ok, f"a healthy log was reported as tampered with: {detail}"


def test_the_anchor_is_read_from_the_chain_not_the_database(tmp_path):
    """The JSONL is the authoritative chain, so it anchors the next hash.

    Reproduced with the stores already divergent -- which a half-completed
    write, a killed process, or a log written by an older build can all
    produce. Anchoring on the database there would hash the next record from a
    link the chain does not end with, breaking it for good.
    """
    audit_dir = tmp_path / "audit"
    log = GovernanceAuditLog(audit_dir=audit_dir)
    log.append(event_id="A1", event_type="t", payload={})
    log.append(event_id="A2", event_type="t", payload={})

    # A chain line the database never received.
    orphan = dict(_chain_lines(log)[-1])
    orphan["event_id"] = "A3"
    orphan["chain_hash"] = "f" * 64
    with open(log.jsonl_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(orphan, separators=(",", ":")) + "\n")

    reopened = GovernanceAuditLog(audit_dir=audit_dir)
    reopened._ensure_initialized()
    assert reopened._prev_hash == "f" * 64, (
        f"the anchor came from the database, not the chain: {reopened._prev_hash[:16]}"
    )

    # And the record appended next links onto that final chain line, so the
    # break is confined to the pre-existing divergence.
    reopened.append(event_id="A4", event_type="t", payload={})
    assert _chain_lines(reopened)[-1]["event_id"] == "A4"
    ok, detail = reopened.verify_chain()
    assert "Chain broken at event A4" not in detail, (
        f"the newly appended record failed to link onto the chain: {detail}"
    )


def test_a_partial_walk_is_not_reported_as_a_verified_chain(log):
    """Reading a prefix must not be described as verifying the chain.

    `verify_chain` stops at `limit`. Saying "Chain intact" after reading the
    first few records of a long log claims coverage it does not have, and the
    count cross-check cannot run on a prefix either.
    """
    for index in range(5):
        log.append(event_id=f"p{index}", event_type="t", payload={"i": index})

    ok, detail = log.verify_chain(limit=2)
    assert ok is True
    assert "limit" in detail.lower(), (
        f"a truncated walk did not disclose that it stopped early: {detail}"
    )
    assert "intact" not in detail.lower(), (
        f"a partial read described the whole chain as intact: {detail}"
    )
