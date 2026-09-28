"""A second, contradicting claim about one event must not vanish.

`append` is idempotent on `event_id`, and SQLite enforces that with
`INSERT OR IGNORE`. But `IGNORE` cannot see *why* a row was rejected: it treats
a genuine replay and a conflicting re-assertion of the same `event_id` as the
same thing, and drops both. So a governance decision that flipped could be
appended and leave no trace anywhere.

Measured before this change, appending `event_id="E1"` twice -- first
`{"verdict": "allow"}` with `allowed=True`, then `{"verdict": "DENY"}` with
`allowed=False`:

    append() returned  payload {'verdict': 'DENY'}  chain_hash 75aea433…
                       -- the hash of the *allow* record, so the returned
                          record did not even hash to its own payload
    audit.jsonl        allow only
    query()            allow only
    verify_chain()     (True, 'Chain intact (1 records verified)')

Every surface reported a healthy log describing one allow. Nothing recorded
that a deny for the same event had been presented and discarded. For a
tamper-evident log that is the worst available failure: the DENY is exactly
the record an audit is kept for.

These tests pin the three-way split `IGNORE` collapses -- a new event, a
replay, and a conflict are different facts and must be reported as different
facts -- and that the conflict becomes durable rather than only logged.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from entroly.governance.audit import GovernanceAuditLog, _safe_json


@pytest.fixture()
def log(tmp_path):
    return GovernanceAuditLog(audit_dir=tmp_path / "audit")


def _chain_lines(log: GovernanceAuditLog) -> list[dict]:
    if not log.jsonl_path.exists():
        return []
    return [
        json.loads(line)
        for line in log.jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _db_rows(log: GovernanceAuditLog) -> int:
    conn = sqlite3.connect(str(log.db_path))
    try:
        return int(conn.execute("SELECT COUNT(*) FROM governance_audit").fetchone()[0])
    finally:
        conn.close()


def _flip(log: GovernanceAuditLog):
    """Append E1 as an allow, then contradict it with a deny."""
    first = log.append(
        event_id="E1", event_type="policy.decision", allowed=True,
        payload={"verdict": "allow", "rule": "r1"},
    )
    second = log.append(
        event_id="E1", event_type="policy.decision", allowed=False,
        payload={"verdict": "DENY", "rule": "r9"},
    )
    return first, second


# ── The three outcomes `INSERT OR IGNORE` collapses ──────────────────


def test_a_new_event_reports_that_it_was_recorded(log):
    record = log.append(event_id="E1", event_type="policy.decision", payload={"n": 1})

    assert record.outcome == "recorded"


def test_a_true_replay_reports_that_it_was_replayed(log):
    """Identical content under one event_id is real idempotence, not a conflict."""
    log.append(event_id="E1", event_type="policy.decision", payload={"n": 1})
    again = log.append(event_id="E1", event_type="policy.decision", payload={"n": 1})

    assert again.outcome == "replayed"


def test_a_true_replay_stays_a_silent_no_op(log):
    """The conflict path must not fire for a replay, or it inflates the log.

    Without this the fix would trade a dropped record for a fabricated one.
    """
    log.append(event_id="E1", event_type="policy.decision", payload={"n": 1})
    log.append(event_id="E1", event_type="policy.decision", payload={"n": 1})

    assert (_db_rows(log), len(_chain_lines(log))) == (1, 1)
    assert [r["event_type"] for r in _chain_lines(log)] == ["policy.decision"]


def test_a_conflicting_claim_reports_a_conflict(log):
    _first, second = _flip(log)

    assert second.outcome == "conflict"


# ── The returned record must describe something real ─────────────────


def test_the_record_returned_for_a_conflict_hashes_to_its_own_payload(log):
    """It carried the rejected payload with the stored record's hash.

    That record describes nothing that exists: its payload was discarded and
    its hash belongs to a different claim. A caller re-verifying what it was
    handed would compute a mismatch and have no way to tell that from tampering.
    """
    _first, second = _flip(log)

    own = hashlib.sha256(
        f"|{second.event_id}|{_safe_json(dict(second.payload))}".encode()
    ).hexdigest()
    assert second.chain_hash == own, (
        "the returned record does not hash to its own payload, so it is not a "
        "record of anything that happened"
    )


# ── The conflict has to outlive the process ──────────────────────────


def test_a_conflicting_claim_becomes_a_durable_record(log):
    """A log line is not an audit record. It must survive in both stores."""
    _flip(log)

    conflicts = [r for r in _chain_lines(log) if r["event_type"] == "audit.conflict"]
    assert len(conflicts) == 1, (
        "the contradicting claim left no durable trace; it was visible only to "
        "whoever was reading logs at the time"
    )
    assert _db_rows(log) == len(_chain_lines(log))


def test_the_conflict_record_carries_the_rejected_claim(log):
    """Recording that a conflict happened is not enough to audit it."""
    _flip(log)

    conflict = next(r for r in _chain_lines(log) if r["event_type"] == "audit.conflict")
    assert conflict["payload"]["conflicting_event_id"] == "E1"
    assert conflict["payload"]["rejected"]["payload"]["verdict"] == "DENY"
    assert conflict["payload"]["rejected"]["allowed"] is False
    assert conflict["payload"]["stored"]["payload"]["verdict"] == "allow"


def test_the_same_conflict_presented_twice_is_recorded_once(log):
    """A retry loop must not turn one disagreement into a thousand records."""
    _flip(log)
    log.append(
        event_id="E1", event_type="policy.decision", allowed=False,
        payload={"verdict": "DENY", "rule": "r9"},
    )

    conflicts = [r for r in _chain_lines(log) if r["event_type"] == "audit.conflict"]
    assert len(conflicts) == 1


def test_two_different_conflicts_are_both_recorded(log):
    """Deduplicating by event_id alone would collapse distinct disagreements."""
    _flip(log)
    log.append(
        event_id="E1", event_type="policy.decision", allowed=False,
        payload={"verdict": "DENY", "rule": "different"},
    )

    conflicts = [r for r in _chain_lines(log) if r["event_type"] == "audit.conflict"]
    assert len(conflicts) == 2


# ── Verification must still hold afterwards ──────────────────────────


def test_the_chain_survives_a_conflict(log):
    """The conflict record is appended through the chain, so it must verify."""
    _flip(log)

    ok, detail = log.verify_chain()
    assert ok, f"recording a conflict broke the chain: {detail}"


def test_the_conflict_is_visible_to_the_query_surface(log):
    """`query` reads the database; an operator must be able to find this."""
    _flip(log)

    found = log.query(event_type="audit.conflict", limit=10)
    assert len(found) == 1
    assert found[0]["payload"]["conflicting_event_id"] == "E1"
