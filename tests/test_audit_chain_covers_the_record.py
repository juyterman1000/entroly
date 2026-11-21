"""The audit chain must commit to the whole record, not just its payload.

The chain hash was ``SHA-256(prev | event_id | payload)``, leaving seven of
the eleven stored fields outside the commitment. Measured against the shipped
verifier, every one of these was accepted as intact *after* the fact was
changed::

    flip allowed False->True : ok=true   <- the authorization decision itself
    rewrite agent_id         : ok=true   <- who performed the action
    rewrite created_at       : ok=true   <- when it happened
    rewrite event_type       : ok=true   <- what it was
    rewrite payload (control): ok=false  <- the one field that was covered

For a log whose purpose is answering "who did what, and were they allowed", a
commitment covering none of those is not an audit trail. The control matters:
it shows the verifier worked and the gap was specific.

A second defect surfaced while fixing it: ``append`` hashed a *stub* record
carrying only event_id, event_type and payload -- no agent, no decision, and
its own ``created_at`` from the default factory rather than the one persisted.
Invisible while the canon covered the payload alone.

The chain remains unkeyed: anyone who can write both sinks can forge a
self-consistent history. These tests pin tamper-evidence against edits, which
is what the structure can provide, not authentication.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from entroly.governance.audit import GovernanceAuditLog, _safe_json


@pytest.fixture
def log(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_AUDIT_DIR", str(tmp_path / "audit"))
    instance = GovernanceAuditLog()
    instance.append(
        event_id="evt-0", event_type="tool_call", agent_id="agent:alice",
        session_id="s1", org_id="acme", trace_id="t1",
        payload={"tool": "delete_prod_db"}, allowed=False,
    )
    instance.append(
        event_id="evt-1", event_type="tool_call", agent_id="agent:alice",
        payload={"tool": "read"}, allowed=True,
    )
    return instance


def _rewrite_first(log_instance, **changes):
    lines = log_instance.jsonl_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    records[0].update(changes)
    log_instance.jsonl_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )


def test_an_untampered_chain_verifies(log) -> None:
    ok, detail = log.verify_chain()
    assert ok, detail


@pytest.mark.parametrize("changes,label", [
    ({"allowed": True}, "a denial rewritten as an approval"),
    ({"allowed": None}, "a denial erased to no-decision"),
    ({"agent_id": "agent:mallory"}, "the actor reassigned"),
    ({"created_at": 1.0}, "the timestamp moved"),
    ({"event_type": "login"}, "the event kind changed"),
    ({"session_id": "other"}, "the session reassigned"),
    ({"org_id": "other"}, "the organisation reassigned"),
    ({"trace_id": "other"}, "the trace reassigned"),
    ({"payload": {"tool": "harmless"}}, "the payload rewritten (control)"),
])
def test_tampering_is_detected(log, changes, label) -> None:
    _rewrite_first(log, **changes)
    ok, _detail = GovernanceAuditLog().verify_chain()
    assert not ok, f"{label} was accepted as intact"


def test_a_legacy_archive_still_verifies_and_says_it_is_weaker(log) -> None:
    """Rejecting every pre-existing archive would be its own false alarm.

    They must keep verifying -- and the report must not call them equal to a
    fully sealed chain, because their agent, decision and timestamp were never
    covered.
    """
    lines = log.jsonl_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    previous = ""
    for record in records:
        legacy = hashlib.sha256(
            f"{previous}|{record['event_id']}|{_safe_json(record['payload'])}".encode()
        ).hexdigest()
        record["chain_hash"] = legacy
        previous = legacy
    log.jsonl_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    connection = sqlite3.connect(log.db_path)
    for record in records:
        connection.execute(
            "UPDATE governance_audit SET chain_hash=? WHERE event_id=?",
            (record["chain_hash"], record["event_id"]),
        )
    connection.commit()
    connection.close()

    ok, detail = GovernanceAuditLog().verify_chain()
    assert ok, detail
    assert "payload edits only" in detail, (
        f"a payload-only chain was reported as fully intact: {detail!r}"
    )


def test_the_stored_hash_describes_the_stored_record(log) -> None:
    """`append` hashed a stub, so the hash described a record never written."""
    record = json.loads(log.jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    allowed = "" if record["allowed"] is None else ("true" if record["allowed"] else "false")
    canon = "\x1f".join((
        "v2", "", record["event_id"], record["event_type"], record["agent_id"],
        record["session_id"], record["org_id"], record["trace_id"], allowed,
        str(int(record["created_at"] * 1000)), _safe_json(record["payload"]),
    ))
    assert record["chain_hash"] == hashlib.sha256(canon.encode("utf-8")).hexdigest()
