"""What `audit verify` claims must match what the chain actually detects.

The chain is unkeyed SHA-256 over `(prev_hash, event_id, payload)`. That is
tamper-*evidence* against accidental corruption and naive edits, and nothing
more: anyone who can write the file can recompute the chain and produce a
history that verifies. Measured against a five-record log, verification does
not detect tail truncation, an edit whose chain was recomputed, or a wholly
fabricated log -- it reported "Chain intact" over attacker-written records
granting admin.

The CLI first said chain verification "proves the recorded entries were not
altered after the fact", which is exactly the overclaim the trust invariants
forbid. These tests pin the real boundary in both directions, so the claim
cannot drift away from the mechanism and the mechanism cannot quietly weaken.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from entroly.governance.audit import GovernanceAuditLog, _safe_json


@pytest.fixture()
def chain(tmp_path):
    log = GovernanceAuditLog(audit_dir=tmp_path / "audit")
    for index in range(5):
        log.append(
            event_id=f"e{index}", event_type="permission.denied",
            agent_id="a", payload={"scope": "write", "i": index}, allowed=False,
        )
    return log


def _records(path: pathlib.Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _write(path: pathlib.Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _rechain(records: list[dict]) -> list[dict]:
    """Recompute the whole chain, as any writer of the file can."""
    prev = ""
    for record in records:
        record["chain_hash"] = hashlib.sha256(
            f"{prev}|{record['event_id']}|{_safe_json(record.get('payload', {}))}".encode()
        ).hexdigest()
        prev = record["chain_hash"]
    return records


def _verify(log: GovernanceAuditLog) -> tuple[bool, str]:
    log._initialized = False          # re-read the file rather than cached state
    return log.verify_chain()


def test_detects_a_payload_edited_in_place(chain):
    records = _records(chain.jsonl_path)
    records[2]["payload"]["scope"] = "admin"
    _write(chain.jsonl_path, records)
    ok, detail = _verify(chain)
    assert ok is False, "an edited payload verified as intact"
    assert "e2" in detail, f"the break was not located: {detail}"


def test_detects_a_record_removed_from_the_middle(chain):
    records = _records(chain.jsonl_path)
    del records[2]
    _write(chain.jsonl_path, records)
    assert _verify(chain)[0] is False, "a deleted record verified as intact"


def test_does_not_detect_tail_truncation(chain):
    """Dropping the newest records leaves every remaining link valid.

    Not a bug in the implementation -- a bare hash chain has no anchor for the
    expected length. Pinned so the CLI never claims otherwise.
    """
    _write(chain.jsonl_path, _records(chain.jsonl_path)[:3])
    assert _verify(chain)[0] is True


def test_does_not_detect_an_edit_whose_chain_was_recomputed(chain):
    """The chain is unkeyed: write access is enough to forge a history."""
    records = _records(chain.jsonl_path)
    records[2]["payload"]["scope"] = "admin"
    _write(chain.jsonl_path, _rechain(records))
    assert _verify(chain)[0] is True


def test_does_not_detect_a_fabricated_self_consistent_log(chain):
    fabricated = [
        {"event_id": f"x{i}", "event_type": "permission.allowed", "agent_id": "attacker",
         "payload": {"scope": "admin"}, "allowed": True, "created_at": 0, "chain_hash": ""}
        for i in range(3)
    ]
    _write(chain.jsonl_path, _rechain(fabricated))
    assert _verify(chain)[0] is True


def test_the_cli_states_the_boundary_it_actually_has(tmp_path, capsys, monkeypatch):
    """The emitted claim must match the measured behaviour above.

    Asserted on the payload the command produces, not on its source text: an
    earlier version grepped the function body and matched the explanatory
    comment quoting the *old* wording, so it failed while the behaviour was
    already correct. A claim test that reads prose tests prose.
    """
    from types import SimpleNamespace

    import entroly.governance.audit as audit_module
    from entroly.cli_governance import _cmd_audit

    # `_cmd_audit` resolves the log through `get_audit_log()`, so isolating it
    # means pointing the resolver at tmp_path and clearing the process-global
    # singleton. Passing an `audit_dir` field on the namespace would be read by
    # nothing and would quietly exercise the operator's real ~/.entroly.
    monkeypatch.setenv("ENTROLY_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setattr(audit_module, "_global_log", None)

    _cmd_audit(SimpleNamespace(audit_action="verify", json_output=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["jsonl_path"].startswith(str(tmp_path)), (
        f"the test escaped its temp directory: {payload['jsonl_path']}"
    )

    assert "unkeyed" in payload["claim_boundary"].lower(), (
        "the claim no longer says the chain is unkeyed; without that, "
        f"'chain intact' reads as integrity it cannot provide: {payload['claim_boundary']}"
    )
    # The three tests above measure exactly these as undetectable.
    undetected = " ".join(payload["does_not_detect"]).lower()
    for limitation in ("truncat", "recomputed", "fabricated"):
        assert limitation in undetected, (
            f"{limitation!r} is undetectable but is not disclosed: {payload['does_not_detect']}"
        )
    assert payload["detects"], "the claim lists no detected tampering at all"
