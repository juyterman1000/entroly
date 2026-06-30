from __future__ import annotations

import dataclasses

import pytest

from entroly.auditable_receipts import AuditableReceiptLog, ReceiptProof, RecordedReceipt
from entroly.receipt_attestation import AttestationKey

pytest.importorskip("cryptography")


RECEIPT = {
    "receipt_id": "cr_manual",
    "selected_context": [{"source_path": "income.pdf", "fingerprint": "sha256:aaa", "reasons": ["match"]}],
    "omitted_context": [{"source_path": "other.pdf", "fingerprint": "sha256:bbb", "reason": "not relevant"}],
}


def test_recorded_receipt_has_portable_proof() -> None:
    log = AuditableReceiptLog(prefer_rust=False)
    recorded = log.record_receipt(RECEIPT)
    assert isinstance(recorded, RecordedReceipt)
    proof = log.prove(recorded.index)
    assert proof.verify(recorded.receipt, operator_public_key=log.public_key)
    restored = ReceiptProof.from_dict(proof.to_dict())
    assert restored.verify(recorded.receipt, operator_public_key=log.public_key)


def test_operator_key_check() -> None:
    log = AuditableReceiptLog(prefer_rust=False)
    recorded = log.record_receipt(RECEIPT)
    proof = log.prove(recorded.index)
    other = AttestationKey.generate()
    assert not proof.verify(recorded.receipt, operator_public_key=other.public_hex())


def test_proof_is_bound_to_receipt_and_signed_timestamp() -> None:
    log = AuditableReceiptLog(prefer_rust=False)
    recorded = log.record_receipt(RECEIPT)
    proof = log.prove(recorded.index)
    changed = {**recorded.receipt, "receipt_id": "substituted"}
    assert not proof.verify(changed, operator_public_key=log.public_key)
    changed_time = dataclasses.replace(proof, timestamp=proof.timestamp + 1)
    assert not changed_time.verify(recorded.receipt, operator_public_key=log.public_key)


def test_recording_uses_an_immutable_snapshot() -> None:
    source = {"receipt_id": "snapshot", "nested": {"value": 1}}
    log = AuditableReceiptLog(prefer_rust=False)
    recorded = log.record_receipt(source)
    source["nested"]["value"] = 2
    exposed = recorded.receipt
    exposed["nested"]["value"] = 3
    assert recorded.receipt["nested"]["value"] == 1
    assert log.prove(recorded.index).verify(
        recorded.receipt, operator_public_key=log.public_key
    )


def test_record_commitment_logs_public_summary() -> None:
    log = AuditableReceiptLog(prefer_rust=False)
    recorded, commitment = log.record_commitment(RECEIPT)
    assert recorded.receipt["commitment_root"] == commitment.root_hex()
    assert recorded.receipt["selected_count"] == 1
    assert recorded.receipt["omitted_count"] == 1
    assert log.prove(recorded.index).verify(
        recorded.receipt, operator_public_key=log.public_key
    )
