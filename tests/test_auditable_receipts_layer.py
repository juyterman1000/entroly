from __future__ import annotations

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
    assert proof.verify(operator_public_key=log.public_key)
    restored = ReceiptProof.from_dict(proof.to_dict())
    assert restored.verify(operator_public_key=log.public_key)


def test_operator_key_check() -> None:
    log = AuditableReceiptLog(prefer_rust=False)
    recorded = log.record_receipt(RECEIPT)
    proof = log.prove(recorded.index)
    other = AttestationKey.generate()
    assert not proof.verify(operator_public_key=other.public_hex())


def test_record_commitment_logs_public_summary() -> None:
    log = AuditableReceiptLog(prefer_rust=False)
    recorded, commitment = log.record_commitment(RECEIPT)
    assert recorded.receipt["commitment_root"] == commitment.root_hex()
    assert recorded.receipt["selected_count"] == 1
    assert recorded.receipt["omitted_count"] == 1
    assert log.prove(recorded.index).verify(operator_public_key=log.public_key)
