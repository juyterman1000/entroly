from __future__ import annotations

import dataclasses

import pytest

from entroly.receipt_attestation import AttestationKey, AttestationLog, verify_attestation, verify_chain

pytest.importorskip("cryptography")


def _receipt(index: int = 1) -> dict:
    return {"receipt_id": f"cr_{index}", "selected_context": ["chunk"], "token_budget": 40}


def test_attestation_chain_validates() -> None:
    key = AttestationKey.generate()
    log = AttestationLog(key)
    for idx in range(4):
        log.append(_receipt(idx))
    result = verify_chain(log.entries, public_key=key.public_hex())
    assert result.valid
    assert result.entries_checked == 4


def test_attestation_detects_changed_receipt_body() -> None:
    key = AttestationKey.generate()
    log = AttestationLog(key)
    entry = log.append(_receipt())
    changed = dataclasses.replace(entry, receipt={**entry.receipt, "token_budget": 999})
    assert not verify_attestation(changed, public_key=key.public_hex())


def test_attestation_rejects_unexpected_key() -> None:
    key = AttestationKey.generate()
    other = AttestationKey.generate()
    entry = AttestationLog(key).append(_receipt())
    assert verify_attestation(entry, public_key=key.public_hex())
    assert not verify_attestation(entry, public_key=other.public_hex())
