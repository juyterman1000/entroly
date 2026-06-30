from __future__ import annotations

import pytest

from entroly.receipt_attestation import AttestationKey
from entroly.receipt_merkle import ReceiptMerkleLog
from entroly.receipt_witness import CosignedTreeHead, ReceiptWitness, WitnessRejection

pytest.importorskip("cryptography")


def test_witness_cosigns_valid_extension() -> None:
    operator = AttestationKey.generate()
    log = ReceiptMerkleLog(operator)
    for idx in range(3):
        log.append({"receipt_id": f"cr_{idx}"})
    old_head = log.signed_tree_head()
    witness = ReceiptWitness(name="w1")
    first = witness.cosign(old_head, operator_key=operator.public_hex())
    assert first.verify()

    log.append({"receipt_id": "cr_3"})
    new_head = log.signed_tree_head()
    second = witness.cosign(
        new_head,
        operator_key=operator.public_hex(),
        consistency_proof=log.prove_consistency(old_head.tree_size),
    )
    assert second.verify()


def test_witness_requires_valid_extension() -> None:
    operator = AttestationKey.generate()
    log = ReceiptMerkleLog(operator)
    for idx in range(3):
        log.append({"receipt_id": f"cr_{idx}"})
    old_head = log.signed_tree_head()
    witness = ReceiptWitness(name="w1")
    witness.cosign(old_head, operator_key=operator.public_hex())

    other = ReceiptMerkleLog(operator)
    for idx in range(4):
        other.append({"receipt_id": f"different_{idx}"})
    with pytest.raises(WitnessRejection):
        witness.cosign(
            other.signed_tree_head(),
            operator_key=operator.public_hex(),
            consistency_proof=other.prove_consistency(old_head.tree_size),
        )


def test_cosigned_threshold() -> None:
    operator = AttestationKey.generate()
    log = ReceiptMerkleLog(operator)
    log.append({"receipt_id": "cr"})
    head = log.signed_tree_head()
    witnesses = [ReceiptWitness(name=f"w{idx}") for idx in range(3)]
    signatures = tuple(w.cosign(head, operator_key=operator.public_hex()) for w in witnesses)
    cosigned = CosignedTreeHead(head, signatures)
    trusted = {w.witness_id for w in witnesses}
    assert cosigned.verify(operator_key=operator.public_hex(), trusted_witnesses=trusted, threshold=2)
    assert not cosigned.verify(operator_key=operator.public_hex(), trusted_witnesses=trusted, threshold=4)
