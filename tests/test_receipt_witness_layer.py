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
    trusted = {w.witness_id: w.public_key for w in witnesses}
    assert cosigned.verify(operator_key=operator.public_hex(), trusted_witnesses=trusted, threshold=2)
    assert not cosigned.verify(operator_key=operator.public_hex(), trusted_witnesses=trusted, threshold=4)
    assert not cosigned.verify(operator_key=operator.public_hex(), trusted_witnesses=trusted, threshold=0)


def test_witness_name_cannot_impersonate_a_trusted_key() -> None:
    operator = AttestationKey.generate()
    log = ReceiptMerkleLog(operator)
    log.append({"receipt_id": "cr"})
    head = log.signed_tree_head()
    trusted = ReceiptWitness(name="auditor")
    impostor = ReceiptWitness(name="auditor")
    cosigned = CosignedTreeHead(
        head,
        (impostor.cosign(head, operator_key=operator.public_hex()),),
    )
    assert not cosigned.verify(
        operator_key=operator.public_hex(),
        trusted_witnesses={trusted.witness_id: trusted.public_key},
        threshold=1,
    )


def test_witness_tracks_extension_from_empty_tree() -> None:
    operator = AttestationKey.generate()
    log = ReceiptMerkleLog(operator)
    witness = ReceiptWitness(name="w1")
    witness.cosign(log.signed_tree_head(), operator_key=operator.public_hex())
    log.append({"receipt_id": "cr"})
    signature = witness.cosign(
        log.signed_tree_head(),
        operator_key=operator.public_hex(),
        consistency_proof=log.prove_consistency(0),
    )
    assert signature.verify(public_key=witness.public_key)
