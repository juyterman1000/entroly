from __future__ import annotations

import dataclasses

import pytest

from entroly.receipt_disclosure import ProvenanceAtom, ReceiptCommitment, receipt_atoms


RECEIPT = {
    "receipt_id": "cr",
    "selected_context": [
        {"source_path": "income.pdf", "fingerprint": "sha256:aaa", "reasons": ["match"]},
    ],
    "omitted_context": [
        {"source_path": "protected.pdf", "fingerprint": "sha256:bbb", "reason": "not relevant"},
    ],
}


def test_receipt_atoms_cover_selected_and_omitted_sources() -> None:
    atoms = receipt_atoms(RECEIPT)
    assert {atom.status for atom in atoms} == {"selected", "omitted"}
    assert {atom.identifier for atom in atoms} == {"income.pdf", "protected.pdf"}


def test_disclosure_verifies_for_one_atom() -> None:
    commitment = ReceiptCommitment.from_receipt(RECEIPT)
    disclosure = commitment.disclose_where(identifier="protected.pdf")[0]
    assert disclosure.verify(commitment_root=commitment.root_hex())
    assert disclosure.atom.identifier == "protected.pdf"


def test_disclosure_serialization_roundtrip() -> None:
    commitment = ReceiptCommitment.from_receipt(RECEIPT)
    disclosure = commitment.disclose(0)
    from entroly.receipt_disclosure import AtomDisclosure
    restored = AtomDisclosure.from_dict(disclosure.to_dict())
    assert restored.verify(commitment_root=commitment.root_hex())


def test_disclosure_is_bound_to_committed_atom() -> None:
    commitment = ReceiptCommitment.from_receipt(RECEIPT)
    disclosure = commitment.disclose(0)
    changed = dataclasses.replace(
        disclosure,
        atom=ProvenanceAtom("source", "other.pdf", "sha256:zzz", "selected", ""),
    )
    assert not changed.verify(commitment_root=commitment.root_hex())


def test_same_atoms_have_different_roots_with_fresh_salts() -> None:
    assert ReceiptCommitment.from_receipt(RECEIPT).root_hex() != ReceiptCommitment.from_receipt(RECEIPT).root_hex()


def test_disclosure_rejects_malformed_hex_without_raising() -> None:
    disclosure = ReceiptCommitment.from_receipt(RECEIPT).disclose(0)
    changed = dataclasses.replace(disclosure, salt="not-hex")
    assert not changed.verify()


def test_commitment_rejects_short_salts() -> None:
    atom = ProvenanceAtom("source", "a", "sha256:a", "selected")
    with pytest.raises(ValueError, match="128 bits"):
        ReceiptCommitment([atom], salts=["00"])


def test_omission_reason_uses_context_receipt_schema() -> None:
    receipt = {
        "omitted_context": [
            {
                "source_path": "omitted.pdf",
                "fingerprint": "sha256:ccc",
                "omission_reason": "budget exhausted",
            }
        ]
    }
    assert receipt_atoms(receipt)[0].detail == "budget exhausted"
