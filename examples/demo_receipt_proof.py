#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from entroly.auditable_receipts import AuditableReceiptLog
from entroly.receipt_witness import CosignedTreeHead, ReceiptWitness


DOCS = [
    ("contract:section-7", "Indemnification survives termination of the Agreement."),
    ("contract:section-12", "Total liability is capped at fees paid in the prior year."),
    ("policy:data", "Customer data remains in EU regions unless approved in writing."),
]


def main() -> None:
    log = AuditableReceiptLog(prefer_rust=False)
    recorded = log.record(DOCS, query="does indemnification survive termination", token_budget=40)
    proof = log.prove(recorded.index)
    head = log.signed_tree_head()
    witnesses = [ReceiptWitness(name=f"witness-{idx}") for idx in range(3)]
    signatures = tuple(w.cosign(head, operator_key=log.public_key) for w in witnesses)
    cosigned = CosignedTreeHead(head, signatures)
    trusted = {w.witness_id for w in witnesses}

    print("receipt_id:", recorded.receipt_id)
    print("tree_size:", proof.tree_size)
    print("proof_valid:", proof.verify(operator_public_key=log.public_key))
    print("witness_quorum_valid:", cosigned.verify(operator_key=log.public_key, trusted_witnesses=trusted, threshold=2))


if __name__ == "__main__":
    main()
