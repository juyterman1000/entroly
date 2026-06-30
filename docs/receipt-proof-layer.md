# Receipt proof layer

This branch adds a small proof layer on top of Context Receipts.

## Included

- `receipt_attestation.py`: signed receipt custody chain.
- `receipt_merkle.py`: Merkle inclusion and prefix proofs for receipt roots.
- `receipt_disclosure.py`: salted commitments for opening one provenance atom at a time.
- `receipt_witness.py`: witness signatures for signed tree heads.
- `auditable_receipts.py`: facade that records context receipts and returns portable proofs.
- `examples/demo_receipt_proof.py`: minimal demo.

## Excluded

The uploaded review package also included edits to `cli.py`, `server.py`, `proxy.py`, and self-healing logic. Those are intentionally excluded from this branch because they touch runtime behavior and should be reviewed separately.

## Usage

```python
from entroly.auditable_receipts import AuditableReceiptLog

log = AuditableReceiptLog(prefer_rust=False)
recorded = log.record_receipt({"receipt_id": "example", "selected_context": []})
proof = log.prove(recorded.index)
assert proof.verify(operator_public_key=log.public_key)
```

The root package exports are intentionally not changed in this PR. Use explicit module imports.
