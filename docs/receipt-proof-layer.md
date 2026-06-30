# Receipt proof layer

This layer adds cryptographic proofs on top of Context Receipts.

Install the optional cryptographic dependency with:

```bash
python -m pip install "entroly[receipt-proof]"
```

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
assert proof.verify(recorded.receipt, operator_public_key=log.public_key)
```

`verify` requires both the expected operator public key and the claimed receipt.
This binds the signed Merkle inclusion proof to the exact receipt contents instead
of only proving membership of an opaque leaf hash.

Witness quorums pin each witness name to its Ed25519 public key:

```python
trusted_witnesses = {witness.witness_id: witness.public_key}
```

The current log and witness state are in memory. Persist tree heads, consistency
proofs, witness checkpoints, and private keys in an access-controlled store before
using this layer across process restarts. A proof is trusted only when its operator
key and witness-key mapping come from a separately trusted configuration.

The root package exports are intentionally unchanged. Use explicit module imports.
