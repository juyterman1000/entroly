# Context Commits

A Context Commit is a portable, content-addressed record of the context Entroly
selected for an agent request. It turns context assembly into an artifact that
can be replayed, inspected, recovered, and independently checked for mutation.

## Contract

`entroly.context-commit.v1` binds these fields to one `ctx_...` identifier:

- the complete Context Receipt;
- exact ordered selected context and its digest;
- omitted-context metadata and a fingerprint-verified recovery bundle;
- the Python or Rust receipt engine and package versions;
- an optional parent Context Commit identifier for session lineage.

Creation is deterministic for the same normalized documents, configuration,
query, engine, and parent. Input document order does not affect the identifier.

## CLI

```bash
entroly context-commit ./repo --query "Where is token rotation enforced?" \
  --budget 8000 --out context-commit.json
entroly context-commit --verify context-commit.json
```

Use `--python` to force the reference receipt implementation. Without it,
Entroly uses the Rust receipt engine when the installed native package exposes
the required symbols and otherwise records the Python fallback explicitly.

## Python

```python
from entroly import create_context_commit, replay_context, verify_context_commit

commit = create_context_commit(
    [("policy.md", policy_text), ("service.py", service_source)],
    query="Which policy controls deployment approval?",
    token_budget=4000,
    parent_commit_id=previous_commit_id,
)

verification = verify_context_commit(commit)
if not verification.valid:
    raise RuntimeError(verification.errors)

selected_context = replay_context(commit)
```

Verification fails closed when the commit identifier, receipt hash, selected
text, recovery text, engine metadata, or lineage is mutated. Every omitted
chunk must recover with matching chunk and content fingerprints.

## Threat Model

Context Commits detect mutation after creation. They do not by themselves prove
who created the artifact or whether an untrusted creator omitted evidence before
Entroly received the documents.

For operator identity and custody, use `AttestationLog` with Ed25519 keys. For
append-only publication and inclusion proofs, use `AuditableReceiptLog`. For
selective disclosure without exposing all source identifiers, use
`ReceiptCommitment`.

Recovery bundles contain source text. Store them under the same access,
retention, and deletion policy as the original source. Do not publish a commit
artifact merely because its hashes verify.

## Conformance Benchmark

The committed benchmark runs 64 deterministic cases per available receipt
engine. Each case creates the same commit from normal and reversed input order,
replays selected context, verifies every omitted chunk, and applies six distinct
mutations.

| Measurement | Result |
|---|---:|
| Engine modes | 2 (Python and Rust) |
| Valid commits | 128 / 128 |
| Deterministic replays | 128 / 128 |
| Exact selected-context replays | 128 / 128 |
| Exact omitted-chunk recoveries | 576 / 576 |
| Tamper mutations detected | 768 / 768 |

```bash
python -m benchmarks.context_commit_conformance
```

Raw evidence:
[`benchmarks/results/context_commit_conformance.json`](../benchmarks/results/context_commit_conformance.json).

The benchmark is synthetic, local, and uses no model calls. It measures artifact
integrity, replay, and recoverability. It does not measure downstream answer
quality and does not claim that Python and Rust select identical context.
