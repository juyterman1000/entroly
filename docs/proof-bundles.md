# Proof bundles

A proof bundle packages evidence Entroly already produces into one file that
someone outside your team can check on their own machine, offline, without
your stores and without trusting you to run the check for them.

```bash
entroly export-proof --generate-key proof.key          # once; prints the public key
ENTROLY_PROOF_SIGNING_KEY=$(cat proof.key) \
  entroly export-proof --sign --format html --out audit.json
entroly verify-proof audit.json --public-key <PUBLIC_KEY_HEX>
```

`export-proof` writes `audit.json` (and `audit.html` with `--format html`).
`verify-proof` exits non-zero if anything fails.

## What goes in

| Evidence | Source | How to include it |
|---|---|---|
| Context Commits | `entroly context-commit` | `--commit PATH` (repeatable) or `--commits-dir DIR`; `.entroly/context-commits/` is used when nothing else is given |
| Session receipt chain | `SessionReceiptChain` (`session_chain.json`) | `--session-chain PATH`; `--session ID` checks the chain belongs to that session |
| WITNESS taint records | `HallucinationTaintTracker` | `--taint PATH`, or `session_taint.json` / `taint.json` beside the chain |

## What the verifier recomputes

- **Each Context Commit**, from its own bytes: commit id, receipt hash,
  selected-context digest, recovery-bundle digest, and recovery of every
  omitted chunk. The result recorded at export must equal the recomputed one.
- **The session chain**: content-derived receipt ids, parent links, chain hash.
- **One Merkle root** over every evidence item, in the order listed.
- **The bundle digest and id**, over everything except the signature.
- **The Ed25519 signature**, when present.

Changing any byte of the evidence, a recorded result, the metadata or the
statement of limits fails verification.

## How far "verified" goes

`verify-proof` reports one of four states. Only the first authenticates who
produced the bundle.

| Status | Meaning |
|---|---|
| `verified` | Intact, signed, and the signer matches the `--public-key` you supplied |
| `signed_unpinned` | Intact and signed, but you did not supply a key, so the signer is unchecked |
| `integrity_only` | Intact and unsigned: self-consistent, anyone could have produced it |
| `invalid` | Something failed; do not rely on it |

`--require-signature` fails an unsigned bundle. `--public-key` fails a bundle
that is unsigned or signed by a different key. Get the public key from the
issuer through a channel you trust; a key read out of the bundle proves
nothing about who made it.

## Signing keys

- `ENTROLY_PROOF_SIGNING_KEY`: a 32-byte Ed25519 private key in hex
  (`--generate-key PATH` creates one with mode 0600 and never overwrites).
- Otherwise the key is derived from `ENTROLY_IDENTITY_KEY` under a fixed
  label. That is convenient if you already hold an identity key, and it is
  exactly as strong as that secret. The bundle records which source was used.
- `--sign` with neither set is an error. Entroly does not fall back to an
  unsigned bundle.
- Signing needs the `cryptography` package; so does verifying a signed bundle.

## Sharing without source text

By default a bundle embeds the Context Commits, including the selected and
omitted text, so the verifier can re-run replay and recovery. `--no-content`
leaves the commits out and keeps their anchors and the exporter's recorded
result. Such a bundle still verifies and still carries queries and risk
summaries, and the verifier states that replay and recovery could not be
re-run.

## What a bundle does not establish

These lines are inside every bundle and covered by its digest and signature:

- Replay is of the context. The bundle shows which bytes were selected for
  the model and that every omitted chunk is recoverable. It does not
  reproduce or vouch for the model's output.
- WITNESS records are Entroly's own local checks, with measured
  false-positive and false-negative rates, not a guarantee of accuracy.
- Timestamps are the exporter's clock; there is no trusted third-party
  timestamp.
- A signature identifies a person or organisation only if you obtained the
  public key from them through a channel you trust.
- The bundle covers the evidence it was given. It cannot show that no other
  commits or turns existed.
- It is technical evidence, not a certification and not a finding of
  compliance with any law, regulation or standard.
