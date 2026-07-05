# Entroly Agentic Harness Audit

Entroly treats an agent session as a traceable harness, not a pile of
independent prompts. The audit surface is designed to answer three production
questions:

1. Which turn introduced unsupported context?
2. Where did that suspect context propagate?
3. Which turn consumed the session budget?

This is the operational framing behind `SessionReceiptChain`,
`HallucinationTaintTracker`, session budget decisions, and `entroly audit`.

## The Runtime Artifact

Each session can persist two JSON artifacts:

| Artifact | Purpose |
| --- | --- |
| `session_chain.json` | Tamper-evident chain of receipt hashes, parent links, budget decisions, and per-turn metadata. |
| `session_taint.json` | WITNESS-derived suspect entities, origin turns, propagated turns, risk labels, and risk decay state. |

The receipt chain owns its content hashes. Caller-supplied receipt ids are kept
as metadata only, so the chain proves the content it actually received instead
of trusting external ids.

## Human Audit

Run:

```bash
entroly audit ./session_chain.json --taint ./session_taint.json
```

The report includes:

- chain integrity status and chain hash
- turn count and head receipt information
- total consumed budget and closing reserve
- the budget hotspot turn
- a per-turn budget ledger
- WITNESS-originated suspect entities and propagation turns

For estimated input cost attribution, pass your provider's input-token price:

```bash
entroly audit ./session_chain.json \
  --taint ./session_taint.json \
  --input-price-per-million 6.00
```

The cost is explicitly labeled as an estimate because output tokens, provider
cache discounts, retries, and hidden reasoning tokens are provider-specific.

## Machine Audit

Run:

```bash
entroly audit ./session_chain.json --taint ./session_taint.json --json
```

The JSON output is intended for CI, incident review, and compliance export. It
contains the same integrity, taint, and budget fields as the text report.

## What This Proves

The audit report can show:

- the session chain was unchanged since serialization
- each link points to the previous receipt id and receipt hash
- duplicate receipt ids or duplicate content hashes are detected
- a suspect entity originated at a specific turn
- the suspect entity propagated into later turns
- a specific turn consumed the most session budget

## What This Does Not Prove

The report is not a legal compliance guarantee and does not judge model quality
by itself. It provides evidence. For regulated deployments, pair it with
retention policy, access controls, external timestamping or signing where
required, and the application's own human review process.

## Product Sentence

Entroly is the harness layer that tells you when an agent session's context
became suspect, which turn it happened in, where it propagated, and what budget
the session consumed.
