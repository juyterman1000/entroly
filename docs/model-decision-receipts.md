# Model Decision Receipts

Entroly emits a deterministic, privacy-safe proof of the model metadata used to
budget each model-bearing proxy request.

## Why

A context window is not merely a model label. It is an operational assumption
that controls compression, output reservation, cost estimates, and whether a
request can fit. Silent or stale assumptions make those decisions impossible to
audit.

A Model Decision Receipt binds the decision to:

- requested and resolved model identifiers;
- provider shape;
- registry trust level;
- exact versus prefix resolution;
- whether a conservative fallback was used;
- effective context window;
- output-token reservation;
- safe input ceiling;
- effective and bundled registry SHA-256 fingerprints.

The receipt digest is SHA-256 over canonical JSON using schema
`entroly.model-decision.v1`.

## Response headers

The existing control-plane header pipeline exposes bounded headers including:

```text
X-Entroly-Model-Receipt-Schema: entroly.model-decision.v1
X-Entroly-Model-Receipt: <sha256>
X-Entroly-Model-Requested: gpt-4o
X-Entroly-Model-Resolved: openai/gpt-4o
X-Entroly-Model-Trust: verified
X-Entroly-Model-Exact: true
X-Entroly-Model-Fallback: false
X-Entroly-Model-Context-Window: 128000
X-Entroly-Model-Input-Budget: 120576
X-Entroly-Model-Output-Reserve: 1024
X-Entroly-Model-Warning: none
X-Entroly-Registry-Digest: <sha256>
X-Entroly-Registry-Base-Digest: <sha256>
```

Header names are generated from control-plane tags, so the same contract applies
to optimized, observed, bypassed, JSON, and streaming responses handled by the
main proxy route.

## Privacy boundary

Receipts never expose:

- prompts or retrieved context;
- credentials or provider headers;
- registry source URLs;
- aliases;
- private override file paths or contents;
- pricing records.

The effective registry fingerprint changes when local discovery or user
overrides change, but the private metadata itself remains undisclosed.

## Verification

`ModelDecisionReceipt.verify()` recomputes the canonical digest locally. A
consumer can persist the receipt headers next to traces or cost records and use
the registry digest to reproduce the exact budgeting environment.

Unknown models and recognized models with unverified context windows are not
silently presented as authoritative. Their receipts explicitly report fallback
use and a machine-readable warning code.
