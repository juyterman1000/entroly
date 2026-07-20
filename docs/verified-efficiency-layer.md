# Verified AI Efficiency Layer

Entroly can now run context optimization as one fail-closed local workflow:

```text
untrusted sources
  -> prompt-injection scan
  -> query-aware context selection
  -> reversible context commit
  -> commit verification
  -> model
  -> evidence-grounded output suppression
  -> signed audit lineage
  -> strong real outcome only
  -> verified world-model update
```

The workflow is designed to reduce context without quietly trading away
evidence, security, or recoverability. It makes no remote call by itself.

## Five-minute integration

```python
from entroly import VerifiedEfficiencyLayer

layer = VerifiedEfficiencyLayer(".entroly/verified-efficiency")

prepared = layer.prepare(
    [
        ("architecture.md", architecture_text),
        ("runbook.md", runbook_text),
    ],
    query="Why did the deployment fail, and what is the recovery sequence?",
    token_budget=4_000,
)

# Send only this exact, committed context to your existing model/provider.
model_output = call_your_model(prepared.context)

# Strict mode removes unsupported claims and marks uncertain claims.
verified = layer.verify_output(
    prepared,
    model_output,
    profile="rag",
    mode="strict",
)
print(verified.output)
```

`prepared.commit_id` identifies a self-contained context commit. The commit
contains the selected spans, their source fingerprints, omitted-span metadata,
and a content-addressed recovery bundle. `prepared.audit` is an Ed25519-signed
local record of the security scans, receipt, commit verification, and exact
context-integrity chain.

## Fail-closed behavior

The default policy blocks instead of silently weakening a guarantee:

- critical or high prompt-injection findings are blocked before compression;
- a context commit that cannot replay or recover exact content is rejected;
- a receipt with `review_level="high"` is not sent onward;
- an output-verifier error withholds the unchecked output;
- an output that still contains high-severity downstream instructions is
  withheld; and
- a weak, synthetic, or agent-self-reported outcome cannot train the world
  model.

Every exception carries an actionable reason. For example,
`ContextRiskError.receipt` contains the coverage, omitted-evidence, dependency,
and novelty evidence needed to decide whether to increase the budget.

Two explicit audit modes exist for deployments that already have a human or
policy review boundary:

```python
review_layer = VerifiedEfficiencyLayer(
    ".entroly/verified-efficiency",
    security_mode="audit",
    context_risk_mode="audit",
)
```

These modes do not erase findings. The findings remain in the signed artifact,
and the returned receipt still exposes its review level and warnings.

## Exact evidence recovery

```python
omitted_id = prepared.receipt["omitted_context"][0]["chunk_id"]
recovered = layer.recover(prepared, omitted_id)

assert recovered.chunks[0]["verified"] is True
print(recovered.chunks[0]["text"])
```

Recovery reloads the persisted commit, verifies the entire commit again, and
checks the recovered text against both its chunk fingerprint and content hash.
A signed recovery artifact records which evidence was revealed without copying
the raw text into the audit receipt.

## Proof-guided fixed-point recovery

For workflows that can make a bounded second model call, Entroly can derive
proof obligations from unsupported claims, select the highest-value exact
omitted evidence under a hard token budget, and retry without changing the
committed prompt prefix. See
[Proof-Guided Context Fixed Point](proof-guided-context-fixed-point.md) for the
provider-neutral callback API, stopping states, guarantees, and limitations.

## Verified-only self-improvement

Self-improvement is an explicit outcome step, not an autonomous self-rating:

```python
from entroly.ravs.events import OutcomeEvent, TraceEvent

trace = TraceEvent(
    request_id="deploy-417",
    policy_decision="verified-efficiency-v1",
)
outcome = OutcomeEvent(
    request_id="deploy-417",
    event_type="test_result",
    value="passed",
    strength="strong",
    source="integration-tests",
    include_in_default_training=True,
)

learning = layer.record_verified_outcome(
    prepared,
    verified,
    trace,
    outcome,
)
print(learning.receipt_hash)
```

The update enters the real, fsync-backed, hash-chained transition ledger.
Repeated delivery of the same outcome is idempotent. Agent self-reports,
behavioral guesses, and dream rollouts are rejected as training evidence.
Entroly's world model may use verified history to propose a reversible real
experiment, but simulated evidence remains in a separate proposal-only ledger
and cannot promote a policy.

## Audit verification

```python
assert layer.verify_audit_artifact(prepared.audit).valid
assert layer.verify_audit_artifact(verified.audit).valid
```

The signing key is generated locally at
`.entroly/verified-efficiency/attestation.key`. Back it up with the state store;
replacing it invalidates custody verification for existing artifacts. Generated
outputs are not persisted by default: audit artifacts store their hashes and
claim-decision commitments, while the caller retains the returned text.

## What this proves—and what it does not

The executable contract at
`benchmarks/verified_efficiency_contract.py` proves that the workflow is wired,
replayable, recoverable, signed, fail-closed, and restricted to strong learning
evidence. It is not a performance benchmark.

Entroly does not claim to eliminate hallucinations. EICV has false positives
and false negatives, and uncalibrated operation uses documented decision bands.
The layer improves the failure boundary by suppressing or flagging unsupported
claims and by making the exact grounding context inspectable.

“First in the world” and superiority over another project remain research
hypotheses until a reproducible comparative benchmark establishes them with a
declared workload, baseline, token budget, quality metric, seeds, and raw
artifacts.
