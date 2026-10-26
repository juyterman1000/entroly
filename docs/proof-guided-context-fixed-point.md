# Proof-Guided Context Fixed Point

One-shot context compression must decide what matters before seeing the model's
answer. Full-context prompting avoids that uncertainty but pays for every token
on every call. Entroly's proof-guided fixed point occupies the space between
those two choices:

1. start with a small, committed context;
2. generate one draft through a caller-supplied model callback;
3. turn unsupported claims into explicit evidence obligations;
4. choose omitted evidence by expected support gain under a hard token budget;
5. recover and fingerprint-verify the selected chunks;
6. append them without changing the original context prefix; and
7. stop when the answer is supported or a declared safety bound is reached.

Entroly does not select a provider and does not make a network call itself. The
application supplies the callback, making every potentially billable operation
explicit.

## Basic use

```python
from entroly import VerifiedEfficiencyLayer

layer = VerifiedEfficiencyLayer(
    ".entroly/verified-efficiency",
    context_risk_mode="audit",  # use only behind an explicit review boundary
)

prepared = layer.prepare(
    [("architecture.md", architecture), ("runbook.md", runbook)],
    query="Why did the deployment fail and how should it recover?",
    token_budget=4_000,
)

def call_model(request):
    # This is the only place a provider call can occur. Keep the stable prefix
    # first if the provider supports prompt-prefix caching.
    return my_model.generate(
        query=request.query,
        stable_context=request.stable_context_prefix,
        appended_evidence=request.appended_evidence,
        previous_draft=request.previous_output,
    )

result = layer.run_fixed_point(
    prepared,
    model_call=call_model,
    max_rounds=3,
    recovery_token_budget=1_200,
    max_chunks_per_round=3,
)

print(result.status)
print(result.final_output.output)
print(result.recovery_tokens_used)
```

`model_call` is invoked exactly once for every item in `result.rounds`. A model
exception or non-string return value raises `FixedPointModelError` with a signed
failure artifact. Verification and exact-recovery failures similarly raise
`FixedPointVerificationError` or `FixedPointRecoveryError`; both retain a signed
record of the failed stage and any safely completed recovery work.

## Product integrations

### Durable host-driven protocol

The same controller has a durable prepare/advance protocol for hosts that own
the model transport. Session files are private local JSON, atomically replaced,
hash-checked on load, and completed snapshots can resume after a process restart.
Runtime v2
responses bind each issued model request to its session, revision, prepare
parameters and context commit with `request_sha256`. Keep that digest and the
issued `revision` beside the actual model invocation; echo them as
`request_sha256` and `expected_revision` when advancing. Never fetch a newer
binding merely to relabel an old reply.

Independent cooperating local processes serialize load/check/save under an
OS-released file lock. At most one new advance may commit for an issued request.
An exact idempotent retry replays its original response at any later revision;
the same key with different output bytes, revision or request digest conflicts.
Missing/malformed bindings fail closed. Stale bindings cannot create a new round.

Prepare a request without calling a provider:

```bash
entroly proof prepare ./docs \
  --query "Which evidence supports the recovery guarantee?" \
  --budget 8000 --idempotency-key request-001 > prepared.json
```

Send `request` through the model route you already operate, then advance the
local verifier:

```bash
entroly proof advance pgs_... \
  --output-file draft.txt --idempotency-key model-round-0 \
  --expected-revision 0 --request-sha256 '<digest-from-prepared.json>'
entroly proof inspect pgs_...
```

`entroly proof run` is available only with an explicit `--model-command`; it
uses `shell=False`, sends request JSON on stdin, and never guesses a provider or
credential. The MCP tools `prepare_proof_guided_context`,
`advance_proof_guided_context`, and `inspect_proof_guided_context` implement the
same protocol. Trusted same-origin localhost sidecar clients can use
`POST /proof/prepare`, `POST /proof/advance`, and `GET /proof/inspect`.
MCP and HTTP advances require the same `expected_revision` and `request_sha256`
fields. The explicit model-command loop echoes the binding issued for each call.
Terminal responses have `request=null` and `request_sha256=null`.

The HTTP sidecar runs synchronous proof-store operations, including lazy
initialization and lock waits, outside its event loop. Cancelling an HTTP request
does not cancel an operation already running in that worker: retry the original
binding and idempotency key rather than assuming no durable transition occurred.

Existing runtime v1 sessions remain inspectable without rewriting. They cannot
be advanced or reused through an old prepare key: re-prepare from the intended
source with a new idempotency key. Upgrade all writers together; older binaries
do not enforce the new contract and locking. Back up v2 state before rolling
back the code, and do not run old writers against it.

This is a request-acceptance contract, not a source-freshness or semantic-safety
certificate. It does not prove that a provider consumed the request, authenticate
an agent, establish that source files are unchanged, or make external actions
exactly-once. Recovered evidence still refers to the committed historical source
snapshot. The host must separately validate current applicability and gate side
effects. Locks cover cooperating processes on a local filesystem, not distributed
writers. Audit files and session state are not one transaction: an interrupted
advance may leave an unreferenced audit artifact, but its session transition and
idempotency record are persisted together.

Known restart limitation: abrupt death during existing attestation-key or audit
artifact creation can strand an inner exclusive-create sentinel lock, blocking
initialization or retry until operator recovery. The new advisory session lock
is OS-released, but does not reclaim those older sentinels. This fails closed;
it is not automatic crash recovery. Never remove a possibly active sentinel
merely because it is old. This separate liveness issue is not repaired here.

The OpenClaw plugin can close the loop automatically through OpenClaw's typed
conversation hooks. It is disabled by default because a revision is another
potentially billable model call. When explicitly enabled, Entroly verifies the
draft locally, recovers exact omitted messages, asks OpenClaw for at most the
configured bounded revision count, and replaces unsafe or still-unsupported
delivery with the locally verified output. OpenClaw remains responsible for
provider choice, authentication, routing, and billing.

## The evidence objective

For unsupported claims `C` and omitted chunks `B`, the planner estimates:

```text
value(B) = sum over c in C of
           claim_risk(c) * lexical_coverage(c, B)
           + small query-alignment prior
```

It then solves a deterministic 0/1 bounded knapsack:

```text
maximize    sum value(chunk)
subject to  sum receipt_token_count(chunk) <= remaining recovery budget
            selected chunks <= per-round chunk bound
```

This is a proof obligation scheduler, not a semantic oracle. The current value
estimate is deterministic and lexical; it can miss paraphrases and cross-lingual
evidence. The EICV verifier also has non-zero false positives and false
negatives. Those limitations are why the controller exposes every obligation,
candidate score, decision, and stop reason instead of claiming certainty.

Planner work is bounded to the 128 highest-utility candidates. The public loop
also enforces:

- 1–8 model rounds;
- 0–100,000 recovered evidence tokens total; and
- 1–16 recovered chunks per round.

The recovery budget counts receipt chunk tokens. Provider-specific wrapper and
message-format overhead is not included and should be added by the application
when estimating billing.

## Monotonic context and cache stability

For every round `t`:

```text
full_context[t] = committed_context || exact_recovered_evidence[0:t]
```

The committed context is a byte-identical prefix in every request. Evidence is
never summarized, rewritten, or removed during the loop. Applications using a
provider with prefix caching should send `stable_context_prefix` before
`appended_evidence`.

Each recovered chunk is verified against both the fingerprint recorded in the
context receipt and the content hash in the recovery bundle. The round receipt
commits:

- the exact grounding-context hash;
- unsupported claim hashes and verifier scores;
- every bounded planner candidate and selected chunk;
- recovery artifact IDs;
- cumulative recovery tokens;
- the previous round artifact; and
- the continue or stop decision.

## Terminal states

| Status | Meaning |
|---|---|
| `supported` | At least one verifiable claim was found and no claim remained unsupported. |
| `no_verifiable_claims` | The response contained no claim that could establish convergence. |
| `no_supporting_omitted_evidence` | Omitted chunks existed, but none overlapped the open proof obligations. |
| `no_omitted_evidence` | No unrecovered chunk remained. |
| `recovery_budget_exhausted` | The budget was zero or no supporting candidate fit. |
| `max_rounds_reached` | The hard model-call bound was reached. |

Every terminal state returns the safest strictly verified output produced in its
last round. Unsupported claims may therefore be removed or marked unverified.

## Safe learning boundary

The final fixed-point output remains compatible with
`record_verified_outcome`. Only a strong external RAVS signal—such as a test,
CI result, executed command, or explicit user acceptance—can update the real
world-model ledger. The number of retries, the model's own confidence, and a
synthetic dream are not success labels.

## Executable evidence

Run:

```bash
python benchmarks/context_fixed_point_contract.py
python benchmarks/proof_guided_runtime_contract.py
```

The contract uses no network and proves the control-flow invariants. It does not
establish a performance or superiority claim. Comparative publication still
requires declared datasets, providers, tokenizers, budgets, baselines, seeds,
quality metrics, latency, costs, and raw artifacts.
