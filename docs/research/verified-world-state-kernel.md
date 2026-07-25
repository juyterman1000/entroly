# Verified World-State Kernel

**Status:** P0 research prototype. This is an additive internal kernel, not a
claim that Entroly already has a correct world model or that graphs improve task
success.

## Research question

Can a bounded local agent use structured state and imperfect transition
predictions without allowing either the graph or the predictor to become an
unverified source of truth?

The P0 kernel implements the minimum control contract required to test that
question:

1. claims are separated into `proposed`, `observed`, `supported`, `verified`, and
   `invalidated` states;
2. only an independent passing receipt can promote a supported claim to the
   verified frontier;
3. repository-bound evidence must identify the same repository revision as the
   claim;
4. invalidating a claim deterministically invalidates only downstream claims
   that depend on it;
5. transition predictions remain advisory and lose planning authority according
   to observed Brier error;
6. checkpoints are canonical, content-addressed, atomically written, and fail
   closed when modified.

Implementation: `entroly/world_state.py`

Tests: `tests/test_world_state.py`

## Authority hierarchy

The kernel enforces this conceptual order:

```text
real observation / tool result
    > deterministic verifier and receipt
    > verified graph claim
    > supported or observed claim
    > world-model prediction
    > model-generated hypothesis
```

A prediction cannot mutate a claim. A language model cannot create a verified
claim directly. A repository-bound receipt from another revision is rejected.

## State model

A claim is the tuple

```text
(claim_id, subject, predicate, object, status, confidence,
 evidence, dependencies, repository revision)
```

Only claims with `status=verified` and verified dependencies appear in the
verified frontier. Invalidation propagates through typed dependencies in a
stable order. It does not restart unrelated work.

The state digest is

```text
sha256(canonical_json(schema, ordered claims, ordered model authority))
```

Claim insertion order therefore does not affect pause/resume identity.

## Prediction reconciliation

A transition predictor returns probabilities over externally observable labels,
for example:

```text
{"tests_pass": 0.8, "regression": 0.1}
```

After executing the action, the kernel computes the mean Brier error across the
union of predicted and observed labels. Planning authority is updated as:

```text
new_authority = old_authority * exp(-learning_rate * brier_error)
```

This is deliberately conservative: accurate predictions do not become facts,
and inaccurate predictions automatically lose influence. Recovery dynamics and
multi-model calibration remain follow-up research.

## What this prototype does not do

It does not yet provide:

- automatic graph extraction from code or tool output;
- a trained latent transition model;
- model routing or autonomous tool execution;
- a general causal-discovery claim;
- cross-process event-ledger integration;
- public APIs or release-surface changes;
- evidence that a graph improves reasoning;
- evidence that predictions improve task success.

The existing Entroly vault and bitemporal belief ledger remain the durable
provenance layer. This branch does not duplicate or replace them.

## P0 acceptance criteria

- Direct insertion of verified or invalidated claims is rejected.
- Observed and supported states require evidence.
- Verification requires a passing receipt from the current repository revision.
- Stale or unversioned repository evidence fails closed.
- Invalidation propagates deterministically to downstream dependencies only.
- State digests are independent of insertion order.
- Snapshot alteration is detected.
- Snapshots with missing dependencies are rejected even when rehashed.
- Incorrect transition predictions reduce model planning authority.

## Next experiment

The next branch should add a deterministic adapter from real Entroly events into
this state model and run an ablation:

1. local coding model alone;
2. model plus Entroly retrieval;
3. retrieval plus verified world state;
4. verified world state plus transition predictions.

Primary metrics:

- verified task success;
- hidden-regression rate;
- repeated-failure rate;
- false-completion rate;
- exact-resume success;
- context tokens per decision;
- prediction calibration;
- graph maintenance overhead.

Kill or redesign the graph layer if it does not improve a protected outcome at
matched compute. Kill or ignore the prediction layer if it does not improve
action selection beyond verified state alone.
