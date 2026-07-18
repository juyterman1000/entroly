# Verified dreaming: real experience, bounded imagination

Entroly can use a learned world model to choose which self-improvement
experiments to run next. The safety boundary is simple:

> Dreams may propose. Only the real environment may promote.

This is a model-based reinforcement-learning loop, not a claim that synthetic
experience is ground truth. It follows the useful part of world-model systems:
learn dynamics from real transitions, rehearse actions cheaply, and return to
the real environment for verification. DreamerV3 similarly improves behavior
by imagining futures in a learned model, while the original World Models work
demonstrated policy learning inside simulated dynamics. Entroly adds a
product-specific evidence boundary: synthetic transitions are structurally
unable to satisfy RAVS/RLVR promotion gates.

Primary background:

- [World Models](https://arxiv.org/abs/1803.10122)
- [Mastering diverse control tasks through world models (DreamerV3)](https://www.nature.com/articles/s41586-025-08744-2)
- [RLVR-World](https://arxiv.org/abs/2505.13934)

## The loop

```text
real state s_t
  -> execute action a_t in the benchmark/environment
  -> observe verified reward r_t and real next state s_(t+1)
  -> append a hash-chained real transition
  -> fit world model only on real verified transitions
  -> simulate short, uncertainty-penalized candidate rollouts
  -> bind every dream to its real parent receipt hashes and model version
  -> append synthetic transitions to a separate proposal-only dream ledger
  -> run the best proposal in the real benchmark/environment
  -> assess promotion only with the ledger-closed, anytime-valid real gate
```

The learned model approximates both dynamics and reward:

```text
p_hat(s_(t+1), r_t | s_t, a_t)
```

Dream planning uses a pessimistic reward:

```text
r_safe = r_hat - lambda * uncertainty
```

Rollouts stop when confidence drops below the configured gate. Increasing the
horizon therefore cannot silently turn low-support extrapolation into trusted
evidence.

Choosing a reversible benchmark experiment is a separate decision. Entroly
uses a confidence-gated UCB acquisition (`predicted Pareto outcome + beta ×
uncertainty`) so the model can value information without gaining any authority
over deployment or promotion. Real benchmark outcomes train a bounded target:
Pareto improvement `+1`, an exact tie `0`, and a dominated result `-1`.

## What Entroly implements

`entroly.ravs.world_model` provides:

- `VerifiedTransition`: typed state, action, reward, next-state, environment,
  verifier, evidence strength, provenance parents, parent receipt commitments,
  model identity, and influence scope;
- `TransitionLedger`: separate append-only, hash-chained files for real and
  synthetic transitions;
- `EmpiricalWorldModel`: a dependency-free, locally weighted dynamics and
  reward model with explicit support IDs and uncertainty;
- `EbbiforgeWorldModelAdapter`: an optional adapter for Ebbiforge's Rust
  `AutoregressivePredictor`;
- `VerifiedDreamController`: bounded rollout, pessimistic ranking, real-only
  fitting, receipt-closed lineage, and an anytime-valid real promotion gate.

Entroly's idle `DreamingLoop` can use this controller to rank a pool of PRISM
configuration mutations. Its state is the normalized tuning configuration,
its action is a bounded configuration delta, its next state is the candidate
configuration actually executed, and its reward is the measured change on the
fixed benchmark. The existing benchmark remains the only path that can save a
new configuration.

## Enable the guarded runtime path

Verified world-model ranking is experimental and off by default until Entroly
has published controlled sample-efficiency evidence.

```bash
export ENTROLY_VERIFIED_DREAMING=1
```

Optional gates:

```bash
export ENTROLY_WORLD_MODEL_MIN_SAMPLES=8
export ENTROLY_WORLD_MODEL_NEIGHBORS=8
export ENTROLY_WORLD_MODEL_MIN_CONFIDENCE=0.55
export ENTROLY_WORLD_MODEL_UNCERTAINTY_PENALTY=0.75
export ENTROLY_WORLD_MODEL_EXPLORATION_BONUS=0.50
```

The dependency-free empirical backend is the default. To use an installed
Ebbiforge Rust world model:

```bash
export ENTROLY_WORLD_MODEL_BACKEND=ebbiforge
export ENTROLY_EBBIFORGE_MAX_VALIDATION_LOSS=1.0
export ENTROLY_EBBIFORGE_RETRAIN_INTERVAL=8
export ENTROLY_EBBIFORGE_TRAIN_EPOCHS=50
```

If `ebbiforge_core` is unavailable or its model cannot satisfy the training and
validation gates, Entroly reports the initialization error and disables the
world-model path rather than silently substituting another backend.

State is stored locally under `.entroly/verified_dreaming/`:

```text
world_model_real.jsonl   # strong, externally verified transitions only
world_model_dream.jsonl  # synthetic predictions; never a training authority
```

If either ledger fails schema or hash-chain validation, the model path fails
closed. The ordinary real benchmark loop remains available and the error is
reported in dreaming statistics.

New dream writes also fail closed unless every support transition resolves to
an exact real-ledger receipt hash and the dream commits its model backend,
model version, and `proposal_only` scope. Older experimental dream entries
without direct parent-hash commitments remain readable for recovery but are
not accepted as new proof-carrying writes.

## Ebbiforge backend

Ebbiforge already exposes a Rust trajectory buffer and autoregressive dynamics
model for `state + action -> next_state`. Entroly does not require Ebbiforge;
the adapter is duck-typed so deployments can inject it when installed.

```python
import ebbiforge_core as ebbi

from entroly.ravs import (
    EbbiforgeWorldModelAdapter,
    EmpiricalWorldModel,
    TransitionLedger,
    VerifiedDreamController,
)

config = ebbi.WorldModelConfig(latent_dim=7)
predictor = ebbi.AutoregressivePredictor(config)

backend = EbbiforgeWorldModelAdapter(
    predictor,
    state_factory=lambda vector: ebbi.LatentState(list(vector), "entroly", 0),
    reward_model=EmpiricalWorldModel(min_samples=8),
)
controller = VerifiedDreamController(
    TransitionLedger(".entroly/verified_dreaming"),
    backend,
)
```

The adapter uses Ebbiforge for next-state dynamics and Entroly's verified
reward head for reward/support estimation. Its confidence is capped by all of:

- Ebbiforge rollout confidence;
- Ebbiforge validation loss;
- Entroly real-transition support.

An untrained predictor, non-finite validation loss, excessive validation loss,
or insufficient verified reward support prevents dreaming.

## RLVR boundary

RLVR means the reward is produced by a check whose result is externally
verifiable—for example a test, CI result, command exit, or accepted/rejected
change. A world-model prediction is not RLVR evidence.

| Evidence | Fits the model | Ranks dreams | Promotes policy/config |
|---|---:|---:|---:|
| Strong real transition | Yes | Yes | Yes, through real sequential gate |
| Medium behavioral signal | No | No | No |
| Agent self-report | No | No | No |
| Synthetic dream transition | No | Yes, as a proposal | No |

## What can be measured honestly

The controller reports:

- real training transitions;
- synthetic dream transitions;
- dream-to-real ratio;
- model backend and version;
- support count and parent transition IDs per prediction;
- uncertainty gate and penalty;
- experiment exploration bonus and Pareto-aligned reward;
- model-guided experiments that were actually run;
- real promotions and rejections through the existing benchmark path.

The firewall contract has an executable, dependency-free check:

```bash
python benchmarks/proof_carrying_dreams_contract.py
```

It verifies receipt closure, proposal-only dream scope, the anytime promotion
boundary, and rejection of a forged real-looking payload. It intentionally
sets `performance_claim` to `false`; it is a conformance check, not a benchmark
result.

The correct sample-efficiency experiment compares identical real-interaction
budgets with verified dreaming on and off, across multiple seeds and common
random numbers. Seeded evaluation commits both the exploration PRNG and the
fresh engine's fragment-ID identity before ingest, and deterministic
tie-breaks prevent process-local hash order from changing a result. Until that
evaluation exists, Entroly should describe this as an implemented experimental
path—not as a proven performance breakthrough.

## Proof-Carrying Dreams boundary

Repeatedly generating candidates, testing them, and checking whether one looks
better creates an optional-stopping problem. A fixed-sample interval is not
valid when an optimizer can keep peeking until noise crosses the release line.

Entroly therefore uses a stitched Hoeffding confidence sequence over bounded
real rewards. It is wider than a one-shot interval because it remains valid at
data-dependent stopping times. The world model can adaptively choose which real
experiment runs next, but the promotion statistic contains only exact real
payloads already committed to the real ledger.

This is the proposed **Proof-Carrying Dreams** contribution: simulator-guided
experiment selection behind a cryptographic and statistical model-influence
firewall. The literature boundary and falsification plan are recorded in
[`research/proof_carrying_dreams.md`](../research/proof_carrying_dreams.md).

The idle `DreamingLoop` currently uses the model only for proposal ranking and
continues to save configurations through its pre-existing fixed benchmark
rule. It does not yet invoke sequential promotion because the shipped fixed
suite does not produce the repeated, conditionally stable samples required by
the confidence-sequence contract. `assess_promotion()` is therefore a guarded
research API until the multi-seed benchmark protocol is implemented.

## Non-claims

- Hash chaining detects modification; it is not encryption.
- An unanchored hash chain is not a signature or proof that a verifier told the
  truth; Proof-Carrying Dreams proves lineage/admissibility, not reality itself.
- The empirical backend is a conservative bootstrap model, not a transformer.
- Ebbiforge supplies learned latent dynamics, but Entroly still requires real
  reward evidence and real holdout verification.
- Synthetic rollouts can reduce wasted experiments only if the learned model
  is calibrated on the deployment distribution.
- The anytime-valid bound assumes bounded rewards and stable conditional means
  under a precommitted benchmark distribution; it does not excuse benchmark
  drift, duplicated trials, or non-independent reward generation.
