# Proof-Carrying Dreams: literature boundary and research specification

Research cutoff: 2026-07-17

Status: research proposal with a local reference implementation. No empirical
sample-efficiency or superiority claim is made here.

## Question

Can a learned world model reduce the number of expensive context-policy
experiments without allowing synthetic evidence, adaptive search, or repeated
peeking to manufacture a promotion?

## Search protocol

The review followed the citation neighborhoods and proceedings search for:

- NeurIPS, ICML/PMLR, ICLR/OpenReview, AISTATS, UAI, COLT and L4DC;
- ACL/EMNLP work on agent context, memory, compression and evidence;
- arXiv papers through the cutoff date for rapidly moving agent-world-model
  work not yet represented in archival proceedings;
- combinations of `world model`, `model-based RL`, `dream`, `synthetic
  experience`, `uncertainty`, `conformal`, `confidence sequence`, `adaptive
  experiment`, `context optimization`, `RLVR`, `provenance`, `receipt`,
  `proof-carrying`, and `self-improving agent`.

This is a systematic prior-art search, not a claim to have enumerated every
paper ever published. Negative keyword searches cannot prove uniqueness.

## What prior work already owns

| Area | Representative primary work | Consequence for Entroly |
|---|---|---|
| Learning and acting in imagined dynamics | [World Models](https://arxiv.org/abs/1803.10122), [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) | “Agents learn in dreams” is not novel. |
| Short, uncertainty-aware model rollouts | [PETS](https://proceedings.neurips.cc/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html), [MBPO](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html), [MOPO](https://proceedings.neurips.cc/paper_files/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html), [MACURA](https://proceedings.mlr.press/v235/frauenknecht24a.html) | Pessimistic reward and confidence-limited horizons are not novel. |
| Calibration and conformal uncertainty | [Calibrated MBRL](https://proceedings.mlr.press/v97/malik19a.html), [PlanCP](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fe318a2b6c699808019a456b706cd845-Abstract-Conference.html), [Conformal policy evaluation](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ce37ba67dd12c4e11ca735da9d60295f-Abstract-Conference.html) | Calibrating a world model alone is not novel. |
| Synthetic replay | [SynthER](https://proceedings.neurips.cc/paper_files/paper/2023/hash/911fc798523e7d4c2e9587129fcf88fc-Abstract-Conference.html) | Separate real and synthetic replay buffers exist; separation alone is not novel. |
| RLVR and world models | [RLVR-World](https://arxiv.org/abs/2505.13934) | “World models plus verifiable rewards” is too broad to claim. |
| World models for language agents | [SimuRA](https://arxiv.org/abs/2507.23773), [Agent World Model](https://arxiv.org/abs/2602.10090), [WorldEvolver](https://arxiv.org/abs/2606.30639) | Consequence simulation, synthetic agent environments and selective foresight are active prior art. |
| Context-policy self-improvement | [ACE](https://arxiv.org/abs/2510.04618), [ACON](https://arxiv.org/abs/2510.00615), [Sculptor](https://openreview.net/forum?id=HPeiH7da0Z), [SUPO](https://aclanthology.org/2026.acl-long.966/), [AdaCoM](https://arxiv.org/abs/2605.30785), [CompactionRL](https://arxiv.org/abs/2607.05378) | Evolving context playbooks, context actions, failure-driven compression and RL-optimized compaction are not novel. |
| Cheap surrogates guiding costly experiments | [Robust multi-fidelity BO](https://proceedings.mlr.press/v206/mikkola23a.html), [Input-dependent MFBO](https://proceedings.mlr.press/v244/fan24a.html) | “Use simulation to choose what to test in reality” is not novel. |
| Safe candidate identification | [Best-arm identification with safety constraints](https://proceedings.mlr.press/v151/wang22h.html), [Top feasible arm identification](https://proceedings.mlr.press/v89/katz-samuels19a.html) | Constrained promotion and sample-efficient candidate selection have mature theory. |
| Inference under adaptive sampling | [Likelihood-ratio confidence sets](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5491280797f3192b895bce84eb83df8d-Abstract-Conference.html), [Anytime-valid prediction-powered inference](https://proceedings.neurips.cc/paper_files/paper/2025/hash/01830c92c6558179fa6d7fb1edff692c-Abstract-Conference.html) | Repeated checking requires time-uniform inference; fixed-sample bounds are insufficient. |
| Proof-carrying agent execution | [Proof-Carrying Agent Actions](https://arxiv.org/abs/2606.04104), [Certified Traces](https://arxiv.org/abs/2605.24462) | Certificates for agent actions exist, but target authorization and execution rather than simulator-guided context-policy learning. |

## Proposed contribution: Proof-Carrying Dreams (PCD)

The proposed contribution is a model-influence firewall for context-policy
self-improvement. It combines three requirements that the reviewed work does
not present together as one context-control protocol:

1. **Receipt-closed imagination.** Every synthetic transition commits to the
   exact hashes of the real receipts that support it, the world-model version,
   backend and an immutable `proposal_only` influence scope. A model output
   with missing, forged or mismatched ancestry is inadmissible.
2. **Selection-only model influence.** The world model may alter the predictable
   order of real experiments. Synthetic rewards never fit the real reward head,
   close an evidence obligation, enter a promotion mean or become RLVR labels.
3. **Anytime-valid real promotion.** Promotion is computed solely from exact
   real payloads already committed to the real ledger, using a time-uniform
   confidence sequence. Repeated dream-test-check cycles therefore cannot gain
   significance merely by stopping on a favorable fluctuation.

The novelty claim is deliberately narrow: **a proof-carrying, receipt-closed
sim-to-real firewall for adaptive optimization of AI context policies**. It is
a candidate systems-and-algorithm contribution, not a claim that its component
mathematics was invented here.

## Formal contract

Let `R` be the append-only real ledger and `D` the append-only dream ledger.
A dream transition `d` is admissible only if:

```text
parents(d) subset IDs(R)
parent_hashes(d) = receipt_hash_R(parents(d))
scope(d) = proposal_only
model_id(d) = hash(backend, model_version)
```

The simulator can choose the next experiment `A_t` as a function of all past
real and synthetic information. Promotion at a data-dependent stopping time
`tau` is allowed only when:

```text
LCB_CS(real rewards of candidate, tau)
  >= UCB_CS(real rewards of incumbent, tau) - margin
```

For rewards in `[-1, 1]`, the reference implementation uses the one-sided
stitched Hoeffding radius

```text
b(n, delta) = sqrt(2 log(pi^2 n^2 / (3 delta)) / n).
```

Allocating `3 delta / (pi^2 n^2)` at time `n` spends `delta / 2` across all
times for one stream. Candidate-lower and incumbent-upper streams together
spend at most `delta`. The guarantee requires stable conditional means under a
precommitted benchmark distribution; it does not repair benchmark drift,
dependent duplicated trials or invalid receipts.

## Why this matters for Entroly

Context optimization creates unusually strong optimizer's-curse pressure:
many cheap mutations can be proposed, evaluated repeatedly and selectively
reported. A strong simulator increases this pressure because it searches more
effectively. PCD makes that optimizer powerful in the proposal plane while
keeping the approval plane independent and auditable.

In plain language: the model may recommend the next experiment, but it cannot
vote on whether its recommendation worked.

The reference implementation treats experiment selection differently from
deployed rollout control. Deployment remains pessimistic. Reversible benchmark
experiments use a confidence-gated UCB acquisition
`predicted_reward + beta * uncertainty`, while the learned target is the
bounded event `Pareto improvement / tie / dominated = +1 / 0 / -1`. This keeps
the acquisition and uncertainty on compatible scales. UCB and binary rewards
are prior art; their role here is to make the selection plane useful without
weakening the receipt or promotion firewall.

## Falsification plan

The contribution is not complete until these experiments are run from raw
receipts and published with all failures:

1. Model-free random/local search versus empirical PCD versus Ebbiforge PCD,
   under equal real-evaluation and compute budgets.
2. Multiple repositories, task families, providers and at least 20 seeds.
   Paired conditions must use common random numbers and the exact native
   benchmark identity so process-local hash ordering cannot masquerade as an
   algorithmic difference.
3. Primary outcomes: real evaluations to a valid promotion decision, final
   task success, evidence retention, hallucination failures, tokens, dollars
   and latency.
4. Calibration: dream-to-real reward error and next-state error by horizon and
   support density.
5. Ablations: no lineage closure, fixed-sample gate, synthetic rewards allowed
   into fitting, no uncertainty cutoff and intentionally adversarial world
   models.
6. Stress tests for optional stopping, forged parents, duplicated receipts,
   benchmark drift and model version replacement.

The result earns a breakthrough claim only if it reduces real evaluations at
equal or better verified quality while the safety ablations demonstrate why
the firewall is necessary.

The local implementation now exposes a fail-closed native benchmark seed that
commits both the exploration stream and pre-ingest engine identity. The
selection path also applies deterministic tie-breaks. A 100-replay smoke check
produced one unique non-timing projection; this validates the harness contract,
not PCD's performance.

## Local directional result (not publishable evidence)

An equal-budget harness pilot used 20 deterministic proposal seeds, 50 shipped
benchmark cases, 32 real candidate evaluations per condition and an eight-item
candidate pool. The empirical PCD controller made 9.7 model-guided choices per
seed on average. Mean recall was identical (`0.9670139`). Mean context
efficiency was `0.0388405` with guidance versus `0.0388314` without it, but the
seed-level comparison produced four guided Pareto wins, six unguided wins and
ten ties/incomparables. This is not evidence of superiority. It falsifies the
stronger claim that the current empirical backend already delivers a reliable
sample-efficiency advantage; the preregistered multi-repository/provider study
and a stronger calibrated backend remain necessary.

## Trust boundary

“Proof-carrying” means machine-checkable lineage and promotion admissibility;
it is not a formal proof that an environment observation is true. Hash chains
detect changes relative to a trusted ledger head but do not authenticate the
verifier, prevent an attacker from rewriting an unanchored ledger, encrypt
data, or replace independent benchmark governance. Strong RAVS collectors and
external anchoring/signatures are separate requirements for higher-assurance
deployments.
