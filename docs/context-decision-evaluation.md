# Paired context decision observations

`entroly.decision_evaluation` records supplied full-context, selected-context
and full-context-repeat outputs. It makes no provider calls. Structured JSON
field projections identify the decision being compared; prose differences are
not treated as decision differences. Malformed or missing fields fail closed.

Each observation binds the model and version, canonical generation settings,
task family, evaluator, projection, tokenizer, context policy, workload digest
and exact Entroly Git SHA. Seeds, latency, token counts and caller-supplied usage
are recorded separately. Query and projected outputs are committed by digest;
source context and raw answers are not copied into the observation. These
digests are integrity checks, not anonymization against dictionary attacks.

For trial t, decision divergence is the indicator that the projected full and
selected decisions differ. `decision_divergence_regret` sums those indicators.
Full/full-repeat divergence is a separate stochastic control. Subtracting its
rate would not establish the causal effect of context reduction, so no such
correction is performed. When a declared truth evaluator returns a bounded
loss, `task_context_regret` sums loss(selected, truth) minus loss(full, truth).
This quantity can be negative. Internal optimizer regret is a different
quantity and cannot substitute for either downstream measure.

`context_regret` accepts only observations with matching scopes and projection
commitments, unique trials, and intact observation hashes. Unavailable truth
loss stays unavailable. Finite observations do not supply a population risk
bound: `risk_bound` remains null. No calibration from WITNESS, RAVS or another
task family is reused.

## Opt-in Decisions API choice trials

`python -m entroly.decisions_trial` collects fixed-choice decisions for the
same task under full context, Entroly-selected context, and a fresh full-context
repeat. The first two expose a choice flip and a Jensen-Shannon divergence
between returned option-probability vectors. The full repeat exposes model
self-variation on the same input. These quantities stay separate; neither a
choice flip nor a probability shift proves that selected context caused an
error. An independently supplied `truth_choice` enables a separate 0/1 task
loss comparison. Without truth, task regret remains null.

The manifest schema is `entroly.decisions-trial-manifest.v1`. Each trial has a
unique `trial_id`, `task_id`, `question`, `full_context`, `selected_context`,
caller-measured `full_context_tokens` and `selected_tokens`, and a `decision`
object with `name`, `instructions`, and at least two distinct `{value,
description}` choices. Top-level fields are `task_family`, `context_policy`,
`tokenizer`, `budget`, and `trials`. Use a frozen selector and workload; the
collector does not perform context selection itself. The token counts are
recorded as caller-supplied and unverified, so this report alone cannot prove
token savings. The provider model is `gpt-6-luna`, currently an alias rather
than a pinned model revision.

To inspect the shape without a provider call, run the synthetic fixture:

```bash
python -m entroly.decisions_trial \
  --manifest benchmarks/fixtures/decisions_trial_manifest.json \
  --responses benchmarks/fixtures/decisions_trial_replay.json \
  --output decisions-trial-report.json
```

An offline replay has schema `entroly.decisions-trial-replay.v1` and exactly
one row per manifest trial, keyed by `trial_id`, with `full`, `selected`, and
`full_repeat` provider-shaped responses. The fixture is synthetic, not
observed OpenAI behavior. Replays may also use captured responses supplied by
the operator; no network call occurs in replay mode.

Live collection requires both an explicit egress flag and a credential:

```bash
OPENAI_API_KEY=... python -m entroly.decisions_trial \
  --manifest private-manifest.json \
  --send-content \
  --output decisions-trial-report.json
```

This sends the full and selected text contexts to OpenAI in three requests per
trial. Review your data and account policy before using it. There are no
automatic retries. A refusal, missing condition, unexpected model/name/choice,
incomplete probability vector, or invalid usage stops the run without writing
a success report. Use a new output path for each run; the command refuses to
overwrite an existing report. The report contains digests and aggregate measures, not raw
contexts or provider answers; small or predictable inputs can still be guessed
from digests. It records provider-observed input tokens when available, with no
price or savings claim. Offline replay latency is unavailable, and live
latency is client wall-clock time.
Per-condition request commitments bind each context, question and rubric. The
`report_id` binds the probability-shift metrics and other report fields; it is
an integrity check, not a provider signature or proof of correct execution.

The report sets `production_authority` to false and does not create a WITNESS
or RAVS certificate. The Decisions API probabilities and confidence are model
outputs, not calibrated correctness probabilities. Production thresholds need
representative labeled data and a frozen evaluation protocol. See the
[OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions)
for the current beta endpoint and response contract.
Each collection gets its own scope ID: a replay is identified by its content
digest, and a live run gets a new random ID. This prevents observations from
separate runs of a mutable model alias from being silently pooled. The report
labels caller-supplied truth and token counts as unverified.

## Offline fixed-policy divergence calibration

`entroly.decision_evaluation.calibrate_fixed_policy_divergence` can now examine
paired observations from one frozen policy. A
`FixedPolicyCalibrationProtocol` declares the exact scope and projection,
dataset identity, sampling design, external precommit reference, disjoint
calibration and holdout task IDs, a risk target and an error probability
`alpha`. The offline command reads that protocol and one
JSON array of sealed observations:

```bash
python -m benchmarks.context_risk_calibration \
  --protocol protocol.json \
  --observations paired-observations.json \
  --output divergence-report.json
```

The command makes no provider calls. It requires every declared task exactly
once, rejects cross-scope or edited observation seals, and outputs aggregate
counts plus source-file digests. The calibration split is descriptive; the
untouched holdout split supplies a one-sided exact binomial upper confidence
limit for the rate of projected full/selected decision disagreement. The
method inverts the binomial CDF at the predeclared `alpha`, including the
zero-divergence case `1 - alpha ** (1 / n)` ([NIST exact binomial limits](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm)).
This is a statement about a fixed
policy's population disagreement rate **conditional** on independently sampled,
identically distributed paired tasks, fixed settings/projection, and a split
declared before holdout outcomes. The protocol hash detects edits; it does not
prove preregistration, independence, provider identity or source authenticity.

Full/full-repeat divergence is reported alongside the bound and is never
subtracted from it. A bound on observed disagreement is not a causal bound on
context-induced error, an answer-quality measure, a per-request guarantee or
evidence for an adaptive budget policy. The report's `production_authority`
field is always false. `require_assurance(decision_risk=True)` still fails
closed; real-model collection, external split provenance and a valid control
policy are needed before a production risk claim can be considered. Existing
WITNESS/RAVS calibration is not transferred to this different loss.

`continuity_debt` measures previously omitted units that become required later.
An exact recovery discharges a missed-unit obligation only when it is verified
and visible before the decision. This declaration-based accounting is not an
automatic detector of future relevance or an adaptive stopping guarantee.

## Frozen local fixture

Run `python -m benchmarks.context_assurance --output PATH`. The frozen
protocol requires the declared `entroly[benchmark]` tokenizer extra; it refuses
to run with heuristic token counts. `benchmarks/context_assurance_protocol.json`
specifies the fixture and gates.
The output records protocol, workload and harness fingerprints, source SHA,
dirty-worktree state, observations and latency percentiles.
Source fingerprints normalize CRLF to LF for Windows checkout portability;
the raw protocol-file digest is also recorded. The frozen policy and gates do
not change when Git converts line endings.

The policy extracts a declared command route and runs an actual local Python
assertion command. A route introduced at turn one becomes relevant after a
19-turn delay in 20/50/100-turn accounting traces. Comparisons include full
context, static prefix truncation, existing receipt selection and exact receipt
recovery. Recovery is addressed by a known oracle chunk ID; it does not test an
automatic recovery planner, real model behavior, memory integration, or the
proof-guided runtime. Full and recovered strategies receive extra context;
this is not an equal-cost efficiency comparison. Repeated decisions in these
traces are dependent, not independent holdout samples.

The fixture reports zero declared-obligation false passes or inexact recoveries
only if its measurements establish that result. Those counts do not establish
zero semantic risk. Performance includes selection, joint audit, exact recovery
and certificate byte/token overhead; unimplemented risk lookup, planning and
resolution costs remain null. Provider usage and spend remain unavailable.
Certificate metadata can outweigh the selected text in small budgets: it must
be included in transport accounting before claiming net token savings.

## Evidence gates

A downstream risk claim requires paired real-model observations, stable
decision projections, independent truth evaluation where applicable, frozen
train/calibration/test separation and adequate scoped samples. A sequential
claim additionally requires a protocol for dependent/adaptive observations.
Adaptive budget and mixed-granularity policies must then demonstrate their
own workload-scoped advantage, including failed cases and recovery costs.
These features remain unavailable here; they are not enabled by this fixture.

Prior work provides methods under explicit statistical assumptions:
[Learn then Test](https://arxiv.org/abs/2110.01052) uses hypothesis testing for
risk control, and [Conformal Risk Control](https://arxiv.org/abs/2208.02814)
addresses expected monotone losses. Neither paper establishes that Entroly's
dependent context traces satisfy the conditions for those guarantees. No
novelty claim or calibrated guarantee is made by this implementation.
