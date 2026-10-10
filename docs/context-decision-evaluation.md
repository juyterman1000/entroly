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
