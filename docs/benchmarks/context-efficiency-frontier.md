# Entroly Context Efficiency Frontier

The Context Efficiency Frontier (CEF) measures whether a context-control system
reduces model input while preserving task outcomes. It does not treat token
savings alone as a product win.

This document preregisters the public protocol before results are collected.
Changing a threshold, workload, scorer, or exclusion after inspecting results
requires a new protocol version and a written reason.

## Research question

For a fixed model, task, prompt, tool set, and sampling configuration, does
Entroly reduce provider-observed context usage without a material loss in task
quality, evidence recall, or grounding?

The benchmark reports a Pareto frontier. It does not collapse quality, cost,
and latency into a promotional composite score.

## Conditions

Every experimental result is paired with `raw` on the same workload version,
task, provider, model, replicate, and scorer.

| Condition | Context behavior |
|---|---|
| `raw` | Complete unmodified task context. This is the required baseline. |
| `native_compaction` | Only the model or agent's documented native context management is enabled. |
| `entroly` | Entroly controls context; model-native compaction is disabled where the provider exposes that control. |
| `combined` | Entroly and documented model-native context management are both enabled. |

Entroly and combined trials must retain the `ctx_...` Context Commit ID for the
exact selected context. The artifact is integrity evidence, not proof that a
task score or model answer is correct.

## Primary hypotheses

The default non-inferiority tolerance is one percentage point.

- **H1 task quality:** lower bound of the paired 95% bootstrap interval for
  task-score delta is at least `-0.01`.
- **H2 evidence:** lower bound of the paired 95% bootstrap interval for
  answer-critical evidence-recall delta is at least `-0.01`.
- **H3 grounding:** upper bound of the paired 95% bootstrap interval for the
  unsupported-claim-rate delta is at most `0.01`.
- **H4 context:** lower bound of the paired 95% bootstrap interval for context
  reduction is greater than zero.

The analyzer emits `quality_preserving_context_win: true` only when all four
hypotheses pass. Larger minimum savings thresholds may be declared by a
workload-specific preregistration, but cannot be lowered after results are seen.

## Measurement contract

Each invocation writes one JSON object conforming to
[`context_efficiency_trial.schema.json`](../../benchmarks/context_efficiency_trial.schema.json).

- `context_tokens`, `reasoning_tokens`, and `output_tokens` come from the
  provider response or an auditable provider log, not a local tokenizer.
- `billed_cost_usd` comes from an invoice, provider usage ledger, or a declared
  immutable pricing snapshot. Reports must identify which source was used.
- `usage_source`, `cost_source`, and `provider_request_id` make that provenance
  machine-readable. A request ID may be replaced by a stable redacted hash.
- `latency_ms` is wall-clock request latency measured at the same boundary for
  all conditions.
- `task_score`, `evidence_recall`, and `unsupported_claim_rate` are bounded in
  `[0, 1]` and use the scorer named in the pairing key.
- Failures, timeouts, refusals, and malformed tool calls remain in the sample
  and receive the preregistered failure score. They are not silently discarded.

## Initial workload matrix

The first public report should include all completed suites, not only favorable
ones.

| Capability | Initial suite | Primary score |
|---|---|---|
| Long-context QA | LongBench HotpotQA and 2WikiMultihopQA | official answer score |
| Tool use | Berkeley Function Calling Leaderboard subset | executable call accuracy |
| Repository coding | SWE-bench Verified or a version-pinned public subset | resolved task rate |
| Long-running agents | Versioned OpenClaw and Hermes traces | terminal task success plus evidence recall |
| Structured operations | Versioned JSON, logs, and API traces | exact field or incident recovery |

Every suite needs enough independent tasks for a meaningful paired interval.
Synthetic needle tests and deterministic conformance fixtures remain useful CI
gates, but they are labeled separately and cannot support the headline product
claim.

## Experimental controls

- Pin model identifiers, provider API versions, workload revisions, prompts,
  tool schemas, temperatures, token limits, and stop conditions.
- Use identical task order and replicate IDs across conditions. Randomize the
  condition execution order within each task to reduce time-of-day bias.
- Preserve provider request IDs and usage records privately when licenses or
  customer data prevent publishing the full trace.
- Score automatically where an executable or exact-answer oracle exists.
  Human or model judging must be blinded to condition and use a frozen rubric.
- Bootstrap over paired task-replicate units with a fixed seed. Do not treat
  multiple tokens, claims, or judge votes from one task as independent samples.
- Report each model and workload separately before any aggregate. An aggregate
  must not hide a failed workload.

## Falsification and exclusions

The claim fails for a condition when any primary hypothesis fails. A point can
remain on the descriptive Pareto frontier while still receiving `NO CLAIM` from
the statistical gate.

Allowed exclusions are limited to failures proven to be benchmark-infrastructure
errors affecting all paired conditions, such as a missing dataset file. Model
timeouts, context-limit errors, tool failures, and safety refusals are outcomes.
All exclusions must be listed with task IDs before analysis.

## Reproduce the analysis

```bash
python -m benchmarks.context_efficiency_frontier trials.jsonl \
  --bootstrap-samples 2000 \
  --quality-tolerance 0.01 \
  --output report.json

python -m benchmarks.context_efficiency_report report.json \
  --output report.md
```

The JSON report is the source of truth. The Markdown table is generated from
that artifact and shows failed conditions and caveats alongside passing ones.

## Publication requirements

A public Entroly CEF result must include:

1. Protocol version and any workload-specific preregistration.
2. Trial JSONL or a license-safe manifest with hashes and provider request IDs.
3. Generated JSON and Markdown reports.
4. Model, provider, scorer, dependency, and Entroly versions.
5. Context Commit artifacts for Entroly-controlled trials, subject to source
   retention and privacy policy.
6. Per-workload results, uncertainty intervals, failures, exclusions, and cost
   provenance.

The preferred headline has this form:

> On the named model and workload, Entroly retained the preregistered quality
> bound while reducing provider-observed context by X% (paired 95% CI: L-U,
> N tasks).

Do not generalize beyond the evaluated models, workloads, budgets, and scorer.
