# Entroly Context Efficiency Frontier v2

Status: post-pilot protocol revision for future frozen runs. This protocol was
written after inspecting calibration behavior, so it is not a retroactive
preregistration. A publishable run must pin this protocol, code revision,
dataset manifest, scorer, model, and experiment configuration before any paid
observations are collected.

## Decision question

For a fixed task distribution, model, prompt, decoding configuration, context
budget, and scorer, does an Entroly condition reduce provider-observed input
while keeping per-task harm below a declared bound?

The primary decision is not percentage of text removed. It is whether context
reduction survives task-success, evidence-retention, grounding, measurement,
and cost-per-success gates. Results apply only to the evaluated operating
point; no single aggregate establishes universal superiority.

## Why v2 exists

The v1 pilot identified four threats to validity:

1. The runner used answer containment instead of the pinned official LongBench
   HotpotQA token-F1 scorer.
2. A failed request with unavailable provider usage looked like zero-token,
   zero-cost inference.
3. A percentile bootstrap can have zero width when every observed delta ties;
   that does not bound the probability of a regression on the next task.
4. Model, decoding, selection, prompt, and budget settings were not bound into
   the pair identity as one canonical experiment configuration.

V2 repairs those defects. V1 artifacts remain v1 and cannot be relabeled.

## Conditions and pairing

Every task-replicate contains the same condition matrix:

| Condition | Behavior |
|---|---|
| `raw` | Complete, unmodified task context; required baseline. |
| `native_compaction` | Only documented model or agent compaction. |
| `entroly` | Entroly context selection with native compaction disabled when controllable. |
| `combined` | Entroly plus documented native compaction. |

The pair key binds workload and manifest version, task ID, provider, actual
model ID, replicate, scorer, and canonical experiment configuration. That
configuration binds the prompt digest, temperature, output limit, reasoning
setting, context budget, random seed, selection policy, dataset revision,
scorer dependency digest, and pricing configuration. Entroly and combined
trials also require the exact
`ctx_...` Context Commit ID.

Condition order is randomized within each task. Task sampling is fixed before
calls begin. A `shortest-context` subset is calibration-only and must be labeled
`SMOKE ONLY`; it cannot support a public product claim.

## Outcomes and estimands

Each trial records these distinct quantities:

- `task_score`: the frozen benchmark metric, which may be fractional.
- `task_success`: a preregistered binary criterion used for cost per successful
  task and per-task regression risk. For the initial answer-only HotpotQA
  runner, success is normalized exact answer match.
- `evidence_recall`: whether answer-critical evidence remains in active context.
- `unsupported_claim_rate`: a scorer-specific grounding quantity. In the
  initial answer-only runner, it is a conservative lexical proxy: whether a
  non-empty response is absent as a normalized contiguous span of supplied
  evidence. It detects extra or reformulated output but is not a semantic
  factuality judgment. A missing answer is a task failure, not automatically a
  fabricated claim.
- provider-observed context, reasoning, and output tokens; latency; observed
  API cost; and cost per successful task.

The official LongBench v1 HotpotQA scorer is token-overlap F1 after lowercase,
ASCII-punctuation removal, English-article removal, and whitespace folding. Its
implementation is pinned by LongBench revision and the SHA-256 digest of
`LongBench/metrics.py`.

Before any provider call, the runner rejects a task whose raw context does not
contain normalized answer evidence. This prevents a retrieval failure from
being blamed on compression when the reference answer was absent at baseline.

## Missingness and failures

Timeouts, refusals, context-limit failures, malformed outputs, and tool errors
remain in the task matrix with failed task success. They are not retried until
successful and are not dropped.

When a failed request has no provider usage record, token and cost fields are
zero-valued placeholders accompanied by `usage_observed: false` and
`cost_observed: false`. The analyzer excludes those placeholders from token and
cost means and blocks an efficiency claim. It never interprets them as free or
zero-token inference. A stable error fingerprint binds the error class, status,
and hashed message without publishing provider text that may contain secrets.

Self-hosted zero API fees exclude hardware, energy, operations, depreciation,
and opportunity cost. Reports must say so.

## Statistical decision rule

The paired bootstrap intervals are descriptive effect summaries. They are not
the only evidence gate.

For each candidate versus raw, v2 additionally computes exact one-sided
Clopper-Pearson bounds for four task-level event rates:

1. raw success followed by candidate failure;
2. candidate evidence recall below raw;
3. candidate unsupported-claim rate above raw;
4. candidate context tokens below raw (a desired event).

The default familywise alpha is `0.05`, Bonferroni-corrected across four primary
gates (`alpha = 0.0125` per gate). The three regression-rate upper bounds must
be at most `0.05`; the context-win-rate lower bound must exceed `0.50`.

The descriptive bounds must also show:

- mean task-score delta lower bound at least `-0.01`;
- mean evidence-recall delta lower bound at least `-0.01`;
- mean unsupported-claim delta upper bound at most `0.01`;
- mean context-reduction lower bound above zero.

The minimum reportable sample is 20 pairs, but that is not sufficient by
itself. With zero observed regressions, the simultaneous 5% regression-risk
gate requires 86 independent task pairs under the default correction. Twenty
clean pairs still have a one-sided 95% upper regression bound of about 13.9%
before multiple-comparison correction.

The analyzer emits:

- `SMOKE ONLY` below the minimum pair count;
- `MEASUREMENT INCOMPLETE` when provider usage is missing;
- `INSUFFICIENT PRECISION` when exact risk bounds do not pass;
- `NO CLAIM` when descriptive effect bounds do not pass; or
- `PASS` only when every gate passes.

All workloads are reported separately. A pooled headline cannot hide a failed
or incomplete workload.

## Workload validity matrix

No single benchmark represents "AI token efficiency." A public evidence set
must cover complementary failure modes:

| Construct | Primary workload | Required control |
|---|---|---|
| Long-context QA | LongBench/LongBench v2 plus HELMET tasks | official scorer, length and answer-position strata |
| Weak lexical cues | NoLiMa | no answer-keyword shortcut |
| Retrieval and aggregation | RULER and noisy-document RAG | answer present in raw context |
| Repository coding | SWE-bench Verified | pinned image, patch/test oracle, resolved rate |
| Terminal agents | Terminal-Bench 2.0 | pinned task image and executable verifier |
| Tool/log/JSON work | frozen public fixtures plus executable schema checks | exact field recovery and valid tool calls |
| Growing sessions | frozen multi-turn traces | repeated-query quality, cumulative usage, recovery audit |

Synthetic needle and deterministic conformance tasks remain engineering tests,
not headline evidence. Stochastic agent workloads require preregistered
replicates and task-clustered analysis; repeated runs of one task are not
independent tasks.

## Bias and contamination controls

- Freeze train/development/test decisions before the public run.
- Report performance by context-length and answer-position strata to detect
  lost-in-the-middle behavior.
- Use immutable repository revisions, container digests, dependency locks, and
  provider model IDs where available.
- Keep prompt and cache prefixes identical across paired conditions except for
  the context transformation under test.
- Preserve append-only trial JSONL, exact response and context digests,
  provider request IDs or stable redactions, and Entroly receipts.
- Declare infrastructure exclusions before unblinding scores. Provider/model
  failures are outcomes unless the same infrastructure fault invalidates every
  paired condition.
- Do not tune thresholds, budgets, or task subsets on the held-out report set.

## Publication packet

A publishable result includes the frozen protocol and repository commit; task
manifest hashes; raw append-only trials or a license-safe hash manifest;
generated JSON and Markdown reports; exact scorer and dependency hashes;
provider usage/cost provenance; all failures and exclusions; Context Commits;
and per-workload results.

Use a bounded claim such as:

> On the named model, frozen workload manifest, budget, and scorer, Entroly met
> the preregistered task-level harm bounds while reducing provider-observed
> context by X% (paired descriptive interval L-U, N independent tasks).

Never shorten that to "best compression tool" or generalize it to unevaluated
models, tasks, providers, or budgets.

## Reproduce

```bash
python -m benchmarks.context_efficiency_frontier trials.jsonl \
  --bootstrap-samples 2000 \
  --quality-tolerance 0.01 \
  --minimum-claim-pairs 20 \
  --familywise-alpha 0.05 \
  --max-regression-rate 0.05 \
  --minimum-context-win-rate 0.50 \
  --output report.json

python -m benchmarks.context_efficiency_report report.json --output report.md
```

The JSON report is the source of truth. The Markdown report is generated from
it and must show caveats, missingness, risk bounds, and failed gates alongside
passing results.

## Research basis

- [LongBench](https://arxiv.org/abs/2308.14508) and
  [LongBench v2](https://arxiv.org/abs/2412.15204)
- [HELMET](https://arxiv.org/abs/2410.02694)
- [RULER](https://arxiv.org/abs/2404.06654)
- [NoLiMa](https://openreview.net/forum?id=A0Ajt0YQEU)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- [LongLLMLingua](https://arxiv.org/abs/2310.06839) and
  [LLMLingua-2](https://arxiv.org/abs/2403.12968)
- [SWE-bench](https://arxiv.org/abs/2310.06770) and
  [Terminal-Bench 2.0](https://arxiv.org/abs/2601.11868)
- [Time-uniform confidence sequences](https://arxiv.org/abs/1810.08240)
- [Estimating bounded means by betting](https://arxiv.org/abs/2010.09686)
- [Anytime-valid inference for count data](https://arxiv.org/abs/2302.10108)
- [How Do AI Agents Spend Your Money?](https://arxiv.org/abs/2604.22750)
- [AI Tokenomics](https://arxiv.org/abs/2606.24616)
