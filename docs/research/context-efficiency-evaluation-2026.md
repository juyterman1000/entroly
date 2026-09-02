# Context-efficiency evaluation: research map and design consequences

Date reviewed: 2026-09-02

This is a scoped, reproducible literature map, not a claim to have enumerated
every paper published by arXiv, ICLR, ICML, NeurIPS, ACL, or EMNLP. The review
targets work that can change Entroly's benchmark design: long-context
evaluation, retrieval and prompt compression, agent task validity, inference
cost, and finite-sample uncertainty. Primary papers, official benchmark sites,
and official repositories are preferred over summaries.

## Conclusions that change the benchmark

### Long context is not one construct

LongBench v2 spans multiple reasoning categories and input lengths rather than
treating context length as the task. HELMET reports that synthetic
needle-in-a-haystack performance does not reliably predict application
performance. RULER adds multi-needle retrieval, tracing, aggregation, and QA,
while NoLiMa reduces lexical overlap between questions and relevant passages.

Design consequence: Entroly cannot support a general quality claim with one
HotpotQA slice. The evidence suite must report retrieval, aggregation,
multi-hop QA, weak-lexical-cue retrieval, repository tasks, tool use, and
growing sessions independently. Synthetic needles remain diagnostics.

Sources: [LongBench v2](https://arxiv.org/abs/2412.15204),
[HELMET](https://arxiv.org/abs/2410.02694),
[RULER](https://arxiv.org/abs/2404.06654), and
[NoLiMa](https://openreview.net/forum?id=A0Ajt0YQEU).

### Position and length can create hidden regressions

Lost in the Middle shows that answer position within a long prompt can
materially change model performance. Reported maximum context length therefore
does not establish effective use of information throughout that context.

Design consequence: freeze and publish context-length and evidence-position
strata. A compressor must not be evaluated only on short examples or examples
whose answer is near a favored boundary. The existing `shortest-context`
selection is explicitly a smoke calibration and cannot produce a public pass.

Source: [Lost in the Middle](https://arxiv.org/abs/2307.03172).

### Compression quality is task- and method-dependent

LongLLMLingua uses question-aware coarse-to-fine compression and addresses
information position. LLMLingua-2 frames compression as token classification
trained from distilled data. Comparative prompt-compression research reports
that simple extractive strategies can be strong and that downstream effects
vary by task.

Design consequence: compare at matched active-token budgets, name the exact
competitor version and configuration, retain loss cases, and avoid treating
compression ratio as the dependent variable. Frozen extractive and algorithmic
baselines are required alongside raw context; learned baselines require pinned
weights and inference settings.

Sources: [LongLLMLingua](https://arxiv.org/abs/2310.06839),
[LLMLingua-2](https://arxiv.org/abs/2403.12968), and
[Characterizing Prompt Compression Methods](https://openreview.net/forum?id=Y3FjKSsfmy).

### Real agent success needs executable oracles

SWE-bench evaluates patches against repository tests. SWE-bench Verified uses
a human-filtered subset, while Terminal-Bench 2.0 uses real terminal tasks and
task verifiers. These are closer to buyer value than a model judging its own
summary.

Design consequence: the repository and terminal gauntlets use pinned execution
environments and executable success checks. Token savings are secondary to
resolved-task rate and cost per resolved task. Model-judged quality is not a
replacement for tests where tests exist.

Sources: [SWE-bench paper](https://arxiv.org/abs/2310.06770),
[SWE-bench Verified](https://www.swebench.com/verified.html), and
[Terminal-Bench 2.0](https://arxiv.org/abs/2601.11868).

### More tokens are not equivalent to more value

Recent analysis of agentic coding traces reports large token-consumption
variation for the same tasks and does not find that higher consumption
necessarily produces higher accuracy. AI-tokenomics work separately models
token expenditure and economic value, including hidden reasoning expenditure.

Design consequence: record input, cached input, reasoning, and output usage
separately when the provider exposes them. The primary economic quantity is
total observed cost divided by successful tasks, reported beside success and
harm—not tokens removed. Provider usage that is missing remains missing.

Sources: [How Do AI Agents Spend Your Money?](https://arxiv.org/abs/2604.22750)
and [AI Tokenomics](https://arxiv.org/abs/2606.24616).

### A narrow bootstrap does not bound unseen harm

Resampling the observed pairs estimates uncertainty in an observed mean under
its assumptions. If all observed differences are identical, a percentile
bootstrap can collapse even when the sample is too small to rule out a
meaningful future regression rate. Work on confidence sequences and bounded
means also makes clear that optional stopping and repeated inspection require
explicit treatment.

Design consequence: v2 keeps paired bootstrap intervals as descriptive effect
summaries and adds exact one-sided binomial bounds for task regression,
evidence regression, unsupported-claim regression, and context-win rates.
Bonferroni correction controls the declared family of four primary gates. Runs
must be frozen before observation; sequential stopping requires a separately
specified anytime-valid design.

Sources: [Time-uniform confidence sequences](https://arxiv.org/abs/1810.08240),
[Estimating bounded means by betting](https://arxiv.org/abs/2010.09686), and
[Anytime-valid inference for count data](https://arxiv.org/abs/2302.10108).

## Implemented now

- Trial/report schema v2 and canonical experiment-configuration binding.
- Official LongBench HotpotQA F1 normalization, pinned to repository revision
  and scorer-file digest.
- Separate fractional score, binary task success, evidence retention, and
  unsupported-output fields.
- Honest labeling of the initial answer-only grounding signal as a conservative
  lexical proxy rather than a semantic factuality oracle.
- Preflight proof that reference answer evidence exists in raw context.
- Missing usage/cost indicators; unknown measurements cannot become zeros in
  efficiency means or Pareto claims.
- Cost per successful task.
- Exact task-level risk bounds with a familywise error policy and explicit
  claim blockers.
- Stable, non-plaintext error fingerprints.
- Historical v1 preservation and explicit v2 post-pilot status.

## Still required before a broad public claim

- Freeze a representative held-out manifest with length and answer-position
  strata; do not use the shortest-context calibration sample.
- Add official adapters for LongBench v2/HELMET, RULER, and NoLiMa.
- Add executable SWE-bench Verified and Terminal-Bench 2.0 conditions with
  pinned containers.
- Add frozen external/algorithmic baselines at matched active-token budgets.
- Add repeated-run, task-clustered analysis for stochastic agents.
- Run at least the sample size required by the declared harm bounds, then
  publish every failure and workload-specific non-pass.
- Obtain independent reproduction. Internal evidence can establish a result;
  third-party reproduction is what makes it credible beyond Entroly's own
  repository.
