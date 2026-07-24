# Preregistration — Determinism Tax on ContextBench

**Status:** criteria locked at the commit that introduces this document, before the
first full run of the harness. Any later edit to the criteria sections invalidates
the preregistration and must be declared as a protocol revision (v2).

- **Harness:** `benchmarks/contextbench_determinism_tax.py`
- **Result artifact:** `benchmarks/results/contextbench_determinism_tax.json`
- **Reference baseline commit:** `77067bc` (deterministic ingest/reconcile ordering)
- **Seed:** 42 (single fixed seed; task subsampling only, no seed selection)

## Protocol revision v2 (declared after the pilot, commit `64d4f71`)

The 5-task Astropy pilot (harness/adapter validated: 5/5 executed, 100%
reproducible, 0 unmapped) surfaced two structural facts that invalidate the v1
**primary** metric:

1. **Size cap:** the default 50 KB source cap skips gold-bearing core files
   (astropy `table.py` = 147 KB) → guaranteed 0 recall. **Runs raise the cap to
   500 KB** (`ENTROLY_MAX_SOURCE_FILE_BYTES=500000`), the native hard ceiling.
2. **Granularity:** Entroly selects **whole files** — file recall 1.0 but line
   precision ~0.002. **Line-level F1 measures selection granularity, not
   determinism**, so it cannot be the primary decision metric: a passage-level
   neural reranker would score high line-F1 purely by selecting narrower spans.

**v2 changes (this supersedes the v1 primary metric in §5):**
- **Primary decision metric → FILE-level F1 determinism tax.** Entroly, BM25, and
  file-rerankers all operate at file granularity, isolating *determinism* from
  *granularity*. The decision table thresholds are unchanged, applied to the
  file-level tax.
- **Line-level F1 → secondary, explicitly flagged granularity-limited.**
- Recorded finding (not a metric): competitive *line*-level retrieval would
  require sub-file selection + stored line offsets in Entroly (the offset gap the
  span adapter exposed). Out of scope for the tax run; motivates a separate build.

All other v1 sections stand.

## 1. Question

Experiment 1 (`docs/research/exp1-reproducibility-matrix.md`) established that
Entroly *reproduces* its own selection byte-for-byte on a fixed architecture. It
did **not** establish that the reproducible selection is *good enough*.

This experiment measures the **determinism tax**: the retrieval-quality gap
between Entroly's strictly-deterministic selector and the best available
nondeterministic (neural / LLM-reranked) selector, on human-annotated gold
contexts. If the tax is large, reproducibility is a compliance niche; if small,
it is a general-purpose advantage.

## 2. Dataset (external, public)

[ContextBench](https://github.com/EuniAI/ContextBench), HuggingFace
`Contextbench/ContextBench` (config `default`, split `train`). Verified schema
per task (streamed 2026-07):

```
instance_id, original_inst_id, repo, repo_url, language, base_commit,
gold_context, patch, test_patch, problem_statement, f2p, p2p, source
```

- **Query** = `problem_statement` (the issue text; verbatim, no rewriting).
- **Corpus** = `repo_url` checked out at `base_commit` (pre-patch state).
- **Gold** = `gold_context`: JSON list of `{file, start_line, end_line, content}`
  spans — human-traced essential dependencies, verified by requiring GPT-5 to
  generate a passing patch from context alone.

**Subset.** ContextBench **Lite** (500 tasks) is the primary evaluation set. A
**pilot** of the first 25 Python tasks (sorted by `instance_id`) is run first to
validate the harness end-to-end; pilot numbers are reported but are **not** the
decision basis. Task subsampling beyond Lite, if any, is by `instance_id` sort —
never by result.

## 3. Systems compared

All systems receive the **identical** corpus snapshot, query, and token budget
`B` (budget parity is mandatory — recall/precision trade against `B`, so an
unequal budget invalidates the comparison). Budgets swept: `B ∈ {2000, 8000}`.

1. **Entroly-deterministic** — `qccr.select` at commit `77067bc`.
2. **BM25-deterministic** — a plain BM25 file/chunk ranker with a total-order
   tie-break (deterministic reference floor).
3. **Neural reranker** — deterministic candidate generation (BM25 top-k) +
   a cross-encoder / embedding reranker (the "best nondeterministic" arm; its
   nondeterminism and compute cost are the point of comparison).
4. **Hybrid** — Entroly candidate generation + neural rerank of the head.
5. **Raw agent retrieval** — an off-the-shelf coding-agent retriever
   (e.g. Agentless embedding retrieval) as an external anchor, if runnable.

Systems 3–5 require compute/API budget and are gated on an explicit go decision.
Systems 1–2 are API-free and run first.

## 4. Metrics (report each SEPARATELY — no composite as the headline)

Following ContextBench, at **file** and **line** granularity via interval
overlap (block/AST granularity is deferred: it requires tree-sitter and is not a
gate). For predicted spans `P` and gold spans `G` (per task, then macro-averaged):

- **Recall** = |P ∩ G| / |G|
- **Precision** = |P ∩ G| / |P|
- **F1** = 2·P·R / (P+R)
- **Efficiency** = rank position at which gold coverage is first achieved
  (earlier = better), normalized to [0,1].
- **Evidence drop** = fraction of retrieved context that never overlaps any gold
  span (retrieved-but-useless mass).
- **Task success (Pass@1)** — SECONDARY, requires running an agent end-to-end;
  reported only if the agent arm is funded.
- **Latency** (wall-clock per task) and **token cost** (selected tokens).
- **Repeatability** — byte-identity rate across 2 runs (Entroly must be 1.000;
  neural arms measured, not assumed).

Line-level mapping: Entroly's selected content is located back to source line
ranges by exact-substring match; unmatched selected content is counted toward
evidence drop, never toward recall (fail-closed).

## 5. Preregistered decision table

Primary decision metric: **block-unavailable ⇒ line-level F1** determinism tax,
in percentage points, `Tax = F1(best nondeterministic) − F1(Entroly-deterministic)`,
macro-averaged over ContextBench Lite at the budget that maximizes each system's
own F1 (each system evaluated at its best budget, reported per budget too).

| Determinism tax (pp) | Decision |
|---|---|
| ≤ 3 | Strong general-purpose thesis |
| > 3 to 7 | Strong high-trust product |
| > 7 to 15 | Compliance / security niche |
| > 15 | Reject broad RCP positioning (unless hybridization closes the gap) |

A **secondary** composite is reported but is **not** the gate:
`Q = F1_line + α·Pass@1 + β·Efficiency − λ·EvidenceDrop`, with preregistered
weights `α = 1.0, β = 0.5, λ = 0.5` (all terms scaled to [0,1]). It is shown to
check whether the composite and the primary F1 tax agree; divergence is itself a
reported finding.

## 6. Confounds explicitly controlled

- **Budget parity** across systems (primary confound).
- **Identical corpus snapshot** (`base_commit`, same ingest filters).
- **Identical query text** (`problem_statement`, verbatim).
- **No gold leakage** into ranking (gold spans are used only for scoring).
- **Macro-average over tasks** (not micro), so large repos don't dominate.
- Report per-language breakdown (Python-heavy dataset could mask other langs).

## 7. Falsification conditions

- **F1** (primary kill): if the line-level determinism tax exceeds 15 pp on Lite,
  the broad "reproducible selection is competitive" thesis is rejected for
  general use; RCP is repositioned as a high-assurance/compliance feature.
- **F2:** if Entroly-deterministic does not beat the BM25-deterministic floor,
  the specific selector adds no value over a trivial deterministic baseline.
- **F3:** if Entroly repeatability is < 1.000 on this real-repo pipeline (vs the
  frozen-corpus result in Exp 1), the end-to-end reproducibility claim fails and
  must be fixed or narrowed before the tax number is meaningful.
- **F4 (evidence quality):** if Entroly's evidence-drop is materially worse than
  neural at equal recall, the reproducibility win comes at a precision cost that
  must be disclosed.

## 8. Known limitations (declared before running)

- Block/AST-level metrics deferred (no tree-sitter); file+line only.
- Cross-architecture reproducibility remains untested (Exp 1 scope).
- Neural/agent arms are gated on compute budget; until funded, only the
  deterministic tax floor (Entroly vs BM25) is measured, which cannot by itself
  decide the table — it can only falsify F2/F3.
