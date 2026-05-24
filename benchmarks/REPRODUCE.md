# Reproducibility Protocol — Entroly Benchmarks

This document specifies the methodology behind every measured number in
`README.md`. It exists so that an independent reviewer — an academic,
an FTC examiner, an enterprise procurement team — can re-run, re-derive,
and cross-check every claim without trusting the maintainer.

This package conforms to the **NeurIPS / ICML / ACL reproducibility
checklist** (code published, data specified, seed fixed, hyperparameters
disclosed, statistical significance reported, per-sample auditability).

---

## TL;DR — one command reproduces everything

```bash
# requires OPENAI_API_KEY in env or .env
python benchmarks/audited_runner.py            # all 7 benchmarks
python benchmarks/audited_runner.py squad      # just one
```

For every benchmark, this writes two files to `benchmarks/results/`:

| File | Purpose |
|---|---|
| `<bench>_accuracy.json` | Aggregated headline metrics + provenance block (model, timestamp, git_sha, seed, sample size, budget). README rows link directly to this. |
| `<bench>_audit.jsonl` | One JSON line per sample with `{index, question, expected, baseline_pred, baseline_correct, entroly_pred, entroly_correct, baseline_total_tokens, entroly_total_tokens, baseline_context_chars, entroly_context_chars}`. A reviewer can `grep` for any question and verify the LLM was shown that context and returned that answer. |

The aggregated JSON can be re-derived from the JSONL — falsifying one
without the other is impossible.

---

## 1. Datasets

All datasets are public and loaded by name from existing
`bench/accuracy.py` helpers. Caches under `bench/.cache/` are gitignored
because they're large and pinned by the dataset publisher's hash, not by
us. First run downloads them; subsequent runs read from cache.

| Benchmark | Dataset | Source | License | Sample size in README |
|---|---|---|---|---|
| NeedleInAHaystack | Synthetic (PaulLusztin-style essay haystack + a needle sentence) | Generated in-process by `bench.accuracy.bench_needle` | n/a (synthetic) | n=20 |
| LongBench (HotpotQA) | THUDM/LongBench `hotpotqa` subtask | https://github.com/THUDM/LongBench | MIT | n=50 |
| Berkeley Function Calling | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` (Simple subset) | https://gorilla.cs.berkeley.edu/leaderboard.html | Apache 2.0 | n=50 |
| SQuAD 2.0 | Stanford SQuAD 2.0 dev set | https://rajpurkar.github.io/SQuAD-explorer/ | CC-BY-SA 4.0 | n=50 |
| GSM8K | OpenAI GSM8K | https://github.com/openai/grade-school-math | MIT | n=20 (pass-through) |
| MMLU | hendrycks/test (MMLU) | https://github.com/hendrycks/test | MIT | n=20 (pass-through) |
| TruthfulQA (MC1) | sylinrl/TruthfulQA | https://github.com/sylinrl/TruthfulQA | Apache 2.0 | n=20 (pass-through) |

Sample selection is **deterministic, no random subsetting**: each loader
returns the first N items in the cached dataset order. Run with the same
N and you get the same N items.

---

## 2. Methodology

For each benchmark item, the runner makes **two LLM calls**:

1. **Baseline call** — the LLM sees the **full original context** unmodified.
2. **Entroly call** — the LLM sees the context after Entroly's QCCR
   compressor has selected fragments to fit under `budget` tokens.

Both calls use:

- **Model**: `gpt-4o-mini` (OpenAI). Same model, same parameters.
- **Temperature**: 0 (deterministic decoding — same input → same output)
- **max_tokens**: derived from the benchmark (most use 64; LongBench
  uses each item's own `max_new_tokens` field, typically 52)
- **No system prompt overrides** — context is delivered as a `system`
  message, question as a `user` message
- **No retries beyond 3 with exponential backoff** for transient API
  errors

The compressor is `entroly.qccr.select(fragments, token_budget=budget,
query=question)` called through
`bench.accuracy._compress_messages_modal(..., mode='entroly')`. This is
the **same code path** that the entroly proxy uses in production.

### Scoring

`bench.accuracy._check_answer(prediction, expected, benchmark, metadata)`
is the single scoring entry point. It dispatches per benchmark:

| Benchmark | Scoring rule |
|---|---|
| needle | Substring match: does the prediction contain the needle's secret token? |
| longbench | Normalized substring match (lowercase, strip articles/punct) against any acceptable answer. |
| squad | Same normalized substring match as longbench. |
| bfcl | AST-level function-call equality against ground-truth call expression. |
| gsm8k | Numeric answer extraction (regex on last `\n#### N` line) + exact match. |
| mmlu | Multiple-choice letter extraction (A/B/C/D) + exact match. |
| truthfulqa | Multiple-choice letter extraction + exact match. |

Read the actual code at `bench/accuracy.py:_check_answer`.

### Statistical reporting

We report **Wilson 95% confidence intervals** for every accuracy figure
(more accurate than the normal-approximation CI for small n and
extreme p). At n=20, the CI half-width at p=0.85 is ~18pp; at n=50 it's
~13pp; at n=100 it's ~9pp. **All differences smaller than the CI
half-width are statistical noise** and should be read as such.

### Token savings

`token_savings_pct = 1 - (avg_entroly_tokens / avg_baseline_tokens) × 100`,
where each `tokens` is the **OpenAI-reported** `prompt_tokens +
completion_tokens` for that call. This is the model's own count, not
our estimate.

---

## 3. Provenance block

Every `<bench>_accuracy.json` includes a `provenance` block. Example:

```json
{
  "provenance": {
    "model": "gpt-4o-mini",
    "seed": 42,
    "budget": 2000,
    "timestamp_utc": "2026-05-23T18:14:22+00:00",
    "git_sha": "0a54982a1b9f",
    "elapsed_s": 80.5,
    "audit_jsonl": "needle_audit.jsonl",
    "audit_record_count": 20,
    "scoring": "bench.accuracy._check_answer (benchmark='needle')",
    "compressor": "entroly.qccr.select (via bench.accuracy._compress_messages_modal mode='entroly')",
    "dataset_loader": "<lambda>"
  }
}
```

Fields:

- `model` — OpenAI model id used (`gpt-4o-mini` for every README row)
- `seed` — random seed (always `42`)
- `budget` — token budget for the compressed call
- `timestamp_utc` — ISO-8601 UTC of when the run started
- `git_sha` — first 12 chars of the commit SHA the runner saw
- `elapsed_s` — wall-clock time for the benchmark
- `audit_jsonl` — path (relative to results/) to the per-sample audit
- `audit_record_count` — number of items in the audit JSONL (must
  equal `samples`)
- `scoring` — the exact scoring function used
- `compressor` — the exact compressor used
- `dataset_loader` — the loader function name

If any of these don't match what the reviewer expects, the run is not
the one the README cites.

---

## 4. Per-sample audit (the strongest form of proof)

`<bench>_audit.jsonl` is a JSON-Lines file. **One sample per line**.
Example line from `squad_audit.jsonl`:

```json
{"index": 0, "question": "In what country is Normandy located?", "expected": ["France", "France", "France", "France"], "baseline_pred": "France", "baseline_correct": true, "baseline_total_tokens": 168, "entroly_pred": "France", "entroly_correct": true, "entroly_total_tokens": 92, "entroly_context_chars": 261, "baseline_context_chars": 762}
```

A reviewer can:

1. `wc -l benchmarks/results/squad_audit.jsonl` → equals
   `samples` in the JSON (currently 50).
2. `grep "Normandy" benchmarks/results/squad_audit.jsonl` → find the
   exact LLM I/O for that question.
3. Re-compute aggregates from the JSONL and verify they match the
   `_accuracy.json`.
4. Cross-check that `sum(entroly_correct) / samples` equals
   `entroly_accuracy` in the JSON.

If aggregates and audit lines disagree, the run is invalid.

---

## 5. Limitations & honest scope

- **Single seed**: results are from `seed=42`. Variance from multiple
  seeds is not yet reported (would be a useful addition; future work).
- **Single model**: only `gpt-4o-mini` is in the table. The compressor
  is model-agnostic, but the LLM accuracy obviously depends on the
  model. Rerunning with a different model is supported (`--model` on
  `bench/accuracy.py`); the README rows specifically claim
  `gpt-4o-mini`.
- **Sample sizes vary by row**: shown in the `n` column. Pass-through
  rows (gsm8k/mmlu/truthfulqa) use n=20 — wider CIs but the claim
  (retention ≥ 94%, savings ≈ 0%) is robust to that width.
- **Drift over time**: OpenAI updates `gpt-4o-mini` periodically. The
  `provenance.timestamp_utc` records when each row was measured. If
  OpenAI ships a major model change, baseline accuracies may shift —
  we re-run and update.
- **Pass-through is tautological**: when the original context already
  fits under budget, Entroly emits the input unchanged, so baseline ≈
  entroly by construction. The pass-through rows verify Entroly does
  no harm in this case — they are not evidence of compression gains.
- **Compressed-context savings are not the same as $-savings**:
  `token_savings_pct` is on input tokens. Total API cost savings
  depend on the input/output token ratio for your specific workload.
  `cost_savings_pct` in the JSON gives a per-call estimate using
  current OpenAI pricing.
- **The compressor changed between releases**: v1.0.4 added entity
  boost and degenerate fallbacks (see `entroly/qccr.py`). Older
  release runs at the same n may give slightly different numbers
  because they used the v1.0.3 compressor. Provenance `git_sha`
  identifies the version.

---

## 6. How to independently verify

Cheapest verification, no API needed:

```bash
# Re-run the answer-survival diagnostic — proves Entroly's compressor
# is preserving the gold answer span at the rate we claim. Zero LLM calls.
python benchmarks/diagnose_anchor_survival.py
```

Full verification with API (~25 min, ~$1 in OpenAI cost at gpt-4o-mini
pricing as of May 2026):

```bash
export OPENAI_API_KEY=sk-...
python benchmarks/audited_runner.py
```

Then audit the produced files:

```bash
# Sanity: every accuracy.json has a matching audit.jsonl whose line
# count equals samples
for f in benchmarks/results/*_accuracy.json; do
  bench=$(basename "$f" _accuracy.json)
  audit="benchmarks/results/${bench}_audit.jsonl"
  lines=$(wc -l < "$audit")
  samples=$(python -c "import json; print(json.load(open('$f'))[0]['samples'])")
  echo "$bench: audit lines=$lines, samples in json=$samples"
done

# Re-derive accuracy from audit
python -c "
import json, sys
audit = sys.argv[1]
n_correct = sum(1 for line in open(audit) if json.loads(line)['entroly_correct'])
n_total = sum(1 for _ in open(audit))
print(f'derived entroly_accuracy from audit: {n_correct/n_total:.4f}')
" benchmarks/results/squad_audit.jsonl
```

If the derived accuracy from the audit equals the JSON's stated
`entroly_accuracy` for every benchmark, the evidence is internally
consistent.

---

## 7. Comparison fairness

Where the README compares Entroly to alternatives (e.g. WITNESS vs
gpt-4o-mini in the hallucination table), the runs use:

- **Identical samples** — both systems see exactly the same 1,200
  decisions (`witness_on_gpt_sample` in `halueval_qa_faithful.json`).
- **Identical evidence** — both systems see the same supporting
  context, with no asymmetric knowledge access.
- **Statistical-tie language** — when 95% CIs overlap, the README says
  "statistical tie", not "beats". See the "Honest reading" note above
  the hallucination table.

The README explicitly disclaims SOTA: "**This is not a global-SOTA
claim**: published methods that score higher on HaluEval-QA do so with
privileged signals (token log-probs, a paired evaluator LLM, or
supervised training on the benchmark); among zero-cost, black-box,
text-only, untrained verifiers we found no verified method that beats
it."

---

## 8. Reporting issues

If you re-run and get results that don't match the README within the
stated CI bands, please open an issue at
https://github.com/juyterman1000/entroly/issues with:

- The `<bench>_accuracy.json` your run produced
- A few lines from the `<bench>_audit.jsonl` showing the discrepancy
- Your `git_sha`, `model`, and `provenance.timestamp_utc`

We will investigate and update the README to match the more recent or
more reliable measurement.
