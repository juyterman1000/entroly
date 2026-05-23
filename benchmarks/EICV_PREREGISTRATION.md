# EICV Pre-Registration

**Frozen at:** 2026-05-20
**Git commit at freeze:** `9587650c3f4288c7e163d7cfbdbe9a5b95ae1cf5`
**Architecture:** EICV (Evidence-Invariant Causal Verification), 6-layer hierarchy estimating
Epistemic Support Density Φ(x) = ∫ P(e|x)·C(e,x)·R(e,x) de
**Status:** Phase 0 — current-state baseline, NO EICV components landed yet.

This document fixes the targets and falsification protocol BEFORE measurement. Per the
project's falsification-first policy, any number reported outside the rules in this
document is "headline only, NOT trust-defensible." Amendments are appended below; no
in-place edits to previously-frozen claims.

---

## 1. Hypothesis (the falsifiable claim)

EICV is a hallucination verifier that, when fully landed (Phases 1–4), exhibits:

> **Provable selective-risk control under adversarial construction shift, at < 5%
> the cost of LLM-judge.**

This is not "highest AUROC on a single benchmark." It is a *category* claim — the
verifier behaves like a SAT solver with uncertainty (constraint satisfaction +
calibrated abstention), and its guarantees compose across layers via e-values.

The claim is *refuted* if any of the failure conditions in §6 fires after a full
Tier-1 run.

---

## 2. Pre-registered metrics and targets

Targets are decided **now**, before EICV components land. They will not be
revised in light of measured numbers; revision becomes a §8 amendment.

### 2.1 Primary metric

| Metric | Target | Rationale |
|---|---:|---|
| **Risk @ 80% coverage, worst-manifold** | ≤ 0.05 | Selective prediction baseline. Beats unconditional NLI (~0.10) at high coverage. |

The primary metric is computed *after* EICV's Layer-5 (e-value CRC + manifold-conditional
calibration) is in place. Phase 0 reports a placeholder; Phase 4 produces the binding
measurement.

### 2.2 Robustness metrics (the trust headline)

| Metric | Target | Beats |
|---|---:|---|
| Falsification-survived AUROC drop, mean (C1 → C4) | ≤ 0.03 | Fusion-4: 0.441 collapse on HaluEval-QA |
| Worst-manifold ECE | ≤ 0.08 | LLM-judge baselines ~0.15, DeBERTa-NLI ~0.12 |
| Conformal coverage violation @ α = 0.10 | ≤ 0.02 | Theoretical bound (by construction under exchangeability) |

### 2.3 Detection AUROC (secondary, NOT the headline)

| Benchmark | Target | Published-baseline anchor |
|---|---:|---|
| HaluEval-QA | ≥ 0.85 | DeBERTa-v3-large NLI ~0.85; current Entroly fusion 0.843 |
| HaluEval-Dialogue F1 | ≥ 0.68 | NLI ~0.78; current 0.584 |
| HaluEval-Summarization F1 | ≥ 0.55 | NLI ~0.65; current 0.301 (retention-first) |
| TruthfulQA detection AUROC | ≥ 0.78 | LLM-judge baselines |
| FActScore atomic precision | ≥ 0.72 | GPT-4 0.76; smaller methods 0.60–0.65 |
| FEVER label accuracy | ≥ 0.75 | DeBERTa-large 0.74; LLM 0.78 |
| TRUE benchmark mean AUROC | ≥ 0.80 | DeBERTa-v3-large 0.79 |
| RAGAS faithfulness | report only | no published comparable baseline |
| HHEM leaderboard hallucination rate | within top decile | leaderboard-relative |

### 2.4 Cost

| Metric | Target |
|---|---:|
| Cost per claim (CPU, P99) | ≤ 20 ms |
| Cost per claim (CPU, P50) | ≤ 8 ms |

Compared to GPT-4-judge ~500 ms + $ per call.

---

## 3. Datasets and provenance

All datasets are pinned to specific HuggingFace dataset revisions (or equivalent
public source) at first run. The pin is recorded in `eicv_baseline.json`.

### 3.1 Tier-1 (must publish on all of these)

| Benchmark | Source | Sample size | Seed |
|---|---|---:|---:|
| HaluEval-QA | `pminervini/HaluEval` config `qa` | 10000 (paired) | 42 |
| HaluEval-Dialogue | `pminervini/HaluEval` config `dialogue` | 200 | 42 |
| HaluEval-Summarization | `pminervini/HaluEval` config `summarization` | 200 | 42 |
| TruthfulQA | `truthful_qa/generation` | 817 | 42 |
| FActScore | Min et al. 2023 bios subset | TBD on adapter land | 42 |
| FEVER | `fever/fever` v1.0 | TBD on adapter land | 42 |
| TRUE benchmark | 11 sub-tasks (see Honovich 2022) | per-task default | 42 |
| HHEM | Vectara HHEM-2.1 test set | per leaderboard | 42 |
| RAGAS faithfulness | per existing benchmark | as currently run | 42 |

### 3.2 Provenance contract

Every benchmark result JSON must include:

- `dataset_hash` — SHA256 of `sorted(item_id || ' ' || canonical_text)` over the
  evaluated subset
- `git_commit` — `git rev-parse HEAD` at run time
- `python_version`, `entroly_version`
- `seed`
- `pin` — HuggingFace dataset revision SHA (when available)
- `start_ts_utc`, `end_ts_utc`

Without all of these, the result is not trust-defensible.

---

## 4. Falsification protocol

Every benchmark runs the four-condition adversarial probe from
`benchmarks/fusion4_falsification.py` (or its per-benchmark equivalent):

| Condition | Negatives | Positives | Tests |
|---|---|---|---|
| C1 | original r | original h | Headline AUROC |
| C2 | original r | entity-controlled h_ctrl | Entity-coverage artifact stripped |
| C3 | paraphrased r_para | original h | Verbatim-copy artifact stripped |
| C4 | paraphrased r_para | entity-controlled h_ctrl | Both artifacts stripped (realistic) |

**Reporting rules:**

- The **trust-defensible AUROC** for a benchmark is `min(C1, C2, C3, C4)`, not C1.
- The **artifact drop** is `C1 − C4`. Report it always.
- A C1-only number is reported with the label `headline_only_not_trust_defensible: true`.
- The `falsification.survives` boolean is true iff `min(C1..C4) ≥ target − 0.03`.

If `survives = false`, no public claim against that target may be made. The result
is still reported (transparency) but with a caveat.

---

## 5. Pre-registered analyses

### 5.1 Per-benchmark
- AUROC, F1, precision, recall, accuracy
- 95% CI for AUROC via DeLong (paired comparisons) or bootstrap (non-paired)
- 95% CI for F1 via Wilson
- ECE binned (10 equal-mass bins) + worst-bin error
- Selective risk curve (risk@coverage, swept α)
- Cost histogram (P50, P95, P99)

### 5.2 Cross-benchmark
- Mean target metric across Tier-1 (8 benchmarks)
- Worst-benchmark metric (the "bound")
- Per-manifold (entity/numeric/compositional/temporal/retrieval-shuffle) min AUROC

### 5.3 Multiple-testing correction
9 benchmarks × ~5 secondary metrics = 45 hypothesis tests. Apply Benjamini-Hochberg
FDR control at q = 0.05 to all AUROC-comparison p-values against published baselines.
Primary metric (Risk@Coverage worst-manifold) is not subject to correction (single test).

### 5.4 Aggregation
The unified runner emits one canonical JSON (`eicv_baseline.json`) per execution.
Schema is `eicv-baseline-v1` (see `benchmarks/eicv_baseline.py` for the exact
field set). Schema changes bump the version suffix and are §8 amendments.

---

## 6. Failure conditions (claim is REFUTED)

Any one of these fires → architecture must be revised before publication.
No selective reporting; the failure is published.

- **F1.** Risk@Coverage(0.80) on worst-manifold > 0.05 → primary claim refuted
- **F2.** Artifact drop C1 → C4 > 0.10 on any Tier-1 benchmark → artifact dominance,
  trust claim refuted on that benchmark
- **F3.** Worst-manifold ECE > 0.15 → calibration claim refuted
- **F4.** Conformal coverage violation @ α = 0.1 > 0.05 (more than 2.5× the bound)
  → coverage guarantee refuted
- **F5.** Cost per claim P99 > 50 ms → cost claim refuted
- **F6.** Mean Tier-1 AUROC < 0.78 → competitive claim refuted

Reporting a refuted claim as success is a §8 violation.

---

## 7. What does NOT get reported

- Cherry-picked benchmark subsets ("we got 0.92 on the subset where it works")
- C1-only numbers presented without C2–C4 context
- Numbers with unpinned dataset versions
- Aggregate means that hide failed benchmarks (must show per-benchmark + aggregate)
- Per-component AUROC presented as the end-to-end EICV claim

---

## 8. Amendments

Any change to §1–§7 after the freeze date is appended here, with date, rationale,
and commit hash. No in-place edits.

(none yet)

---

## 9. Publication contract

When the EICV trust report is published (Phase 5 deliverable):

- All raw `eicv_baseline.json` outputs published alongside
- Git commit hash published
- Dataset revisions published (HF revision SHAs)
- Per-benchmark falsification tables included as supplementary
- This pre-registration document published unchanged, with all amendments
- Any §6 failure condition that fired is reported in the body, not buried
