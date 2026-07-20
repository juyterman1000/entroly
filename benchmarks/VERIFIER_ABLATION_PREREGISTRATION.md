# Preregistration — Experiment 1: Verifier Recall Under Controlled Evidence Ablation

**Status:** criteria locked at the commit that introduces this document, before the
first full run of the harness. Any later edit to the criteria sections invalidates
the preregistration and must be declared as a protocol revision (v2).

- **Harness:** `benchmarks/verifier_ablation_recall.py`
- **Result artifact:** `benchmarks/results/verifier_ablation_recall.json`
- **Seed:** 42 (single fixed seed; no seed selection)

## 1. Question

Does EICV detect **compression-induced evidence gaps** — a *correct* claim whose
supporting evidence has been removed from the context?

This gates the closed-loop design
(`compress -> answer -> verify -> retrieve omitted span -> retry once`,
implemented in `entroly/compression_verification_loop.py`): the loop converts
verification into recovered evidence **only if** the verifier fires when evidence
was dropped and stays quiet when it was not.

## 2. Known prior (public, committed)

`benchmarks/results/eicv_endtoend.json`: AUROC 0.9976–0.9979 (SQuAD v2) and
0.944 (HaluEval-QA) distinguishing right vs hallucinated answers **with full
evidence present**, surviving falsification conditions C1–C4. That result does
NOT establish sensitivity to *missing evidence for a correct answer*, which is
the closed-loop failure case. The endtoend artifact also shows the decision
zones are miscalibrated for this regime (`supported: 0` even on grounded
pairs), so this experiment treats φ as a continuous score and reports
decision-based flag rates without pass/fail weight.

## 3. Dataset construction (SQuAD v2 validation, seed 42)

Pool: items with non-empty context and answer length > 5 characters, shuffled
with `random.Random(42)` (mirrors `eicv_endtoend.load_squad`). The question
text is retained for the engine-gated secondary condition.

Sentence split: `re.split(r"(?<=[.!?])\s+", context)`, empty segments dropped.

**Eligibility cascade** — every exclusion is counted and reported in the result
artifact:

| Gate | Rule |
|---|---|
| E1 | answer occurs (case-insensitive substring) in the context |
| E1b | answer occurs inside at least one single sentence (not only across a boundary) |
| E2 | context has ≥ 4 sentences |
| E3 | critical ablation removes ≤ 50% of context characters |
| E4 | after critical ablation the answer no longer occurs in the remaining context |
| E5 | a random length-matched ablation exists that avoids all answer-bearing sentences (total removed characters within ±30% of the critical ablation, ≤ 50 randomized attempts) |

- **Eval set:** first 200 eligible items in pool order.
- **Dev/holdout split:** eval items 1–100 are dev, 101–200 are holdout.
- **Calibration set:** the next 200 pool items *after* the pool cursor at which
  the eval set completed (disjoint from eval by construction), used as grounded
  `(context, answer)` pairs for `EICVAnalyzer.fit_calibrators`.

HaluEval-QA is excluded a priori: its knowledge fields are typically 2–4
sentences, so critical ablation would routinely violate E2/E3 and collapse n.

## 4. Conditions

Claim is always the **gold answer span**. Evidence varies:

| Condition | Evidence |
|---|---|
| INTACT | full context |
| CRITICAL | context minus **all** answer-bearing sentences |
| RANDOM | context minus non-answer sentences, total removed characters matched to CRITICAL within ±30% |
| QCCR-REAL (secondary, engine-gated) | QCCR selection (`entroly.qccr.select`) over the context sentences with `query = question` and token budget matched to CRITICAL's kept length; run only when the native engine is available, otherwise recorded as `{"status": "skipped"}` |

One `EICVAnalyzer` instance, calibrated once, scores all conditions.
Score = `1 − φ` (higher = more suspect).

## 5. Metrics

- mean φ per condition; φ-gap (INTACT − CRITICAL) and (INTACT − RANDOM)
- AUROC(INTACT vs CRITICAL) and AUROC(INTACT vs RANDOM), rank statistic, full eval set
- operating threshold τ: smallest score threshold with dev FPR ≤ 0.20; recall
  and FPR then measured on holdout at that τ
- decision-based flag rates (strict = `hallucinated`; liberal = `≠ supported`)
  with Wilson 95% CIs — reported for operations, no pass/fail weight
- per-verify latency

## 6. Preregistered criteria (locked)

| ID | Criterion |
|---|---|
| **P1** (primary) | AUROC(INTACT vs CRITICAL) ≥ 0.85 on the full eval set (n = 200 per condition) |
| **P2** (operating point) | holdout recall ≥ 0.75 **and** holdout FPR ≤ 0.25 at the dev-selected τ |
| **F1** (falsification) | the experiment **fails regardless of P1/P2** if AUROC(INTACT vs RANDOM) > 0.65 — the verifier would be detecting "something was removed" (length/ablation artifact) rather than "this claim's evidence is missing" |
| **S1** (secondary, engine-gated) | on QCCR-REAL items where the answer span did not survive selection: flag rate at τ ≥ 0.75. Skipping (no native engine) does not affect P1/P2/F1 |

Exit code: 0 iff P1 ∧ P2 ∧ ¬F1.

## 7. Interpretation commitments

- **P1 ∧ P2 ∧ ¬F1** → the verifier link in the closed loop is validated at the
  span level; proceed to task-level closed-loop trials
  (`benchmarks/AGENTIC_TASKS_PREREGISTRATION.md`).
- **F1 fires** → EICV's apparent recall is confounded with ablation artifacts;
  before any closed-loop claim, isolate evidence-gap-specific signals (e.g.
  ESG `unsupported_fraction` alone) and re-preregister.
- **¬P1** → the closed loop must not rely on post-hoc verification to trigger
  retrieval; the trigger must come from selection-time receipts (known-omitted
  spans), which changes the loop design, not just its tuning.

## 8. Scope limits

- No LLM is called; this measures the verifier, not model behavior.
- Claims are gold spans, not model outputs; full-sentence model answers are
  covered by Experiment 0.
- Single dataset (SQuAD v2), single seed; results are span-level evidence, not
  agent-session evidence.

---

## 9. Protocol revision v2 (declared before the v2 run)

**v1 outcome (committed):** P1 and P2 passed (AUROC 0.9928; holdout recall
0.960 at FPR 0.080), F1 fired (AUROC intact-vs-random 0.9334) →
`verdict_pass: false` per the locked rules
(`benchmarks/results/verifier_ablation_recall.json`).

**Diagnosis (committed, dev items only):**
`benchmarks/results/diagnose_ablation_artifact.json` shows the F1 firing was a
**metric defect, not a verifier property**:

- Two semantically identical evidence variants (raw context vs
  whitespace-rejoined) "separated" at AUROC 0.9431 — impossible for a real
  signal.
- ~90% of intact and random scores are exactly 0.0 (heavy ties); the v1
  `_auroc` helper had no midrank tie correction, and Python's tuple sort broke
  score-ties by label, systematically ranking label-1 items above label-0
  items at every tie.
- Operationally the verifier is clean at the deployed threshold: flag rates
  intact 6%, random 5%, critical 97%.

**v2 changes (methodology only — all P1/P2/F1/S1 thresholds unchanged):**

1. **Metric fix:** midrank tie-corrected Mann-Whitney AUROC (identical
   distributions ⇒ 0.5 exactly).
2. **Construction fix:** INTACT is `" ".join(sentences(context))` so all
   conditions share one formatting pipeline (removes the raw-vs-rejoined
   confound as a matter of hygiene; the diagnosis shows it was not the driver
   once ties are handled, since means and flag rates were identical).
3. **Fresh evaluation items:** the v2 eval set (200 items, dev/holdout 100/100)
   is drawn from the pool *after* the v1 calibration range (pool index ≥ 492),
   so no item used in the v1 run or the diagnosis contributes to the v2
   verdict. v2 calibration uses the next 200 pool items after the v2 eval
   cursor.
4. **Artifact:** `benchmarks/results/verifier_ablation_recall_v2.json`,
   schema `verifier-ablation-recall-v2`. The v1 artifact remains committed.

The v1 verdict stands as recorded. The v2 run is the binding verdict for the
Section 7 interpretation commitments.
