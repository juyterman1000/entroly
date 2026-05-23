# Conformal Selective-Verification Cascade — Honest Result

**Idea.** Cheap deterministic WITNESS emits a class-conditional
split-conformal p-value; only the *uncertain band* escalates to an LLM;
the band edge is exactly `escalation.py` rule (★) with the conformal
local error rate as the operative risk. Goal: Pareto-dominate both a
$0 verifier and a per-call LLM judge, with a finite-sample
selective-risk guarantee.

**Protocol.** HaluEval-QA, 600-item / 1,200-decision shared sample,
balanced. Band fitted on a 600-decision calibration split, measured on
the disjoint 600-decision test split. WITNESS scores deterministic;
gpt-4o-mini judge on the same items (cached).

## What the data actually shows (test split, n=600)

| System | Error | Cost (LLM calls/item) |
|---|---|---|
| WITNESS-only (shipped τ) | 0.1550 | 0.00 |
| gpt-4o-mini-only | 0.1433 | 1.00 |
| **Cascade @ε=0.02–0.05** | **0.1333** | **~0.91** |
| Cascade @ε=0.08 | 0.1500 | 0.62 |
| Cascade @ε≥0.10 | ≥0.157 | 0.54–0.61 |

## Honest verdict — a partial success and a useful negative

**Validated (solid):**
- The **conformal selective-risk guarantee holds empirically** on real
  HaluEval-QA: realized selective risk ≤ ε within the finite-sample
  ⌈(n+1)(1−α)⌉ slack at every tested ε. The math is correct and
  falsification-tested (7/7, incl. a negative control that already
  caught one unit bug in the harness).
- The band edge **provably equals** `escalation.py` rule (★).

**Real but NOT statistically established:**
- The cascade's point-estimate error (0.1333) is **below both**
  WITNESS-only (0.1550) and gpt-4o-mini-only (0.1433) — a genuine
  fusion effect. But at n=600 the standard error on an ~0.14 error
  rate is ≈0.014; the margins (0.010 vs LLM, 0.022 vs WITNESS) are
  **within ~1σ**. Suggestive, underpowered, **not** a proven win.

**The hoped-for breakthrough does NOT replicate on real HaluEval-QA:**
- At the only operating point that beats both verifiers, the cascade
  still escalates **91.5%** of items — a mere **~8% LLM-cost cut**, not
  the large saving the synthetic concentrated-error regime produced.
- Diagnosed cause (this is the valuable part): gpt-4o-mini-only (0.143)
  is barely better than WITNESS-only (0.155), and WITNESS's
  confidently-correct region is *small* — its errors are **not
  score-concentrated**. This is the same root cause as the collapsed
  calibrated τ≈0.0004 in the first benchmark: WITNESS's scalar risk
  does not pack its errors into an identifiable low-confidence band, so
  there is little "easy mass" to keep cheaply.

## The real, defensible contribution

1. A correct, reusable, falsification-tested module
   (`entroly/conformal_cascade.py`) that converts *any* well-calibrated
   cheap verifier into a cost-bounded cascade with a conformal
   selective-risk guarantee — the proven `escalation.py` bound, now
   operational on a measured two-verifier system.
2. A precise, measured **negative finding**: on HaluEval-QA the
   bottleneck is **not** the cascade architecture; it is WITNESS's
   score concentration/calibration. The synthetic test proves the
   cascade yields large savings *when* the cheap verifier's errors are
   concentrated; real WITNESS-QA is only weakly so.
3. Therefore the genuine lever for a hallucination-cost breakthrough is
   **improving the cheap verifier's calibration/score-concentration**
   (so a large fraction is confidently keep-cheap), after which the
   cascade math — already proven — converts that directly into large,
   guaranteed LLM-cost savings.

## Not claimed

Statistical superiority over either verifier (underpowered at n=600);
large cost savings on QA (refuted here, ~8% only); conditional
coverage; robustness to distribution shift; LLM-as-oracle (its measured
error is carried explicitly).
