# Experiment 1 — Verifier Recall Under Controlled Evidence Ablation

**EICV detects compression-induced evidence gaps: holdout recall 0.940 at FPR 0.040, and is exactly blind (AUROC 0.500) to benign length-matched ablation.**

Preregistration: [VERIFIER_ABLATION_PREREGISTRATION.md](../VERIFIER_ABLATION_PREREGISTRATION.md)
(criteria locked at `c45d3073` before the first run; protocol v2 declared at `ac0178c7`).
Harness: `benchmarks/verifier_ablation_recall.py`. Seed 42, SQuAD v2 validation,
200 fresh eval items (dev/holdout 100/100), 200 disjoint calibration pairs.

## Question

The committed EICV results (AUROC 0.998/0.944, `eicv_endtoend.json`) measure
right-vs-hallucinated answers **with full evidence present**. The closed loop
(`compress -> answer -> verify -> retrieve omitted span -> retry`) depends on a
different property: does the verifier fire when a *correct* claim's supporting
evidence was **removed from context** — the compression failure case?

## Result (protocol v2, fresh items)

| Criterion | Value | Locked threshold | Outcome |
|---|---:|---|---|
| P1 AUROC(intact vs critical ablation) | 0.9748 | ≥ 0.85 | PASS |
| P2 holdout recall at dev-selected τ | 0.940 [0.875, 0.972] | ≥ 0.75 | PASS |
| P2 holdout FPR at τ | 0.040 [0.016, 0.098] | ≤ 0.25 | PASS |
| F1 AUROC(intact vs random length-matched ablation) | 0.5000 | fails if > 0.65 | clean |
| S1 flag rate on real-QCCR-dropped answers | 11/11 = 1.00 [0.741, 1.0] | ≥ 0.75 | PASS (small n) |

Mean scores (1−φ): intact 0.0325, random **0.0325**, critical 0.8504.
Latency: 1.86 ms/verify. Wilson 95% CIs. Exit criteria: P1 ∧ P2 ∧ ¬F1 → **PASS**.

## S1 — real compression, not synthetic ablation

With the native engine built (`entroly_core` 1.0.62 local wheel), the S1
condition ran real QCCR selection (`qccr.select`, query = the SQuAD question)
at a token budget matched to the critical ablation's kept length:

- **Answer survived selection in 189/200 items (94.5%)** — independently
  consistent with the committed matched-cap frontier result (93.3% retention
  at 4x, `compression_frontier.md`).
- **All 11 dropped-answer cases were flagged by EICV (11/11)** at the
  preregistered τ. Point estimate 1.00 passes the ≥ 0.75 criterion; n = 11 is
  small and the Wilson 95% lower bound is 0.741 — the sample is too small to
  claim recall above ~0.74 on real-compression gaps with confidence.
- Observed span-level silent failures (answer dropped AND not flagged) across
  200 real compressions: **0**.

## The v1 story (kept, on purpose)

The v1 run **failed** its own falsification gate: AUROC(intact vs random) came
out 0.9334, which per the locked rules voided the result
(`verifier_ablation_recall.json`, committed as FAIL). Diagnosis
(`diagnose_ablation_artifact.json`) traced it to a **metric defect, not a
verifier property**: the rank-AUROC helper had no midrank tie correction, and
with ~90% of scores tied at exactly 0.0, Python's tuple sort broke ties by
label — fabricating AUROC ≈ 0.94 between *semantically identical* inputs.
The same uncorrected pattern exists in 7 other files (including
`eicv_endtoend.py`, `esg_falsification.py`, `fever_esg.py`,
`stave_benchmark.py`, `entroly/semantic_entropy.py`, `entroly/rnr.py`,
`entroly/adversarial_calibration.py`) and is tracked for a separate audit of
the committed AUROC artifacts.

## What this licenses — and what it does not

- **Licensed:** the verifier link in the closed-loop design is validated at
  span level. When compression drops a claim's evidence, EICV flags it 94% of
  the time at a 4% false-alarm rate, and does not fire merely because context
  got shorter.
- **Not licensed:** any claim about model behavior or agent sessions. Claims
  here are gold spans, not model outputs; no LLM was called. Task-level
  validation is Experiment 0 (`AGENTIC_TASKS_PREREGISTRATION.md`), which this
  result gates.
- S1's n = 11 dropped-answer sample bounds real-compression recall only at
  ≥ 0.74 (95% lower bound); a larger dropped-answer sample (tighter budgets or
  more items) is needed before quoting a recall number for real compression.

## Reproduce

```bash
python benchmarks/verifier_ablation_recall.py
python benchmarks/diagnose_ablation_artifact.py
```
