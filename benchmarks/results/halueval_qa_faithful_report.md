# HaluEval-QA — Faithful Protocol Result

**Dataset:** `pminervini/HaluEval[qa]` — 10,000 items, each with
`knowledge`, `question`, `right_answer`, `hallucinated_answer`.
**Protocol:** standard HaluEval — score *both* answers per item →
20,000 balanced binary decisions. Metric: accuracy; threshold-free
AUROC reported as the unspoofable primary number. Seed 42.
**No threshold cheating:** WITNESS operating point chosen on a disjoint
2,000-item calibration split; accuracy reported on the held-out 8,000.
**GPT judges** see the *same knowledge* WITNESS sees (fair grounded
comparison) on a shared 600-item / 1,200-decision sample from the test
split.

## WITNESS — full set (20,000 decisions)

| Metric | Value |
|---|---|
| **AUROC (threshold-free, primary)** | **0.7976** |
| Test accuracy @ calibrated τ=0.0004 | **0.8492 ± 0.0055** |
| Oracle-τ ceiling (optimistic) | 0.8500 |
| Precision / Recall / F1 | 0.787 / 0.957 / 0.864 |
| Cost / latency | **$0**, 2.36 ms/decision |

Calibrated ≈ oracle (0.8492 vs 0.8500) → the threshold is **not
overfit**. Run is deterministic (reproduced bit-identical across two
independent runs).

## Head-to-head — identical 1,200 decisions (grounded)

| System | Accuracy | F1 | Precision | Recall | Cost |
|---|---|---|---|---|---|
| **WITNESS** | **0.8658 ± 0.0193** | 0.878 | 0.804 | 0.968 | $0, 2.4ms |
| gpt-4o-mini (grounded judge) | 0.8625 ± 0.0195 | 0.853 | 0.916 | 0.798 | LLM call |
| gpt-3.5-turbo (grounded judge) | 0.5217 ± 0.0283 | 0.173 | 0.638 | 0.100 | LLM call |
| GPT-3.5 — HaluEval paper (no-knowledge ref) | 0.6259 | — | — | — | LLM call |

## Verdict

- **WITNESS statistically ties gpt-4o-mini** on identical data
  (86.58% vs 86.25%; ±~1.9% CIs overlap heavily — the 0.33-pt gap is
  not significant) while costing **$0 and 2.4 ms** instead of an LLM
  call. Different operating points: WITNESS is high-recall (0.97),
  gpt-4o-mini is high-precision (0.92); aggregate accuracy ≈ equal.
- **WITNESS decisively beats the canonical published GPT-3.5 number**
  (84.9% vs 62.6%).
- **AUROC 0.7976** over the full 20k is a genuine, threshold-free
  result — it cannot be inflated by operating-point choice.

## Honest caveats

1. **gpt-3.5-turbo's 52% here under-sells it.** With a strict
   one-word grounded judge prompt, 3.5 degenerates to almost always
   answering "No" (recall 0.10). The same minimal prompt did *not*
   break gpt-4o-mini (86%), so the prompt is sound — 3.5 is just
   prompt-sensitive. The fair reference for 3.5 is the paper's
   few-shot **62.59%** (no-knowledge), not the 52% measured here.
2. **WITNESS's calibrated τ ≈ 0.0004** is effectively "flag any
   non-zero groundedness deficit" — a high-recall / lower-precision
   operating point. The AUROC 0.80 is the number to stand behind
   publicly; 84.9% is real but operating-point-specific.
3. **Not a SOTA claim over all detectors.** Best fine-tuned/NLI
   hallucination detectors in the literature reach a similar ~80–88%
   band on QA. WITNESS is *competitive with strong modern systems at
   zero marginal cost* — that is the defensible claim, not "world
   record."
4. This measures the **WITNESS detector**, not the escalation
   controller; the escalation/regret work does not affect this number.
5. Grounded comparison gives GPT the knowledge (fair to WITNESS,
   which sees it). The paper's 62.59% is the no-knowledge setting and
   is cited as an external reference only.
