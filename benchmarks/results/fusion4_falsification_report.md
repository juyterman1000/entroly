# Fusion-4 "0.90 AUROC" — Caught and Shelved (Process Win)

**Status: REJECTED before shipping. README stays at WITNESS AUROC 0.798.**

This documents a benchmark-gaming-via-dataset-artifact that the
falsification-first discipline caught *before* it reached any
user-facing surface. Recording it as a methodology success.

## The claim

A grid-searched 4-signal fusion (weights W=.05 E=.05 **G=.80** S=.10)
scored **AUROC 0.9003** on HaluEval-QA (full 10K, 16K test decisions)
vs WITNESS-only 0.798 — a +0.10 "breakthrough" — defended as
not-overfitting because cal 0.898 → test 0.900 on disjoint splits.

## Why the defense was invalid

A cal/test split *inside HaluEval* cannot detect exploitation of how
HaluEval is built. Both splits share the construction, so an artifact
signal generalizes perfectly across them; "test ≥ cal" proves nothing.

## The falsification (pre-registered, frozen weights, n=400)

GPT-4o-mini broke the construction two ways, then the SAME frozen
fusion was re-measured:

| Condition | Fusion AUROC |
|---|---|
| C1 original (reproduces the claim) | 0.894 |
| C2 entity-controlled positives | 0.850 |
| C3 faithful-paraphrased vs hallucinated | 0.572 |
| C4 realistic (both broken) | 0.453 |

**Mechanism table — deterministic, label-noise-free, decisive:**

| Answer type | Mean entity-gap G |
|---|---|
| faithful `r` (HaluEval correct) | **0.000** |
| hallucinated `h` (entity-swapped) | 0.188 |
| entity-controlled `h_ctrl` | 0.067 |
| paraphrased-but-correct `r_para` | **0.137** |

`G(faithful) = exactly 0.000` over all 400 items — no real signal is
exactly constant on a class unless it measures how that class was
constructed. HaluEval's `right_answer` is extracted from the knowledge;
the signal detects *extraction vs fabrication*, not faithfulness. A
merely paraphrased **correct** answer (G=0.137) looks almost identical
to a real hallucination (G=0.188).

## Verdict & decision

- The 0.90 is a **HaluEval entity-swap synthesis artifact**. It does
  not generalize; in the realistic condition it is at/below chance.
- **README unchanged**: WITNESS AUROC **0.798** remains the only
  user-facing hallucination-accuracy number. Fusion-4 / 0.90 is not
  added anywhere user-facing.
- Entity-grounding remains a *legitimate idea* but is not defensible
  until it (a) survives paraphrase and (b) holds on a dataset whose
  negatives are not entity-swapped (e.g. RAGTruth). Shelved, on record.

## Why this is the headline success

We shipped **honesty**: a +0.10 "win" was killed by our own harness
before a single user saw it. The threshold-free, falsification-tested
WITNESS 0.798 is worth more than an artifact-inflated 0.90 — because it
is true. Reproduce: `python benchmarks/fusion4_falsification.py`.
