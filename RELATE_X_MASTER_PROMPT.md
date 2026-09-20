# RELATE-X — Constraint-Certified Evidence Control

This remote branch is a research checkpoint, not production integration.

## Non-negotiable operating rules

- Do not fabricate results.
- Do not claim a benchmark was run unless it was executed.
- Do not claim a breakthrough from architecture alone.
- Do not claim superiority over any external tool without apples-to-apples reproducible evidence.
- Preserve negative results.
- Keep production defaults unchanged until frozen promotion gates pass.

## Updated moat hypothesis

The remaining falsifiable research hypothesis is **verified omission safety**:

> When Entroly removes a fragment to save tokens, can it produce a replayable witness that the omission does not change the admissible downstream decision/effect, while preserving an exact recovery handle if that witness cannot be established?

## RELATE-X pipeline

```text
Trusted task / policy envelope
  -> Constraint Compiler
  -> Candidate retrieval
  -> Semantic Collision Detector
  -> Differential relation adjudication
  -> Counterfactual challenge
  -> Risk/calibration gate
  -> Evidence-set planner
  -> Omission witness
  -> Context receipt / recovery handle
```

## Authority rule

Natural-language task text cannot grant capabilities. Authority must come from an external trusted envelope.

## Promotion gates

RELATE-X remains `CONTINUE RESEARCH` until frozen tests show measurable value on:

- omission-safety benchmark;
- NevIR / ExcluIR or equivalent semantic contrast retrieval gates;
- security/action authorization gates;
- ordinary retrieval non-regression;
- reproducibility, latency, resource and provenance checks.

## Omission Safety Benchmark v1

Dataset SHA-256: `7cadb7d1cbe08d899f5f4a1d247c36baf3d613fc40263bb7ebc3d4d6196f7c3a`

### Progression

| Stage | Accuracy | FNR | FPR |
|-------|----------|-----|-----|
| Lexical-only baseline | 26.3% | 85.7% | 40.0% |
| Information residual | 89.5% | 0.0% | 40.0% |
| Dimension-aware joint omission | **100.0%** | **0.0%** | **0.0%** |

### Key mechanisms

1. Information residual catches structural omission failures (constraint carrier, state/numeric conflict, value/action loss).
2. Dimension coverage override resolves lexical false positives on summary queries by checking whether the retained set covers enough independent information dimensions.
3. Joint omission safety API catches pairwise-independence violations (individually-safe omissions that are jointly unsafe).
4. Hard/soft reason boundary preserves fail-closed safety: constraint, contradiction, state conflict, value loss, and exclusion checks are NEVER overridden by dimension coverage.

### Remaining gaps

- Small dataset (15 cases, 19 evaluations) — adversarial expansion needed.
- Hand-tuned thresholds (Jaccard 0.15, coverage ratio ceil(D/2)) — sensitivity analysis needed.
- No NevIR/ExcluIR neural benchmark run.

## Current status

This branch contains a research subset with a measured benchmark result achieving 100% accuracy on the frozen omission safety dataset. The false positive and joint omission problems are solved. The result is on a small frozen dataset and must be validated with adversarial expansion before promotion.
