# RELATE-X — Constraint-Certified Evidence Control

This remote branch is a research checkpoint, not production integration.

## Non-negotiable operating rules

- Do not fabricate results.
- Do not claim a benchmark was run unless it was executed.
- Do not claim a breakthrough from architecture alone.
- Do not claim Jev superiority without apples-to-apples reproducible evidence.
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

Lexical-only baseline: 85.7% false negative rate (12/14 unsafe omissions approved).
Information residual witness: 0.0% false negative rate (0/14 unsafe omissions approved).

Remaining gap: 40% false positive rate from lexical obligation matching on "summarize" tasks.
Remaining architectural gap: no joint omission safety check.

## Current status

This branch contains a research subset with a measured benchmark result. The information residual witness eliminates all false negatives on the frozen omission safety benchmark. It does not contain a breakthrough claim — the false positive rate and joint omission problem are open.
