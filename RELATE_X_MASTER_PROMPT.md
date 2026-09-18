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

## Current status

This branch contains a reconstructed research subset and unit/invariant tests. It does not contain a benchmark win or breakthrough claim.
