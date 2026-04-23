---
claim_id: 3d3ed66a-8cbf-4ab7-8f14-4ddb2e75f06c
entity: accuracy
status: inferred
confidence: 0.75
sources:
  - bench\accuracy.py:51
  - bench\accuracy.py:66
  - bench\accuracy.py:243
  - bench\accuracy.py:445
  - bench\accuracy.py:560
last_checked: 2026-04-23T03:07:07.778548+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: accuracy

**Language:** python
**Lines of code:** 655

## Types
- `class BenchmarkResult()` — Result of a single benchmark run.
- `class RetentionReport()` — Accuracy retention: Entroly vs baseline.

## Functions
- `def bench_needle(model: str, samples: int = 20) -> list[dict]` — NeedleInAHaystack: can the LLM find a fact in compressed context?
- `def run_benchmark(
    benchmark: str,
    model: str = "gpt-4o-mini",
    samples: int = 50,
    budget: int = 50_000,
) -> RetentionReport`
- `def main()`

## Dependencies
- `__future__`
- `dataclasses`
- `json`
- `os`
- `pathlib`
- `random`
- `re`
- `sys`
- `time`
- `typing`
