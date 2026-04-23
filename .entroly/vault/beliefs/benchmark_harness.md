---
claim_id: ab3bfa2d-f8ff-4784-809c-c47cb8ec37bf
entity: benchmark_harness
status: inferred
confidence: 0.75
sources:
  - entroly\benchmark_harness.py:43
  - entroly\benchmark_harness.py:26
  - entroly\benchmark_harness.py:39
  - entroly\benchmark_harness.py:40
last_checked: 2026-04-23T03:07:07.792191+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: benchmark_harness

**Language:** python
**Lines of code:** 120


## Functions
- `def run_benchmark(engine: Any, budget_seconds: float = 10.0) -> dict[str, Any]` — Run the fixed evaluation payload and return the context_efficiency score. READ ONLY — this function is the ground truth metric. autotune.py calls this but never modifies it. The engine and benchmark c

## Dependencies
- `__future__`
- `gc`
- `time`
- `typing`
