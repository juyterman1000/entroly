---
claim_id: 87a6bc53-8f9b-4ad6-8aa5-021a5f1d26cc
entity: test_production_e2e
status: inferred
confidence: 0.75
sources:
  - tests\test_production_e2e.py:57
  - tests\test_production_e2e.py:69
  - tests\test_production_e2e.py:73
  - tests\test_production_e2e.py:391
  - tests\test_production_e2e.py:37
  - tests\test_production_e2e.py:51
  - tests\test_production_e2e.py:52
  - tests\test_production_e2e.py:53
  - tests\test_production_e2e.py:54
  - tests\test_production_e2e.py:82
last_checked: 2026-04-23T03:07:07.939323+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_production_e2e

**Language:** python
**Lines of code:** 556


## Functions
- `def ok(label: str, cond: bool, detail: str = "") -> bool`
- `def section(name: str)`
- `def fresh(max_frags: int = 10_000) -> EntrolyEngine`
- `def worker(thread_id: int)`

## Dependencies
- `gc`
- `json`
- `os`
- `pathlib`
- `sys`
- `tempfile`
- `threading`
- `time`
