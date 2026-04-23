---
claim_id: d694e98a-df7f-4e8a-941b-05ef0db8d9e8
entity: test_real_user
status: inferred
confidence: 0.75
sources:
  - tests\test_real_user.py:44
  - tests\test_real_user.py:54
  - tests\test_real_user.py:60
  - tests\test_real_user.py:72
  - tests\test_real_user.py:88
  - tests\test_real_user.py:99
  - tests\test_real_user.py:6
  - tests\test_real_user.py:36
  - tests\test_real_user.py:37
  - tests\test_real_user.py:39
last_checked: 2026-04-23T03:07:07.941701+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_real_user

**Language:** python
**Lines of code:** 418


## Functions
- `def check(label: str, condition: bool, detail: str = "")`
- `def invariant(name: str)`
- `def collect_sources() -> list[tuple[str, Path]]`
- `def ingest_all(engine, sources) -> dict[str, str]` — Ingest all real source files. Returns label→fragment_id.
- `def scores_from_explain(engine) -> dict[str, float]` — Extract fragment_id→score from explain_selection().
- `def run()`

## Dependencies
- `os`
- `pathlib`
- `sys`
