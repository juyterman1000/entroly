---
claim_id: 0f77d78c-3fc7-4771-818a-cc3b0443d509
entity: test_functional
status: inferred
confidence: 0.75
sources:
  - tests\test_functional.py:50
  - tests\test_functional.py:63
  - tests\test_functional.py:69
  - tests\test_functional.py:81
  - tests\test_functional.py:86
  - tests\test_functional.py:92
  - tests\test_functional.py:100
  - tests\test_functional.py:110
  - tests\test_functional.py:146
  - tests\test_functional.py:175
last_checked: 2026-04-23T03:07:07.934131+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_functional

**Language:** python
**Lines of code:** 824


## Functions
- `def check(label: str, condition: bool, detail: str = "") -> bool`
- `def skip(label: str, reason: str = "")`
- `def section(name: str)`
- `def opt_selected(opt: dict) -> list[dict]` — Extract the list of selected fragments from an optimize result.
- `def opt_total_tokens(opt: dict) -> int` — Extract total_tokens from an optimize result.
- `def opt_effective_budget(opt: dict, fallback: int) -> int` — Extract effective budget from an optimize result.
- `def real_sources() -> list[tuple[str, Path]]` — Return all real .py and .rs files from the project.
- `def fresh_engine(tmp_dir: str | None = None, **cfg_kwargs)` — Create a EntrolyEngine backed by a private temp checkpoint dir.
- `def test_cold_start()`
- `def test_single_file()`
- `def test_ingest_contract()`
- `def test_dedup_boundary()`
- `def test_feedback_loop()`
- `def score_of(fid: str) -> float | None`
- `def test_multi_turn_lifecycle()`
- `def test_query_sensitivity()`
- `def test_score_stability()`
- `def test_advance_turn()`
- `def get_turn()`
- `def test_stats_contract()`
- `def test_explain_selection()`
- `def test_checkpoint_crash_sim()`
- `def test_large_budget()`
- `def test_tiny_budget()`
- `def test_recall_vs_optimize_universe()`
- `def test_mixed_feedback()`
- `def test_prefetch_prediction()`
- `def test_zero_query()`
- `def run()`

## Dependencies
- `os`
- `pathlib`
- `sys`
- `tempfile`
- `time`
