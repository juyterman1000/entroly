---
claim_id: 717c5529-9af0-473a-be44-0640e24402db
entity: test_intensive_functional
status: inferred
confidence: 0.75
sources:
  - tests\test_intensive_functional.py:52
  - tests\test_intensive_functional.py:64
  - tests\test_intensive_functional.py:70
  - tests\test_intensive_functional.py:78
  - tests\test_intensive_functional.py:87
  - tests\test_intensive_functional.py:122
  - tests\test_intensive_functional.py:140
  - tests\test_intensive_functional.py:172
  - tests\test_intensive_functional.py:204
  - tests\test_intensive_functional.py:255
last_checked: 2026-04-23T03:07:07.935117+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_intensive_functional

**Language:** python
**Lines of code:** 831


## Functions
- `def check(label: str, condition: bool, detail: str = "") -> bool`
- `def skip(label: str, reason: str = "")`
- `def section(code: str, title: str)`
- `def real_sources() -> list[tuple[str, Path]]`
- `def fresh_engine(tmp=None, **cfg_overrides)`
- `def test_token_count_accuracy()`
- `def test_feedback_idempotency()`
- `def test_relevance_ordering()`
- `def test_checkpoint_file_format()`
- `def test_resume_full_state()`
- `def test_multi_checkpoint_cycle()`
- `def test_budget_utilization_math()`
- `def test_sufficiency_contract()`
- `def test_provenance_chain()`
- `def test_entropy_signal()`
- `def test_advance_turn_decay()`
- `def recency_score()`
- `def test_stats_after_eviction()`
- `def frag_count()`
- `def test_record_success_monotone()`
- `def test_dedup_tokens_saved()`
- `def test_optimize_fields_contract()`
- `def test_recall_top_k_exact()`
- `def test_recall_scores_ordered()`
- `def test_stats_after_feedback()`
- `def test_config_propagates()`
- `def test_empty_record_success()`
- `def test_unknown_fragment_id()`
- `def test_large_corpus_performance()`
- `def run()`

## Dependencies
- `gzip`
- `json`
- `os`
- `pathlib`
- `sys`
- `tempfile`
- `time`
