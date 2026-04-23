---
claim_id: 39850f8a-5ac0-41fd-826d-9144abf425a4
entity: test_integration
status: inferred
confidence: 0.75
sources:
  - entroly-core\tests\test_integration.py:661
  - entroly-core\tests\test_integration.py:15
  - entroly-core\tests\test_integration.py:34
  - entroly-core\tests\test_integration.py:40
  - entroly-core\tests\test_integration.py:49
  - entroly-core\tests\test_integration.py:59
  - entroly-core\tests\test_integration.py:72
  - entroly-core\tests\test_integration.py:78
  - entroly-core\tests\test_integration.py:90
  - entroly-core\tests\test_integration.py:96
last_checked: 2026-04-23T03:07:07.863374+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_integration

**Language:** python
**Lines of code:** 822

## Types
- `class DataPipeline()` — ETL pipeline for processing events.

## Functions
- `def test(name, fn)`
- `def test_default_constructor()`
- `def test_custom_params()`
- `def test_exploration_rate_clamp()`
- `def test_basic_ingest()`
- `def test_ingest_with_explicit_tokens()`
- `def test_token_estimation_code_vs_prose()`
- `def test_pinned_ingest()`
- `def test_empty_content()`
- `def test_large_content()`
- `def test_multiple_ingests()`
- `def test_criticality(path, expected_crit)`
- `def test_license_safety()`
- `def test_security_warning_safety()`
- `def test_normal_code_not_pinned()`
- `def test_entropy_honest_for_critical()` — Critical files should have honest entropy, not inflated.
- `def test_entropy_varies_with_content()`
- `def test_exact_duplicate()`
- `def test_different_content_not_duplicate()`
- `def test_near_duplicate()`
- `def test_dep_graph_auto_link()` — Ingest definition then usage — should create dep edge.
- `def test_dep_graph_import_detection()` — Python import statements should create strong dep edges.
- `def test_dep_graph_order_matters()` — Usage before definition: no edge. Then ingest definition: edge created on next usage.
- `def test_dep_graph_empty()`
- `def test_bug_tracing()`
- `def test_refactoring()`
- `def test_code_generation()`
- `def test_testing()`
- `def test_unknown_task()`
- `def test_optimize_empty()`
- `def test_optimize_selects_within_budget()`
- `def test_optimize_adaptive_budget()` — Bug tracing query surfaces a 1.5x recommended budget (advisory).
- `def test_optimize_pinned_always_included()`
- `def test_optimize_sufficiency()`
- `def test_optimize_returns_ordered()`
- `def test_feedback_success()`
- `def test_feedback_failure()`
- `def test_feedback_affects_ranking()` — Fragments with positive feedback should rank higher.
- `def test_feedback_empty_ids()`
- `def test_feedback_nonexistent_id()`
- `def test_explain_before_optimize()`
- `def test_explain_after_optimize()`
- `def test_recall_empty()`
- `def test_recall_returns_ranked()`
- `def test_advance_turn()`
- `def test_decay_evicts_stale()`
- `def test_pinned_survives_decay()`
- `def test_critical_file_survives_decay()`
- `def test_export_import_roundtrip()`
- `def test_import_invalid_json()`
- `def test_stats_structure()`
- `def test_shannon_entropy()`
- `def test_simhash()`
- `def test_hamming_distance()`
- `def test_normalized_entropy()`
- `def test_boilerplate_ratio()`
- `def test_unicode_content()`
- `def test_binary_like_content()`
- `def test_very_long_source_path()`
- `def test_optimize_zero_budget()`
- `def test_recall_zero_k()`
- `def __init__(self, config: dict)`
- `def ingest_event(self, event: dict) -> bool` — Ingest a single event into the buffer.
- `def flush(self) -> int` — Flush buffered events to storage.
- `def test_skeleton_populated_on_ingest_python()`
- `def test_skeleton_token_count_less_than_full()`
- `def test_no_skeleton_for_non_code()`
- `def test_skeleton_present_for_js()`
- `def test_optimize_uses_skeleton_when_budget_tight()` — With a very tight budget, skeleton variants should appear in results.
- `def test_optimize_prefers_full_when_budget_allows()` — With a large budget, all fragments should be 'full', not skeletons.
- `def test_optimize_variant_field_always_present()` — Every selected fragment should have a 'variant' field.

## Dependencies
- `axios`
- `entroly_core`
- `pathlib`
- `sys`
- `traceback`

## Linked Beliefs
- [[entroly_core]]
