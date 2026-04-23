---
claim_id: e91535ae-74cb-4442-bffb-3ff753aaf3e5
entity: test_zero_token_autonomy
status: inferred
confidence: 0.75
sources:
  - tests\test_zero_token_autonomy.py:24
  - tests\test_zero_token_autonomy.py:97
  - tests\test_zero_token_autonomy.py:195
  - tests\test_zero_token_autonomy.py:249
  - tests\test_zero_token_autonomy.py:301
  - tests\test_zero_token_autonomy.py:334
  - tests\test_zero_token_autonomy.py:415
  - tests\test_zero_token_autonomy.py:472
  - tests\test_zero_token_autonomy.py:510
  - tests\test_zero_token_autonomy.py:567
last_checked: 2026-04-23T03:07:07.944322+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_zero_token_autonomy

**Language:** python
**Lines of code:** 617

## Types
- `class TestEvolutionBudget()` — Test the 'Tax on Savings' guardrail in ValueTracker.
- `class TestStructuralSynthesizer()` — Test zero-token skill synthesis from code structure.
- `class TestDreamingLoop()` — Test the autonomous idle-time self-play optimization.
- `class TestEvolutionDaemon()` — Test the daemon that orchestrates all 3 pillars.
- `class TestEvolutionLoggerEnhancements()` — Test source file tracking in MissRecord.
- `class TestComponentFeedbackBus()` — Test the gradient-free self-improvement bus.
- `class TestEpistemicRouterSelfTuning()` — Test that the router adaptively tunes its thresholds.
- `class TestPrefetchSelfImprovement()` — Test adaptive co_access_window tuning.
- `class TestCacheAligner()` — Test KV-cache prefix alignment for free token savings.
- `class TestFlowOrchestratorFeedback()` — Test that flow outcomes feed back to the router for self-tuning.

## Functions
- `def test_fresh_tracker_has_zero_budget(self)`
- `def test_budget_grows_with_savings(self)`
- `def test_spend_rejected_when_over_budget(self)`
- `def test_spend_accepted_within_budget(self)`
- `def test_budget_invariant_strictly_token_negative(self)` — C_spent(t) ≤ τ · S(t) must hold at all times.
- `def test_returns_none_for_no_source_files(self)`
- `def test_extracts_function_signatures(self)`
- `def test_detects_classes(self)`
- `def test_detects_imports(self)`
- `def test_entropy_closure_ranks_by_information(self)`
- `def test_synthesize_produces_valid_tool(self)`
- `def test_should_not_dream_when_active(self)`
- `def test_should_dream_after_idle(self)`
- `def test_generates_synthetic_queries(self)`
- `def test_generates_failure_replays(self)`
- `def test_stats_reports_idle_time(self)`
- `def test_run_once_with_no_gaps(self)`
- `def test_structural_synthesis_preferred_over_llm(self)`
- `def test_budget_rejection_when_no_savings(self)` — Daemon should reject LLM synthesis when budget is $0.
- `def test_stats_include_all_pillars(self)`
- `def test_miss_record_includes_source_files(self)`
- `def test_log_and_get_trend(self)`
- `def test_empty_trend_returns_zeros(self)`
- `def test_suggest_adjustment_increases_when_improving(self)`
- `def test_suggest_adjustment_decreases_when_degrading(self)`
- `def test_suggest_no_change_with_insufficient_data(self)`
- `def test_stats_reports_all_components(self)`
- `def test_persistence_to_disk(self)`
- `def test_record_outcome_tracks_history(self)`
- `def test_high_success_fast_answer_lowers_confidence(self)`
- `def test_low_success_fast_answer_raises_confidence(self)`
- `def test_low_success_compile_shrinks_freshness(self)`
- `def test_self_tune_respects_bounds(self)`
- `def test_hit_rate_tracking(self)`
- `def test_stats_include_hit_rate(self)`
- `def test_component_bus_wiring(self)`
- `def test_identical_context_hits_cache(self)`
- `def test_different_context_misses(self)`
- `def test_similar_context_hits(self)`
- `def test_stats_tracks_hits_and_misses(self)`
- `def test_invalidate_clears_client_cache(self)`
- `def test_execute_records_outcome_to_router(self)`
- `def test_component_bus_receives_feedback(self)`

## Dependencies
- `math`
- `os`
- `pathlib`
- `pytest`
- `tempfile`
- `time`

## Key Invariants
- test_budget_invariant_strictly_token_negative: C_spent(t) ≤ τ · S(t) must hold at all times.
