---
claim_id: f07e6239-a8ee-40a8-bea3-7844eb3fe08d
entity: test_ios
status: inferred
confidence: 0.75
sources:
  - tests\test_ios.py:65
  - tests\test_ios.py:138
  - tests\test_ios.py:163
  - tests\test_ios.py:192
  - tests\test_ios.py:290
  - tests\test_ios.py:338
  - tests\test_ios.py:383
  - tests\test_ios.py:406
  - tests\test_ios.py:439
  - tests\test_ios.py:504
last_checked: 2026-04-23T03:07:07.936246+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_ios

**Language:** python
**Lines of code:** 805

## Types
- `class TestSDSDiversityPenalty()` — Verify that SDS penalizes redundant information.
- `class TestSDSBudgetRespect()` — Verify SDS always respects the token budget.
- `class TestSDSFeedbackIntegration()` — Verify SDS respects Wilson score feedback multipliers.
- `class TestMRKResolutionSelection()` — Verify MRK selects optimal resolution per fragment.
- `class TestMRKCoverageImprovement()` — Verify MRK covers more files than standard knapsack.
- `class TestECDBQueryFactor()` — Verify ECDB scales budget with query vagueness.
- `class TestECDBCodebaseFactor()` — Verify ECDB scales budget with codebase size.
- `class TestECDBBounds()` — Verify ECDB respects minimum and maximum budget bounds.
- `class TestContextBlockFormatting()` — Verify context block correctly formats multi-resolution output.
- `class TestIOSEndToEnd()` — Full pipeline tests with real engine + IOS.
- `class TestIOSPerformance()` — Verify IOS doesn't degrade performance significantly.
- `class TestMathProperties()` — Verify mathematical properties of the algorithms.
- `class TestEdgeCases()` — Edge cases and boundary conditions.
- `class TestIOSRegressionFixes()` — Regression coverage for IOS-specific production bugs.

## Functions
- `def make_engine(**kwargs)` — Create an EntrolyEngine with IOS enabled by default.
- `def ingest_fragment(engine, content, source="test.py", token_count=0, is_pinned=False)` — Helper to ingest and return the fragment ID.
- `def test_diverse_selection_over_redundant(self)` — Given duplicate + unique fragments, SDS should prefer diverse set.
- `def test_diversity_score_reported(self)` — IOS should report a diversity score.
- `def test_similar_fragments_low_diversity(self)` — Selecting similar (but not identical) fragments should yield low diversity score.
- `def test_all_unique_content_high_diversity(self)` — Completely different fragments should yield high diversity score.
- `def test_budget_never_exceeded(self, budget)` — Total selected tokens must never exceed budget.
- `def test_zero_budget(self)` — Zero budget should select nothing (or only pinned).
- `def test_boosted_fragment_preferred(self)` — Fragments with positive feedback should be preferred.
- `def test_full_resolution_with_generous_budget(self)` — With plenty of budget, fragments should be full resolution.
- `def test_skeleton_resolution_with_tight_budget(self)` — With tight budget, should use skeleton resolution for some fragments.
- `def test_reference_resolution_exists(self)` — Reference fragments should appear when budget is very tight.
- `def test_mrk_disabled_uses_full_only(self)` — With MRK disabled, all fragments should be full resolution.
- `def test_more_files_covered_with_mrk(self)` — MRK should cover more unique files than legacy knapsack.
- `def test_specific_query_small_budget(self)` — Specific queries (low vagueness) should get smaller budgets.
- `def test_vague_query_large_budget(self)` — Vague queries (high vagueness) should get larger budgets.
- `def test_medium_vagueness_near_static(self)` — Medium vagueness should produce budget near the static value.
- `def test_budget_monotonic_in_vagueness(self, vagueness)` — Budget should increase monotonically with vagueness.
- `def test_larger_codebase_larger_budget(self)` — Larger codebases should get larger budgets.
- `def test_codebase_factor_caps_at_2x(self)` — Codebase factor should cap at 2.0 (300+ fragments).
- `def test_minimum_budget(self)` — Budget should never drop below 500 tokens.
- `def test_maximum_budget(self)` — Budget should never exceed 30% of context window.
- `def test_model_aware_budget(self, model, window)` — Budget should scale with model's context window.
- `def test_full_fragments_in_code_fences(self)` — Full resolution fragments should appear in code fences.
- `def test_skeleton_fragments_grouped(self)` — Skeleton fragments should appear under 'Structural Outlines'.
- `def test_reference_fragments_listed(self)` — Reference fragments should appear under 'Also relevant'.
- `def test_empty_fragments_returns_empty(self)` — No fragments = empty string.
- `def test_resolution_ordering(self)` — Full fragments should appear before skeleton, skeleton before reference.
- `def test_pipeline_produces_valid_output(self)` — Full pipeline: ingest → optimize → format should produce valid context.
- `def test_ios_vs_legacy_both_valid(self)` — Both IOS and legacy paths should produce valid results.
- `def test_ios_enabled_flag_in_result(self)` — When IOS is enabled, result should include ios_enabled flag.
- `def test_ios_disabled_no_flag(self)` — When IOS is disabled, result should not include ios_enabled.
- `def test_1000_fragments_under_100ms(self)` — IOS should handle 1000 fragments in under 100ms.
- `def test_diversity_factor_bounds(self)` — Diversity factor should be in [0.1, 1.0].
- `def test_ecdb_sigmoid_shape(self)` — ECDB query factor should follow sigmoid shape.
- `def test_ecdb_query_factor_formula(self)` — Verify ECDB query factor matches the documented formula.
- `def test_single_fragment(self)` — Single fragment should always be selected at full resolution.
- `def test_all_pinned(self)` — All pinned fragments should always be included.
- `def test_budget_smaller_than_smallest(self)` — Budget smaller than any fragment shouldn't crash.
- `def test_empty_query(self)` — Empty query shouldn't crash IOS.
- `def test_unicode_content(self)` — Unicode content shouldn't crash IOS.
- `def test_negative_feedback_does_not_backfill_bad_reference(self)` — Strongly down-ranked fragments should not reappear as cheap references.
- `def test_ios_exploration_can_vary_selected_set(self)` — Exploration must still fire when IOS is enabled.
- `def test_exploration_does_not_seed_exploit_cache(self)` — Exploratory selections must not become exact-hit exploit cache entries.
- `def test_explain_selection_includes_compressed_variants(self)` — Explainability should stay in sync with reference/skeleton selections.

## Dependencies
- `entroly.proxy_config`
- `entroly.proxy_transform`
- `entroly_core`
- `math`
- `pytest`
- `unittest.mock`

## Linked Beliefs
- [[entroly_core]]

## Key Invariants
- test_budget_never_exceeded: Total selected tokens must never exceed budget.
- test_ios_exploration_can_vary_selected_set: Exploration must still fire when IOS is enabled.
- test_exploration_does_not_seed_exploit_cache: Exploratory selections must not become exact-hit exploit cache entries.
