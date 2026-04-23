---
claim_id: b3a96640-7638-4c9a-a220-d93583278e5e
entity: test_apa
status: inferred
confidence: 0.75
sources:
  - tests\test_apa.py:26
  - tests\test_apa.py:86
  - tests\test_apa.py:146
  - tests\test_apa.py:204
  - tests\test_apa.py:299
  - tests\test_apa.py:29
  - tests\test_apa.py:32
  - tests\test_apa.py:41
  - tests\test_apa.py:49
  - tests\test_apa.py:56
last_checked: 2026-04-23T03:07:07.921128+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_apa

**Language:** python
**Lines of code:** 311

## Types
- `class TestCalibratedTokenCount()` — Per-language char/token ratio estimation.
- `class TestBuildPreamble()` — Task-aware preamble generation.
- `class TestDeduplicateFragments()` — Content-hash deduplication.
- `class TestFormatContextBlockAPA()` — Integration: preamble + dedup in context block.
- `class TestTokenBudgetAccuracy()` — Verify calibrated estimation is more accurate than len/4.

## Functions
- `def test_empty_content_returns_zero(self)`
- `def test_python_denser_than_default(self)`
- `def test_rust_less_dense_than_python(self)`
- `def test_json_densest(self)`
- `def test_unknown_extension_uses_default(self)`
- `def test_never_returns_zero_for_nonempty(self)`
- `def test_no_source_uses_default(self)`
- `def test_all_languages_return_positive(self, ext, lang)`
- `def test_no_signals_returns_empty(self)` — Unknown task, low vagueness, no security → no preamble.
- `def test_security_issues_trigger_warning(self)`
- `def test_single_security_issue_singular(self)`
- `def test_high_vagueness_triggers_clarification(self)`
- `def test_low_vagueness_no_clarification(self)`
- `def test_bugtracing_hint(self)`
- `def test_refactoring_hint(self)`
- `def test_testing_hint(self)`
- `def test_codereview_hint(self)`
- `def test_codegeneration_no_hint(self)` — CodeGeneration doesn't have a specific hint — no preamble.
- `def test_combined_signals(self)` — Security + vagueness + task should all appear.
- `def test_no_duplicates_keeps_all(self)`
- `def test_exact_duplicates_removed(self)`
- `def test_keeps_first_occurrence(self)`
- `def test_near_duplicates_kept(self)` — Fragments with same prefix but different suffix are kept.
- `def test_empty_list(self)`
- `def test_uses_preview_field(self)` — Should use preview field when content is not available.
- `def test_preamble_appears_for_bugtracing(self)`
- `def test_no_preamble_for_unknown_low_vagueness(self)`
- `def test_security_preamble_with_issues(self)`
- `def test_dedup_removes_duplicate_fragments(self)`
- `def test_backward_compatible_without_kwargs(self)` — Existing callers without task_type/vagueness should still work.
- `def test_vagueness_threshold(self)` — Vagueness at exactly 0.6 should NOT trigger (> 0.6 needed).
- `def test_empty_fragments_and_memories_returns_empty(self)`
- `def test_preamble_before_fragments(self)` — Preamble should appear before code fragments.
- `def test_reasonable_estimates(self, code, source, expected_range)`

## Dependencies
- `entroly.proxy_transform`
- `pytest`
