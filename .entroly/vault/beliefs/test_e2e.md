---
claim_id: bb054fbc-1de6-4309-b718-fceffe12c061
entity: test_e2e
status: inferred
confidence: 0.75
sources:
  - tests\test_e2e.py:48
  - tests\test_e2e.py:72
  - tests\test_e2e.py:88
  - tests\test_e2e.py:106
  - tests\test_e2e.py:112
  - tests\test_e2e.py:120
  - tests\test_e2e.py:134
  - tests\test_e2e.py:146
  - tests\test_e2e.py:155
  - tests\test_e2e.py:164
last_checked: 2026-04-23T03:07:07.929262+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_e2e

**Language:** python
**Lines of code:** 643


## Functions
- `def test_knapsack_selects_optimal_subset()` — Verify that the DP solver picks the highest-value subset.
- `def test_knapsack_respects_pinned()` — Pinned fragments must always be included.
- `def test_ebbinghaus_decay()` — Verify exponential decay math matches the Ebbinghaus curve.
- `def test_entropy_all_same_chars()` — A string of identical characters has zero entropy.
- `def test_entropy_increases_with_diversity()` — More character diversity → higher entropy.
- `def test_boilerplate_detection()` — Import-heavy code should have high boilerplate ratio.
- `def test_cross_fragment_redundancy()` — Identical fragments should have redundancy = 1.0.
- `def test_simhash_identical_texts()` — Identical texts must produce identical fingerprints.
- `def test_simhash_similar_texts_close()` — Similar texts should have small Hamming distance.
- `def test_dedup_index_catches_duplicates()` — DedupIndex should detect near-identical content.
- `def test_dedup_index_allows_different()` — DedupIndex should NOT flag very different content as duplicates.
- `def test_import_extraction()` — Extract Python imports from source code.
- `def test_test_file_inference()` — Infer test file paths from source file paths.
- `def test_co_access_learning()` — Pre-fetcher should learn co-access patterns.
- `def test_checkpoint_save_and_load()` — Checkpoint should roundtrip fragments correctly.
- `def test_full_engine_pipeline()` — End-to-end test of the Entroly engine.
- `def test_positive_feedback_raises_fragment_value()` — After repeated positive feedback on a fragment, recall_relevant must rank it higher relative to a fragment that received no feedback. Proves the PRISM RL weight update actually changes scoring.
- `def test_negative_feedback_suppresses_fragment()` — After repeated negative feedback on a fragment, its Wilson lower-bound multiplier should drop below 1.0, reducing its effective relevance score.
- `def test_recall_correct_after_eviction()` — After advance_turn() applies Ebbinghaus decay and rebuilds the LSH index, pinned live fragments must still be found correctly. This tests that rebuild_lsh_index() correctly re-slots remaining fragment
- `def test_fragment_guard_flags_secrets()` — FragmentGuard must detect hardcoded secrets in ingested fragments. The 'quality_issues' key in the ingest response must be non-empty.
- `def test_fragment_guard_passes_clean_code()` — FragmentGuard must not flag clean Rust code.
- `def test_provenance_hallucination_risk()` — ContextProvenance must produce a hallucination_risk between 0 and 1, and verified_fraction must reflect the fraction of fragments with known sources. Build provenance directly from a real optimize_con
- `def test_export_import_preserves_prism_covariance()` — After export_state/import_state roundtrip, the PRISM optimizer covariance must survive (not reset to identity), proving the learned RL state persists across restarts.
- `def test_token_budget_zero_uses_default()` — Calling optimize_context with token_budget=0 must not crash and must use the server's configured default budget.

## Dependencies
- `entroly.checkpoint`
- `entroly.config`
- `entroly.prefetch`
- `entroly.server`
- `entroly_core`
- `json`
- `math`
- `os`
- `pandas`
- `pathlib`
- `sys`
- `tempfile`
- `utils.helpers`

## Linked Beliefs
- [[entroly_core]]
- [[checkpoint]]
- [[config]]

## Key Invariants
- test_knapsack_respects_pinned: Pinned fragments must always be included.
- test_simhash_identical_texts: Identical texts must produce identical fingerprints.
- test_positive_feedback_raises_fragment_value: After repeated positive feedback on a fragment, recall_relevant must rank it higher relative to a fr
- test_recall_correct_after_eviction: After advance_turn() applies Ebbinghaus decay and rebuilds the LSH index, pinned live fragments must
- test_fragment_guard_flags_secrets: FragmentGuard must detect hardcoded secrets in ingested fragments. The 'quality_issues' key in the i
- test_fragment_guard_passes_clean_code: FragmentGuard must not flag clean Rust code.
- test_provenance_hallucination_risk: ContextProvenance must produce a hallucination_risk between 0 and 1, and verified_fraction must refl
- test_export_import_preserves_prism_covariance: After export_state/import_state roundtrip, the PRISM optimizer covariance must survive (not reset to
- test_token_budget_zero_uses_default: Calling optimize_context with token_budget=0 must not crash and must use the server's configured def
