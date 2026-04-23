---
claim_id: 1cc3e866-9879-418c-abe7-54099ba60184
entity: test_federation
status: inferred
confidence: 0.75
sources:
  - tests\test_federation.py:111
  - tests\test_federation.py:173
  - tests\test_federation.py:210
  - tests\test_federation.py:263
  - tests\test_federation.py:314
  - tests\test_federation.py:345
  - tests\test_federation.py:387
  - tests\test_federation.py:407
  - tests\test_federation.py:440
  - tests\test_federation.py:463
last_checked: 2026-04-23T03:07:07.932993+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_federation

**Language:** python
**Lines of code:** 1038

## Types
- `class TestGaussianMechanism()`
- `class TestTrimmedMean()`
- `class TestContributionEligibility()`
- `class TestFileExchange()` — Export → import should preserve packet structure.
- `class TestAggregation()` — Multiple identical contributions should converge.
- `class TestMergePolicy()` — Merge should call update_weights on the optimizer.
- `class TestContributionPacket()`
- `class TestPrivacy()` — With DP noise, individual contributions should be statistically indistinguishable from the aggregate. We inject one outlier among many honest values and verify the aggregate doesn't leak the outlier's
- `class TestStats()`
- `class TestContributeViaOptimizer()`
- `class TestFallbacks()` — NaN weights must be rejected — no silent corruption.
- `class TestIntegrationReal()` — These tests use the actual ArchetypeOptimizer class, not mocks. Proves the federation module works end-to-end with the real system.
- `class TestPrivacyAccountant()` — Upgrade 3: Cumulative privacy budget tracking.
- `class TestTTLAndDedup()` — Upgrade 2: Contribution expiry and per-client dedup.
- `class TestFedProx()` — Upgrade 4: FedProx proximal regularization.
- `class TestMergeOutcomeTracking()` — Upgrade 6: Convergence tracking.
- `class TestGitHubTransport()` — Zero-cost global federation via GitHub Issues API.

## Functions
- `def tmp_data_dir(tmp_path)` — Create a temporary data directory for federation.
- `def sample_weights()`
- `def mock_archetype_optimizer(sample_weights)` — Create a mock ArchetypeOptimizer for testing.
- `def test_noise_sigma_positive(self)`
- `def test_noise_sigma_scales_with_epsilon(self)`
- `def test_noise_sigma_decreases_with_participants(self)`
- `def test_noise_preserves_all_keys(self)`
- `def test_noise_clips_to_valid_range(self)`
- `def test_noise_is_non_zero(self)` — Verify noise is actually added (not identity).
- `def test_noise_has_minimum_floor(self)` — Even with huge participant count, noise σ ≥ 0.01.
- `def test_basic_average(self)`
- `def test_empty_list(self)`
- `def test_single_value(self)`
- `def test_two_values(self)`
- `def test_outlier_resilience(self)` — Byzantine values at extremes should be trimmed.
- `def test_symmetric_trim(self)`
- `def test_disabled_returns_none(self, tmp_data_dir)`
- `def test_low_confidence_returns_none(self, tmp_data_dir)`
- `def test_low_samples_returns_none(self, tmp_data_dir)`
- `def test_eligible_returns_packet(self, tmp_data_dir, sample_weights)`
- `def test_contribution_weights_are_noised(self, tmp_data_dir, sample_weights)` — Noised weights should differ from originals.
- `def test_export_import_roundtrip(self, tmp_data_dir, sample_weights)` — Export → import should preserve packet structure.
- `def test_import_invalid_file_returns_none(self, tmp_data_dir)`
- `def test_save_and_load_contributions(self, tmp_data_dir)`
- `def test_anti_echo_filters_own(self, tmp_data_dir)` — Client should not load its own contributions.
- `def test_aggregate_converges_to_mean(self)` — Multiple identical contributions should converge.
- `def test_aggregate_handles_diversity(self)` — Different weight values should produce a reasonable mean.
- `def test_aggregate_empty_returns_empty(self)`
- `def test_merge_with_optimizer(self, tmp_data_dir, mock_archetype_optimizer)` — Merge should call update_weights on the optimizer.
- `def test_merge_disabled_returns_false(self, tmp_data_dir, mock_archetype_optimizer)`
- `def test_merge_no_global_returns_false(self, tmp_data_dir, mock_archetype_optimizer)` — When no contributions exist, merge should return False.
- `def test_merge_insufficient_contributors(self, tmp_data_dir, mock_archetype_optimizer)` — Below MIN_CONTRIBUTORS, merge should not apply.
- `def test_to_dict_roundtrip(self)`
- `def test_json_serializable(self)`
- `def test_individual_unrecoverable_from_aggregate(self)` — With DP noise, individual contributions should be statistically indistinguishable from the aggregate. We inject one outlier among many honest values and verify the aggregate doesn't leak the outlier's
- `def test_client_id_is_hashed(self, tmp_data_dir)` — Client ID should be a hash, not the raw hostname.
- `def test_stats_structure(self, tmp_data_dir)`
- `def test_enable_disable(self, tmp_data_dir)`
- `def test_contribute_happy_path(self, tmp_data_dir, mock_archetype_optimizer)`
- `def test_contribute_no_archetype(self, tmp_data_dir)`
- `def test_contribute_disabled(self, tmp_data_dir, mock_archetype_optimizer)`
- `def test_validate_rejects_nan(self)` — NaN weights must be rejected — no silent corruption.
- `def test_validate_rejects_inf(self)` — Inf weights must be rejected.
- `def test_validate_rejects_degenerate_zeros(self)` — All-zero PRISM weights = degenerate, must reject.
- `def test_validate_rejects_negative_weight(self)` — Negative PRISM weights must be rejected.
- `def test_validate_rejects_too_few_keys(self)` — Truncated packets with < 3 keys must be rejected.
- `def test_validate_accepts_good_weights(self)` — Valid weights should pass validation.
- `def test_alpha_capped_at_70_percent(self, tmp_data_dir)` — Local weights must retain ≥30% influence even with huge global pool.
- `def test_rollback_on_apply_failure(self, tmp_data_dir)` — If update_weights raises, rollback to pre-merge snapshot.
- `def test_real_contribute_roundtrip(self, tmp_path)` — Real ArchetypeOptimizer → contribute → packet saved to disk.
- `def test_real_merge_changes_weights(self, tmp_path)` — Real merge: global weights should actually change the optimizer's state.
- `def test_real_merge_persists_to_disk(self, tmp_path)` — After merge, loading a fresh optimizer should see the merged weights.
- `def test_fresh_accountant_can_contribute(self)`
- `def test_budget_grows_sublinearly(self)` — ε_total grows as √k, not k — advanced composition.
- `def test_budget_exhaustion_blocks_contributions(self, tmp_data_dir)` — Once budget is exhausted, prepare_contribution returns None.
- `def test_remaining_budget_decreases(self)`
- `def test_expired_contributions_removed(self, tmp_data_dir)` — Contributions older than 30 days should be garbage-collected.
- `def test_per_client_dedup_keeps_latest(self, tmp_data_dir)` — Multiple contributions from same client: keep only the latest.
- `def test_staleness_decay_reduces_confidence(self, tmp_data_dir)` — Older contributions should have reduced confidence via decay.
- `def test_fedprox_pulls_toward_local(self, tmp_data_dir)` — Merged weights should be closer to local than pure linear blend.
- `def test_merge_records_outcome(self, tmp_data_dir, mock_archetype_optimizer)` — Merge should append a MergeOutcome.
- `def test_outcome_saved_to_disk(self, tmp_data_dir, mock_archetype_optimizer)` — Outcomes should be persisted to JSONL file.
- `def test_stats_includes_v2_fields(self, tmp_data_dir)` — Stats should report privacy budget, FedProx μ, TTL, etc.
- `def test_init_without_token(self)` — Transport should be read-only without a token.
- `def test_init_with_token(self)` — Transport should have write access with a token.
- `def test_push_without_token_returns_false(self)` — Push should fail gracefully without a token.
- `def test_pull_without_issue_returns_empty(self)` — Pull should return empty list when no Issue is found.
- `def test_sync_to_local_saves_packets(self, tmp_data_dir)` — sync_to_local should save pulled packets to the client's local dir.
- `def test_sync_skips_own_contributions(self, tmp_data_dir)` — sync_to_local should skip packets from the same client.

## Dependencies
- `entroly.federation`
- `json`
- `math`
- `os`
- `pathlib`
- `pytest`
- `statistics`
- `tempfile`
- `time`
- `unittest.mock`

## Key Invariants
- TestFallbacks: NaN weights must be rejected — no silent corruption.
- test_validate_rejects_nan: NaN weights must be rejected — no silent corruption.
- test_validate_rejects_inf: Inf weights must be rejected.
- test_validate_rejects_degenerate_zeros: All-zero PRISM weights = degenerate, must reject.
- test_validate_rejects_negative_weight: Negative PRISM weights must be rejected.
- test_validate_rejects_too_few_keys: Truncated packets with < 3 keys must be rejected.
- test_alpha_capped_at_70_percent: Local weights must retain ≥30% influence even with huge global pool.
