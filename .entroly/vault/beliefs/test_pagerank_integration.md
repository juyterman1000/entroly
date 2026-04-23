---
claim_id: f901f81d-6d00-4425-b1e7-f3c09f844142
entity: test_pagerank_integration
status: inferred
confidence: 0.75
sources:
  - tests\test_pagerank_integration.py:24
  - tests\test_pagerank_integration.py:133
  - tests\test_pagerank_integration.py:261
  - tests\test_pagerank_integration.py:346
  - tests\test_pagerank_integration.py:432
  - tests\test_pagerank_integration.py:468
  - tests\test_pagerank_integration.py:28
  - tests\test_pagerank_integration.py:40
  - tests\test_pagerank_integration.py:44
  - tests\test_pagerank_integration.py:52
last_checked: 2026-04-23T03:07:07.938139+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_pagerank_integration

**Language:** python
**Lines of code:** 506

## Types
- `class TestRustPageRank()` — Test the Rust EntrolyEngine.compute_pagerank() method.
- `class TestDaemonPageRankPrioritization()` — Test that the daemon sorts gaps by PageRank centrality.
- `class TestEvolutionLoggerSourceTracking()` — Test source file deduplication and tracking in gaps.
- `class TestDaemonLifecycle()` — Test daemon start/stop and background thread management.
- `class TestVaultConfigIntegration()` — Test VaultConfig/VaultManager construction patterns used in server.py.
- `class TestPythonFallbackEngine()` — Test the Python EntrolyEngine without Rust dependency.

## Functions
- `def rust_engine(self)` — Create a Rust engine if available, skip if not.
- `def test_empty_engine_returns_empty_dict(self, rust_engine)`
- `def test_single_fragment_returns_score(self, rust_engine)`
- `def test_multiple_fragments_return_scores(self, rust_engine)`
- `def test_hub_file_gets_higher_score(self, rust_engine)` — A file imported by many others should have higher PageRank.
- `def test_scores_are_deterministic(self, rust_engine)` — Same input → same output.
- `def test_scores_sum_close_to_one(self, rust_engine)` — PageRank scores should sum to approximately 1.0.
- `def test_compute_pagerank_after_dep_graph_built(self, rust_engine)` — PageRank should work after dep graph auto_link runs during ingest.
- `def test_daemon_processes_gaps_without_pagerank(self)` — Daemon should work even without Rust engine (no PageRank).
- `def test_daemon_stats_after_gap_processing(self)`
- `def test_gaps_include_source_files(self)` — Verify gaps returned by EvolutionLogger include source_files.
- `def test_pagerank_sorting_order(self)` — Test the PageRank sorting logic directly.
- `def test_gap_with_no_source_files_gets_zero_priority(self)`
- `def test_gap_with_multiple_source_files_uses_max(self)` — When a gap has multiple source files, max PageRank wins.
- `def test_source_files_deduplication(self)`
- `def test_source_files_preserves_order(self)`
- `def test_empty_source_files(self)`
- `def test_gap_threshold_respected(self)`
- `def test_multiple_entities_tracked_independently(self)`
- `def test_stats_accurate(self)`
- `def test_vault_write_creates_file(self)`
- `def test_start_creates_thread(self)`
- `def test_stop_terminates_thread(self)`
- `def test_double_start_is_safe(self)`
- `def test_stats_while_running(self)`
- `def test_stats_while_stopped(self)`
- `def test_cooldown_prevents_rapid_processing(self)`
- `def test_vault_config_base_path(self)`
- `def test_vault_config_default_path(self)`
- `def test_vault_manager_creates_structure(self)`
- `def test_vault_manager_idempotent_init(self)`
- `def test_ingest_and_optimize_cycle(self)`
- `def test_advance_turn(self)`
- `def test_record_success_failure(self)`
- `def test_get_stats(self)`

## Dependencies
- `os`
- `pathlib`
- `pytest`
- `tempfile`
- `time`
