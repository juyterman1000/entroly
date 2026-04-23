---
claim_id: b2bf10f3-4e22-4109-8cb4-de706b31581c
entity: test_context_bridge
status: inferred
confidence: 0.75
sources:
  - tests\test_context_bridge.py:30
  - tests\test_context_bridge.py:126
  - tests\test_context_bridge.py:173
  - tests\test_context_bridge.py:207
  - tests\test_context_bridge.py:260
  - tests\test_context_bridge.py:321
  - tests\test_context_bridge.py:343
  - tests\test_context_bridge.py:379
  - tests\test_context_bridge.py:32
  - tests\test_context_bridge.py:38
last_checked: 2026-04-23T03:07:07.926126+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_context_bridge

**Language:** python
**Lines of code:** 439

## Types
- `class TestLODManager()`
- `class TestNkbeAllocator()`
- `class TestCognitiveBus()`
- `class TestHCCEngine()`
- `class TestAutoTune()`
- `class TestMemoryBridge()` — MemoryBridge should work even without hippocampus installed.
- `class TestCompression()` — code = import os def process_file(path): \"\"\"Process a file.\"\"\" content = open(path).read() # Parse the content lines = content.split('\\n') for line in lines: handle_line(line) return True
- `class TestIntegrationPipeline()` — LOD budget weights feed into NKBE allocation.

## Functions
- `def test_register_sets_dormant(self)`
- `def test_register_with_parent_increments_depth(self)`
- `def test_hysteresis_prevents_premature_transition(self)` — ACTIVE→SATURATED requires ACTIVE_MIN_TICKS (3) ticks at high load.
- `def test_dormant_to_active_promotion(self)`
- `def test_active_to_dormant_demotion(self)`
- `def test_saturation_alert(self)`
- `def test_unregister_reparents_children(self)`
- `def test_fibonacci_hash_scatter(self)` — Fibonacci scatter should distribute agents across [0, 1000).
- `def test_budget_weights(self)`
- `def test_single_agent_gets_full_budget(self)`
- `def test_two_equal_agents_roughly_equal_split(self)`
- `def test_higher_weight_gets_more(self)`
- `def test_minimum_budget_respected(self)`
- `def test_reinforce_adjusts_weights(self)`
- `def test_publish_and_drain(self)`
- `def test_novelty_dedup(self)`
- `def test_stats(self)`
- `def test_all_reference_when_tight_budget(self)`
- `def test_high_relevance_gets_full(self)`
- `def test_marginal_gain_ordering(self)` — Greedy should pick highest gain-ratio first.
- `def test_retention_values_correct(self)`
- `def test_initial_weights(self)`
- `def test_positive_outcome_reinforces(self)`
- `def test_negative_outcome_decreases(self)`
- `def test_drift_penalty_prevents_runaway(self)`
- `def test_polyak_averaging_is_slower(self)`
- `def test_ema_outcome_tracking(self)`
- `def test_graceful_degradation(self)` — MemoryBridge should work even without hippocampus installed.
- `def test_salience_mapping(self)`
- `def test_skeleton_keeps_definitions(self)` — code = import os def process_file(path): \"\"\"Process a file.\"\"\" content = open(path).read() # Parse the content lines = content.split('\\n') for line in lines: handle_line(line) return True
- `def process_file(path)`
- `def test_reference_is_one_line(self)`
- `def test_skeleton_keeps_key_markers(self)`
- `def test_lod_nkbe_integration(self)` — LOD budget weights feed into NKBE allocation.
- `def test_hcc_autotune_feedback_loop(self)` — AutoTune adjusts weights, HCC uses them for compression.
- `def test_bus_lod_cron_pipeline(self)` — CognitiveBus events trigger LOD transitions.

## Dependencies
- `entroly.context_bridge`
- `math`
- `pytest`
- `time`
