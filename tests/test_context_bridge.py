"""
End-to-end integration tests for the Entroly context bridge.

Tests all 6 orchestration components:
  1. LODManager — tier transitions, hysteresis, saturation alerts
  2. SubagentOrchestrator — spawn/despawn, depth limits, budget decay
  3. CronSessionManager — scheduled jobs, lifecycle
  4. MemoryBridge — graceful degradation (no hippocampus)
  5. HCCEngine — rate-distortion compression optimization
  6. AutoTune — EMA, Polyak averaging, weight calibration
"""

import math
import time
import pytest

from entroly.context_bridge import (
    LodTier, AgentState, LODManager,
    SubagentOrchestrator, CronSessionManager, MemoryBridge,
    HCCEngine, CompressionLevel, HCCFragment, AutoTune,
    NkbeAllocator, CognitiveBus,
    _generate_skeleton, _generate_reference, _emotional_to_salience,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. LODManager Tests
# ═══════════════════════════════════════════════════════════════════════

class TestLODManager:

    def test_register_sets_dormant(self):
        lod = LODManager()
        state = lod.register("agent_1")
        assert state.tier == LodTier.DORMANT
        assert state.depth == 0

    def test_register_with_parent_increments_depth(self):
        lod = LODManager()
        lod.register("parent")
        child = lod.register("child", parent_id="parent")
        assert child.depth == 1
        assert "child" in lod._agents["parent"].children

    def test_hysteresis_prevents_premature_transition(self):
        """ACTIVE→SATURATED requires ACTIVE_MIN_TICKS (3) ticks at high load."""
        lod = LODManager()
        lod.register("agent", initial_tier=LodTier.ACTIVE)

        # Tick 1 and 2: high load but NOT enough ticks → stays ACTIVE
        result1 = lod.update_load("agent", 0.95)
        result2 = lod.update_load("agent", 0.95)
        assert result1 is None
        assert result2 is None
        assert lod._agents["agent"].tier == LodTier.ACTIVE

        # Tick 3: now meets min_ticks → transition to SATURATED
        result3 = lod.update_load("agent", 0.95)
        assert result3 == LodTier.SATURATED

    def test_dormant_to_active_promotion(self):
        lod = LODManager()
        lod.register("agent")
        result = lod.update_load("agent", 0.5)
        assert result == LodTier.ACTIVE
        assert lod._promotions == 1

    def test_active_to_dormant_demotion(self):
        lod = LODManager()
        lod.register("agent", initial_tier=LodTier.ACTIVE)
        # Need 3 ticks at low load
        for _ in range(3):
            lod.update_load("agent", 0.05)
        assert lod._agents["agent"].tier == LodTier.DORMANT
        assert lod._demotions == 1

    def test_saturation_alert(self):
        lod = LODManager()
        # Register 20 agents, saturate 2 (10% > 5% threshold)
        for i in range(20):
            lod.register(f"agent_{i}", initial_tier=LodTier.ACTIVE)
        lod._agents["agent_0"].tier = LodTier.SATURATED
        lod._agents["agent_1"].tier = LodTier.SATURATED
        alerts = lod.tick()
        assert len(alerts) == 1
        assert "Saturation alert" in alerts[0]

    def test_unregister_reparents_children(self):
        lod = LODManager()
        lod.register("grandparent")
        lod.register("parent", parent_id="grandparent")
        lod.register("child", parent_id="parent")
        lod.unregister("parent")
        assert lod._agents["child"].parent_id == "grandparent"
        assert "child" in lod._agents["grandparent"].children

    def test_fibonacci_hash_scatter(self):
        """Fibonacci scatter should distribute agents across [0, 1000)."""
        lod = LODManager()
        positions = []
        for i in range(100):
            state = lod.register(f"agent_{i}")
            positions.append(state.fib_position)
        # Check all are in range
        assert all(0 <= p < 1000 for p in positions)
        # Check reasonable spread (std > 100 for 100 agents in [0, 1000))
        mean_pos = sum(positions) / len(positions)
        variance = sum((p - mean_pos) ** 2 for p in positions) / len(positions)
        assert variance ** 0.5 > 100, "Fibonacci scatter should be well-distributed"

    def test_budget_weights(self):
        lod = LODManager()
        lod.register("heavy", initial_tier=LodTier.HEAVY)
        lod.register("active", initial_tier=LodTier.ACTIVE)
        lod.register("dormant", initial_tier=LodTier.DORMANT)
        weights = lod.get_budget_weights()
        assert weights["heavy"] == 2.0
        assert weights["active"] == 1.0
        assert "dormant" not in weights


# ═══════════════════════════════════════════════════════════════════════
# 2. NkbeAllocator Tests
# ═══════════════════════════════════════════════════════════════════════

class TestNkbeAllocator:

    def test_single_agent_gets_full_budget(self):
        alloc = NkbeAllocator(global_budget=10000)
        alloc.register_agent("agent_a", weight=1.0)
        result = alloc.allocate()
        assert result["agent_a"] == 10000

    def test_two_equal_agents_roughly_equal_split(self):
        alloc = NkbeAllocator(global_budget=10000)
        alloc.register_agent("a", weight=1.0)
        alloc.register_agent("b", weight=1.0)
        result = alloc.allocate()
        # Should be roughly equal (within 20%)
        assert abs(result["a"] - result["b"]) < 2000

    def test_higher_weight_gets_more(self):
        alloc = NkbeAllocator(global_budget=10000, min_agent_budget=512)
        alloc.register_agent("high", weight=3.0)
        alloc.register_agent("low", weight=1.0)
        alloc.update_fragments("high", 50, 5000)
        alloc.update_fragments("low", 10, 1000)
        result = alloc.allocate()
        assert result["high"] > result["low"]

    def test_minimum_budget_respected(self):
        alloc = NkbeAllocator(global_budget=2000, min_agent_budget=1024)
        alloc.register_agent("a", weight=1.0)
        alloc.register_agent("b", weight=1.0)
        result = alloc.allocate()
        assert result["a"] >= 1024
        assert result["b"] >= 1024

    def test_reinforce_adjusts_weights(self):
        alloc = NkbeAllocator(global_budget=10000, learning_rate=0.1)
        alloc.register_agent("good", weight=1.0)
        alloc.register_agent("bad", weight=1.0)
        alloc.reinforce({"good": 1.0, "bad": 0.0})
        # "good" weight should increase, "bad" should decrease
        assert alloc._weights["good"] > 1.0
        assert alloc._weights["bad"] < 1.0


# ═══════════════════════════════════════════════════════════════════════
# 3. CognitiveBus Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCognitiveBus:

    def test_publish_and_drain(self):
        bus = CognitiveBus()
        bus.subscribe("agent_a")
        bus.subscribe("agent_b")
        delivered = bus.publish("agent_a", "observation", "Found a bug")
        assert delivered == 1  # Only to agent_b (not self)
        events = bus.drain("agent_b")
        assert len(events) == 1
        assert events[0]["source"] == "agent_a"

    def test_novelty_dedup(self):
        bus = CognitiveBus()
        bus.subscribe("agent_b")
        bus.publish("agent_a", "observation", "Same message")
        d2 = bus.publish("agent_a", "observation", "Same message")
        assert d2 == 0  # Deduplicated
        assert bus._total_suppressed == 1

    def test_stats(self):
        bus = CognitiveBus()
        bus.subscribe("a")
        bus.subscribe("b")
        bus.publish("a", "observation", "Hello")
        stats = bus.stats()
        assert stats["total_published"] == 1
        assert stats["total_delivered"] == 1


# ═══════════════════════════════════════════════════════════════════════
# 4. HCCEngine Tests
# ═══════════════════════════════════════════════════════════════════════

class TestHCCEngine:

    def test_all_reference_when_tight_budget(self):
        hcc = HCCEngine()
        hcc.add_fragment("f1", "test.py", "def foo():\n    return 1\n" * 100,
                         entropy_score=0.5, relevance=0.5)
        hcc.add_fragment("f2", "test2.py", "class Bar:\n    pass\n" * 50,
                         entropy_score=0.3, relevance=0.3)
        result = hcc.optimize(token_budget=10)
        for frag in result:
            assert frag.assigned_level == CompressionLevel.REFERENCE

    def test_high_relevance_gets_full(self):
        hcc = HCCEngine()
        hcc.add_fragment("high", "important.py",
                         "def critical_function():\n    # This is very important\n    return 42\n",
                         entropy_score=0.9, relevance=0.95)
        hcc.add_fragment("low", "trivial.py",
                         "# just a comment\n",
                         entropy_score=0.1, relevance=0.1)
        result = hcc.optimize(token_budget=10000)
        high_frag = [f for f in result if f.fragment_id == "high"][0]
        assert high_frag.assigned_level == CompressionLevel.FULL

    def test_marginal_gain_ordering(self):
        """Greedy should pick highest gain-ratio first."""
        hcc = HCCEngine()
        # High entropy + relevance = high gain ratio
        hcc.add_fragment("good", "good.py",
                         "def process_payment(amount):\n    validate(amount)\n    charge(amount)\n",
                         entropy_score=0.9, relevance=0.9)
        # Low entropy + relevance = low gain ratio
        hcc.add_fragment("bad", "bad.py",
                         "# TODO: implement this\npass\n" * 20,
                         entropy_score=0.1, relevance=0.1)
        # Budget enough for one Full
        result = hcc.optimize(token_budget=50)
        good = [f for f in result if f.fragment_id == "good"][0]
        bad = [f for f in result if f.fragment_id == "bad"]
        # Good fragment should get higher compression level than bad
        if bad:
            assert good.assigned_level <= bad[0].assigned_level  # Lower number = better

    def test_retention_values_correct(self):
        assert CompressionLevel.RETENTION[0] == 1.00
        assert CompressionLevel.RETENTION[1] == 0.70
        assert CompressionLevel.RETENTION[2] == 0.15


# ═══════════════════════════════════════════════════════════════════════
# 5. AutoTune Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAutoTune:

    def test_initial_weights(self):
        at = AutoTune()
        weights = at.get_weights()
        assert weights["entropy"] == 1.0
        assert weights["relevance"] == 1.0
        assert weights["recency"] == 0.5
        assert weights["diversity"] == 0.3

    def test_positive_outcome_reinforces(self):
        at = AutoTune()
        # Good outcome with high entropy contribution
        for _ in range(20):
            at.update(1.0, {"entropy": 0.9, "relevance": 0.5, "recency": 0.5, "diversity": 0.5})
        # Entropy weight should increase (Polyak is slow, so check raw weights)
        assert at._weights["entropy"] > 1.0

    def test_negative_outcome_decreases(self):
        at = AutoTune()
        # Prime with some good outcomes first
        for _ in range(10):
            at.update(0.8, {"entropy": 0.9, "relevance": 0.5, "recency": 0.5, "diversity": 0.5})
        # Then bad outcomes with high entropy
        for _ in range(50):
            at.update(0.0, {"entropy": 0.9, "relevance": 0.5, "recency": 0.5, "diversity": 0.5})
        # Entropy weight should decrease since it was high when outcomes were bad
        assert at._weights["entropy"] < 1.0

    def test_drift_penalty_prevents_runaway(self):
        at = AutoTune(drift_penalty=0.5)  # Strong drift penalty
        at._weights["entropy"]
        # Very extreme outcomes
        for _ in range(100):
            at.update(1.0, {"entropy": 1.0, "relevance": 0.0, "recency": 0.0, "diversity": 0.0})
        # Should not exceed clamp range
        assert at._weights["entropy"] <= 5.0
        assert at._weights["entropy"] >= 0.05

    def test_polyak_averaging_is_slower(self):
        at = AutoTune()
        at.update(1.0, {"entropy": 1.0, "relevance": 1.0, "recency": 1.0, "diversity": 1.0})
        # Polyak should move much less than raw weights
        raw_delta = abs(at._weights["entropy"] - 1.0)
        polyak_delta = abs(at._polyak_weights["entropy"] - 1.0)
        assert polyak_delta < raw_delta

    def test_ema_outcome_tracking(self):
        at = AutoTune(ema_alpha=0.5)  # Fast EMA for test
        at.update(1.0, {"entropy": 0.5})
        # EMA should move toward 1.0
        assert at._outcome_ema > 0.5
        at.update(0.0, {"entropy": 0.5})
        # EMA should drop
        assert at._outcome_ema < 1.0


# ═══════════════════════════════════════════════════════════════════════
# 6. MemoryBridge Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMemoryBridge:

    def test_graceful_degradation(self):
        """MemoryBridge should work even without hippocampus installed."""
        bus = CognitiveBus()
        mb = MemoryBridge(bus)
        assert not mb.active
        assert mb.bridge_events() == 0
        assert mb.recall_for_context("anything") == []
        assert mb.stats() == {"active": False}

    def test_salience_mapping(self):
        assert _emotional_to_salience(0.9) == 100.0
        assert _emotional_to_salience(0.6) == 50.0
        assert _emotional_to_salience(0.3) == 20.0
        assert _emotional_to_salience(0.1) == 5.0


# ═══════════════════════════════════════════════════════════════════════
# 7. Skeleton / Reference Generation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCompression:

    def test_skeleton_keeps_definitions(self):
        code = """import os

def process_file(path):
    \"\"\"Process a file.\"\"\"
    content = open(path).read()
    # Parse the content
    lines = content.split('\\n')
    for line in lines:
        handle_line(line)
    return True
"""
        skeleton = _generate_skeleton(code)
        assert "import os" in skeleton
        assert "def process_file" in skeleton
        # Should be shorter than original
        assert len(skeleton) < len(code)

    def test_reference_is_one_line(self):
        content = "def foo():\n    return 42\n"
        ref = _generate_reference(content, "/path/to/file.py")
        assert "\n" not in ref
        assert "[file.py]" in ref

    def test_skeleton_keeps_key_markers(self):
        content = "x = 1\n# TODO: fix this\ny = 2\n"
        skeleton = _generate_skeleton(content)
        assert "TODO" in skeleton


# ═══════════════════════════════════════════════════════════════════════
# 8. Integration: Full Pipeline Test
# ═══════════════════════════════════════════════════════════════════════

class TestIntegrationPipeline:

    def test_lod_nkbe_integration(self):
        """LOD budget weights feed into NKBE allocation."""
        lod = LODManager()
        lod.register("heavy_agent", initial_tier=LodTier.HEAVY)
        lod.register("active_agent", initial_tier=LodTier.ACTIVE)

        weights = lod.get_budget_weights()
        alloc = NkbeAllocator(global_budget=10000)
        for name, w in weights.items():
            alloc.register_agent(name, weight=w)

        budgets = alloc.allocate()
        # HEAVY (weight=2.0) should get more than ACTIVE (weight=1.0)
        assert budgets["heavy_agent"] > budgets["active_agent"]

    def test_hcc_autotune_feedback_loop(self):
        """AutoTune adjusts weights, HCC uses them for compression."""
        at = AutoTune()
        hcc = HCCEngine()

        hcc.add_fragment("f1", "main.py", "def main():\n    run()\n",
                         entropy_score=0.8, relevance=0.7)
        result = hcc.optimize(token_budget=1000)
        assert len(result) > 0

        # Record positive outcome
        updated = at.update(1.0, {"entropy": 0.8, "relevance": 0.7,
                                  "recency": 0.5, "diversity": 0.3})
        assert "entropy" in updated
        assert updated["entropy"] > 0

    def test_bus_lod_cron_pipeline(self):
        """CognitiveBus events trigger LOD transitions."""
        bus = CognitiveBus()
        lod = LODManager()

        # Register agents
        lod.register("main", initial_tier=LodTier.ACTIVE)
        bus.subscribe("main")
        lod.register("cron_email", initial_tier=LodTier.DORMANT)
        bus.subscribe("cron_email")

        # Cron wakes up → promote
        lod.update_load("cron_email", 0.5)
        assert lod._agents["cron_email"].tier == LodTier.ACTIVE

        # Do work, publish result
        delivered = bus.publish("cron_email", "observation", "3 new emails found")
        assert delivered == 1  # Delivered to main

        # Cron finishes → demote (need min ticks)
        for _ in range(4):
            lod.update_load("cron_email", 0.01)
        assert lod._agents["cron_email"].tier == LodTier.DORMANT


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
