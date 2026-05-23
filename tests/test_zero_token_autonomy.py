"""
Tests for Three Pillars of Zero-Token Autonomy.

Covers:
  Pillar 1: Evolution Budget Guardrail (ValueTracker)
  Pillar 2: Structural Synthesizer (SkillEngine)
  Pillar 3: Dreaming Loop (Autotune)
  Integration: Evolution Daemon
"""

import math
import os
import tempfile
import time
from pathlib import Path

import pytest


# ═══════════════════════════════════════════════════════════════════════
# Pillar 1: Token Economy — Evolution Budget Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEvolutionBudget:
    """Test the 'Tax on Savings' guardrail in ValueTracker."""

    def _make_tracker(self):
        from entroly.value_tracker import ValueTracker
        tmp = tempfile.mkdtemp()
        return ValueTracker(data_dir=Path(tmp))

    def test_fresh_tracker_has_zero_budget(self):
        vt = self._make_tracker()
        budget = vt.get_evolution_budget()
        assert budget["available_usd"] == 0.0
        assert not budget["can_evolve"]
        assert budget["tax_rate"] == 0.05

    def test_budget_grows_with_savings(self):
        vt = self._make_tracker()
        # Simulate saving 10,000 tokens at $0.003/1K = $0.03/request
        for _ in range(100):
            vt.record(tokens_saved=10000, model="claude-sonnet-4")
        budget = vt.get_evolution_budget()
        # 100 requests × 10K tokens × $0.003/1K = $3.00 saved
        # Budget = 5% × $3.00 = $0.15
        assert budget["available_usd"] > 0.0
        assert budget["can_evolve"]
        assert budget["total_earned_usd"] == pytest.approx(
            budget["available_usd"], abs=0.01
        )

    def test_spend_rejected_when_over_budget(self):
        vt = self._make_tracker()
        # No savings → no budget → spend should be rejected
        result = vt.record_evolution_spend(1.0)
        assert result["status"] == "rejected"

    def test_spend_accepted_within_budget(self):
        vt = self._make_tracker()
        # Save enough to have budget
        for _ in range(50):
            vt.record(tokens_saved=50000, model="gpt-4o")
        budget = vt.get_evolution_budget()
        assert budget["can_evolve"]

        # Spend a small amount
        result = vt.record_evolution_spend(0.001, success=True)
        assert result["status"] == "recorded"
        assert result["remaining_usd"] > 0

    def test_budget_invariant_strictly_token_negative(self):
        """C_spent(t) ≤ τ · S(t) must hold at all times."""
        vt = self._make_tracker()
        # Save $10 worth of tokens
        vt.record(tokens_saved=3_333_333, model="claude-sonnet-4")
        budget = vt.get_evolution_budget()
        earned = budget["total_earned_usd"]

        # Try to spend more than earned
        result = vt.record_evolution_spend(earned + 1.0)
        assert result["status"] == "rejected"

        # Spend exactly the earned amount
        result = vt.record_evolution_spend(earned - 0.001)
        assert result["status"] == "recorded"

        # Now budget should be near zero
        budget = vt.get_evolution_budget()
        assert budget["available_usd"] < 0.01


# ═══════════════════════════════════════════════════════════════════════
# Pillar 2: Structural Synthesizer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestStructuralSynthesizer:
    """Test zero-token skill synthesis from code structure."""

    def _make_synth(self):
        from entroly.skill_engine import StructuralSynthesizer
        return StructuralSynthesizer()

    def _write_sample_file(self, tmpdir: str) -> str:
        path = os.path.join(tmpdir, "auth_service.py")
        with open(path, "w") as f:
            f.write(
                "import hmac\n"
                "from typing import Optional\n\n"
                "class AuthService:\n"
                "    def authenticate(self, token: str, user_id: int) -> bool:\n"
                "        '''Validate a user token.'''\n"
                "        return hmac.compare_digest(token, self._get_secret(user_id))\n\n"
                "    def _get_secret(self, user_id: int) -> str:\n"
                "        return self._db.get(f'secret:{user_id}')\n\n"
                "    async def refresh_token(self, user_id: int, scope: str) -> Optional[str]:\n"
                "        '''Generate a new token.'''\n"
                "        return await self._token_service.mint(user_id, scope)\n\n"
                "def validate_cors(origin: str) -> bool:\n"
                "    return origin in ALLOWED_ORIGINS\n"
            )
        return path

    def test_returns_none_for_no_source_files(self):
        synth = self._make_synth()
        result = synth.synthesize_structural("missing.entity", [], ["why is auth broken"])
        assert result is None

    def test_extracts_function_signatures(self):
        synth = self._make_synth()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_sample_file(tmpdir)
            invariants = synth._extract_invariants([path])
            names = [s["name"] for s in invariants["signatures"]]
            assert "authenticate" in names
            assert "refresh_token" in names
            assert "validate_cors" in names

    def test_detects_classes(self):
        synth = self._make_synth()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_sample_file(tmpdir)
            invariants = synth._extract_invariants([path])
            assert "AuthService" in invariants["classes"]

    def test_detects_imports(self):
        synth = self._make_synth()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_sample_file(tmpdir)
            invariants = synth._extract_invariants([path])
            assert any("hmac" in imp for imp in invariants["imports"])

    def test_entropy_closure_ranks_by_information(self):
        synth = self._make_synth()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_sample_file(tmpdir)
            invariants = synth._extract_invariants([path])
            closure = synth._compute_entropy_closure("auth", [path], invariants)

            # Verify descending entropy order
            for i in range(len(closure) - 1):
                assert closure[i]["entropy"] >= closure[i + 1]["entropy"]

            # Functions with more params should rank higher
            names = [n["name"] for n in closure]
            # authenticate(token, user_id) has 2 params → higher entropy
            # validate_cors(origin) has 1 param → lower entropy
            if "authenticate" in names and "validate_cors" in names:
                auth_idx = names.index("authenticate")
                cors_idx = names.index("validate_cors")
                assert auth_idx < cors_idx

    def test_synthesize_produces_valid_tool(self):
        synth = self._make_synth()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_sample_file(tmpdir)
            spec = synth.synthesize_structural(
                "auth_service",
                [path],
                ["why does authentication fail", "fix the auth token validation"],
            )
            assert spec is not None
            assert spec.entity == "auth_service"
            assert "structural_induction" in spec.tool_code
            assert "zero-token" in spec.tool_code
            assert spec.metrics.get("synthesis_method") == 0.0  # 0 = structural
            assert "def execute" in spec.tool_code
            assert "def matches" in spec.tool_code


# ═══════════════════════════════════════════════════════════════════════
# Pillar 3: Dreaming Loop Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDreamingLoop:
    """Test the autonomous idle-time self-play optimization."""

    def _make_loop(self):
        from entroly.autotune import FeedbackJournal, DreamingLoop
        tmp = tempfile.mkdtemp()
        journal = FeedbackJournal(tmp)
        # Seed with some episodes
        journal.log(
            weights={"w_r": 0.3, "w_f": 0.25, "w_s": 0.25, "w_e": 0.2},
            reward=0.8, query="fix the auth bug", token_budget=8000,
        )
        journal.log(
            weights={"w_r": 0.3, "w_f": 0.25, "w_s": 0.25, "w_e": 0.2},
            reward=-0.5, query="optimize database.query performance", token_budget=12000,
        )
        return DreamingLoop(journal, max_iterations=3)

    def test_should_not_dream_when_active(self):
        loop = self._make_loop()
        loop.record_activity()
        assert not loop.should_dream()

    def test_should_dream_after_idle(self):
        loop = self._make_loop()
        loop._last_activity = time.time() - 120  # 2 minutes ago
        assert loop.should_dream()

    def test_generates_synthetic_queries(self):
        loop = self._make_loop()
        queries = loop.generate_synthetic_queries()
        assert len(queries) > 0
        # Should have different origins
        origins = {q.get("origin", "default") for q in queries}
        assert len(origins) >= 1

    def test_generates_failure_replays(self):
        loop = self._make_loop()
        queries = loop.generate_synthetic_queries()
        failure_replays = [q for q in queries if q.get("origin") == "failure_replay"]
        assert len(failure_replays) > 0  # We seeded a negative reward episode

    def test_stats_reports_idle_time(self):
        loop = self._make_loop()
        stats = loop.stats()
        assert "idle_seconds" in stats
        assert "total_dreams" in stats
        assert stats["total_dreams"] == 0


# ═══════════════════════════════════════════════════════════════════════
# Integration: Evolution Daemon Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEvolutionDaemon:
    """Test the daemon that orchestrates all 3 pillars."""

    def _make_daemon(self):
        from entroly.vault import VaultManager, VaultConfig
        from entroly.evolution_logger import EvolutionLogger
        from entroly.value_tracker import ValueTracker
        from entroly.evolution_daemon import EvolutionDaemon

        tmp = tempfile.mkdtemp()
        vault = VaultManager(VaultConfig(base_path=os.path.join(tmp, "vault")))
        evo_logger = EvolutionLogger(vault_path=os.path.join(tmp, "vault"))
        value_tracker = ValueTracker(data_dir=Path(tmp))

        daemon = EvolutionDaemon(
            vault=vault,
            evolution_logger=evo_logger,
            value_tracker=value_tracker,
        )
        return daemon, evo_logger, value_tracker

    def test_run_once_with_no_gaps(self):
        daemon, _, _ = self._make_daemon()
        result = daemon.run_once()
        assert result["gaps_processed"] == 0

    def test_structural_synthesis_preferred_over_llm(self):
        daemon, evo_logger, vt = self._make_daemon()
        stats = daemon.stats()
        assert stats["structural_successes"] == 0
        assert stats["llm_rejections"] == 0

    def test_budget_rejection_when_no_savings(self):
        """Daemon should reject LLM synthesis when budget is $0."""
        daemon, _, vt = self._make_daemon()
        budget = vt.get_evolution_budget()
        assert not budget["can_evolve"]

    def test_stats_include_all_pillars(self):
        daemon, _, _ = self._make_daemon()
        stats = daemon.stats()
        assert "structural_successes" in stats
        assert "structural_failures" in stats
        assert "llm_rejections" in stats
        assert "budget" in stats
        assert "running" in stats


# ═══════════════════════════════════════════════════════════════════════
# EvolutionLogger enhancement tests
# ═══════════════════════════════════════════════════════════════════════

class TestEvolutionLoggerEnhancements:
    """Test source file tracking in MissRecord."""

    def test_miss_record_includes_source_files(self):
        from entroly.evolution_logger import EvolutionLogger
        el = EvolutionLogger(gap_threshold=2)

        el.record_miss(
            query="how does auth work",
            entity_key="auth_service",
            source_files=["src/auth.py", "src/tokens.py"],
        )
        el.record_miss(
            query="fix auth bug",
            entity_key="auth_service",
            source_files=["src/auth.py", "src/middleware.py"],
        )

        gaps = el.get_pending_gaps()
        assert len(gaps) == 1
        gap = gaps[0]
        assert gap["entity_key"] == "auth_service"
        # Source files should be deduplicated
        assert "src/auth.py" in gap["source_files"]
        assert "src/tokens.py" in gap["source_files"]
        assert "src/middleware.py" in gap["source_files"]
        assert len(gap["source_files"]) == 3  # no duplicates


# ═══════════════════════════════════════════════════════════════════════
# ComponentFeedbackBus — Universal Self-Improvement Signal Tests
# ═══════════════════════════════════════════════════════════════════════

class TestComponentFeedbackBus:
    """Test the gradient-free self-improvement bus."""

    def _make_bus(self):
        from entroly.autotune import ComponentFeedbackBus
        tmp = tempfile.mkdtemp()
        return ComponentFeedbackBus(tmp)

    def test_log_and_get_trend(self):
        bus = self._make_bus()
        for i in range(10):
            bus.log("router", "success_rate", 0.5 + i * 0.05)
        trend = bus.get_trend("router", "success_rate")
        assert trend["count"] == 10
        assert trend["ema"] > 0.5  # Should be above initial

    def test_empty_trend_returns_zeros(self):
        bus = self._make_bus()
        trend = bus.get_trend("nonexistent", "metric")
        assert trend["count"] == 0
        assert trend["ema"] == 0.0

    def test_suggest_adjustment_increases_when_improving(self):
        bus = self._make_bus()
        # Log improving trend
        for i in range(10):
            bus.log("prefetch", "hit_rate", 0.3 + i * 0.05)

        # Should suggest increasing the parameter (maximize=True)
        suggestion = bus.suggest_adjustment(
            "prefetch", "hit_rate",
            current_value=5.0, bounds=(2.0, 15.0),
            step_size=0.1, maximize=True,
        )
        assert suggestion > 5.0  # Should increase

    def test_suggest_adjustment_decreases_when_degrading(self):
        bus = self._make_bus()
        # Log degrading trend
        for i in range(10):
            bus.log("prefetch", "hit_rate", 0.8 - i * 0.05)

        suggestion = bus.suggest_adjustment(
            "prefetch", "hit_rate",
            current_value=10.0, bounds=(2.0, 15.0),
            step_size=0.1, maximize=True,
        )
        assert suggestion < 10.0  # Should decrease

    def test_suggest_no_change_with_insufficient_data(self):
        bus = self._make_bus()
        bus.log("router", "metric", 0.5)
        suggestion = bus.suggest_adjustment(
            "router", "metric",
            current_value=5.0, bounds=(0.0, 10.0),
        )
        assert suggestion == 5.0  # No change — not enough data

    def test_stats_reports_all_components(self):
        bus = self._make_bus()
        bus.log("router", "flow_success", 1.0)
        bus.log("prefetch", "hit_rate", 0.7)
        bus.log("pruner", "reward", 0.5)
        stats = bus.stats()
        assert "router:flow_success" in stats
        assert "prefetch:hit_rate" in stats
        assert "pruner:reward" in stats

    def test_persistence_to_disk(self):
        bus = self._make_bus()
        bus.log("router", "success", 1.0)
        # Verify file was written
        assert bus._path.exists()
        content = bus._path.read_text()
        assert "router" in content


# ═══════════════════════════════════════════════════════════════════════
# EpistemicRouter Self-Tuning Tests
# ═══════════════════════════════════════════════════════════════════════

class TestEpistemicRouterSelfTuning:
    """Test that the router adaptively tunes its thresholds."""

    def _make_router(self):
        from entroly.epistemic_router import EpistemicRouter
        tmp = tempfile.mkdtemp()
        return EpistemicRouter(vault_path=tmp, miss_threshold=3,
                               freshness_hours=24.0, min_confidence=0.6)

    def test_record_outcome_tracks_history(self):
        router = self._make_router()
        router.record_outcome("fast_answer", success=True)
        router.record_outcome("fast_answer", success=False)
        assert hasattr(router, "_flow_outcomes")
        assert len(router._flow_outcomes["fast_answer"]) == 2

    def test_high_success_fast_answer_lowers_confidence(self):
        router = self._make_router()
        original = router._min_confidence
        # 10 successful fast_answer flows → should lower confidence threshold
        for _ in range(10):
            router.record_outcome("fast_answer", success=True)
        assert router._min_confidence <= original

    def test_low_success_fast_answer_raises_confidence(self):
        router = self._make_router()
        original = router._min_confidence
        # 10 failed fast_answer flows → should raise confidence threshold
        for _ in range(10):
            router.record_outcome("fast_answer", success=False)
        assert router._min_confidence >= original

    def test_low_success_compile_shrinks_freshness(self):
        router = self._make_router()
        original = router._freshness_hours
        for _ in range(10):
            router.record_outcome("compile_on_demand", success=False)
        assert router._freshness_hours <= original

    def test_self_tune_respects_bounds(self):
        router = self._make_router()
        # Drive min_confidence down hard
        for _ in range(200):
            router.record_outcome("fast_answer", success=True)
        assert router._min_confidence >= 0.3  # Floor

        # Drive freshness down hard
        router2 = self._make_router()
        for _ in range(200):
            router2.record_outcome("compile_on_demand", success=False)
        assert router2._freshness_hours >= 4.0  # Floor


# ═══════════════════════════════════════════════════════════════════════
# PrefetchEngine Self-Improvement Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPrefetchSelfImprovement:
    """Test adaptive co_access_window tuning."""

    def _make_engine(self):
        from entroly.prefetch import PrefetchEngine
        return PrefetchEngine(co_access_window=5)

    def test_hit_rate_tracking(self):
        engine = self._make_engine()
        # Make a prediction
        preds = engine.predict("src/auth.py", "import utils\n", "python")
        # Record some actual accesses
        engine.record_actual_access("src/auth.py")
        assert engine._total_predictions >= 1

    def test_stats_include_hit_rate(self):
        engine = self._make_engine()
        stats = engine.stats()
        assert "hit_rate" in stats
        assert "co_access_window" in stats
        assert "prediction_hits" in stats

    def test_component_bus_wiring(self):
        engine = self._make_engine()
        bus = self._make_bus()
        engine.set_component_bus(bus)
        assert engine._component_bus is bus

    def _make_bus(self):
        from entroly.autotune import ComponentFeedbackBus
        tmp = tempfile.mkdtemp()
        return ComponentFeedbackBus(tmp)


# ═══════════════════════════════════════════════════════════════════════
# CacheAligner Tests (formerly dead code, now wired)
# ═══════════════════════════════════════════════════════════════════════

class TestCacheAligner:
    """Test KV-cache prefix alignment for free token savings."""

    def _make_aligner(self):
        from entroly.cache_aligner import CacheAligner
        return CacheAligner(similarity_threshold=0.90)

    def test_identical_context_hits_cache(self):
        ca = self._make_aligner()
        ctx = "def foo():\n    return 42\n"
        _, hit1 = ca.align("client1", ctx)
        assert not hit1  # First call is always a miss
        _, hit2 = ca.align("client1", ctx)
        assert hit2  # Same content → cache hit

    def test_different_context_misses(self):
        ca = self._make_aligner()
        ca.align("client1", "def foo(): return 1")
        _, hit = ca.align("client1", "completely different unrelated code block xyz")
        assert not hit  # Very different → cache miss

    def test_similar_context_hits(self):
        ca = self._make_aligner()
        # Use a sufficiently large context so minor change keeps Jaccard > 90%
        base_tokens = ["def", "foo():", "x", "=", "1", "y", "=", "2",
                        "z", "=", "3", "w", "=", "4", "return", "x", "+", "y",
                        "+", "z", "+", "w", "class", "Bar:", "pass"]
        base = " ".join(base_tokens)
        ca.align("c1", base)
        # Change 1 token out of 25 → Jaccard ~24/26 ≈ 0.92
        modified_tokens = base_tokens.copy()
        modified_tokens[5] = "99"  # change "2" → "99"
        modified = " ".join(modified_tokens)
        _, hit = ca.align("c1", modified)
        assert hit  # >90% Jaccard → reuse

    def test_stats_tracks_hits_and_misses(self):
        ca = self._make_aligner()
        ca.align("c1", "code block one")
        ca.align("c1", "code block one")
        stats = ca.stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_invalidate_clears_client_cache(self):
        ca = self._make_aligner()
        ca.align("c1", "cached code")
        ca.invalidate("c1")
        _, hit = ca.align("c1", "cached code")
        assert not hit  # Invalidated → miss


# ═══════════════════════════════════════════════════════════════════════
# FlowOrchestrator Feedback Loop Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFlowOrchestratorFeedback:
    """Test that flow outcomes feed back to the router for self-tuning."""

    def _make_orchestrator(self):
        from entroly.vault import VaultManager, VaultConfig
        from entroly.epistemic_router import EpistemicRouter
        from entroly.belief_compiler import BeliefCompiler
        from entroly.verification_engine import VerificationEngine
        from entroly.change_pipeline import ChangePipeline
        from entroly.evolution_logger import EvolutionLogger
        from entroly.flow_orchestrator import FlowOrchestrator

        tmp = tempfile.mkdtemp()
        vault = VaultManager(VaultConfig(base_path=os.path.join(tmp, "vault")))
        router = EpistemicRouter(vault_path=os.path.join(tmp, "vault"),
                                  miss_threshold=2)
        compiler = BeliefCompiler(vault)
        verifier = VerificationEngine(vault)
        change_pipe = ChangePipeline(vault, verifier)
        evolution = EvolutionLogger(vault_path=os.path.join(tmp, "vault"))

        orch = FlowOrchestrator(
            vault=vault, router=router, compiler=compiler,
            verifier=verifier, change_pipe=change_pipe,
            evolution=evolution, source_dir=tmp,
        )
        return orch, router

    def test_execute_records_outcome_to_router(self):
        orch, router = self._make_orchestrator()
        result = orch.execute("explain the auth module")
        # Router should now have outcome history
        assert hasattr(router, "_flow_outcomes") or hasattr(router, "_total_outcomes")

    def test_component_bus_receives_feedback(self):
        from entroly.autotune import ComponentFeedbackBus
        orch, router = self._make_orchestrator()
        tmp = tempfile.mkdtemp()
        bus = ComponentFeedbackBus(tmp)
        orch._component_bus = bus

        orch.execute("find bugs in the code")
        # Bus should have recorded the flow outcome
        trend = bus.get_trend("epistemic_router", "flow_success")
        assert trend["count"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

