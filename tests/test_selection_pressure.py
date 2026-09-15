"""Tests for selection pressure maintenance — PRISM 5D.

Verifies:
  1. Resonance homeostasis — w_resonance adapts to content weight health
  2. Query-conditioned modulation — weights shift per-query in log-space
  3. Structured reward — smoothed precision, log coverage, recovery gate
  4. 5D contribution estimates — w_resonance gets credit for diversity
  5. Collapse detection — fires on Gini threshold, uses resonance lever
  6. Weight metrics — Gini and entropy on arbitrary distributions
"""

from __future__ import annotations

import math
import random

from entroly.selection_pressure import (
    CONTENT_DIMS,
    PRISM_5D,
    CollapseDetector,
    PressureSignals,
    ResonanceHomeostasis,
    compute_pressure_reward,
    compute_query_modulation,
    estimate_contributions_5d,
    modulate_weights,
    weight_entropy,
    weight_gini,
)


# ═══════════════════════════════════════════════════════════════════════
# Resonance Homeostasis
# ═══════════════════════════════════════════════════════════════════════


class TestResonanceHomeostasis:
    def test_healthy_weights_return_base(self):
        h = ResonanceHomeostasis(base_resonance=0.10)
        cw = {"w_recency": 0.25, "w_frequency": 0.25,
              "w_semantic": 0.25, "w_entropy": 0.25}
        res = h.compute_resonance(cw)
        assert abs(res - 0.10) < 1e-9

    def test_collapsed_weights_boost_resonance(self):
        h = ResonanceHomeostasis(base_resonance=0.10, gain=0.5)
        cw = {"w_recency": 0.85, "w_frequency": 0.05,
              "w_semantic": 0.05, "w_entropy": 0.05}
        res = h.compute_resonance(cw)
        assert res > 0.10

    def test_max_resonance_cap(self):
        h = ResonanceHomeostasis(base_resonance=0.10, gain=10.0, max_resonance=0.35)
        cw = {"w_recency": 0.97, "w_frequency": 0.01,
              "w_semantic": 0.01, "w_entropy": 0.01}
        res = h.compute_resonance(cw)
        assert res <= 0.35

    def test_proportional_response(self):
        """More collapse -> more resonance (proportional gain)."""
        h = ResonanceHomeostasis(base_resonance=0.10, gain=0.5)
        mild = {"w_recency": 0.40, "w_frequency": 0.25,
                "w_semantic": 0.20, "w_entropy": 0.15}
        severe = {"w_recency": 0.85, "w_frequency": 0.05,
                  "w_semantic": 0.05, "w_entropy": 0.05}
        r_mild = h.compute_resonance(mild)
        r_severe = h.compute_resonance(severe)
        assert r_severe > r_mild

    def test_negative_feedback_property(self):
        """Resonance boost makes the FULL 5D distribution more uniform."""
        h = ResonanceHomeostasis(base_resonance=0.10, gain=0.5)
        collapsed_4d = {"w_recency": 0.85, "w_frequency": 0.05,
                        "w_semantic": 0.05, "w_entropy": 0.05}
        new_res = h.compute_resonance(collapsed_4d)
        full_5d = dict(collapsed_4d)
        full_5d["w_resonance"] = new_res
        total = sum(full_5d.values())
        normalized = {k: v / total for k, v in full_5d.items()}
        assert weight_entropy(normalized) > weight_entropy(
            {k: v / sum(collapsed_4d.values()) for k, v in collapsed_4d.items()}
        )

    def test_stats(self):
        h = ResonanceHomeostasis(base_resonance=0.10)
        cw = {"w_recency": 0.85, "w_frequency": 0.05,
              "w_semantic": 0.05, "w_entropy": 0.05}
        h.compute_resonance(cw)
        s = h.stats()
        assert s["n_adjustments"] >= 1
        assert s["last_resonance"] > s["base_resonance"]


# ═══════════════════════════════════════════════════════════════════════
# Query-Conditioned Modulation
# ═══════════════════════════════════════════════════════════════════════


class TestQueryModulation:
    def test_empty_query_gives_zero_sigma(self):
        sigma = compute_query_modulation("")
        assert all(v == 0.0 for v in sigma.values())
        assert set(sigma.keys()) == set(PRISM_5D)

    def test_debug_query_boosts_recency_and_resonance(self):
        sigma = compute_query_modulation("why does this function crash?")
        assert sigma["w_recency"] > 0
        assert sigma["w_resonance"] > 0

    def test_architecture_query_boosts_entropy_and_resonance(self):
        sigma = compute_query_modulation("explain the system architecture")
        assert sigma["w_entropy"] > 0
        assert sigma["w_resonance"] > 0

    def test_refactor_query_boosts_frequency(self):
        sigma = compute_query_modulation("refactor the payment handler")
        assert sigma["w_frequency"] > 0

    def test_comparison_query_boosts_semantic_and_resonance(self):
        sigma = compute_query_modulation("what is the difference between A and B?")
        assert sigma["w_semantic"] > 0
        assert sigma["w_resonance"] > 0

    def test_recent_query_boosts_recency(self):
        sigma = compute_query_modulation("what changed in the latest commit?")
        assert sigma["w_recency"] > 0

    def test_modulate_identity_on_zero_sigma(self):
        base = {"w_recency": 0.30, "w_frequency": 0.25,
                "w_semantic": 0.25, "w_entropy": 0.10, "w_resonance": 0.10}
        sigma = {d: 0.0 for d in PRISM_5D}
        result = modulate_weights(base, sigma)
        for d in PRISM_5D:
            assert abs(result[d] - base[d]) < 1e-6

    def test_modulate_sums_to_one(self):
        base = {"w_recency": 0.30, "w_frequency": 0.25,
                "w_semantic": 0.25, "w_entropy": 0.10, "w_resonance": 0.10}
        sigma = compute_query_modulation("debug the crash in auth flow")
        result = modulate_weights(base, sigma)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_modulate_shifts_dominant_dimension(self):
        base = {"w_recency": 0.20, "w_frequency": 0.20,
                "w_semantic": 0.20, "w_entropy": 0.20, "w_resonance": 0.20}
        sigma = compute_query_modulation("explain the system design")
        result = modulate_weights(base, sigma)
        assert result["w_entropy"] > base["w_entropy"]
        assert result["w_resonance"] > base["w_resonance"]


# ═══════════════════════════════════════════════════════════════════════
# Structured Reward
# ═══════════════════════════════════════════════════════════════════════


class TestSmoothedPrecision:
    def test_zero_claims_gives_zero(self):
        r = compute_pressure_reward(PressureSignals())
        assert r.r_precision == 0.0

    def test_all_grounded(self):
        r = compute_pressure_reward(PressureSignals(n_grounded=10))
        assert abs(r.r_precision - 10 / 11) < 1e-9

    def test_contradiction_penalty(self):
        clean = compute_pressure_reward(
            PressureSignals(n_grounded=8, n_unsupported=2)
        )
        dirty = compute_pressure_reward(
            PressureSignals(n_grounded=8, n_contradicted=2)
        )
        assert dirty.r_precision < clean.r_precision

    def test_all_contradicted_zeroes_precision(self):
        r = compute_pressure_reward(PressureSignals(n_contradicted=10))
        assert r.r_precision == 0.0


class TestLogDiscountedCoverage:
    def test_zero_selected_gives_zero(self):
        r = compute_pressure_reward(PressureSignals())
        assert r.r_coverage == 0.0

    def test_all_adequate(self):
        r = compute_pressure_reward(
            PressureSignals(n_adequate_fragments=10, n_selected_fragments=10)
        )
        assert abs(r.r_coverage - 1.0) < 1e-9

    def test_diminishing_returns_per_fragment(self):
        r0 = compute_pressure_reward(
            PressureSignals(n_adequate_fragments=0, n_selected_fragments=50)
        )
        r5 = compute_pressure_reward(
            PressureSignals(n_adequate_fragments=5, n_selected_fragments=50)
        )
        r10 = compute_pressure_reward(
            PressureSignals(n_adequate_fragments=10, n_selected_fragments=50)
        )
        assert (r5.r_coverage - r0.r_coverage) / 5 > (r10.r_coverage - r5.r_coverage) / 5

    def test_concavity(self):
        prev_marginal = float("inf")
        for a in range(1, 20):
            ra = compute_pressure_reward(
                PressureSignals(n_adequate_fragments=a, n_selected_fragments=20)
            )
            ra_prev = compute_pressure_reward(
                PressureSignals(n_adequate_fragments=a - 1, n_selected_fragments=20)
            )
            marginal = ra.r_coverage - ra_prev.r_coverage
            assert marginal <= prev_marginal + 1e-12
            prev_marginal = marginal


class TestRecoveryGate:
    def test_gate_fires_on_recovery(self):
        r = compute_pressure_reward(
            PressureSignals(
                n_grounded=10, n_adequate_fragments=10,
                n_selected_fragments=10, n_distinct_sources=5,
                utilization=0.8, recovery_triggered=True,
            )
        )
        assert r.gated
        assert r.reward <= 0.4

    def test_gate_overrides_perfect_signals(self):
        r = compute_pressure_reward(
            PressureSignals(
                n_grounded=100, n_adequate_fragments=100,
                n_selected_fragments=100, n_distinct_sources=50,
                utilization=0.85, recovery_triggered=True,
            )
        )
        assert r.reward <= 0.4

    def test_gate_does_not_inflate_low_reward(self):
        r = compute_pressure_reward(
            PressureSignals(recovery_triggered=True)
        )
        assert r.reward < 0.4
        assert not r.gated


class TestSourceDiversity:
    def test_all_same_source_gives_low_diversity(self):
        r = compute_pressure_reward(
            PressureSignals(
                n_selected_fragments=10, n_distinct_sources=1,
            )
        )
        assert r.r_diversity == 0.1

    def test_all_distinct_sources_gives_high_diversity(self):
        r = compute_pressure_reward(
            PressureSignals(
                n_selected_fragments=10, n_distinct_sources=10,
            )
        )
        assert abs(r.r_diversity - 1.0) < 1e-9

    def test_diversity_in_reward(self):
        low_div = compute_pressure_reward(
            PressureSignals(
                n_grounded=10, n_adequate_fragments=10,
                n_selected_fragments=10, n_distinct_sources=1,
                utilization=0.8,
            )
        )
        high_div = compute_pressure_reward(
            PressureSignals(
                n_grounded=10, n_adequate_fragments=10,
                n_selected_fragments=10, n_distinct_sources=10,
                utilization=0.8,
            )
        )
        assert high_div.reward > low_div.reward


# ═══════════════════════════════════════════════════════════════════════
# 5D Contribution Estimates
# ═══════════════════════════════════════════════════════════════════════


class TestContributions5D:
    def test_all_five_dimensions_present(self):
        c = estimate_contributions_5d()
        assert set(c.keys()) == set(PRISM_5D)

    def test_sums_to_one(self):
        c = estimate_contributions_5d(
            witness_score=0.9, evidence_adequacy=0.8,
            utilization=0.85, source_diversity=0.7,
        )
        assert abs(sum(c.values()) - 1.0) < 1e-9

    def test_high_diversity_boosts_resonance(self):
        low = estimate_contributions_5d(source_diversity=0.1)
        high = estimate_contributions_5d(source_diversity=0.8)
        assert high["w_resonance"] > low["w_resonance"]

    def test_no_negative_contributions(self):
        rng = random.Random(42)
        for _ in range(100):
            c = estimate_contributions_5d(
                witness_score=rng.random(),
                evidence_adequacy=rng.random(),
                utilization=rng.random(),
                n_recovered=rng.randint(0, 5),
                source_diversity=rng.random(),
            )
            assert all(v >= 0 for v in c.values())


# ═══════════════════════════════════════════════════════════════════════
# Weight Metrics
# ═══════════════════════════════════════════════════════════════════════


class TestWeightMetrics:
    def test_uniform_gini_zero(self):
        assert abs(weight_gini(dict.fromkeys(PRISM_5D, 0.2))) < 1e-9

    def test_concentrated_gini_high(self):
        w = {"w_recency": 0.96, "w_frequency": 0.01, "w_semantic": 0.01,
             "w_entropy": 0.01, "w_resonance": 0.01}
        assert weight_gini(w) > 0.7

    def test_uniform_entropy_one(self):
        assert abs(weight_entropy(dict.fromkeys(PRISM_5D, 0.2)) - 1.0) < 1e-9

    def test_concentrated_entropy_low(self):
        w = {"w_recency": 0.96, "w_frequency": 0.01, "w_semantic": 0.01,
             "w_entropy": 0.01, "w_resonance": 0.01}
        assert weight_entropy(w) < 0.3


# ═══════════════════════════════════════════════════════════════════════
# Collapse Detector
# ═══════════════════════════════════════════════════════════════════════


class TestCollapseDetector:
    def test_healthy_no_collapse(self):
        det = CollapseDetector()
        w = dict.fromkeys(PRISM_5D, 0.2)
        res, event = det.check(w)
        assert event is None

    def test_collapsed_triggers_event(self):
        det = CollapseDetector()
        w = {"w_recency": 0.02, "w_frequency": 0.02, "w_semantic": 0.90,
             "w_entropy": 0.02, "w_resonance": 0.04}
        res, event = det.check(w)
        assert event is not None
        assert event.dominant_dim == "w_semantic"
        assert event.resonance_after > event.resonance_before

    def test_resonance_lever_is_used(self):
        det = CollapseDetector()
        w = {"w_recency": 0.02, "w_frequency": 0.02, "w_semantic": 0.90,
             "w_entropy": 0.02, "w_resonance": 0.04}
        res, event = det.check(w)
        assert res > w["w_resonance"]

    def test_cooldown_suppresses(self):
        det = CollapseDetector(cooldown=3)
        w = {"w_recency": 0.02, "w_frequency": 0.02, "w_semantic": 0.90,
             "w_entropy": 0.02, "w_resonance": 0.04}
        _, e1 = det.check(w)
        _, e2 = det.check(w)
        assert e1 is not None
        assert e2 is None

    def test_stats(self):
        det = CollapseDetector(cooldown=1)
        healthy = dict.fromkeys(PRISM_5D, 0.2)
        collapsed = {"w_recency": 0.02, "w_frequency": 0.02,
                     "w_semantic": 0.90, "w_entropy": 0.02, "w_resonance": 0.04}
        det.check(healthy)
        det.check(collapsed)
        det.check(healthy)
        s = det.stats()
        assert s["n_observed"] == 3
        assert s["n_collapses"] == 1


# ═══════════════════════════════════════════════════════════════════════
# Invariants
# ═══════════════════════════════════════════════════════════════════════


class TestRewardBounds:
    def test_fuzz_unit_interval(self):
        rng = random.Random(42)
        for _ in range(1000):
            signals = PressureSignals(
                n_grounded=rng.randint(0, 100),
                n_contradicted=rng.randint(0, 100),
                n_unsupported=rng.randint(0, 100),
                n_adequate_fragments=rng.randint(0, 100),
                n_selected_fragments=rng.randint(0, 100),
                n_distinct_sources=rng.randint(0, 100),
                recovery_triggered=rng.random() > 0.5,
                utilization=rng.random(),
            )
            r = compute_pressure_reward(signals)
            assert 0.0 <= r.reward <= 1.0

    def test_fuzz_gated_never_exceeds_cap(self):
        rng = random.Random(99)
        for _ in range(500):
            signals = PressureSignals(
                n_grounded=rng.randint(0, 100),
                n_contradicted=rng.randint(0, 100),
                n_unsupported=rng.randint(0, 100),
                n_adequate_fragments=rng.randint(0, 100),
                n_selected_fragments=rng.randint(0, 100),
                n_distinct_sources=rng.randint(0, 100),
                recovery_triggered=True,
                utilization=rng.random(),
            )
            r = compute_pressure_reward(signals, gate_cap=0.3)
            assert r.reward <= 0.3 + 1e-9

    def test_fuzz_modulation_sums_to_one(self):
        rng = random.Random(77)
        for _ in range(500):
            base = {d: max(0.01, rng.random()) for d in PRISM_5D}
            total = sum(base.values())
            base = {d: v / total for d, v in base.items()}
            sigma = {d: rng.uniform(-1, 1) for d in PRISM_5D}
            result = modulate_weights(base, sigma)
            assert abs(sum(result.values()) - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════════
# Integration: wiring into OnlinePrism and SelfImprovingLoop
# ═══════════════════════════════════════════════════════════════════════


class TestOnlinePrismIntegration:
    """Verify that 5D contributions are safe to pass to the 4D OnlinePrism."""

    def test_5d_contributions_harmless_to_4d_prism(self):
        from entroly.online_learner import OnlinePrism
        prism = OnlinePrism()
        c5 = estimate_contributions_5d(
            witness_score=0.8,
            evidence_adequacy=0.7,
            utilization=0.85,
            source_diversity=0.6,
        )
        assert "w_resonance" in c5
        pre = prism.weights()
        post = prism.observe(0.7, c5)
        assert set(post.keys()) == set(pre.keys())
        assert "w_resonance" not in post

    def test_collapse_detector_reads_prism_weights(self):
        from entroly.online_learner import OnlinePrism
        prism = OnlinePrism()
        det = CollapseDetector()
        for _ in range(10):
            w = prism.weights()
            res, evt = det.check(w)
            assert isinstance(res, float)
            assert 0.0 < res <= 0.35
            prism.observe(0.6, {d: 0.25 for d in CONTENT_DIMS})

    def test_modulated_weights_usable_as_prism_prior(self):
        from entroly.online_learner import OnlinePrism
        base = {d: 0.2 for d in PRISM_5D}
        sigma = compute_query_modulation("debug the crash in auth")
        modulated = modulate_weights(base, sigma)
        content_only = {
            k: modulated[k] for k in CONTENT_DIMS
        }
        total = sum(content_only.values())
        content_only = {k: v / total for k, v in content_only.items()}
        prism = OnlinePrism(prior_weights=content_only)
        w = prism.weights()
        assert abs(sum(w.values()) - 1.0) < 1e-9


class TestSelfImprovingIntegration:
    """Verify that the 5D upgrade to SelfImprovingLoop is backward-compatible."""

    def test_observe_witness_produces_5d_contributions(self):
        from unittest.mock import MagicMock
        from entroly.self_improving import SelfImprovingLoop

        loop = SelfImprovingLoop()
        witness = MagicMock()
        witness.summary_score = 0.8
        witness.n_grounded = 5
        witness.n_contradicted = 0
        witness.n_unsupported = 1
        witness.certificates = []
        reward = loop.observe_witness(witness, tokens_used=800, token_budget=1000)
        assert 0.0 <= reward <= 1.0
        last_fb = loop._history[-1]
        assert "w_resonance" in last_fb.contributions

    def test_observe_recovery_produces_5d_contributions(self):
        from entroly.self_improving import SelfImprovingLoop

        loop = SelfImprovingLoop()
        reward = loop.observe_recovery(n_recovered=2, n_total_omissions=5)
        assert 0.0 <= reward <= 1.0
        last_fb = loop._history[-1]
        assert "w_resonance" in last_fb.contributions

    def test_prism_convergence_prevents_collapse(self):
        """Dirichlet-REINFORCE converges — Gini stays below threshold."""
        det = CollapseDetector(gini_threshold=0.40, cooldown=1)
        from entroly.online_learner import OnlinePrism
        prism = OnlinePrism(prior_strength=2.0)
        skewed = {"w_recency": 0.05, "w_frequency": 0.05,
                  "w_semantic": 0.80, "w_entropy": 0.10}
        for _ in range(100):
            prism.observe(0.9, skewed)
            w = prism.weights()
            _, evt = det.check(w)
            assert evt is None, "Robbins-Monro decay prevents collapse"

    def test_collapse_from_external_weight_override(self):
        """Collapse detector fires when external override skews weights."""
        det = CollapseDetector(gini_threshold=0.40, cooldown=1)
        from entroly.online_learner import OnlinePrism
        prism = OnlinePrism()
        prism.observe(0.7, {d: 0.25 for d in CONTENT_DIMS})
        collapsed = {"w_recency": 0.02, "w_frequency": 0.02,
                     "w_semantic": 0.90, "w_entropy": 0.06}
        new_res, evt = det.check(collapsed)
        assert evt is not None
        assert evt.dominant_dim == "w_semantic"
        assert new_res > 0.10
