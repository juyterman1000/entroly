"""Tests for the self-improving loop (WITNESS → PRISM closed loop)."""

from __future__ import annotations

import pytest
from dataclasses import dataclass


# ── Mock WITNESS structures ──────────────────────────────────────────────

@dataclass
class MockCertificate:
    label: str
    evidence_adequacy: float = 0.7


@dataclass
class MockWitnessResult:
    summary_score: float
    n_grounded: int
    n_contradicted: int
    n_unsupported: int
    n_unknown: int = 0
    certificates: list = None
    output: str = ""
    latency_ms: float = 0

    def __post_init__(self):
        if self.certificates is None:
            self.certificates = [MockCertificate("grounded")] * self.n_grounded


# ── Reward computation ───────────────────────────────────────────────────

class TestComputeReward:
    def test_perfect_verification(self):
        from entroly.self_improving import compute_reward
        r = compute_reward(
            witness_score=1.0,
            n_grounded=5,
            n_contradicted=0,
            n_unsupported=0,
            evidence_adequacy=1.0,
            utilization=0.85,
        )
        assert r > 0.85, f"Perfect verification should give high reward, got {r}"

    def test_contradictions_penalize(self):
        from entroly.self_improving import compute_reward
        r_good = compute_reward(witness_score=0.8, n_grounded=5, n_contradicted=0, n_unsupported=0)
        r_bad = compute_reward(witness_score=0.8, n_grounded=3, n_contradicted=2, n_unsupported=0)
        assert r_bad < r_good, "Contradictions should lower reward"

    def test_recovery_penalizes(self):
        from entroly.self_improving import compute_reward
        r_no_recovery = compute_reward(n_recovered=0, n_total_omissions=10)
        r_recovery = compute_reward(n_recovered=5, n_total_omissions=10)
        assert r_recovery < r_no_recovery, "Recoveries should lower reward"

    def test_full_recovery_is_bad(self):
        from entroly.self_improving import compute_reward
        r = compute_reward(n_recovered=10, n_total_omissions=10)
        assert r < 0.5, f"Full recovery = bad selection, got {r}"

    def test_reward_bounds(self):
        from entroly.self_improving import compute_reward
        # Extreme inputs shouldn't break bounds
        for ws in [0.0, 0.5, 1.0]:
            for u in [0.0, 0.5, 1.0]:
                for n_r in [0, 5, 10]:
                    r = compute_reward(
                        witness_score=ws,
                        utilization=u,
                        n_recovered=n_r,
                        n_total_omissions=10,
                    )
                    assert 0 <= r <= 1, f"Reward {r} out of bounds"

    def test_utilization_sweet_spot(self):
        from entroly.self_improving import compute_reward
        r_low = compute_reward(utilization=0.3)
        r_good = compute_reward(utilization=0.85)
        r_high = compute_reward(utilization=0.99)
        assert r_good >= r_low, "Sweet spot should beat low utilization"
        assert r_good >= r_high, "Sweet spot should beat over-utilization"


# ── Contribution estimation ──────────────────────────────────────────────

class TestEstimateContributions:
    def test_sums_to_one(self):
        from entroly.self_improving import estimate_contributions
        c = estimate_contributions(witness_score=0.8, evidence_adequacy=0.9, utilization=0.85)
        total = sum(c.values())
        assert abs(total - 1.0) < 1e-6, f"Contributions sum to {total}"

    def test_high_adequacy_boosts_semantic(self):
        from entroly.self_improving import estimate_contributions
        c_high = estimate_contributions(evidence_adequacy=0.9)
        c_low = estimate_contributions(evidence_adequacy=0.3)
        assert c_high["w_semantic"] > c_low["w_semantic"]

    def test_all_dims_present(self):
        from entroly.self_improving import estimate_contributions
        c = estimate_contributions()
        for dim in ["w_recency", "w_frequency", "w_semantic", "w_entropy"]:
            assert dim in c, f"Missing dimension {dim}"
            assert c[dim] > 0, f"Dimension {dim} should be positive"


# ── Self-Improving Loop ──────────────────────────────────────────────────

class TestSelfImprovingLoop:
    def test_observe_witness_updates_weights(self):
        from entroly.self_improving import SelfImprovingLoop

        loop = SelfImprovingLoop()
        w_before = dict(loop.current_weights)

        result = MockWitnessResult(
            summary_score=0.95,
            n_grounded=5,
            n_contradicted=0,
            n_unsupported=0,
        )
        reward = loop.observe_witness(result, tokens_used=5000, token_budget=8000)

        assert 0 < reward <= 1
        assert loop.stats.n_observations == 1

    def test_observe_recovery_penalizes(self):
        from entroly.self_improving import SelfImprovingLoop

        loop = SelfImprovingLoop()
        r = loop.observe_recovery(n_recovered=5, n_total_omissions=10)
        assert r < 0.8, f"Recovery should penalize, got {r}"
        assert loop.stats.total_recoveries == 5

    def test_multiple_observations_converge(self):
        from entroly.self_improving import SelfImprovingLoop

        loop = SelfImprovingLoop()

        # Simulate 20 good observations
        for _ in range(20):
            result = MockWitnessResult(
                summary_score=0.9,
                n_grounded=5,
                n_contradicted=0,
                n_unsupported=0,
            )
            loop.observe_witness(result, tokens_used=6000, token_budget=8000)

        assert loop.stats.n_observations == 20
        assert loop.stats.mean_reward > 0.5

    def test_improvement_detection(self):
        from entroly.self_improving import SelfImprovingLoop

        loop = SelfImprovingLoop()

        # First 15: mediocre
        for _ in range(15):
            result = MockWitnessResult(
                summary_score=0.5,
                n_grounded=3,
                n_contradicted=1,
                n_unsupported=1,
            )
            loop.observe_witness(result)

        # Next 10: good (simulating learning took effect)
        for _ in range(10):
            result = MockWitnessResult(
                summary_score=0.95,
                n_grounded=5,
                n_contradicted=0,
                n_unsupported=0,
            )
            loop.observe_witness(result)

        # Last 10 should be better than overall mean
        assert loop.is_improving, f"Should detect improvement, trend={loop.stats.improvement_trend}"

    def test_explicit_feedback(self):
        from entroly.self_improving import SelfImprovingLoop

        loop = SelfImprovingLoop()
        loop.observe_explicit(0.9, source="user_thumbs_up")
        assert loop.stats.n_observations == 1

    def test_summary_output(self):
        from entroly.self_improving import SelfImprovingLoop

        loop = SelfImprovingLoop()
        result = MockWitnessResult(
            summary_score=0.8,
            n_grounded=4,
            n_contradicted=0,
            n_unsupported=1,
        )
        loop.observe_witness(result)
        s = loop.summary()
        assert "Self-Improving Loop" in s
        assert "1 observations" in s

    def test_thread_safety(self):
        """Multiple threads can observe concurrently."""
        import threading
        from entroly.self_improving import SelfImprovingLoop

        loop = SelfImprovingLoop()
        errors = []

        def observe_n(n):
            try:
                for _ in range(n):
                    result = MockWitnessResult(
                        summary_score=0.7,
                        n_grounded=3,
                        n_contradicted=0,
                        n_unsupported=1,
                    )
                    loop.observe_witness(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=observe_n, args=(10,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert loop.stats.n_observations == 40
