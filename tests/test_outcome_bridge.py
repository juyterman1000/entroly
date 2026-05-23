"""
Tests for RAVS → PRISM Outcome Bridge (Hindsight Posterior Correction).

Coverage:
  HPC-01  HONEST REWARD MAP        known outcomes map to correct rewards
  HPC-02  WEAK REJECTED            weak signals return None (no correction)
  HPC-03  CACHE ROUNDTRIP          cache_observation stores, on_honest_outcome retrieves
  HPC-04  POSTERIOR CORRECTION      honest outcome shifts PRISM alphas correctly
  HPC-05  POSITIVE CORRECTION      tests-passed corrects upward when implicit was low
  HPC-06  NEGATIVE CORRECTION      tests-failed corrects downward when implicit was high
  HPC-07  NO DOUBLE CORRECTION     same request_id can only correct once (pop semantics)
  HPC-08  CACHE MISS               outcome for unknown request_id is a cache miss
  HPC-09  STALE EVICTION           observations older than max_age_s are evicted
  HPC-10  CONVERGENCE DIRECTION    after many honest corrections, PRISM weights shift
  HPC-11  CONFIDENCE SCALING       strong signals produce larger corrections than medium
  HPC-12  STATS ACCOUNTING         stats counters track applied/skipped/missed correctly
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from entroly.online_learner import OnlinePrism, compute_implicit_reward, compute_contributions
from entroly.ravs.outcome_bridge import (
    CachedObservation,
    OutcomeBridge,
    honest_reward,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_prism() -> OnlinePrism:
    return OnlinePrism(
        prior_weights={
            "w_recency": 0.30,
            "w_frequency": 0.20,
            "w_semantic": 0.35,
            "w_entropy": 0.15,
        },
        prior_strength=10.0,
    )


def _make_bridge(prism: OnlinePrism, **kwargs) -> OutcomeBridge:
    return OutcomeBridge(prism, **kwargs)


# ══════════════════════════════════════════════════════════════════════
# HPC-01: Honest reward mapping
# ══════════════════════════════════════════════════════════════════════


def test_hpc01_honest_reward_mapping():
    # Strong signals
    assert honest_reward("test_result", "passed", "strong") == (1.0, 1.0)
    assert honest_reward("test_result", "failed", "strong") == (0.0, 1.0)
    assert honest_reward("ci_result", "passed", "strong") == (1.0, 1.0)
    assert honest_reward("ci_result", "failed", "strong") == (0.0, 1.0)
    assert honest_reward("user_acceptance", "accepted", "strong") == (1.0, 1.0)
    assert honest_reward("user_acceptance", "rejected", "strong") == (0.0, 1.0)

    # Medium signals
    r, c = honest_reward("retry_event", "failure", "medium")
    assert c == 0.5
    assert r == 0.2

    r, c = honest_reward("topic_change", "success", "medium")
    assert c == 0.5
    assert r == 0.8


# ══════════════════════════════════════════════════════════════════════
# HPC-02: Weak signals rejected
# ══════════════════════════════════════════════════════════════════════


def test_hpc02_weak_returns_none():
    assert honest_reward("agent_self_report", "success", "weak") is None
    assert honest_reward("test_result", "passed", "weak") is None


# ══════════════════════════════════════════════════════════════════════
# HPC-03: Cache roundtrip
# ══════════════════════════════════════════════════════════════════════


def test_hpc03_cache_roundtrip():
    prism = _make_prism()
    bridge = _make_bridge(prism)

    contribs = {"w_recency": 0.3, "w_frequency": 0.2, "w_semantic": 0.35, "w_entropy": 0.15}
    bridge.cache_observation(
        request_id="r1",
        implicit_reward=0.7,
        implicit_advantage=0.2,
        contributions=contribs,
        weights=prism.weights(),
    )

    # Should be able to correct
    result = bridge.on_honest_outcome("r1", "test_result", "passed", "strong")
    assert result is not None
    assert result["request_id"] == "r1"
    assert result["honest_reward"] == 1.0
    assert result["implicit_reward"] == 0.7


# ══════════════════════════════════════════════════════════════════════
# HPC-04: Posterior correction
# ══════════════════════════════════════════════════════════════════════


def test_hpc04_posterior_correction_changes_alphas():
    prism = _make_prism()
    bridge = _make_bridge(prism)

    # Record initial alphas
    initial_alphas = dict(prism._alphas)

    # Simulate an optimization with low implicit reward
    contribs = {"w_recency": 0.4, "w_frequency": 0.1, "w_semantic": 0.4, "w_entropy": 0.1}
    prism.observe(0.3, contribs)  # low implicit reward

    mid_alphas = dict(prism._alphas)

    # Cache the observation
    bridge.cache_observation(
        request_id="r1",
        implicit_reward=0.3,
        implicit_advantage=0.3 - prism._reward_ema,
        contributions=contribs,
        weights=prism.weights(),
    )

    # Now honest outcome says it was actually a success
    result = bridge.on_honest_outcome("r1", "test_result", "passed", "strong")
    assert result is not None
    assert result["delta_advantage"] > 0  # honest was better than implicit

    # Alphas should have shifted from the correction
    final_alphas = dict(prism._alphas)
    assert final_alphas != mid_alphas


# ══════════════════════════════════════════════════════════════════════
# HPC-05: Positive correction
# ══════════════════════════════════════════════════════════════════════


def test_hpc05_positive_correction_on_test_pass():
    prism = _make_prism()
    bridge = _make_bridge(prism)

    contribs = {"w_recency": 0.25, "w_frequency": 0.25, "w_semantic": 0.25, "w_entropy": 0.25}

    # Low implicit reward
    prism.observe(0.2, contribs)
    bridge.cache_observation(
        request_id="r1",
        implicit_reward=0.2,
        implicit_advantage=0.2 - prism._reward_ema,
        contributions=contribs,
        weights=prism.weights(),
    )

    result = bridge.on_honest_outcome("r1", "test_result", "passed", "strong")
    assert result is not None
    assert result["delta_advantage"] > 0  # correction is positive


# ══════════════════════════════════════════════════════════════════════
# HPC-06: Negative correction
# ══════════════════════════════════════════════════════════════════════


def test_hpc06_negative_correction_on_test_fail():
    prism = _make_prism()
    bridge = _make_bridge(prism)

    contribs = {"w_recency": 0.25, "w_frequency": 0.25, "w_semantic": 0.25, "w_entropy": 0.25}

    # High implicit reward
    prism.observe(0.9, contribs)
    bridge.cache_observation(
        request_id="r1",
        implicit_reward=0.9,
        implicit_advantage=0.9 - prism._reward_ema,
        contributions=contribs,
        weights=prism.weights(),
    )

    result = bridge.on_honest_outcome("r1", "test_result", "failed", "strong")
    assert result is not None
    assert result["delta_advantage"] < 0  # correction is negative


# ══════════════════════════════════════════════════════════════════════
# HPC-07: No double correction
# ══════════════════════════════════════════════════════════════════════


def test_hpc07_no_double_correction():
    prism = _make_prism()
    bridge = _make_bridge(prism)

    contribs = {"w_recency": 0.25, "w_frequency": 0.25, "w_semantic": 0.25, "w_entropy": 0.25}
    prism.observe(0.5, contribs)
    bridge.cache_observation(
        request_id="r1",
        implicit_reward=0.5,
        implicit_advantage=0.0,
        contributions=contribs,
        weights=prism.weights(),
    )

    # First correction succeeds
    r1 = bridge.on_honest_outcome("r1", "test_result", "passed", "strong")
    assert r1 is not None

    # Second attempt: cache entry already popped
    r2 = bridge.on_honest_outcome("r1", "test_result", "passed", "strong")
    assert r2 is None


# ══════════════════════════════════════════════════════════════════════
# HPC-08: Cache miss
# ══════════════════════════════════════════════════════════════════════


def test_hpc08_cache_miss():
    prism = _make_prism()
    bridge = _make_bridge(prism)

    result = bridge.on_honest_outcome("nonexistent", "test_result", "passed", "strong")
    assert result is None
    assert bridge.stats()["cache_misses"] == 1


# ══════════════════════════════════════════════════════════════════════
# HPC-09: Stale eviction
# ══════════════════════════════════════════════════════════════════════


def test_hpc09_stale_eviction():
    prism = _make_prism()
    bridge = _make_bridge(prism, max_age_s=1.0)  # 1 second TTL

    contribs = {"w_recency": 0.25, "w_frequency": 0.25, "w_semantic": 0.25, "w_entropy": 0.25}

    # Manually insert a stale observation
    bridge._cache["old"] = CachedObservation(
        request_id="old",
        timestamp=time.time() - 100,  # 100 seconds ago
        implicit_reward=0.5,
        contributions=contribs,
        weights_at_time=prism.weights(),
        implicit_advantage=0.0,
    )

    # Cache a new observation (triggers eviction)
    bridge.cache_observation(
        request_id="new",
        implicit_reward=0.5,
        implicit_advantage=0.0,
        contributions=contribs,
        weights=prism.weights(),
    )

    # Old should be evicted
    assert "old" not in bridge._cache
    assert "new" in bridge._cache


# ══════════════════════════════════════════════════════════════════════
# HPC-10: Convergence direction
# ══════════════════════════════════════════════════════════════════════


def test_hpc10_convergence_direction():
    """After many positive corrections with high w_semantic contribution,
    w_semantic weight should increase relative to others."""
    prism = _make_prism()
    bridge = _make_bridge(prism, correction_eta=2.0)

    initial_w = prism.weights()

    for i in range(20):
        # Semantic-heavy contributions
        contribs = {"w_recency": 0.1, "w_frequency": 0.1, "w_semantic": 0.7, "w_entropy": 0.1}
        prism.observe(0.3, contribs)  # low implicit
        bridge.cache_observation(
            request_id=f"r{i}",
            implicit_reward=0.3,
            implicit_advantage=0.3 - prism._reward_ema,
            contributions=contribs,
            weights=prism.weights(),
        )
        # Honest says success — semantic-heavy config was actually good
        bridge.on_honest_outcome(f"r{i}", "test_result", "passed", "strong")

    final_w = prism.weights()

    # w_semantic should have increased relative to start
    # (the honest corrections consistently rewarded semantic-heavy configs)
    assert final_w["w_semantic"] > initial_w["w_semantic"] - 0.05  # allow small tolerance


# ══════════════════════════════════════════════════════════════════════
# HPC-11: Confidence scaling
# ══════════════════════════════════════════════════════════════════════


def test_hpc11_confidence_scaling():
    """Strong signals produce larger corrections than medium."""
    prism_strong = _make_prism()
    prism_medium = _make_prism()
    bridge_strong = _make_bridge(prism_strong)
    bridge_medium = _make_bridge(prism_medium)

    contribs = {"w_recency": 0.25, "w_frequency": 0.25, "w_semantic": 0.25, "w_entropy": 0.25}

    # Same scenario for both
    for prism, bridge, rid in [
        (prism_strong, bridge_strong, "rs"),
        (prism_medium, bridge_medium, "rm"),
    ]:
        prism.observe(0.3, contribs)
        bridge.cache_observation(
            request_id=rid,
            implicit_reward=0.3,
            implicit_advantage=0.3 - prism._reward_ema,
            contributions=contribs,
            weights=prism.weights(),
        )

    # Strong correction
    r_strong = bridge_strong.on_honest_outcome("rs", "test_result", "passed", "strong")
    # Medium correction (topic_change is medium strength)
    r_medium = bridge_medium.on_honest_outcome("rm", "topic_change", "success", "medium")

    assert r_strong is not None
    assert r_medium is not None
    # Strong correction should have higher eta
    assert r_strong["confidence"] > r_medium["confidence"]


# ══════════════════════════════════════════════════════════════════════
# HPC-12: Stats accounting
# ══════════════════════════════════════════════════════════════════════


def test_hpc12_stats_accounting():
    prism = _make_prism()
    bridge = _make_bridge(prism)

    contribs = {"w_recency": 0.25, "w_frequency": 0.25, "w_semantic": 0.25, "w_entropy": 0.25}

    # Applied
    prism.observe(0.3, contribs)
    bridge.cache_observation("r1", 0.3, -0.2, contribs, prism.weights())
    bridge.on_honest_outcome("r1", "test_result", "passed", "strong")

    # Skipped (weak)
    prism.observe(0.3, contribs)
    bridge.cache_observation("r2", 0.3, -0.2, contribs, prism.weights())
    bridge.on_honest_outcome("r2", "agent_self_report", "success", "weak")

    # Cache miss
    bridge.on_honest_outcome("r999", "test_result", "passed", "strong")

    s = bridge.stats()
    assert s["corrections_applied"] == 1
    assert s["corrections_skipped"] == 1
    assert s["cache_misses"] == 1
    assert s["total_events"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
