"""
RAVS → PRISM Outcome Bridge — Honest Reward Injection
========================================================

The core algorithmic contribution: retroactive Bayesian posterior
correction using delayed honest outcomes.

Problem statement:
  OnlinePrism currently learns from compute_implicit_reward(), which
  estimates reward from budget utilization and selectivity. This is a
  *proxy*: it measures "did we fill the budget well?" not "did the
  answer succeed?" The proxy reward is available immediately (at
  optimize_context return time) but has unknown correlation with actual
  task success.

  RAVS V1 honest outcomes (test_result, ci_result, command_exit,
  user_acceptance) arrive *later* — sometimes seconds, sometimes
  minutes after the optimization call. But they are ground truth.

Solution: Hindsight Posterior Correction (HPC)
  When a RAVS honest outcome arrives for request_id R:

    1. Look up the PRISM observation that was recorded for R
       (we cached the contributions and implicit reward at optimize time).

    2. Compute the "honest advantage": the difference between the honest
       reward and the implicit reward that was originally applied.

    3. Apply a corrective Dirichlet update using the cached contributions
       and the honest advantage. This is mathematically equivalent to:

         α_i ← α_i - η_old · advantage_old · c_i   (undo implicit)
         α_i ← α_i + η_new · advantage_new · c_i   (apply honest)

       But since undoing is numerically unstable with decayed learning
       rates, we instead apply a *differential* update:

         Δ_advantage = honest_advantage - implicit_advantage
         α_i ← α_i + η_correction · Δ_advantage · c_i

       The correction learning rate η_correction is scaled by a
       confidence factor based on outcome strength (strong > medium).

    4. The posterior now reflects honest outcomes, not just proxy signals.
       Over time, PRISM weights converge on configurations that produce
       REAL successes, not just "good budget utilization."

Why this matters:
  Without this bridge, PRISM optimizes for proxy quality — a debugging
  session that fills 85% of the budget looks identical to one that
  solves the bug. With honest outcomes, PRISM learns that certain weight
  configurations actually produce correct code, passing tests, accepted
  diffs. The optimization surface shifts from "context shape" to
  "task success."

Integration:
  The bridge is a pure-function module. It does not touch production
  routing. It reads from the RAVS event log and writes corrections
  into OnlinePrism's posterior. The engine calls
  ``bridge.on_honest_outcome()`` whenever a RAVS honest signal arrives
  via the MCP tools (record_test_result, record_command_exit, etc.).

Thread safety:
  All mutable state is in OnlinePrism (already thread-safe).
  The bridge holds an LRU cache of recent observations, protected
  by its own lock.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("entroly.ravs.outcome_bridge")


# ── Cached observation from optimize_context time ─────────────────────

@dataclass
class CachedObservation:
    """What PRISM saw at optimize_context() time for a given request."""

    request_id: str
    timestamp: float

    # The implicit reward that was already applied
    implicit_reward: float

    # The per-dimension contributions (REINFORCE gradient estimate)
    contributions: dict[str, float]

    # The PRISM weights at observation time (for diagnostics)
    weights_at_time: dict[str, float]

    # The advantage that was applied (reward - baseline at the time)
    implicit_advantage: float


# ── Honest reward mapping ─────────────────────────────────────────────

# Map RAVS outcome (event_type, value) to a reward signal.
# Strong signals get full weight; medium get reduced weight.
_OUTCOME_REWARD: dict[tuple[str, str], float] = {
    # Strong: ground truth
    ("test_result", "passed"):       1.0,
    ("test_result", "failed"):       0.0,
    ("ci_result", "passed"):         1.0,
    ("ci_result", "failed"):         0.0,
    ("command_exit", "success"):     0.9,   # exit code 0 ≠ full success
    ("command_exit", "failure"):     0.1,   # exit code ≠ 0 ≠ total failure
    ("user_acceptance", "accepted"): 1.0,
    ("user_acceptance", "rejected"): 0.0,
    # Medium: behavioral inference
    ("retry_event", "failure"):      0.2,   # rephrase suggests prior was wrong
    ("topic_change", "success"):     0.8,   # moved on suggests prior was ok
    ("escalation_event", "failure"): 0.15,  # escalation suggests prior was inadequate
}

# Confidence scaling by strength level.
# Strong signals get full correction weight; medium is discounted.
_STRENGTH_CONFIDENCE: dict[str, float] = {
    "strong": 1.0,
    "medium": 0.5,
    "weak":   0.0,    # never correct from self-report
}


def honest_reward(event_type: str, value: str, strength: str) -> tuple[float, float] | None:
    """Map a RAVS outcome to (reward, confidence).

    Returns None if the outcome shouldn't drive learning (weak strength
    or unknown event_type/value pair).
    """
    confidence = _STRENGTH_CONFIDENCE.get(strength, 0.0)
    if confidence <= 0.0:
        return None

    reward = _OUTCOME_REWARD.get((event_type, value))
    if reward is None:
        # Unknown pair — don't guess
        return None

    return (reward, confidence)


# ── The Bridge ────────────────────────────────────────────────────────


class OutcomeBridge:
    """Connects RAVS honest outcomes to PRISM's Dirichlet posterior.

    Lifecycle:
      1. Engine calls ``cache_observation()`` at optimize_context() time.
      2. Engine calls ``on_honest_outcome()`` when a RAVS honest signal
         arrives (via record_test_result, record_ci_result, etc.).
      3. The bridge computes the differential advantage and applies a
         corrective update to the OnlinePrism instance.

    The bridge holds an LRU cache of recent observations. Observations
    older than ``max_age_s`` are evicted to bound memory.
    """

    def __init__(
        self,
        prism: Any,           # OnlinePrism instance
        *,
        max_cache_size: int = 500,
        max_age_s: float = 3600.0,      # 1 hour
        correction_eta: float = 0.8,    # base correction learning rate
    ):
        self._prism = prism
        self._max_cache_size = max_cache_size
        self._max_age_s = max_age_s
        self._correction_eta = correction_eta
        self._cache: dict[str, CachedObservation] = {}
        self._lock = threading.Lock()

        # Counters for diagnostics
        self._corrections_applied = 0
        self._corrections_skipped = 0
        self._cache_misses = 0

    def cache_observation(
        self,
        request_id: str,
        implicit_reward: float,
        implicit_advantage: float,
        contributions: dict[str, float],
        weights: dict[str, float],
    ) -> None:
        """Cache the PRISM observation at optimize_context() time.

        Called by the engine immediately after OnlinePrism.observe().
        """
        now = time.time()
        obs = CachedObservation(
            request_id=request_id,
            timestamp=now,
            implicit_reward=implicit_reward,
            contributions=dict(contributions),
            weights_at_time=dict(weights),
            implicit_advantage=implicit_advantage,
        )
        with self._lock:
            self._cache[request_id] = obs
            self._evict_stale(now)

    def on_honest_outcome(
        self,
        request_id: str,
        event_type: str,
        value: str,
        strength: str,
    ) -> dict[str, Any] | None:
        """Process a RAVS honest outcome and apply posterior correction.

        Returns a diagnostic dict if correction was applied, None otherwise.
        """
        # 1. Map to (reward, confidence)
        mapped = honest_reward(event_type, value, strength)
        if mapped is None:
            self._corrections_skipped += 1
            return None

        honest_r, confidence = mapped

        # 2. Look up cached observation
        with self._lock:
            obs = self._cache.pop(request_id, None)

        if obs is None:
            self._cache_misses += 1
            logger.debug(
                "OutcomeBridge: no cached observation for %s (cache miss)",
                request_id,
            )
            return None

        # 3. Compute differential advantage
        # honest_advantage = honest_reward - baseline (we use the same
        # EMA baseline that was current at observation time, accessed
        # via the PRISM instance).
        baseline = self._prism._reward_ema  # noqa: SLF001
        honest_advantage = honest_r - baseline
        delta_advantage = honest_advantage - obs.implicit_advantage

        if abs(delta_advantage) < 1e-6:
            # Implicit and honest agree — no correction needed
            self._corrections_skipped += 1
            return None

        # 4. Compute correction learning rate
        # Scale by confidence (strong=1.0, medium=0.5) and by a
        # decay factor based on how many observations PRISM has seen
        # (we want corrections to have less impact as PRISM converges).
        n = self._prism._n  # noqa: SLF001
        eta = self._correction_eta * confidence / math.sqrt(max(n, 1) + 1)

        # 5. Apply corrective Dirichlet update
        with self._prism._lock:  # noqa: SLF001
            for dim, alpha in self._prism._alphas.items():  # noqa: SLF001
                c = obs.contributions.get(dim, 0.0)
                correction = eta * delta_advantage * c
                self._prism._alphas[dim] = max(0.1, alpha + correction)  # noqa: SLF001

        self._corrections_applied += 1

        result = {
            "request_id": request_id,
            "event_type": event_type,
            "value": value,
            "strength": strength,
            "honest_reward": round(honest_r, 4),
            "implicit_reward": round(obs.implicit_reward, 4),
            "delta_advantage": round(delta_advantage, 4),
            "correction_eta": round(eta, 6),
            "confidence": confidence,
            "n_at_correction": n,
        }

        logger.debug(
            "OutcomeBridge: corrected PRISM posterior for %s: "
            "honest=%.3f implicit=%.3f Δadv=%.3f η=%.4f",
            request_id, honest_r, obs.implicit_reward,
            delta_advantage, eta,
        )

        return result

    def _evict_stale(self, now: float) -> None:
        """Evict entries older than max_age_s + LRU if over capacity."""
        # Age-based eviction
        stale = [
            rid for rid, obs in self._cache.items()
            if (now - obs.timestamp) > self._max_age_s
        ]
        for rid in stale:
            del self._cache[rid]

        # LRU eviction if still over capacity
        if len(self._cache) > self._max_cache_size:
            by_age = sorted(
                self._cache.items(),
                key=lambda kv: kv[1].timestamp,
            )
            to_remove = len(self._cache) - self._max_cache_size
            for rid, _ in by_age[:to_remove]:
                del self._cache[rid]

    def stats(self) -> dict[str, Any]:
        """Diagnostic stats for dashboard/logging."""
        with self._lock:
            cache_size = len(self._cache)
        return {
            "corrections_applied": self._corrections_applied,
            "corrections_skipped": self._corrections_skipped,
            "cache_misses": self._cache_misses,
            "cache_size": cache_size,
            "total_events": (
                self._corrections_applied
                + self._corrections_skipped
                + self._cache_misses
            ),
        }
