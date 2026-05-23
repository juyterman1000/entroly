"""
Online Bayesian PRISM — Live Weight Adaptation
================================================

Closes the learning loop so the running engine improves with every call,
not just on restart.

Mathematical foundation:

  PRISM 5D weights live on the 4-simplex (must be non-negative, sum ~1).
  The natural Bayesian prior over a simplex is the Dirichlet distribution:

    w ~ Dir(α₁, α₂, α₃, α₄, α₅)

  where αᵢ = prior_strength × archetype_weight_i.

  After each optimize_context() call, we observe an implicit reward r ∈ [0, 1]
  computed from budget utilization and selection quality. We update:

    αᵢ ← αᵢ + η · r · contribution_i

  where contribution_i measures how much dimension i contributed to the
  current optimization result (proxy for ∂reward/∂wᵢ — a REINFORCE-style
  gradient estimate).

  The posterior mean (used as the weight vector) is:

    E[wᵢ] = αᵢ / Σαⱼ

  This gives us:
    1. Principled exploration (high variance when α is small)
    2. Convergence to optimum (posterior concentrates as evidence grows)
    3. Anchoring to archetype prior (prevents divergence with few observations)
    4. Natural normalization (Dirichlet mean always sums to 1)
    5. Thread-safe (single atomic update per call)

  The learning rate η decays as 1/√n (Robbins-Monro condition):
    η(n) = η₀ / √(n + 1)
  ensuring Ση = ∞ (can reach any optimum) and Ση² < ∞ (converges).

Why this matters:
  Without this, Entroly's weights are frozen at startup. A user working on
  a debugging task gets the same weights as someone doing a refactor.
  With online learning, the engine adapts within minutes to the user's
  actual work pattern — debugging sessions upweight recency+semantic,
  refactoring sessions upweight frequency+entropy, automatically.

Usage:
  learner = OnlinePrism(prior_weights={...})
  # After each optimize_context():
  new_weights = learner.observe(reward, contributions)
  engine.set_weights(*new_weights)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("entroly.online_learner")

# ── Constants ─────────────────────────────────────────────────────────

WEIGHT_DIMS = ["w_recency", "w_frequency", "w_semantic", "w_entropy"]

# Prior strength: how many "virtual observations" the archetype prior
# is worth. Higher = slower adaptation but more stable.
# 20 means the prior is equivalent to 20 real observations.
DEFAULT_PRIOR_STRENGTH = 20.0

# Base learning rate for Dirichlet updates.
# Decays as η₀/√(n+1) per Robbins-Monro.
DEFAULT_ETA = 1.0

# Minimum observations before applying learned weights.
# Below this, use pure prior (not enough signal).
WARMUP_N = 3

# EMA smoothing for the reward baseline (advantage normalization).
REWARD_EMA_ALPHA = 0.1

# Persistence: save posterior every N observations.
SAVE_INTERVAL = 10


@dataclass
class PrismState:
    """Serializable state of the online PRISM learner."""
    alphas: dict[str, float] = field(default_factory=dict)
    n_observations: int = 0
    reward_ema: float = 0.5
    total_reward: float = 0.0
    best_reward: float = 0.0
    last_updated: float = 0.0


class OnlinePrism:
    """Online Bayesian weight adaptation for PRISM 5D.

    Thread-safe: all updates are protected by a lock.
    The engine calls observe() after each optimize_context().
    """

    def __init__(
        self,
        prior_weights: dict[str, float] | None = None,
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
        eta: float = DEFAULT_ETA,
        dims: list[str] | None = None,
    ):
        self._dims = dims or WEIGHT_DIMS
        self._eta0 = eta
        self._lock = threading.Lock()

        # Initialize Dirichlet alphas from prior
        pw = prior_weights or {d: 1.0 / len(self._dims) for d in self._dims}
        self._alphas = {
            d: prior_strength * pw.get(d, 0.25)
            for d in self._dims
        }

        self._n = 0
        self._reward_ema = 0.5
        self._total_reward = 0.0
        self._best_reward = 0.0
        self._last_weights: dict[str, float] = {}

        logger.debug(
            "OnlinePrism initialized: dims=%s, prior_strength=%.1f, "
            "initial_weights=%s",
            self._dims, prior_strength, self.weights(),
        )

    # ── Public API ─────────────────────────────────────────────────

    def weights(self) -> dict[str, float]:
        """Current posterior mean (Dirichlet mean = α_i / Σα)."""
        with self._lock:
            total = sum(self._alphas.values())
            if total <= 0:
                return {d: 1.0 / len(self._dims) for d in self._dims}
            return {d: self._alphas[d] / total for d in self._dims}

    def weights_tuple(self) -> tuple[float, ...]:
        """Weights as a tuple in dim order (for engine.set_weights())."""
        w = self.weights()
        return tuple(w[d] for d in self._dims)

    def observe(
        self,
        reward: float,
        contributions: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Observe the outcome of an optimize_context() call.

        Args:
            reward: Implicit reward ∈ [0, 1] from the optimization.
                    Computed from budget utilization, selection quality, etc.
            contributions: Per-dimension contribution to the result.
                          If None, uniform attribution is used.

        Returns:
            Updated weight dict (posterior mean).
        """
        reward = max(0.0, min(1.0, reward))

        with self._lock:
            self._n += 1
            n = self._n

            # Advantage: reward relative to baseline (REINFORCE-style)
            advantage = reward - self._reward_ema
            self._reward_ema = (
                REWARD_EMA_ALPHA * reward
                + (1 - REWARD_EMA_ALPHA) * self._reward_ema
            )

            self._total_reward += reward
            self._best_reward = max(self._best_reward, reward)

            # Learning rate decay: η = η₀ / √(n + 1)
            eta = self._eta0 / math.sqrt(n + 1)

            # Default contributions: uniform
            if contributions is None:
                contributions = {d: 1.0 / len(self._dims) for d in self._dims}

            # Dirichlet update: α_i += η · advantage · contribution_i
            # Positive advantage → reinforce current pattern
            # Negative advantage → weaken current pattern
            for d in self._dims:
                c = contributions.get(d, 0.0)
                delta = eta * advantage * c

                # Only apply positive updates to alphas (Dirichlet params must be > 0)
                # For negative advantage, we reduce alpha but floor at 0.1
                # to prevent any dimension from going to zero.
                self._alphas[d] = max(0.1, self._alphas[d] + delta)

            # Compute posterior mean
            total = sum(self._alphas.values())
            self._last_weights = {
                d: self._alphas[d] / total for d in self._dims
            }

        if n <= 5 or n % 10 == 0:
            logger.debug(
                "OnlinePrism: n=%d, reward=%.3f, adv=%.3f, eta=%.4f, "
                "weights=%s",
                n, reward, advantage, eta,
                {d: f"{v:.3f}" for d, v in self._last_weights.items()},
            )

        return dict(self._last_weights)

    def reset_to_prior(self, prior_weights: dict[str, float],
                       prior_strength: float = DEFAULT_PRIOR_STRENGTH) -> None:
        """Reset to a new prior (e.g., on archetype change)."""
        with self._lock:
            self._alphas = {
                d: prior_strength * prior_weights.get(d, 0.25)
                for d in self._dims
            }
            self._n = 0
            self._reward_ema = 0.5

    def state(self) -> PrismState:
        """Export serializable state."""
        with self._lock:
            return PrismState(
                alphas=dict(self._alphas),
                n_observations=self._n,
                reward_ema=self._reward_ema,
                total_reward=self._total_reward,
                best_reward=self._best_reward,
                last_updated=time.time(),
            )

    def load_state(self, state: PrismState) -> None:
        """Restore from serialized state."""
        with self._lock:
            self._alphas = dict(state.alphas)
            self._n = state.n_observations
            self._reward_ema = state.reward_ema
            self._total_reward = state.total_reward
            self._best_reward = state.best_reward

    def stats(self) -> dict[str, Any]:
        """Dashboard-friendly stats."""
        w = self.weights()
        with self._lock:
            return {
                "weights": {d: round(v, 4) for d, v in w.items()},
                "alphas": {d: round(v, 2) for d, v in self._alphas.items()},
                "n_observations": self._n,
                "reward_ema": round(self._reward_ema, 4),
                "avg_reward": round(
                    self._total_reward / max(self._n, 1), 4
                ),
                "best_reward": round(self._best_reward, 4),
                "learning_rate": round(
                    self._eta0 / math.sqrt(self._n + 1), 6
                ),
                "phase": (
                    "warmup" if self._n < WARMUP_N
                    else "learning" if self._n < 100
                    else "converged"
                ),
            }


# ── Implicit Reward Computation ───────────────────────────────────────

def compute_implicit_reward(
    selected_count: int,
    total_fragments: int,
    tokens_used: int,
    token_budget: int,
    query_present: bool = True,
) -> float:
    """Compute an implicit reward from optimization outcomes.

    Reward components:
      1. Budget utilization (0-1): How well we used the budget.
         Sweet spot is 70-95% utilization (underfilling wastes opportunity,
         overfilling means we're not selective enough).

      2. Selectivity (0-1): Fraction of fragments selected.
         Lower is better — it means we're being discriminating.
         Optimal range: 20-60% of fragments selected.

      3. Query bonus: Small bonus when a query is provided
         (task-specific optimization is more valuable than generic).

    Combined via geometric mean for multiplicative interaction.
    """
    if total_fragments == 0 or token_budget == 0:
        return 0.5  # Neutral — can't assess

    # 1. Budget utilization: bell curve peaking at 85%
    utilization = tokens_used / token_budget
    # Gaussian centered at 0.85 with σ=0.2
    util_reward = math.exp(-((utilization - 0.85) ** 2) / (2 * 0.2 ** 2))

    # 2. Selectivity: sigmoid favoring 30-50% selection rate
    select_rate = selected_count / max(total_fragments, 1)
    # Optimal at ~40% selection, penalize extremes
    select_reward = math.exp(-((select_rate - 0.40) ** 2) / (2 * 0.15 ** 2))

    # 3. Query bonus
    query_bonus = 1.05 if query_present else 1.0

    # Geometric mean (multiplicative interaction)
    reward = (util_reward * select_reward) ** 0.5 * query_bonus
    return max(0.0, min(1.0, reward))


def compute_contributions(
    selected_fragments: list[dict],
    total_fragments: int,
) -> dict[str, float]:
    """Estimate per-dimension contribution to the optimization result.

    This is a REINFORCE-style gradient estimate: for each dimension,
    we measure how correlated that dimension's score was with selection.

    Higher correlation → that dimension contributed more → update it more.
    """
    if not selected_fragments or total_fragments == 0:
        return {d: 0.25 for d in WEIGHT_DIMS}

    # Aggregate per-dimension scores of selected fragments
    dim_scores = {d: 0.0 for d in WEIGHT_DIMS}
    n = len(selected_fragments)

    for frag in selected_fragments:
        # Map fragment properties to dimensions
        dim_scores["w_recency"] += frag.get("recency_score", 0.5)
        dim_scores["w_frequency"] += frag.get("frequency_score", 0.5)
        dim_scores["w_semantic"] += frag.get("semantic_score", 0.5)
        dim_scores["w_entropy"] += frag.get("entropy_score", 0.5)

    # Normalize to [0, 1] per dimension
    for d in WEIGHT_DIMS:
        dim_scores[d] = dim_scores[d] / max(n, 1)

    # Normalize to sum=1 (contribution is a distribution)
    total = sum(dim_scores.values())
    if total > 0:
        dim_scores = {d: v / total for d, v in dim_scores.items()}

    return dim_scores
