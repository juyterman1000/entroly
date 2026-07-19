"""Closed-loop self-improvement: WITNESS → PRISM → better selection.

This module wires three existing Entroly subsystems into a flywheel:

    1. QCCR/knapsack selects context using PRISM weights
    2. Agent uses the selected context to answer
    3. WITNESS verifies the answer against the selected evidence
    4. This module converts the WITNESS verdict into a PRISM reward
    5. PRISM updates weights → next selection is better

Neither Hermes nor OpenClaw can replicate this because they lack:
  - Evidence-locked compression (what was selected, what was omitted)
  - WITNESS (proof-carrying verification of the answer)
  - PRISM (online Bayesian weight learning)

The more you use Entroly, the better it gets at selecting context
for YOUR codebase.  This is the self-improving moat.

Usage::

    from entroly.self_improving import SelfImprovingLoop

    loop = SelfImprovingLoop()

    # After each optimize_context + LLM response:
    loop.observe(
        witness_result=witness_result,       # from WITNESS verification
        receipt=receipt,                     # from context selection
        recovered_omissions=recovered_ids,   # user recovered these = bad selection
    )

    # The loop automatically updates PRISM weights.
    # Next optimize_context() uses improved weights.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("entroly.self_improving")


# ── Reward signals ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class SelectionFeedback:
    """A single feedback observation from the closed loop.

    Combines multiple reward signals into a single PRISM reward.
    """

    timestamp: float
    source: str  # "witness", "receipt_recovery", "budget_utilization", "explicit"

    # WITNESS signals
    witness_score: float = 0.0       # WitnessResult.summary_score ∈ [0, 1]
    n_grounded: int = 0              # Claims verified as grounded
    n_contradicted: int = 0          # Claims contradicted by evidence
    n_unsupported: int = 0           # Claims without evidence support
    evidence_adequacy: float = 0.0   # Mean evidence adequacy across claims

    # Receipt recovery signals
    n_recovered: int = 0             # User recovered omitted spans = bad selection
    n_total_omissions: int = 0       # Total omissions available

    # Budget utilization signals
    tokens_used: int = 0             # Tokens actually used
    token_budget: int = 0            # Token budget available
    utilization: float = 0.0         # tokens_used / token_budget

    # Combined reward ∈ [0, 1] for PRISM
    reward: float = 0.0

    # Per-dimension contribution estimates for REINFORCE gradient
    contributions: dict[str, float] = field(default_factory=dict)


def compute_reward(
    *,
    witness_score: float = 0.5,
    n_grounded: int = 0,
    n_contradicted: int = 0,
    n_unsupported: int = 0,
    evidence_adequacy: float = 0.5,
    n_recovered: int = 0,
    n_total_omissions: int = 0,
    utilization: float = 0.5,
) -> float:
    """Compute a combined PRISM reward from multiple feedback signals.

    Reward decomposition:
        40% — WITNESS verification score (grounded claims are good)
        25% — Evidence adequacy (relevant evidence was selected)
        20% — Recovery penalty (recovered omissions = selection was wrong)
        15% — Budget utilization (use the budget efficiently)

    Returns:
        Reward ∈ [0, 1].
    """
    # WITNESS component: high score = claims are grounded
    r_witness = max(0.0, min(1.0, witness_score))

    # Contradiction penalty: each contradiction is a strong negative signal
    total_claims = n_grounded + n_contradicted + n_unsupported
    if total_claims > 0:
        contradiction_rate = n_contradicted / total_claims
        # Contradictions are much worse than unsupported claims.
        # A contradiction means we selected WRONG context.
        r_witness = r_witness * (1.0 - 2.0 * contradiction_rate)
        r_witness = max(0.0, r_witness)

    # Evidence adequacy component
    r_adequacy = max(0.0, min(1.0, evidence_adequacy))

    # Recovery penalty: if user recovered omissions, our selection was bad
    if n_total_omissions > 0 and n_recovered > 0:
        recovery_rate = n_recovered / n_total_omissions
        # More recoveries = worse selection.  Full recovery = reward 0.
        r_recovery = 1.0 - recovery_rate
    else:
        # No recoveries = good selection (or user didn't need to check)
        r_recovery = 1.0

    # Budget utilization: sweet spot is 70-95%.
    # Under 50% = wasted budget.  Over 95% = too tight.
    if utilization < 0.5:
        r_budget = utilization / 0.5  # Linear ramp 0→1 from 0→50%
    elif utilization <= 0.95:
        r_budget = 1.0  # Sweet spot
    else:
        r_budget = max(0.5, 1.0 - (utilization - 0.95) * 10)  # Gentle penalty

    # Weighted combination
    reward = (
        0.40 * r_witness
        + 0.25 * r_adequacy
        + 0.20 * r_recovery
        + 0.15 * r_budget
    )

    return max(0.0, min(1.0, reward))


def estimate_contributions(
    *,
    witness_score: float = 0.5,
    evidence_adequacy: float = 0.5,
    utilization: float = 0.5,
    n_recovered: int = 0,
) -> dict[str, float]:
    """Estimate per-dimension PRISM contributions for REINFORCE gradient.

    Maps WITNESS signals to which PRISM dimensions likely caused
    the outcome:
        - High evidence adequacy → semantic similarity was right
        - Good utilization → entropy scoring was right
        - Grounded claims → recency (recent context was relevant)
        - No recoveries → frequency (popular fragments were right)

    These are noisy proxies, but Dirichlet posterior concentrates
    with enough observations.
    """
    # Start uniform
    c = {
        "w_recency": 0.25,
        "w_frequency": 0.25,
        "w_semantic": 0.25,
        "w_entropy": 0.25,
    }

    # High evidence adequacy → semantic matching worked
    if evidence_adequacy > 0.6:
        c["w_semantic"] += 0.15
        c["w_entropy"] -= 0.05

    # Good budget utilization → entropy scoring calibrated well
    if 0.7 <= utilization <= 0.95:
        c["w_entropy"] += 0.10

    # No recoveries → the frequency signal (popular = important) was right
    if n_recovered == 0:
        c["w_frequency"] += 0.10

    # High WITNESS score → recent context was relevant
    if witness_score > 0.7:
        c["w_recency"] += 0.10

    # Normalize to sum to 1
    total = sum(c.values())
    return {k: v / total for k, v in c.items()}


# ── Closed loop ──────────────────────────────────────────────────────────

@dataclass
class LoopStats:
    """Aggregate statistics for the self-improving loop."""

    n_observations: int = 0
    mean_reward: float = 0.0
    best_reward: float = 0.0
    worst_reward: float = 1.0
    mean_witness_score: float = 0.0
    total_recoveries: int = 0
    total_contradictions: int = 0
    improvement_trend: float = 0.0  # Positive = getting better
    last_10_mean: float = 0.0


class SelfImprovingLoop:
    """Wires WITNESS verdicts + receipt recoveries → PRISM weight updates.

    This is the flywheel that makes Entroly self-improving:

        selection → answer → verification → reward → better selection

    Thread-safe.  Persists feedback history to disk for cross-session
    learning continuity.
    """

    def __init__(
        self,
        *,
        learner: Any | None = None,
        persist_dir: Path | str | None = None,
        max_history: int = 1000,
    ) -> None:
        """Initialize the self-improving loop.

        Args:
            learner: An ``OnlinePrism`` instance.  If None, creates one
                with default prior weights.
            persist_dir: Directory to persist feedback history.
                Defaults to ``~/.entroly/feedback/``.
            max_history: Maximum feedback entries to keep in memory.
        """
        self._lock = threading.Lock()
        self._max_history = max_history
        self._history: list[SelectionFeedback] = []
        self._stats = LoopStats()

        # Lazy import to avoid circular deps
        if learner is not None:
            self._learner = learner
        else:
            self._learner = None  # Created lazily on first observe()

        if persist_dir is not None:
            self._persist_dir = Path(persist_dir)
        else:
            self._persist_dir = Path.home() / ".entroly" / "feedback"

    def _ensure_learner(self):
        """Lazy-initialize the OnlinePrism learner."""
        if self._learner is None:
            from .online_learner import OnlinePrism

            self._learner = OnlinePrism()
        return self._learner

    # ── Core feedback API ─────────────────────────────────────────────

    def observe_witness(
        self,
        witness_result: Any,
        *,
        tokens_used: int = 0,
        token_budget: int = 0,
    ) -> float:
        """Observe a WITNESS verification result → update PRISM weights.

        This is the primary feedback signal.  Call after every
        ``apply_witness_policy()`` or ``WitnessAnalyzer.verify()``.

        Args:
            witness_result: A ``WitnessResult`` from the witness module.
            tokens_used: Tokens consumed by the selected context.
            token_budget: Token budget that was available.

        Returns:
            The computed reward ∈ [0, 1].
        """
        # Extract signals from WitnessResult
        summary_score = getattr(witness_result, "summary_score", 0.5)
        n_grounded = getattr(witness_result, "n_grounded", 0)
        n_contradicted = getattr(witness_result, "n_contradicted", 0)
        n_unsupported = getattr(witness_result, "n_unsupported", 0)

        # Mean evidence adequacy across certificates
        certs = getattr(witness_result, "certificates", [])
        if certs:
            adequacy = sum(
                getattr(c, "evidence_adequacy", 0.5) for c in certs
            ) / len(certs)
        else:
            adequacy = 0.5

        utilization = (tokens_used / token_budget) if token_budget > 0 else 0.5

        reward = compute_reward(
            witness_score=summary_score,
            n_grounded=n_grounded,
            n_contradicted=n_contradicted,
            n_unsupported=n_unsupported,
            evidence_adequacy=adequacy,
            utilization=utilization,
        )

        contributions = estimate_contributions(
            witness_score=summary_score,
            evidence_adequacy=adequacy,
            utilization=utilization,
        )

        feedback = SelectionFeedback(
            timestamp=time.time(),
            source="witness",
            witness_score=summary_score,
            n_grounded=n_grounded,
            n_contradicted=n_contradicted,
            n_unsupported=n_unsupported,
            evidence_adequacy=adequacy,
            tokens_used=tokens_used,
            token_budget=token_budget,
            utilization=utilization,
            reward=reward,
            contributions=contributions,
        )

        self._record(feedback)
        return reward

    def observe_recovery(
        self,
        n_recovered: int,
        n_total_omissions: int,
    ) -> float:
        """Observe receipt recovery events → penalize PRISM weights.

        When a user recovers omitted spans, that's a direct signal
        the selection was wrong.

        Args:
            n_recovered: Number of omissions the user recovered.
            n_total_omissions: Total omissions available in the receipt.

        Returns:
            The computed reward ∈ [0, 1] (lower when more recovered).
        """
        reward = compute_reward(
            n_recovered=n_recovered,
            n_total_omissions=n_total_omissions,
        )

        contributions = estimate_contributions(n_recovered=n_recovered)

        feedback = SelectionFeedback(
            timestamp=time.time(),
            source="receipt_recovery",
            n_recovered=n_recovered,
            n_total_omissions=n_total_omissions,
            reward=reward,
            contributions=contributions,
        )

        self._record(feedback)
        return reward

    def observe_explicit(self, reward: float, source: str = "explicit") -> None:
        """Record an explicit reward (e.g., from user feedback or tests).

        Args:
            reward: Reward value ∈ [0, 1].
            source: Description of the feedback source.
        """
        feedback = SelectionFeedback(
            timestamp=time.time(),
            source=source,
            reward=max(0.0, min(1.0, reward)),
        )
        self._record(feedback)

    # ── Internal ──────────────────────────────────────────────────────

    def _record(self, feedback: SelectionFeedback) -> None:
        """Record feedback, update PRISM, update stats."""
        with self._lock:
            # Update PRISM weights
            learner = self._ensure_learner()
            learner.observe(
                reward=feedback.reward,
                contributions=feedback.contributions or None,
            )

            # Append to history
            self._history.append(feedback)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            # Update stats
            self._update_stats(feedback)

            # Persist periodically
            if self._stats.n_observations % 10 == 0:
                self._persist()

        logger.info(
            "Self-improving loop: %s reward=%.3f "
            "(n=%d, mean=%.3f, trend=%+.3f)",
            feedback.source,
            feedback.reward,
            self._stats.n_observations,
            self._stats.mean_reward,
            self._stats.improvement_trend,
        )

    def _update_stats(self, feedback: SelectionFeedback) -> None:
        """Update aggregate statistics."""
        s = self._stats
        s.n_observations += 1
        n = s.n_observations

        # Running mean
        s.mean_reward = s.mean_reward + (feedback.reward - s.mean_reward) / n
        s.best_reward = max(s.best_reward, feedback.reward)
        s.worst_reward = min(s.worst_reward, feedback.reward)

        if feedback.source == "witness":
            s.mean_witness_score = (
                s.mean_witness_score + (feedback.witness_score - s.mean_witness_score) / n
            )
        s.total_recoveries += feedback.n_recovered
        s.total_contradictions += feedback.n_contradicted

        # Last 10 mean for trend detection
        last_10 = self._history[-10:]
        s.last_10_mean = sum(f.reward for f in last_10) / len(last_10) if last_10 else 0

        # Improvement trend: last 10 vs overall mean
        if n >= 20:
            s.improvement_trend = s.last_10_mean - s.mean_reward

    # ── Persistence ───────────────────────────────────────────────────

    def _persist(self) -> None:
        """Save recent feedback to disk."""
        try:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            path = self._persist_dir / "feedback_log.jsonl"
            # Append last 10 entries
            entries = self._history[-10:]
            with open(path, "a", encoding="utf-8") as f:
                for entry in entries:
                    record = {
                        "ts": entry.timestamp,
                        "src": entry.source,
                        "reward": round(entry.reward, 4),
                        "witness": round(entry.witness_score, 4),
                        "adequacy": round(entry.evidence_adequacy, 4),
                        "recovered": entry.n_recovered,
                        "contradicted": entry.n_contradicted,
                        "grounded": entry.n_grounded,
                    }
                    f.write(json.dumps(record) + "\n")
        except OSError:
            pass  # Persistence is best-effort.

    # ── Query ─────────────────────────────────────────────────────────

    @property
    def stats(self) -> LoopStats:
        """Current loop statistics."""
        with self._lock:
            return LoopStats(**{
                k: getattr(self._stats, k) for k in self._stats.__dataclass_fields__
            })

    @property
    def current_weights(self) -> dict[str, float]:
        """Current PRISM weights (posterior mean)."""
        return self._ensure_learner().weights()

    @property
    def is_improving(self) -> bool:
        """True if the last 10 observations are above the overall mean."""
        return self._stats.improvement_trend > 0.01

    def summary(self) -> str:
        """Human-readable summary of the self-improving loop."""
        s = self._stats
        w = self.current_weights
        trend = "improving" if self.is_improving else "stable" if abs(s.improvement_trend) < 0.02 else "degrading"

        lines = [
            f"Self-Improving Loop: {s.n_observations} observations, {trend}",
            f"  Mean reward:    {s.mean_reward:.3f} (best: {s.best_reward:.3f})",
            f"  Last 10 mean:   {s.last_10_mean:.3f} (trend: {s.improvement_trend:+.3f})",
            f"  WITNESS mean:   {s.mean_witness_score:.3f}",
            f"  Contradictions: {s.total_contradictions}",
            f"  Recoveries:     {s.total_recoveries}",
            f"  Current weights: {', '.join(f'{k}={v:.3f}' for k, v in w.items())}",
        ]
        return "\n".join(lines)
