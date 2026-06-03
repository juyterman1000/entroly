"""
Safe Adaptive Behavior: Promotion Gate
======================================

Evaluates shadow policies against the live policy.
Only promotes a shadow policy if it meets non-inferiority criteria on
success rate, repair rate, and retry rate. Also implements auto-rollback.
"""

import collections
import logging
import threading
from typing import Any

logger = logging.getLogger("entroly.promotion_gate")

class PromotionGate:
    def __init__(
        self,
        holdout_window: int = 50,
        epsilon: float = 0.05,  # Max allowable success rate degradation
        delta: float = 0.20,    # Max allowable increase in repair/retry rate
    ):
        self.holdout_window = holdout_window
        self.epsilon = epsilon
        self.delta = delta

        self._lock = threading.Lock()

        # State
        self._live_outcomes = collections.deque(maxlen=self.holdout_window)
        self._shadow_outcomes = collections.deque(maxlen=self.holdout_window)
        self._post_promotion_outcomes = collections.deque(maxlen=20)

        self._shadow_weights: dict[str, float] | None = None
        self._previous_live_weights: dict[str, float] | None = None

    def set_shadow_candidate(self, weights: dict[str, float]):
        """Propose a new shadow weight configuration."""
        with self._lock:
            self._shadow_weights = dict(weights)
            self._shadow_outcomes.clear()
            self._live_outcomes.clear()

    def record_live_outcome(self, success: bool, repair_count: int, retry_count: int):
        """Record an outcome for the currently active live policy."""
        with self._lock:
            self._live_outcomes.append((success, repair_count, retry_count))
            # Also record for rollback monitoring if we recently promoted
            if self._previous_live_weights is not None:
                self._post_promotion_outcomes.append((success, repair_count, retry_count))

    def record_shadow_outcome(self, success: bool, repair_count: int, retry_count: int):
        """Record an outcome for the shadow policy."""
        with self._lock:
            self._shadow_outcomes.append((success, repair_count, retry_count))

    def evaluate_promotion(self) -> dict[str, float] | None:
        """Evaluate if the shadow policy should be promoted to live.
        
        Returns the new weights if promoted, None otherwise.
        """
        with self._lock:
            if not self._shadow_weights:
                return None
            if len(self._shadow_outcomes) < self.holdout_window or len(self._live_outcomes) < self.holdout_window:
                return None

            live_success = sum(1 for o in self._live_outcomes if o[0]) / self.holdout_window
            shadow_success = sum(1 for o in self._shadow_outcomes if o[0]) / self.holdout_window

            live_repair = sum(o[1] for o in self._live_outcomes) / self.holdout_window
            shadow_repair = sum(o[1] for o in self._shadow_outcomes) / self.holdout_window

            live_retry = sum(o[2] for o in self._live_outcomes) / self.holdout_window
            shadow_retry = sum(o[2] for o in self._shadow_outcomes) / self.holdout_window

            logger.debug(
                f"Promotion evaluate: Live (succ={live_success:.2f}, rep={live_repair:.2f}, ret={live_retry:.2f}) vs "
                f"Shadow (succ={shadow_success:.2f}, rep={shadow_repair:.2f}, ret={shadow_retry:.2f})"
            )

            # Criteria:
            # 1. Non-inferior success rate
            if shadow_success < live_success - self.epsilon:
                return None
            # 2. No regression in repairs
            if shadow_repair > live_repair * (1 + self.delta) and shadow_repair > live_repair + 0.1:
                return None
            # 3. No regression in retries
            if shadow_retry > live_retry * (1 + self.delta) and shadow_retry > live_retry + 0.1:
                return None

            # Promote!
            promoted_weights = self._shadow_weights
            self._shadow_weights = None
            self._shadow_outcomes.clear()
            self._post_promotion_outcomes.clear()

            logger.info("PromotionGate: Shadow policy promoted to Live.")
            return promoted_weights

    def check_rollback(self) -> dict[str, float] | None:
        """Check if we need to rollback after a recent promotion."""
        with self._lock:
            if not self._previous_live_weights or len(self._post_promotion_outcomes) < 20:
                return None

            post_repair = sum(o[1] for o in self._post_promotion_outcomes) / 20.0
            post_retry = sum(o[2] for o in self._post_promotion_outcomes) / 20.0
            post_success = sum(1 for o in self._post_promotion_outcomes if o[0]) / 20.0

            # If repair/retry rates explode, or success plummets
            if post_repair > 1.5 or post_retry > 1.5 or post_success < 0.5:
                logger.warning(f"PromotionGate: Auto-rollback triggered. repair={post_repair:.2f}, retry={post_retry:.2f}, success={post_success:.2f}")
                rollback = self._previous_live_weights
                self._previous_live_weights = None
                self._post_promotion_outcomes.clear()
                return rollback

            return None

    def commit_promotion(self, live_weights: dict[str, float]):
        """Acknowledge promotion was applied to store rollback state."""
        with self._lock:
            self._previous_live_weights = dict(live_weights)
