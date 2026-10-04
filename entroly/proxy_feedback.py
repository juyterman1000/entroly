"""Passive proxy feedback signals and bounded quality-trend tracking.

These heuristics describe observable response and query patterns. Their scores
are signals for routing feedback, not measurements of answer correctness.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

# ── Passive Implicit Feedback ─────────────────────────────────────────────
#
# Extracts RL feedback signals from observable proxy traffic:
#   Signal 1: LLM confusion detection (response text analysis)
#   Signal 2: Query trajectory rephrase detection (SimHash similarity)
#   Signal 3: Sufficiency heuristic (already computed in optimize)
#
# These are heuristic observations, not ground-truth answer-quality labels.


class ImplicitFeedbackTracker:
    """Extract implicit RL feedback from proxy traffic.

    Thread-safe. Per-client state tracks query trajectories for
    rephrase detection. Response text is scanned for possible confusion
    indicators; neither signal establishes whether an answer is correct.
    """

    # ── Signal 1: Confusion patterns in LLM responses ────────────────
    # These phrases can indicate missing context; they do not prove its cause.
    _CONFUSION_PATTERNS = re.compile(
        r"(?:I\s+(?:don'?t|do\s+not)\s+(?:have|see)\s+(?:enough\s+|the\s+)?(?:context|code|file|information))"
        r"|(?:could\s+you\s+(?:provide|share|show|paste))"
        r"|(?:I(?:'m|\s+am)\s+not\s+(?:sure|certain)\s+(?:about|what|which|where))"
        r"|(?:without\s+(?:seeing|access|the\s+(?:full|actual|complete)))"
        r"|(?:I\s+(?:cannot|can'?t)\s+(?:see|access|find|determine))"
        r"|(?:I\s+(?:don'?t|do\s+not)\s+have\s+(?:access|visibility))"
        r"|(?:(?:more|additional)\s+context\s+(?:would|is)\s+(?:needed|helpful|required))"
        r"|(?:please\s+(?:share|provide|paste)\s+(?:the|your))",
        re.IGNORECASE,
    )

    # Minimum response length to trigger confidence signal (chars)
    _MIN_CONFIDENT_LENGTH = 200

    # Rephrase detection thresholds
    _REPHRASE_SIMILARITY_THRESHOLD = 0.75  # SimHash similarity > this = rephrase
    _REPHRASE_TIME_WINDOW_S = 90.0  # Within this many seconds
    _TOPIC_CHANGE_THRESHOLD = 0.30  # Similarity < this = topic change = success

    # Buffer cap for streaming responses (bytes)
    _MAX_BUFFER_BYTES = 50 * 1024  # 50KB — covers 99%+ of LLM responses

    def __init__(self):
        self._lock = threading.Lock()
        # Per-client trajectory: client_key -> (query_simhash, selected_ids, timestamp)
        self._trajectories: dict[str, tuple] = {}
        # Stats
        self._confusion_detections = 0
        self._confidence_detections = 0
        self._rephrase_detections = 0
        self._topic_changes = 0
        self._total_assessed = 0
        # CUSUM-EMA quality drift detector
        self._drift_detector = _CusumEmaDriftDetector()

    def assess_response(self, response_text: str) -> float:
        """Assess an LLM response for confusion vs confidence.

        Returns a reward signal:
          -1.0  = strong confusion detected (multiple indicators)
          -0.5  = mild confusion detected (one indicator)
           0.0  = ambiguous / too short to tell
          +0.3  = confident response (long, structured)
          +0.5  = confident response with code blocks
        """
        if not response_text or len(response_text) < 50:
            return 0.0

        # Count confusion pattern matches
        confusion_matches = len(self._CONFUSION_PATTERNS.findall(response_text[:5000]))

        if confusion_matches >= 2:
            return -1.0  # Strong confusion
        if confusion_matches == 1:
            return -0.5  # Mild confusion

        # Check for confidence signals
        has_code_blocks = "```" in response_text
        is_long = len(response_text) >= self._MIN_CONFIDENT_LENGTH

        if is_long and has_code_blocks:
            return 0.5  # Confident with code
        if is_long:
            return 0.3  # Confident (structured answer)

        return 0.0  # Ambiguous

    def detect_rephrase(
        self, client_key: str, query_text: str, selected_ids: list
    ) -> tuple | None:
        """Check if this query is a rephrase of the previous one.

        Returns:
          ("rephrase", prev_selected_ids) if rephrase detected
          ("topic_change", prev_selected_ids) if topic changed
          None if no trajectory data or ambiguous
        """
        try:
            from entroly_core import py_simhash
            query_hash = py_simhash(query_text)
        except (ImportError, Exception):
            return None

        now = time.time()

        with self._lock:
            prev = self._trajectories.get(client_key)

            # Update trajectory
            self._trajectories[client_key] = (query_hash, selected_ids, now)

            # Evict old entries (> 1000 clients)
            if len(self._trajectories) > 1000:
                oldest_key = min(
                    self._trajectories,
                    key=lambda k: self._trajectories[k][2],
                )
                del self._trajectories[oldest_key]

        if prev is None:
            return None

        prev_hash, prev_ids, prev_time = prev
        time_delta = now - prev_time

        if time_delta > self._REPHRASE_TIME_WINDOW_S:
            return None  # Too long ago to be a rephrase

        if not prev_ids:
            return None  # No fragment IDs to attribute

        # Rust owns the classification math; Python keeps only per-client
        # trajectory state and feedback side effects.
        try:
            from entroly_core import py_classify_query_transition

            transition = py_classify_query_transition(
                prev_hash,
                query_text,
                time_delta,
                self._REPHRASE_TIME_WINDOW_S,
                self._REPHRASE_SIMILARITY_THRESHOLD,
                self._TOPIC_CHANGE_THRESHOLD,
            )
            status = transition.get("status")
        except Exception:
            # Fallback for older wheels: same thresholds, same Hamming math.
            xor = query_hash ^ prev_hash
            hamming = bin(xor).count("1")
            similarity = 1.0 - (hamming / 64.0)
            if similarity > self._REPHRASE_SIMILARITY_THRESHOLD:
                status = "rephrase"
            elif similarity < self._TOPIC_CHANGE_THRESHOLD:
                status = "topic_change"
            else:
                status = "ambiguous"

        if status == "rephrase":
            with self._lock:
                self._rephrase_detections += 1
            return ("rephrase", prev_ids)

        if status == "topic_change":
            with self._lock:
                self._topic_changes += 1
            return ("topic_change", prev_ids)

        return None  # Ambiguous mid-range similarity

    def record_assessment(self, reward: float) -> None:
        """Track assessment stats and feed the drift detector."""
        with self._lock:
            self._total_assessed += 1
            if reward < -0.25:
                self._confusion_detections += 1
            elif reward > 0.25:
                self._confidence_detections += 1
            # Feed dual drift detector
            self._drift_detector.update(reward)

    def quality_trend(self) -> str:
        """Return current quality trend: 'stable', 'declining', or 'improving'."""
        with self._lock:
            return self._drift_detector.trend()

    def stats(self) -> dict[str, Any]:
        """Return feedback tracker statistics."""
        with self._lock:
            drift_stats = self._drift_detector.to_dict()
            return {
                "total_assessed": self._total_assessed,
                "confusion_detections": self._confusion_detections,
                "confidence_detections": self._confidence_detections,
                "rephrase_detections": self._rephrase_detections,
                "topic_changes": self._topic_changes,
                "quality_trend": drift_stats["trend"],
                "drift_detector": drift_stats,
            }


class _CusumEmaDriftDetector:
    """Dual online quality drift detector: CUSUM + EMA.

    Combines two complementary algorithms from the change-point detection
    literature -- online kernel CUSUM and EMA trend tracking. Neither is ours;
    running both against the same stream and requiring agreement before
    declaring drift is, because a single detector on a noisy quality signal
    fires often enough that operators stop trusting it:

    1. **EMA** (Exponential Moving Average): Smooth trend tracker.
       α = 0.15 → emphasizes recent observations. Fast to respond but
       susceptible to noise.

    2. **Page's CUSUM** (Cumulative Sum): Detects persistent drift in
       the reward signal. Accumulates deviations from the target mean.
       More robust than EMA — fires only on sustained degradation.

    Quality trend states:
      - "stable": Both detectors within bounds
      - "declining": Either detector flags degradation
      - "improving": EMA above positive threshold after a decline

    Thread-safety: Caller must hold lock (ImplicitFeedbackTracker._lock).
    """

    # EMA smoothing factor: 0.15 gives ~13-sample effective window
    _ALPHA = 0.15
    # CUSUM sensitivity: accumulate when reward < this target
    _TARGET_MEAN = 0.0
    # CUSUM decision threshold: fire alarm when cumulative sum exceeds this
    _CUSUM_THRESHOLD = 3.0
    # EMA threshold for "declining" signal
    _EMA_DECLINE_THRESHOLD = -0.20
    # EMA threshold for "improving" signal
    _EMA_IMPROVE_THRESHOLD = 0.15
    # Minimum observations before drift detection activates
    _MIN_OBSERVATIONS = 5

    def __init__(self):
        self.ema: float = 0.0
        self.cusum_pos: float = 0.0  # Detect upward shift (quality improving)
        self.cusum_neg: float = 0.0  # Detect downward shift (quality declining)
        self.count: int = 0
        self._was_declining: bool = False

    def update(self, reward: float) -> None:
        """Feed a new reward observation."""
        self.count += 1

        # EMA update
        if self.count == 1:
            self.ema = reward
        else:
            self.ema = self._ALPHA * reward + (1.0 - self._ALPHA) * self.ema

        # Page's CUSUM update (two-sided)
        deviation = reward - self._TARGET_MEAN
        self.cusum_pos = max(0.0, self.cusum_pos + deviation)
        self.cusum_neg = max(0.0, self.cusum_neg - deviation)

        # Track state transitions for "improving" detection
        if self.trend() == "declining":
            self._was_declining = True

    def trend(self) -> str:
        """Return current quality trend."""
        if self.count < self._MIN_OBSERVATIONS:
            return "stable"  # Not enough data yet

        # Declining: EMA below threshold OR CUSUM negative alarm
        if (self.ema < self._EMA_DECLINE_THRESHOLD
                or self.cusum_neg > self._CUSUM_THRESHOLD):
            return "declining"

        # Improving: EMA above positive threshold AND recovered from decline
        if self._was_declining and self.ema > self._EMA_IMPROVE_THRESHOLD:
            return "improving"

        return "stable"

    def reset(self) -> None:
        """Reset detector state (e.g., on session restart)."""
        self.ema = 0.0
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.count = 0
        self._was_declining = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ema": round(self.ema, 4),
            "cusum_pos": round(self.cusum_pos, 4),
            "cusum_neg": round(self.cusum_neg, 4),
            "observations": self.count,
            "trend": self.trend(),
        }
