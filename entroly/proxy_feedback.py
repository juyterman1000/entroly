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
        # Monitor the change in the observed confusion rate.
        self._drift_detector = _FrozenBaselineDriftDetector()

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


class _FrozenBaselineDriftDetector:
    """Monitor the confusion heuristic against a fixed startup reference.

    Two complete, non-overlapping windows are needed to signal a material
    increase. This is a finite-window heuristic, with no claimed false-alarm
    guarantee for arbitrary correlated traffic. The EMA is diagnostic only.
    Calls are serialized by ImplicitFeedbackTracker.
    """

    _BASELINE_SIZE = 2048
    _WINDOW_SIZE = 1024
    _INCREASE = 0.10
    _RECOVERY_MARGIN = 0.05
    _ALPHA = 0.15

    def __init__(self) -> None:
        self.reset()

    def update(self, reward: float) -> None:
        self.count += 1
        self.ema = (
            reward if self.count == 1 else self.ema + self._ALPHA * (reward - self.ema)
        )
        confused = int(reward < -0.25)
        if self.count <= self._BASELINE_SIZE:
            self._baseline_confused += confused
            return
        self._window_count += 1
        self._window_confused += confused
        if self._window_count < self._WINDOW_SIZE:
            return

        self._last_window_rate = self._window_confused / self._WINDOW_SIZE
        delta = self._last_window_rate - self._baseline_confused / self._BASELINE_SIZE
        self._window_count = 0
        self._window_confused = 0
        if self._trend == "declining":
            self._consecutive = (
                self._consecutive + 1 if delta <= self._RECOVERY_MARGIN else 0
            )
            if self._consecutive == 2:
                self._trend = "improving"
                self._consecutive = 0
        else:
            if self._trend == "improving":
                self._trend = "stable"
            self._consecutive = self._consecutive + 1 if delta >= self._INCREASE else 0
            if self._consecutive == 2:
                self._trend = "declining"
                self._consecutive = 0

    def trend(self) -> str:
        return self._trend

    def reset(self) -> None:
        self.count = 0
        self.ema = 0.0
        self._baseline_confused = 0
        self._window_count = 0
        self._window_confused = 0
        self._last_window_rate: float | None = None
        self._consecutive = 0
        self._trend = "stable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "frozen_confusion_rate_v1",
            "ema": round(self.ema, 4),
            "observations": self.count,
            "baseline_ready": self.count >= self._BASELINE_SIZE,
            "baseline_confusion_rate": (
                self._baseline_confused / self._BASELINE_SIZE
                if self.count >= self._BASELINE_SIZE
                else None
            ),
            "latest_window_confusion_rate": self._last_window_rate,
            "trend": self.trend(),
        }
