"""
EscalationCollector — detect "same query → different (stronger) model"
within a TTL window.

When a user re-asks the same query against a stronger model shortly
after a first attempt, that's a strong signal the first attempt was
inadequate. We emit ``escalation_event`` (medium strength) attached to
the FIRST request's id, with metadata recording the original/new model
and the time delta.

Strength = "medium" because:
  - It's behavioral inference, not ground truth (the user might be
    A/B-testing models, not signaling failure).
  - But it's stronger than agent self-report and stronger than rephrase
    detection (which can fire on benign clarification).

Why a separate collector from RetryCollector: rephrase detects "same
query, same model" (weak signal — could be clarification). Escalation
detects "same query, *different* model, where new model is stronger"
(model swap is an active, deliberate signal).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from ..events import AppendOnlyEventLog, OutcomeEvent

logger = logging.getLogger(__name__)


DEFAULT_TIME_WINDOW_S = 300.0  # 5 min — broader than retry: escalations take longer
DEFAULT_SIMILARITY_THRESHOLD = 0.70


@dataclass
class _Attempt:
    request_id: str
    query_hash: int
    model: str
    timestamp: float


class EscalationCollector:
    """Per-client tracker of (query_hash, model) attempts.

    Thread-safe. Bounded memory.
    """

    # Subset of known model orderings used to recognize "stronger". Conservative:
    # only emit escalation when we're confident the new model is stronger.
    # Sourced from RAVS ModelCapability table (see entroly/ravs/models.py if/when
    # extracted). Any unknown ordering returns "unknown" → no event.
    _STRENGTH_ORDER: dict[str, int] = {
        # OpenAI
        "gpt-4o-mini": 1,    "gpt-4o": 3,    "gpt-4-turbo": 3,
        "o1-mini": 4,        "o1": 5,
        "o3-mini": 4,        "o3": 5,
        # Anthropic
        "claude-haiku-4": 1, "claude-3-5-haiku": 1, "claude-haiku": 1, "haiku": 1,
        "claude-sonnet-4": 3, "claude-3-5-sonnet": 3, "claude-sonnet": 3, "sonnet": 3,
        "claude-opus-4": 5,  "claude-3-opus": 5, "claude-opus": 5, "opus": 5,
        # Google
        "gemini-2.5-flash": 1, "gemini-2.5-pro": 3,
    }

    def __init__(
        self,
        log: AppendOnlyEventLog,
        *,
        time_window_s: float = DEFAULT_TIME_WINDOW_S,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_clients: int = 1000,
    ):
        self._log = log
        self._time_window_s = time_window_s
        self._similarity_threshold = similarity_threshold
        self._max_clients = max_clients
        # client_key -> list of recent attempts (most recent first, capped at 4)
        self._history: dict[str, list[_Attempt]] = {}
        self._lock = threading.Lock()

    def observe(
        self,
        *,
        client_key: str,
        request_id: str,
        query_text: str,
        model: str,
        timestamp: float | None = None,
    ) -> str | None:
        """Record this attempt; emit escalation_event if the pattern matches."""
        if not client_key or not request_id or not model:
            return None
        ts = timestamp if timestamp is not None else time.time()
        qhash = self._simhash(query_text)
        if qhash is None:
            return None

        emitted: str | None = None
        with self._lock:
            attempts = self._history.setdefault(client_key, [])
            # Look for any prior attempt within window with similar query but
            # weaker model. Most-recent first wins (the most plausible original).
            for prior in attempts:
                if prior.request_id == request_id:
                    continue
                if (ts - prior.timestamp) > self._time_window_s:
                    continue
                xor = qhash ^ prior.query_hash
                hamming = bin(xor).count("1")
                similarity = 1.0 - (hamming / 64.0)
                if similarity < self._similarity_threshold:
                    continue
                # Same query, prior different model — is this an escalation?
                if prior.model == model:
                    continue  # same model → that's RetryCollector's job
                if not self._is_stronger(model, prior.model):
                    continue  # downgrade or unknown — don't emit (no signal)
                # ✓ Escalation pattern detected
                try:
                    self._log.write_outcome(OutcomeEvent(
                        request_id=prior.request_id,
                        timestamp=ts,
                        event_type="escalation_event",
                        value="failure",  # original wasn't sufficient
                        strength="medium",
                        source="escalation_collector",
                        include_in_default_training=True,
                        metadata={
                            "original_model": prior.model,
                            "escalated_to": model,
                            "similarity": round(similarity, 4),
                            "time_delta_s": round(ts - prior.timestamp, 2),
                        },
                    ))
                    emitted = "escalation_event"
                except Exception as e:
                    logger.debug("EscalationCollector: write failed: %s", e)
                break  # only emit once per observation

            # Append current attempt; cap history depth + LRU evict
            attempts.insert(0, _Attempt(
                request_id=request_id, query_hash=qhash,
                model=model, timestamp=ts,
            ))
            if len(attempts) > 4:
                attempts.pop()
            if len(self._history) > self._max_clients:
                oldest_key = min(
                    self._history,
                    key=lambda k: (
                        self._history[k][-1].timestamp if self._history[k] else 0.0
                    ),
                )
                if oldest_key != client_key:
                    self._history.pop(oldest_key, None)

        return emitted

    @classmethod
    def _is_stronger(cls, candidate: str, baseline: str) -> bool:
        """True iff ``candidate`` is known to be stronger than ``baseline``.

        Returns False on unknown models — never emits a false-positive
        escalation event from a model we don't have ordering info for.
        """
        c = cls._lookup_strength(candidate)
        b = cls._lookup_strength(baseline)
        if c is None or b is None:
            return False
        return c > b

    @classmethod
    def _lookup_strength(cls, model: str) -> int | None:
        if not model:
            return None
        m = model.lower().strip()
        if m in cls._STRENGTH_ORDER:
            return cls._STRENGTH_ORDER[m]
        # Substring match (e.g. "claude-3-5-sonnet-20240620" → "sonnet")
        for key, val in cls._STRENGTH_ORDER.items():
            if key in m:
                return val
        return None

    @staticmethod
    def _simhash(text: str) -> int | None:
        # Reuse the same impl as RetryCollector to keep similarity comparable
        from .retry import RetryCollector
        return RetryCollector._simhash(text)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tracked_clients": len(self._history),
                "time_window_s": self._time_window_s,
                "similarity_threshold": self._similarity_threshold,
            }
