"""
RetryCollector — detect query-rephrase / topic-change at request boundary.

Emits OutcomeEvents into the RAVS log for the *previous* request, based
on whether the *current* query looks like a rephrase (signals previous
attempt failed) or a topic change (signals previous attempt was good
enough to move on).

Reuses SimHash-based similarity detection (already implemented in
``entroly_core.py_simhash``). Falls back gracefully when the Rust core
isn't installed.

Important: this collector does NOT act on the signal — it only records.
The proxy's existing ``ImplicitFeedbackTracker`` keeps firing
``record_success/record_failure`` independently. This isolation ensures
RAVS instrumentation never silently changes production behavior; it
only adds an honest event stream that offline eval can consume.

Strength = "medium". Behavioral inference is reliable but not
ground-truth — the user might rephrase for clarity, not because the
prior answer was wrong.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from ..events import AppendOnlyEventLog, OutcomeEvent

logger = logging.getLogger(__name__)


# Tunables — same defaults as the proxy's ImplicitFeedbackTracker so
# the two systems agree on what counts as "rephrase" vs "topic change".
DEFAULT_TIME_WINDOW_S = 120.0
DEFAULT_SIMILARITY_THRESHOLD = 0.70   # Hamming-similarity on 64-bit SimHash


@dataclass
class _PriorRequest:
    request_id: str
    query_hash: int
    timestamp: float


class RetryCollector:
    """Tracks per-client request trajectories; emits retry/topic_change events.

    Thread-safe. Bounded memory (LRU eviction beyond ``max_clients``).
    """

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
        self._priors: dict[str, _PriorRequest] = {}
        self._lock = threading.Lock()

    def observe(
        self,
        *,
        client_key: str,
        request_id: str,
        query_text: str,
        timestamp: float | None = None,
    ) -> str | None:
        """Record this request and possibly emit an outcome for the prior.

        Returns the event_type emitted (``"retry_event"`` |
        ``"topic_change"`` | ``None``) so callers can log/inspect.
        """
        if not client_key or not request_id:
            return None
        ts = timestamp if timestamp is not None else time.time()
        qhash = self._simhash(query_text)
        if qhash is None:
            return None

        with self._lock:
            prior = self._priors.get(client_key)
            self._priors[client_key] = _PriorRequest(
                request_id=request_id,
                query_hash=qhash,
                timestamp=ts,
            )
            # LRU eviction
            if len(self._priors) > self._max_clients:
                oldest_key = min(
                    self._priors, key=lambda k: self._priors[k].timestamp,
                )
                if oldest_key != client_key:
                    self._priors.pop(oldest_key, None)

        if prior is None:
            return None

        time_delta = ts - prior.timestamp
        if time_delta < 0 or time_delta > self._time_window_s:
            return None

        # Hamming-similarity on 64-bit hashes
        xor = qhash ^ prior.query_hash
        hamming = bin(xor).count("1")
        similarity = 1.0 - (hamming / 64.0)

        if similarity > self._similarity_threshold:
            event_type = "retry_event"
            value = "failure"   # rephrase ⇒ prior likely failed
        else:
            event_type = "topic_change"
            value = "success"   # topic move-on ⇒ prior good enough

        try:
            self._log.write_outcome(OutcomeEvent(
                request_id=prior.request_id,
                timestamp=ts,
                event_type=event_type,
                value=value,
                strength="medium",
                source="retry_collector",
                include_in_default_training=True,
                metadata={
                    "similarity": round(similarity, 4),
                    "time_delta_s": round(time_delta, 2),
                },
            ))
        except Exception as e:
            logger.debug("RetryCollector: failed to write outcome: %s", e)
            return None
        return event_type

    @staticmethod
    def _simhash(text: str) -> int | None:
        """64-bit SimHash via Rust core, or pure-Python fallback.

        Both implementations agree to within statistical noise on long
        texts. Short queries (< ~5 tokens) have high variance — that's
        a fundamental property of SimHash, not a bug.
        """
        if not text:
            return 0
        try:
            from entroly_core import py_simhash
            return py_simhash(text)
        except ImportError:
            pass
        # Pure-Python fallback (matches server.py:99 algorithm)
        try:
            import hashlib
            tokens = [
                t for t in text.lower().split()
                if t.isalnum() or any(c.isalnum() for c in t)
            ]
            if not tokens:
                return 0
            features = (
                [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                 for i in range(len(tokens) - 2)]
                if len(tokens) >= 3 else tokens
            )
            bits = [0] * 64
            for feat in features:
                h = int(hashlib.md5(feat.encode("utf-8", errors="replace")).hexdigest(), 16)
                for i in range(64):
                    if (h >> i) & 1:
                        bits[i] += 1
                    else:
                        bits[i] -= 1
            out = 0
            for i in range(64):
                if bits[i] > 0:
                    out |= 1 << i
            return out
        except Exception:
            return None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tracked_clients": len(self._priors),
                "time_window_s": self._time_window_s,
                "similarity_threshold": self._similarity_threshold,
            }
