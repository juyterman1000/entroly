"""Immediate in-process Traffic Value for fresh Entroly users.

The durable executive dashboard already keeps Today / 7D / 30D / 60D / 90D
and All Time rollups in ``ValueTracker``. This module adds a deliberately
non-durable **This session** view on top of the same Traffic Receipt stream so a
user can see value after a few requests instead of waiting for a longer window.

No second savings ledger or database is introduced. The session accumulator is
bounded, content-blind, process-local, and resets when the proxy process
restarts. Durable accounting remains owned by :mod:`entroly.proxy_traffic_value`.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from . import proxy_traffic_value as _value

_MAX_SESSION_SEEN = 2048
_SESSION_LOCK = threading.RLock()
_SESSION_STARTED_AT = time.time()
_SESSION_METRICS: dict[str, Any] = _value._empty_metrics()
_SESSION_SEEN: deque[str] = deque()
_SESSION_SEEN_SET: set[str] = set()

_ORIGINAL_RECORD = _value.record_traffic_value_receipt
_ORIGINAL_SNAPSHOT = _value.build_traffic_value_snapshot


def _remember_receipt(receipt_id: str) -> bool:
    """Return True only once per bounded in-process receipt id."""
    if not receipt_id:
        return False
    with _SESSION_LOCK:
        if receipt_id in _SESSION_SEEN_SET:
            return False
        if len(_SESSION_SEEN) >= _MAX_SESSION_SEEN:
            old = _SESSION_SEEN.popleft()
            _SESSION_SEEN_SET.discard(old)
        _SESSION_SEEN.append(receipt_id)
        _SESSION_SEEN_SET.add(receipt_id)
        return True


def _record_session_receipt(receipt: Any) -> bool:
    """Accumulate one verified Traffic Receipt into this process session."""
    if not getattr(receipt, "verify", lambda: False)():
        return False
    receipt_id = str(getattr(receipt, "receipt_id", "") or "")
    if not _remember_receipt(receipt_id):
        return False
    delta = _value._receipt_delta(receipt)
    with _SESSION_LOCK:
        _value._accumulate(_SESSION_METRICS, delta)
    return True


def _record_with_session(
    receipt: Any,
    *,
    tracker: Any | None = None,
) -> bool:
    # Session value should remain useful even if durable telemetry is temporarily
    # unavailable. Idempotency is enforced independently by the bounded receipt
    # id set above; durable idempotency remains owned by ValueTracker.
    _record_session_receipt(receipt)
    return _ORIGINAL_RECORD(receipt, tracker=tracker)


def _session_rollup(*, now: float | None = None) -> dict[str, Any]:
    now_ts = time.time() if now is None else float(now)
    with _SESSION_LOCK:
        result: dict[str, Any] = {
            "key": "session",
            "label": "This session",
            **_value._empty_metrics(),
        }
        _value._accumulate(result, _SESSION_METRICS)
    result["session_started_at"] = _SESSION_STARTED_AT
    result["session_elapsed_seconds"] = max(0, int(now_ts - _SESSION_STARTED_AT))
    result["window_start"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(_SESSION_STARTED_AT)
    )
    result["window_end"] = "now"
    return _value._finalize_metrics(result)


def _snapshot_with_session(
    tracker: Any | None = None,
    *,
    today: Any | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    snapshot = _ORIGINAL_SNAPSHOT(tracker, today=today, now=now)
    session = _session_rollup(now=now)

    windows = dict(snapshot.get("windows", {}))
    windows["session"] = session
    snapshot["windows"] = windows

    order = [str(key) for key in snapshot.get("window_order", [])]
    snapshot["window_order"] = ["session", *[key for key in order if key != "session"]]
    snapshot["session_started_at"] = _SESSION_STARTED_AT
    snapshot["session_elapsed_seconds"] = session["session_elapsed_seconds"]

    # Immediate proof matters most during the first day. After that, retain the
    # age-adaptive durable default while keeping This session one click away.
    if session["requests_observed"] > 0 and int(snapshot.get("installed_days", 0) or 0) == 0:
        snapshot["default_window"] = "session"

    truth = dict(snapshot.get("truth", {}))
    truth["session"] = (
        "This session is an in-process rollup of verified Traffic Receipts. "
        "It resets on proxy restart and is not added again to All Time."
    )
    snapshot["truth"] = truth
    return snapshot


def _reset_session_state_for_tests(*, started_at: float | None = None) -> None:
    global _SESSION_STARTED_AT, _SESSION_METRICS
    with _SESSION_LOCK:
        _SESSION_STARTED_AT = time.time() if started_at is None else float(started_at)
        _SESSION_METRICS = _value._empty_metrics()
        _SESSION_SEEN.clear()
        _SESSION_SEEN_SET.clear()


def install_session_value() -> None:
    """Patch the existing value module's public seams exactly once."""
    if _value.record_traffic_value_receipt is not _record_with_session:
        _value.record_traffic_value_receipt = _record_with_session
    if _value.build_traffic_value_snapshot is not _snapshot_with_session:
        _value.build_traffic_value_snapshot = _snapshot_with_session


install_session_value()


__all__ = [
    "_reset_session_state_for_tests",
    "_session_rollup",
    "install_session_value",
]
