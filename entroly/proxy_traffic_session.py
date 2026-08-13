"""Compatibility seams for process-local AI Traffic Value.

``This session`` is now implemented natively by :mod:`entroly.proxy_traffic_value`.
This module intentionally performs no monkey patching. It remains as a small
compatibility surface for integrations/tests that imported the earlier session
module while all accounting and snapshot ownership stays in one place.
"""

from __future__ import annotations

from typing import Any

from . import proxy_traffic_value as _value


def _record_session_receipt(receipt: Any) -> bool:
    return _value._record_session_receipt(receipt)


def _record_with_session(
    receipt: Any,
    *,
    tracker: Any | None = None,
) -> bool:
    return _value.record_traffic_value_receipt(receipt, tracker=tracker)


def _session_rollup(*, now: float | None = None) -> dict[str, Any]:
    return _value._session_rollup(now=now)


def _snapshot_with_session(
    tracker: Any | None = None,
    *,
    today: Any | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    return _value.build_traffic_value_snapshot(tracker, today=today, now=now)


def _reset_session_state_for_tests(*, started_at: float | None = None) -> None:
    _value._reset_session_state_for_tests(started_at=started_at)


def install_session_value() -> None:
    """Compatibility no-op: session value is already native in the value module."""
    return None


__all__ = [
    "_record_session_receipt",
    "_record_with_session",
    "_reset_session_state_for_tests",
    "_session_rollup",
    "_snapshot_with_session",
    "install_session_value",
]
