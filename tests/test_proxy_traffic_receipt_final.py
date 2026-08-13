from __future__ import annotations

from starlette.requests import Request

from entroly.proxy_traffic_receipt_final import (
    _ensure_request_id,
    _nested_recovery_depth,
    _request_local_coverage_unavailable,
)


def _request(headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/messages",
            "raw_path": b"/v1/messages",
            "query_string": b"",
            "headers": list(headers or []),
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 9377),
        }
    )


def test_request_id_is_injected_once_and_visible_to_starlette_headers() -> None:
    request = _request()
    first = _ensure_request_id(request)
    second = _ensure_request_id(request)

    assert first
    assert second == first
    assert request.headers["x-request-id"] == first
    ids = [value for key, value in request.scope["headers"] if key.lower() == b"x-request-id"]
    assert len(ids) == 1


def test_existing_request_id_is_preserved() -> None:
    request = _request([(b"x-request-id", b"client-owned-id")])
    assert _ensure_request_id(request) == "client-owned-id"
    assert request.headers["x-request-id"] == "client-owned-id"


def test_nested_recovery_depth_supports_keyword_and_positional_calls() -> None:
    assert _nested_recovery_depth((), {}) == 0
    assert _nested_recovery_depth((), {"recovery_depth": 1}) == 1
    assert _nested_recovery_depth((None, "", "openai", None, "req", 2), {}) == 2


def test_shared_coverage_is_withheld_from_per_request_receipts() -> None:
    pct, source, risk = _request_local_coverage_unavailable(object())
    assert pct is None
    assert source == "withheld_shared_state"
    assert risk == "UNKNOWN"
