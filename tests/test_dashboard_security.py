from __future__ import annotations

import http.client
import json
import re
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import entroly.controls_html as controls_html
import entroly.dashboard as dashboard
import entroly.dashboard_security as security


@contextmanager
def _server(*, max_workers: int = 4):
    server = security.BoundedThreadingHTTPServer(
        ("127.0.0.1", 0),
        security.SafeDashboardHandler,
        max_workers=max_workers,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _request(
    server,
    method: str,
    path: str,
    *,
    body: bytes | str | None = None,
    headers: dict[str, str] | None = None,
    encode_chunked: bool = False,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        server.server_port,
        timeout=3,
    )
    request_headers = dict(headers or {})
    request_headers.setdefault("Host", f"127.0.0.1:{server.server_port}")
    connection.request(
        method,
        path,
        body=body,
        headers=request_headers,
        encode_chunked=encode_chunked,
    )
    response = connection.getresponse()
    payload = response.read()
    returned_headers = {key.casefold(): value for key, value in response.getheaders()}
    status = response.status
    connection.close()
    return status, returned_headers, payload


def test_security_boundary_is_active_and_contract_matched() -> None:
    assert security.install_dashboard_security()
    assert security._HARDENING_ERRORS == ()
    assert dashboard.DashboardHandler is security.SafeDashboardHandler
    assert dashboard.start_dashboard is security.start_dashboard


def test_control_token_is_unpredictable_header_safe_and_embedded_once() -> None:
    token = security.control_token()

    assert re.fullmatch(r"[A-Za-z0-9._~-]{32,512}", token)
    assert controls_html.CONTROLS_HTML.count(json.dumps(token)) == 1
    assert "X-Entroly-Control-Token" in controls_html.CONTROLS_HTML
    assert "window.renderRepos = function" in controls_html.CONTROLS_HTML
    assert "window.refreshLogs = async function" in controls_html.CONTROLS_HTML


@pytest.mark.parametrize(
    "value",
    [
        None,
        "short",
        "x" * 513,
        "x" * 31 + "\n",
        "<script>" + "x" * 32,
        "x" * 31 + " ",
    ],
)
def test_invalid_control_tokens_are_rejected(value) -> None:
    assert security._validated_control_token(value) is None


def test_display_copy_escapes_values_and_keys_without_mutating_input() -> None:
    source = {
        "<img src=x onerror=alert(1)>": "</script><script>alert(2)</script>",
        "safe": ["A&B", 7, True, None],
    }

    result = security._display_safe(source)

    assert "<" not in next(iter(result))
    assert "<script>" not in result[next(iter(result))]
    assert result["safe"] == ["A&amp;B", 7, True, None]
    assert next(iter(source)) == "<img src=x onerror=alert(1)>"


def test_display_copy_handles_cycles_depth_nonfinite_and_large_strings() -> None:
    cycle: list[object] = []
    cycle.append(cycle)
    payload = {
        "cycle": cycle,
        "nan": float("nan"),
        "infinity": float("inf"),
        "long": "<" + "x" * 20_000,
    }

    result = security._display_safe(payload)

    assert result["cycle"] == ["[cycle]"]
    assert result["nan"] == 0.0
    assert result["infinity"] == 0.0
    assert result["long"].startswith("&lt;")
    assert result["long"].endswith("…[truncated]")
    assert len(result["long"]) < 9_000


def test_metrics_route_never_returns_active_markup(monkeypatch) -> None:
    attack = "<img src=x onerror=fetch('/api/control/stop',{method:'POST'})>"
    monkeypatch.setattr(
        dashboard,
        "_get_full_snapshot",
        lambda: {
            "recent_requests": [{"model": attack, "query": attack}],
            "security": {"findings": [{"file": attack, "message": attack}]},
            "health": {"top_recommendation": attack},
            "explain": {"included": [{"source": attack, "reason": attack}]},
        },
    )

    with _server() as server:
        status, headers, body = _request(server, "GET", "/api/metrics")

    text = body.decode("utf-8")
    assert status == 200
    assert "<img" not in text
    assert "&lt;img" in text
    assert headers["content-length"] == str(len(body))
    assert headers["cache-control"].startswith("no-store")
    assert headers["cross-origin-resource-policy"] == "same-origin"
    assert headers["x-frame-options"] == "DENY"


def test_host_header_rebinding_is_rejected() -> None:
    with _server() as server:
        status, headers, body = _request(
            server,
            "GET",
            "/health",
            headers={"Host": f"attacker.example:{server.server_port}"},
        )

    assert status == 403
    assert b"untrusted dashboard request context" in body
    assert headers["content-length"] == str(len(body))


def test_cross_origin_browser_request_is_rejected() -> None:
    with _server() as server:
        status, headers, body = _request(
            server,
            "GET",
            "/api/metrics",
            headers={
                "Origin": "https://attacker.example",
                "Sec-Fetch-Site": "cross-site",
            },
        )

    assert status == 403
    assert b"untrusted dashboard request context" in body
    assert headers["content-length"] == str(len(body))


def test_missing_control_capability_cannot_mutate_state(monkeypatch) -> None:
    called = False

    def unexpected_post(_handler) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(security, "_ORIGINAL_DO_POST", unexpected_post)
    with _server() as server:
        status, headers, body = _request(
            server,
            "POST",
            "/api/control/stop",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )

    assert status == 403
    assert not called
    assert b"invalid control capability" in body
    assert headers["content-length"] == str(len(body))


def test_valid_same_origin_capability_reaches_control_handler(monkeypatch) -> None:
    called = False

    def accepted_post(handler) -> None:
        nonlocal called
        called = True
        handler._send_json(200, {"ok": True, "result": "accepted"})

    monkeypatch.setattr(security, "_ORIGINAL_DO_POST", accepted_post)
    with _server() as server:
        status, headers, body = _request(
            server,
            "POST",
            "/api/control/quality",
            body=b'{"mode":"balanced"}',
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Origin": f"http://127.0.0.1:{server.server_port}",
                "Sec-Fetch-Site": "same-origin",
                "X-Entroly-Control-Token": security.control_token(),
            },
        )

    assert status == 200
    assert called
    assert json.loads(body) == {"ok": True, "result": "accepted"}
    assert headers["content-length"] == str(len(body))


def test_wrong_content_type_and_chunked_controls_fail_before_mutation(monkeypatch) -> None:
    called = False

    def unexpected_post(_handler) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(security, "_ORIGINAL_DO_POST", unexpected_post)
    common = {"X-Entroly-Control-Token": security.control_token()}
    with _server() as server:
        wrong_type, wrong_headers, wrong_body = _request(
            server,
            "POST",
            "/api/control/bypass",
            body=b"enabled=true",
            headers={**common, "Content-Type": "application/x-www-form-urlencoded"},
        )
        chunked, chunked_headers, chunked_body = _request(
            server,
            "POST",
            "/api/control/bypass",
            body=b"{}",
            headers={
                **common,
                "Content-Type": "application/json",
                "Transfer-Encoding": "chunked",
            },
            encode_chunked=True,
        )

    assert wrong_type == 415
    assert wrong_headers["content-length"] == str(len(wrong_body))
    assert chunked == 400
    assert chunked_headers["content-length"] == str(len(chunked_body))
    assert b"chunked control requests are unsupported" in chunked_body
    assert not called


def test_oversized_request_target_is_rejected_before_route_dispatch() -> None:
    with _server() as server:
        status, headers, body = _request(
            server,
            "GET",
            "/" + "a" * 9000,
        )

    assert status == 414
    assert b"request target too long" in body
    assert headers["content-length"] == str(len(body))


class _FakeSocket:
    def __init__(self) -> None:
        self.sent = b""
        self.closed = False
        self.timeout = None

    def settimeout(self, value: float) -> None:
        self.timeout = value

    def sendall(self, payload: bytes) -> None:
        self.sent += payload

    def shutdown(self, _how: int) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True


def test_worker_saturation_returns_bounded_503_without_spawning_thread() -> None:
    server = security.BoundedThreadingHTTPServer(
        ("127.0.0.1", 0),
        security.SafeDashboardHandler,
        max_workers=1,
    )
    fake = _FakeSocket()
    try:
        assert server._worker_slots.acquire(blocking=False)
        server.process_request(fake, ("127.0.0.1", 12345))
    finally:
        server._worker_slots.release()
        server.server_close()

    assert b"503 Service Unavailable" in fake.sent
    assert b"dashboard_busy" in fake.sent
    assert fake.timeout == 1.0
    assert fake.closed


@pytest.mark.parametrize("workers", [0, -1, 129, True, "bad", float("nan")])
def test_invalid_worker_limits_fail_before_server_bind(workers) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        security.BoundedThreadingHTTPServer(
            ("127.0.0.1", 0),
            security.SafeDashboardHandler,
            max_workers=workers,
        )
