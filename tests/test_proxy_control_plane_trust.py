from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

import entroly.proxy as proxy_module
from entroly.proxy_control_plane_safe import (
    _secure_catch_all,
    _secure_sidecar_guard,
)
from entroly.proxy_transport_safe import CircuitBreaker, _CURRENT_QUERY


def _request(
    *,
    method: str = "GET",
    path: str = "/stats",
    query: bytes = b"",
    body: bytes = b"",
    headers: dict[str, str] | None = None,
    client_host: str = "127.0.0.1",
    app=None,
) -> Request:
    header_pairs = [(b"host", b"127.0.0.1:9377")]
    for name, value in (headers or {}).items():
        header_pairs.append((name.lower().encode("ascii"), value.encode("latin-1")))
    if body and not any(name.lower() == "content-length" for name in (headers or {})):
        header_pairs.append((b"content-length", str(len(body)).encode("ascii")))
    messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": query,
            "headers": header_pairs,
            "client": (client_host, 43123),
            "server": ("127.0.0.1", 9377),
            "app": app,
        },
        receive,
    )


def test_control_plane_hardening_is_active() -> None:
    assert proxy_module._sidecar_guard is _secure_sidecar_guard
    assert proxy_module._catch_all is _secure_catch_all


def test_direct_loopback_same_origin_sidecar_request_is_allowed(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_SIDECAR_TOKEN", raising=False)

    async def handler(_request: Request):
        return JSONResponse({"ok": True})

    guarded = _secure_sidecar_guard(handler)
    response = asyncio.run(guarded(_request()))

    assert response.status_code == 200
    assert json.loads(response.body) == {"ok": True}
    assert response.headers["cache-control"].startswith("no-store")
    assert response.headers["x-content-type-options"] == "nosniff"


def test_reverse_proxy_headers_are_not_identity(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_SIDECAR_TOKEN", raising=False)
    called = False

    async def handler(_request: Request):
        nonlocal called
        called = True
        return JSONResponse({"ok": True})

    request = _request(headers={"X-Forwarded-For": "198.51.100.9"})
    response = asyncio.run(_secure_sidecar_guard(handler)(request))

    assert response.status_code == 403
    assert not called


def test_remote_client_is_denied_without_token(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_SIDECAR_TOKEN", raising=False)

    async def handler(_request: Request):
        raise AssertionError("remote request reached sidecar handler")

    response = asyncio.run(
        _secure_sidecar_guard(handler)(
            _request(client_host="198.51.100.17")
        )
    )

    assert response.status_code == 403


def test_configured_sidecar_token_is_mandatory_even_on_loopback(monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_SIDECAR_TOKEN", "control-plane-secret")

    async def handler(_request: Request):
        return JSONResponse({"ok": True})

    guarded = _secure_sidecar_guard(handler)
    denied = asyncio.run(guarded(_request()))
    allowed = asyncio.run(
        guarded(
            _request(
                headers={"X-Entroly-Sidecar-Token": "control-plane-secret"},
                client_host="198.51.100.17",
            )
        )
    )

    assert denied.status_code == 403
    assert allowed.status_code == 200


def test_wrong_sidecar_token_has_same_bounded_denial(monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_SIDECAR_TOKEN", "correct-token")

    async def handler(_request: Request):
        raise AssertionError("wrong token reached sidecar handler")

    response = asyncio.run(
        _secure_sidecar_guard(handler)(
            _request(headers={"Authorization": "Bearer wrong-token"})
        )
    )

    assert response.status_code == 403
    payload = json.loads(response.body)
    assert payload["error"] == "sidecar_forbidden"
    assert "correct-token" not in response.body.decode("utf-8")


def test_sidecar_request_body_is_bounded_before_handler(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_SIDECAR_TOKEN", raising=False)
    monkeypatch.setenv("ENTROLY_SIDECAR_MAX_REQUEST_BYTES", "8")
    called = False

    async def handler(_request: Request):
        nonlocal called
        called = True
        return JSONResponse({"ok": True})

    response = asyncio.run(
        _secure_sidecar_guard(handler)(
            _request(method="POST", body=b"{" + b"x" * 20 + b"}")
        )
    )

    assert response.status_code == 413
    assert not called


def test_sidecar_invalid_json_becomes_400_not_500(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_SIDECAR_TOKEN", raising=False)

    async def handler(request: Request):
        await request.json()
        return JSONResponse({"ok": True})

    response = asyncio.run(
        _secure_sidecar_guard(handler)(
            _request(method="POST", body=b"not-json")
        )
    )

    assert response.status_code == 400
    assert json.loads(response.body)["error"] == "invalid_json"


class _FakeClient:
    def __init__(self, response: httpx.Response):
        self.response = response
        self.calls: list[tuple[str, str, dict]] = []

    async def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.response


class _FakeProxy:
    def __init__(self, response: httpx.Response):
        self._rate_limiter = None
        self._breaker = CircuitBreaker(failure_threshold=3, cooldown_s=30)
        self._client = _FakeClient(response)
        self.observed = []

    def _usage_dimensions(self, _headers):
        return {}

    def _resolve_target(self, _provider: str, path: str) -> str:
        query = _CURRENT_QUERY.get()
        suffix = f"?{query}" if query else ""
        return f"https://api.openai.com{path}{suffix}"

    def _build_headers(self, _headers, _provider):
        return {"content-type": "application/json"}

    async def _ensure_client(self):
        return self._client

    def _apply_outbound_redaction(self, body):
        return body, {"X-Entroly-Redacted": "false"}

    async def _observe_json_usage(self, **kwargs):
        self.observed.append(kwargs)

    async def _stream_response(self, *_args, **_kwargs):
        return JSONResponse({"stream": True})


def _catch_all_request(proxy: _FakeProxy, **kwargs) -> Request:
    app = SimpleNamespace(state=SimpleNamespace(proxy=proxy))
    return _request(app=app, path="/v1/models", **kwargs)


def test_catch_all_preserves_query_on_same_origin() -> None:
    request_stub = httpx.Request("GET", "https://api.openai.com/v1/models")
    proxy = _FakeProxy(
        httpx.Response(
            200,
            json={"data": []},
            headers={"content-type": "application/json"},
            request=request_stub,
        )
    )

    response = asyncio.run(
        _secure_catch_all(
            _catch_all_request(proxy, query=b"api-version=2026-07-01")
        )
    )

    assert response.status_code == 200
    assert proxy._client.calls[0][1].endswith("?api-version=2026-07-01")
    assert _CURRENT_QUERY.get() == ""


def test_catch_all_rejects_oversized_body_before_upstream(monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_PROXY_MAX_REQUEST_BYTES", "8")
    request_stub = httpx.Request("POST", "https://api.openai.com/v1/models")
    proxy = _FakeProxy(
        httpx.Response(
            200,
            json={"ok": True},
            headers={"content-type": "application/json"},
            request=request_stub,
        )
    )

    response = asyncio.run(
        _secure_catch_all(
            _catch_all_request(
                proxy,
                method="POST",
                body=b"{" + b"x" * 20 + b"}",
            )
        )
    )

    assert response.status_code == 413
    assert proxy._client.calls == []


def test_catch_all_rejects_non_object_json() -> None:
    request_stub = httpx.Request("POST", "https://api.openai.com/v1/models")
    proxy = _FakeProxy(
        httpx.Response(
            200,
            json={"ok": True},
            headers={"content-type": "application/json"},
            request=request_stub,
        )
    )

    response = asyncio.run(
        _secure_catch_all(
            _catch_all_request(proxy, method="POST", body=b"[]")
        )
    )

    assert response.status_code == 400
    assert proxy._client.calls == []


def test_catch_all_honors_open_circuit_before_upstream() -> None:
    request_stub = httpx.Request("GET", "https://api.openai.com/v1/models")
    proxy = _FakeProxy(
        httpx.Response(
            200,
            json={"data": []},
            headers={"content-type": "application/json"},
            request=request_stub,
        )
    )
    proxy._breaker.force_open()

    response = asyncio.run(_secure_catch_all(_catch_all_request(proxy)))

    assert response.status_code == 503
    assert proxy._client.calls == []


def test_catch_all_transport_error_is_generic() -> None:
    class FailingClient(_FakeClient):
        async def request(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            request = httpx.Request(method, url)
            raise httpx.ConnectError(
                f"failed at {url}?secret=must-not-leak",
                request=request,
            )

    request_stub = httpx.Request("GET", "https://api.openai.com/v1/models")
    proxy = _FakeProxy(
        httpx.Response(
            200,
            json={"data": []},
            headers={"content-type": "application/json"},
            request=request_stub,
        )
    )
    proxy._client = FailingClient(proxy._client.response)

    response = asyncio.run(_secure_catch_all(_catch_all_request(proxy)))

    assert response.status_code == 502
    assert "must-not-leak" not in response.body.decode("utf-8")
