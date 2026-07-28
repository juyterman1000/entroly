from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

import entroly.proxy as proxy_module
from entroly.proxy_config import ProxyConfig
from entroly.proxy_transport_safe import (
    BoundedAsyncClient,
    CircuitBreaker,
    _CURRENT_QUERY,
    _read_request_body_bounded,
    _safe_build_headers,
    _safe_forward_response,
    _safe_handle_proxy,
    _safe_http_client_kwargs,
    _safe_resolve_target,
    _safe_retry_after,
    _safe_target_url,
    _sanitize_proxy_config,
)


def _request(body_chunks: list[bytes], *, query: bytes = b"", content_length=None) -> Request:
    headers = [(b"content-type", b"application/json")]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(body_chunks) - 1,
        }
        for index, chunk in enumerate(body_chunks)
    ]
    if not messages:
        messages = [{"type": "http.request", "body": b"", "more_body": False}]

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": query,
            "headers": headers,
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 9377),
        },
        receive,
    )


def test_transport_hardening_is_activated_on_proxy_module() -> None:
    assert proxy_module._CircuitBreaker is CircuitBreaker
    assert proxy_module._http_client_kwargs is _safe_http_client_kwargs
    assert proxy_module.PromptCompilerProxy.handle_proxy is _safe_handle_proxy


def test_http_client_ignores_ambient_proxy_and_redirects_by_default(monkeypatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://untrusted-proxy.invalid:8080")
    monkeypatch.delenv("ENTROLY_TRUST_PROXY_ENV", raising=False)

    kwargs = _safe_http_client_kwargs()

    assert kwargs["trust_env"] is False
    assert kwargs["follow_redirects"] is False


def test_ambient_proxy_requires_explicit_operator_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_TRUST_PROXY_ENV", "1")

    assert _safe_http_client_kwargs()["trust_env"] is True


def test_provider_redirect_is_converted_to_explicit_block() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            307,
            headers={"location": "https://attacker.invalid/collect"},
            content=b"redirect",
        )

    async def run() -> httpx.Response:
        async with BoundedAsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            return await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"authorization": "Bearer top-secret"},
                json={"model": "test"},
            )

    response = asyncio.run(run())

    assert response.status_code == 424
    assert response.json()["error"] == "upstream_redirect_blocked"
    assert "attacker.invalid" not in response.text


def test_non_streaming_upstream_response_is_bounded() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 4096)

    async def run() -> httpx.Response:
        async with BoundedAsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=False,
            trust_env=False,
            max_response_bytes=1024,
        ) as client:
            return await client.get("https://api.openai.com/v1/models")

    response = asyncio.run(run())

    assert response.status_code == 424
    assert response.json()["error"] == "upstream_response_too_large"
    assert response.json()["max_bytes"] == 1024


def test_transport_error_does_not_echo_query_credentials() -> None:
    secret = "query-key-must-not-appear"

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed to connect to {request.url}", request=request)

    async def run() -> str:
        async with BoundedAsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            try:
                await client.get(f"https://api.example.invalid/v1?key={secret}")
            except httpx.ConnectError as exc:
                return str(exc)
        raise AssertionError("connect failure was not raised")

    message = asyncio.run(run())

    assert message == "upstream connection failed"
    assert secret not in message


@pytest.mark.parametrize(
    "url",
    [
        "ftp://api.example.com/v1",
        "https://user:password@api.example.com/v1",
        "https://metadata.google.internal/computeMetadata/v1",
        "https://169.254.169.254/latest/meta-data",
        "https://example.com/v1#fragment",
        "http://example.com/v1",
    ],
)
def test_unsafe_upstream_targets_are_rejected(monkeypatch, url: str) -> None:
    monkeypatch.delenv("ENTROLY_ALLOW_INSECURE_UPSTREAM", raising=False)
    monkeypatch.delenv("ENTROLY_ALLOW_PRIVATE_UPSTREAM", raising=False)

    with pytest.raises(ValueError):
        _safe_target_url(url, "/chat/completions")


def test_loopback_http_upstream_remains_supported_for_local_models() -> None:
    target = _safe_target_url(
        "http://127.0.0.1:11434",
        "/v1/chat/completions",
    )

    assert target == "http://127.0.0.1:11434/v1/chat/completions"


def test_private_upstream_requires_explicit_operator_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_ALLOW_PRIVATE_UPSTREAM", raising=False)
    with pytest.raises(ValueError, match="private upstream"):
        _safe_target_url("https://10.0.0.8", "/v1/chat/completions")

    monkeypatch.setenv("ENTROLY_ALLOW_PRIVATE_UPSTREAM", "1")
    assert _safe_target_url(
        "https://10.0.0.8", "/v1/chat/completions"
    ) == "https://10.0.0.8/v1/chat/completions"


def test_request_path_cannot_select_another_origin() -> None:
    for path in ("//attacker.invalid/x", "/https://attacker.invalid/x", "/../admin"):
        with pytest.raises(ValueError):
            _safe_target_url("https://api.openai.com", path)


def test_provider_query_string_is_preserved_on_same_origin() -> None:
    target = _safe_target_url(
        "https://generativelanguage.googleapis.com",
        "/v1beta/models/gemini:generateContent",
        "key=provider-query-key&alt=sse",
    )

    assert target.startswith("https://generativelanguage.googleapis.com/")
    assert target.endswith("?key=provider-query-key&alt=sse")


def test_resolver_uses_request_scoped_query_without_cross_task_state() -> None:
    dummy = SimpleNamespace(
        config=ProxyConfig(
            gemini_base_url="https://generativelanguage.googleapis.com"
        )
    )
    token = _CURRENT_QUERY.set("key=scoped")
    try:
        target = _safe_resolve_target(
            dummy,
            "gemini",
            "/v1beta/models/gemini:generateContent",
        )
    finally:
        _CURRENT_QUERY.reset(token)

    assert target.endswith("?key=scoped")
    assert _CURRENT_QUERY.get() == ""


def test_chunked_request_body_is_bounded_before_json_parse() -> None:
    request = _request([b"{" + b"x" * 10, b"y" * 10 + b"}"])

    with pytest.raises(OverflowError):
        asyncio.run(_read_request_body_bounded(request, 12))


def test_declared_oversized_request_is_rejected_without_reading_body() -> None:
    request = _request([b"{}"], content_length=999)

    with pytest.raises(OverflowError):
        asyncio.run(_read_request_body_bounded(request, 100))


def test_json_array_root_is_rejected_before_core_handler(monkeypatch) -> None:
    called = False

    async def should_not_run(_self, _request):
        nonlocal called
        called = True
        raise AssertionError("invalid root reached core proxy handler")

    monkeypatch.setattr(
        "entroly.proxy_transport_safe._ORIGINAL_HANDLE_PROXY", should_not_run
    )
    request = _request([b"[]"], content_length=2)

    response = asyncio.run(_safe_handle_proxy(SimpleNamespace(), request))

    assert response.status_code == 400
    assert not called
    assert json.loads(response.body)["error"] == "invalid_json_shape"


def test_query_string_is_available_during_core_handler(monkeypatch) -> None:
    async def fake_core(self, request):
        target = _safe_resolve_target(self, "gemini", request.url.path)
        return JSONResponse({"target": target})

    monkeypatch.setattr(
        "entroly.proxy_transport_safe._ORIGINAL_HANDLE_PROXY", fake_core
    )
    dummy = SimpleNamespace(
        config=ProxyConfig(
            gemini_base_url="https://generativelanguage.googleapis.com"
        )
    )
    request = _request([b"{}"], query=b"key=abc%20123", content_length=2)

    response = asyncio.run(_safe_handle_proxy(dummy, request))
    target = json.loads(response.body)["target"]

    assert target.endswith("?key=abc%20123")
    assert _CURRENT_QUERY.get() == ""


def test_connection_named_headers_and_cookies_are_not_forwarded() -> None:
    original = {
        "Connection": "x-request-debug",
        "X-Request-Debug": "must-not-forward",
        "Cookie": "session=private",
        "X-Forwarded-For": "203.0.113.5",
        "Authorization": "Bearer provider-key",
        "Content-Type": "application/json",
    }

    forwarded = _safe_build_headers(SimpleNamespace(), original, "openai")

    lowered = {name.casefold(): value for name, value in forwarded.items()}
    assert "x-request-debug" not in lowered
    assert "cookie" not in lowered
    assert "x-forwarded-for" not in lowered
    assert lowered["authorization"] == "Bearer provider-key"
    assert lowered["content-type"] == "application/json"


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "", "abc"])
def test_non_finite_or_invalid_retry_after_is_rejected(raw: str) -> None:
    assert _safe_retry_after(raw) is None


def test_retry_after_is_bounded() -> None:
    assert _safe_retry_after("999999999") == 86400.0
    assert _safe_retry_after("-5") == 0.0


def test_circuit_breaker_allows_only_one_half_open_probe() -> None:
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=0.01)
    breaker.record_failure()
    assert breaker.state == "open"
    time.sleep(0.02)

    assert breaker.allow_request() is True
    assert breaker.state == "half_open"
    results: list[bool] = []

    def contender() -> None:
        results.append(breaker.allow_request())

    thread = threading.Thread(target=contender)
    thread.start()
    thread.join()

    assert results == [False]
    assert breaker.allow_request() is True
    breaker.record_success()
    assert breaker.state == "closed"


def test_open_breaker_prevents_nonstream_upstream_call(monkeypatch) -> None:
    called = False

    async def should_not_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("open breaker reached upstream forwarding")

    monkeypatch.setattr(
        "entroly.proxy_transport_safe._ORIGINAL_FORWARD_RESPONSE", should_not_run
    )
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=60)
    breaker.record_failure()
    dummy = SimpleNamespace(_breaker=breaker)

    response = asyncio.run(_safe_forward_response(dummy))

    assert response.status_code == 503
    assert not called


def test_nonfinite_proxy_configuration_restores_safe_defaults() -> None:
    config = ProxyConfig(
        context_fraction=float("nan"),
        fisher_scale=float("inf"),
        trajectory_lambda=-1.0,
    )

    sanitized = _sanitize_proxy_config(config)

    defaults = ProxyConfig()
    assert sanitized.context_fraction == defaults.context_fraction
    assert sanitized.fisher_scale == defaults.fisher_scale
    assert sanitized.trajectory_lambda == defaults.trajectory_lambda
