"""Control-plane hardening for the Entroly HTTP proxy.

This layer protects two paths that are easy to overlook during transport review:

* sidecar routes expose recovered context, local files, model feedback, runtime
  mutation, and proof sessions; they must not trust reverse-proxy topology;
* the catch-all provider route must use the same request limits, target policy,
  query preservation, circuit breaker, and redaction as explicit API routes.

The module patches global route factories before ``create_proxy_app`` builds the
Starlette route table. Public provider behavior remains unchanged for valid
loopback clients and valid provider JSON requests.
"""

from __future__ import annotations

import hmac
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from . import proxy as _proxy
from . import proxy_transport_safe as _transport
from .proxy_transform import detect_provider

_DEFAULT_SIDECAR_MAX_REQUEST_BYTES = 2 * 1024 * 1024
_FORWARDED_REQUEST_HEADERS = frozenset(
    {
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-forwarded-port",
        "x-real-ip",
    }
)


def _presented_sidecar_token(request: Request) -> str:
    explicit = request.headers.get("x-entroly-sidecar-token", "").strip()
    if explicit:
        return explicit
    authorization = request.headers.get("authorization", "").strip()
    scheme, separator, value = authorization.partition(" ")
    if separator and scheme.casefold() == "bearer":
        return value.strip()
    return ""


def _sidecar_token_valid(request: Request) -> bool:
    configured = os.environ.get("ENTROLY_SIDECAR_TOKEN", "")
    if not configured:
        return False
    presented = _presented_sidecar_token(request)
    if not presented:
        return False
    return hmac.compare_digest(
        configured.encode("utf-8", errors="strict"),
        presented.encode("utf-8", errors="strict"),
    )


def _has_forwarding_headers(request: Request) -> bool:
    return any(name in request.headers for name in _FORWARDED_REQUEST_HEADERS)


def _direct_local_sidecar_request(request: Request) -> bool:
    if _has_forwarding_headers(request):
        return False
    return _proxy._is_trusted_sidecar_request(request)


def _sidecar_security_headers(response: Response) -> Response:
    response.headers.setdefault("Cache-Control", "no-store, max-age=0")
    response.headers.setdefault("Pragma", "no-cache")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


def _sidecar_denied() -> JSONResponse:
    return JSONResponse(
        {
            "error": "sidecar_forbidden",
            "detail": (
                "Entroly sidecar endpoints require a direct local same-origin "
                "request or a valid sidecar token."
            ),
        },
        status_code=403,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _secure_sidecar_guard(
    handler: Callable[[Request], Awaitable[Response]],
) -> Callable[[Request], Awaitable[Response]]:
    """Protect one sidecar endpoint with topology-safe authorization.

    If ``ENTROLY_SIDECAR_TOKEN`` is configured, it is mandatory for every
    request. Otherwise only a direct loopback request with no forwarding headers
    and same-origin browser metadata is accepted. Reverse-proxy deployment must
    configure a token; loopback proxy hops alone are never treated as identity.
    """

    async def _guarded(request: Request) -> Response:
        token_configured = bool(os.environ.get("ENTROLY_SIDECAR_TOKEN", ""))
        token_valid = _sidecar_token_valid(request)
        if token_configured:
            if not token_valid:
                return _sidecar_denied()
        elif not _direct_local_sidecar_request(request):
            return _sidecar_denied()

        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            limit = _transport._bounded_positive_int(
                "ENTROLY_SIDECAR_MAX_REQUEST_BYTES",
                _DEFAULT_SIDECAR_MAX_REQUEST_BYTES,
            )
            try:
                await _transport._read_request_body_bounded(request, limit)
            except OverflowError:
                return JSONResponse(
                    {
                        "error": "sidecar_request_too_large",
                        "max_bytes": limit,
                    },
                    status_code=413,
                    headers={"Cache-Control": "no-store"},
                )
            except (ConnectionError, ValueError) as exc:
                return JSONResponse(
                    {
                        "error": "invalid_sidecar_request",
                        "detail": str(exc)[:300],
                    },
                    status_code=400,
                    headers={"Cache-Control": "no-store"},
                )

        try:
            response = await handler(request)
        except json.JSONDecodeError:
            response = JSONResponse(
                {"error": "invalid_json"},
                status_code=400,
            )
        return _sidecar_security_headers(response)

    _guarded.__name__ = getattr(handler, "__name__", "sidecar_handler")
    _guarded.__doc__ = getattr(handler, "__doc__", None)
    return _guarded


def _request_query_string(request: Request) -> str:
    raw = request.scope.get("query_string", b"")
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError("query string must be bytes")
    if len(raw) > _transport._MAX_QUERY_CHARS or any(byte < 32 for byte in raw):
        raise ValueError("query string is not safe bounded text")
    try:
        return bytes(raw).decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("query string must be ASCII percent-encoded") from exc


def _record_breaker_outcome(breaker: Any, status_code: int) -> None:
    if status_code in {429, 500, 502, 503, 504} or status_code == 424:
        breaker.record_failure()
    else:
        breaker.record_success()


def _safe_upstream_error() -> JSONResponse:
    return JSONResponse(
        {
            "error": "upstream_unavailable",
            "detail": "The upstream provider request failed.",
        },
        status_code=502,
    )


async def _secure_catch_all(request: Request) -> Response:
    """Forward unmatched provider paths through the hardened boundary."""
    proxy = request.app.state.proxy
    headers = {key: value for key, value in request.headers.items()}
    request_id = headers.get("x-request-id") or os.urandom(6).hex()
    usage_dimensions = proxy._usage_dimensions(headers)

    if proxy._rate_limiter is not None and not proxy._rate_limiter.try_consume():
        return JSONResponse(
            {"error": "rate_limit_exceeded"},
            status_code=429,
            headers={"Retry-After": "1"},
        )
    if not proxy._breaker.allow_request():
        return _transport._circuit_open_response(proxy._breaker)

    try:
        query = _request_query_string(request)
    except ValueError as exc:
        return JSONResponse(
            {"error": "invalid_query_string", "detail": str(exc)},
            status_code=400,
        )

    token = _transport._CURRENT_QUERY.set(query)
    try:
        provider = detect_provider(request.url.path, headers)
        target_url = proxy._resolve_target(provider, request.url.path)
        forward_headers = proxy._build_headers(headers, provider)

        if request.method in {"GET", "HEAD", "OPTIONS"}:
            try:
                client = await proxy._ensure_client()
                response = await client.request(
                    request.method,
                    target_url,
                    headers=forward_headers,
                )
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError):
                proxy._breaker.record_failure()
                return _safe_upstream_error()
            _record_breaker_outcome(proxy._breaker, response.status_code)
            if request.method == "HEAD":
                return Response(
                    status_code=response.status_code,
                    headers={
                        "Content-Type": response.headers.get(
                            "content-type", "application/octet-stream"
                        )
                    },
                )
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    content = response.json()
                except (json.JSONDecodeError, ValueError):
                    return JSONResponse(
                        {
                            "error": "invalid_upstream_json",
                            "status": response.status_code,
                        },
                        status_code=502,
                    )
                return JSONResponse(content=content, status_code=response.status_code)
            return JSONResponse(
                content={"data": response.text},
                status_code=response.status_code,
            )

        limit = _transport._bounded_positive_int(
            "ENTROLY_PROXY_MAX_REQUEST_BYTES",
            _transport._DEFAULT_MAX_REQUEST_BYTES,
        )
        try:
            body_bytes = await _transport._read_request_body_bounded(request, limit)
        except OverflowError:
            return JSONResponse(
                {
                    "error": "request_body_too_large",
                    "max_bytes": limit,
                },
                status_code=413,
            )
        except (ConnectionError, ValueError) as exc:
            return JSONResponse(
                {"error": "invalid_request_body", "detail": str(exc)[:300]},
                status_code=400,
            )

        if not body_bytes and request.method == "DELETE":
            body: dict[str, Any] = {}
        else:
            try:
                decoded = json.loads(body_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return JSONResponse(
                    {"error": "invalid_request_body"},
                    status_code=400,
                )
            if not isinstance(decoded, dict):
                return JSONResponse(
                    {
                        "error": "invalid_json_shape",
                        "detail": "Provider requests must use a JSON object root.",
                    },
                    status_code=400,
                )
            body = decoded

        body, redaction_headers = proxy._apply_outbound_redaction(body)
        provider = detect_provider(request.url.path, headers, body)
        target_url = proxy._resolve_target(provider, request.url.path)
        forward_headers = proxy._build_headers(headers, provider)
        is_streaming = bool(body.get("stream", False)) or (
            "streamGenerateContent" in request.url.path
        )
        if is_streaming:
            return await proxy._stream_response(
                target_url,
                forward_headers,
                body,
                provider=provider,
                request_id=request_id,
                extra_headers=redaction_headers,
                usage_dimensions=usage_dimensions,
            )

        try:
            client = await proxy._ensure_client()
            response = await client.request(
                request.method,
                target_url,
                json=body,
                headers=forward_headers,
            )
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError):
            proxy._breaker.record_failure()
            return _safe_upstream_error()
        _record_breaker_outcome(proxy._breaker, response.status_code)

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                content = response.json()
            except (json.JSONDecodeError, ValueError):
                return JSONResponse(
                    {
                        "error": "invalid_upstream_json",
                        "status": response.status_code,
                    },
                    status_code=502,
                    headers=redaction_headers,
                )
            if response.status_code < 400 and isinstance(content, dict):
                await proxy._observe_json_usage(
                    body=body,
                    provider=provider,
                    request_id=request_id,
                    payload=content,
                    path=request.url.path,
                    usage_dimensions=usage_dimensions,
                )
            return JSONResponse(
                content=content,
                status_code=response.status_code,
                headers=redaction_headers,
            )
        return JSONResponse(
            content={"data": response.text},
            status_code=response.status_code,
            headers=redaction_headers,
        )
    finally:
        _transport._CURRENT_QUERY.reset(token)


_proxy._sidecar_guard = _secure_sidecar_guard
_proxy._catch_all = _secure_catch_all

__all__ = [
    "_direct_local_sidecar_request",
    "_secure_catch_all",
    "_secure_sidecar_guard",
    "_sidecar_token_valid",
]
