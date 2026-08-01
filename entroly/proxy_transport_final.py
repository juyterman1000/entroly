"""Final integrity refinements for the hardened provider transport."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from urllib.parse import urlunsplit

import httpx

from . import proxy as _proxy
from . import proxy_transport_safe as _safe

_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_STALE_ENTITY_HEADERS = frozenset(
    {"content-encoding", "content-length", "transfer-encoding"}
)


class _LazyDirectTransport(httpx.AsyncBaseTransport):
    """Direct HTTP transport that initializes its CA context on first use."""

    def __init__(self, verify: str):
        self._verify = verify
        self._transport: httpx.AsyncHTTPTransport | None = None
        self._lock = asyncio.Lock()

    async def _get_transport(self) -> httpx.AsyncHTTPTransport:
        if self._transport is None:
            async with self._lock:
                if self._transport is None:
                    # AsyncHTTPTransport never consults HTTP_PROXY/HTTPS_PROXY;
                    # proxy routing is a client-level trust_env behavior.
                    self._transport = httpx.AsyncHTTPTransport(verify=self._verify)
        return self._transport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = await self._get_transport()
        return await transport.handle_async_request(request)

    async def aclose(self) -> None:
        if self._transport is not None:
            await self._transport.aclose()


class BoundedAsyncClient(_safe.BoundedAsyncClient):
    """Bound decoded bodies while preserving coherent response headers."""

    async def send(self, request: httpx.Request, *args, **kwargs) -> httpx.Response:
        caller_streaming = bool(kwargs.get("stream", False))
        if caller_streaming:
            try:
                response = await httpx.AsyncClient.send(self, request, *args, **kwargs)
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
                raise _safe._generic_transport_exception(exc, request) from exc
            if 300 <= response.status_code < 400:
                await response.aclose()
                return httpx.Response(
                    502,
                    headers={"content-type": "application/json"},
                    content=json.dumps(
                        {
                            "error": "upstream_redirect_blocked",
                            "detail": (
                                "Provider redirects are blocked so credentials "
                                "cannot change origin."
                            ),
                        }
                    ).encode("utf-8"),
                    request=request,
                )
            return response

        kwargs["stream"] = True
        try:
            response = await httpx.AsyncClient.send(self, request, *args, **kwargs)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
            raise _safe._generic_transport_exception(exc, request) from exc

        if 300 <= response.status_code < 400:
            await response.aclose()
            return httpx.Response(
                424,
                headers={"content-type": "application/json"},
                json={
                    "error": "upstream_redirect_blocked",
                    "detail": (
                        "Provider redirects are blocked so credentials cannot "
                        "change origin."
                    ),
                    "upstream_status": response.status_code,
                },
                request=request,
            )

        payload = bytearray()
        too_large = False
        try:
            # aiter_bytes() applies HTTP content decoding. Bound the decoded form
            # to prevent compressed response bombs from bypassing the limit.
            async for chunk in response.aiter_bytes():
                if len(payload) + len(chunk) > self._entroly_max_response_bytes:
                    too_large = True
                    break
                payload.extend(chunk)
        finally:
            await response.aclose()

        if too_large:
            return httpx.Response(
                424,
                headers={
                    "content-type": "application/json",
                    "x-entroly-upstream-response-limit": str(
                        self._entroly_max_response_bytes
                    ),
                },
                json={
                    "error": "upstream_response_too_large",
                    "detail": (
                        "Upstream response exceeded the configured proxy "
                        "safety limit."
                    ),
                    "max_bytes": self._entroly_max_response_bytes,
                },
                request=request,
            )

        headers = [
            (name, value)
            for name, value in response.headers.multi_items()
            if name.casefold() not in _STALE_ENTITY_HEADERS
        ]
        return httpx.Response(
            response.status_code,
            headers=headers,
            content=bytes(payload),
            request=request,
            extensions=response.extensions,
        )


def _safe_http_client_kwargs() -> dict[str, Any]:
    """Return safe HTTPX kwargs with explicit CA and proxy policies separated.

    An explicitly configured CA bundle is passed through ``verify`` and does not
    require ``trust_env=True``. Ambient HTTP(S)_PROXY variables therefore remain
    ignored unless the operator independently opts in with
    ``ENTROLY_TRUST_PROXY_ENV=1``.
    """
    ca_bundle = _proxy._resolve_ca_bundle_from_env()
    trust_proxy_env = _safe._env_flag("ENTROLY_TRUST_PROXY_ENV")
    kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        "follow_redirects": False,
        "trust_env": trust_proxy_env,
        "limits": httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        ),
    }
    if ca_bundle:
        kwargs["verify"] = ca_bundle
    return kwargs


def _safe_target_url(base_url: str, path: str, query: str = "") -> str:
    if "#" in query:
        raise ValueError("provider query string must not contain a fragment marker")
    scheme, netloc, _port, base_path, base_query = _safe._validate_upstream_base(
        base_url
    )
    request_path = _safe._validate_relative_provider_path(path)
    if (
        len(query) > _safe._MAX_QUERY_CHARS
        or any(ord(character) < 32 for character in query)
    ):
        raise ValueError("provider query string is not safe bounded text")
    combined_path = f"{base_path}{request_path}" if base_path else request_path
    combined_query = "&".join(part for part in (base_query, query) if part)
    return urlunsplit((scheme, netloc, combined_path, combined_query, ""))


def _safe_build_headers(
    self, original: dict[str, str], provider: str
) -> dict[str, str]:
    if not isinstance(original, dict) or len(original) > _safe._MAX_HEADER_COUNT:
        raise ValueError("request headers exceed the proxy safety limit")
    dynamic_hop = _safe._dynamic_connection_headers(original)
    cleaned: dict[str, str] = {}
    for raw_name, raw_value in original.items():
        name = str(raw_name)
        value = str(raw_value)
        lower = name.casefold()
        if (
            not _HEADER_NAME_RE.fullmatch(name)
            or len(name) > 256
            or len(value) > _safe._MAX_HEADER_VALUE_CHARS
            or any(ord(character) < 32 and character != "\t" for character in value)
            or "\r" in value
            or "\n" in value
            or lower in dynamic_hop
            or lower.startswith("proxy-")
            or lower == "forwarded"
            or lower.startswith("x-forwarded-")
            or lower in {"cookie", "set-cookie"}
        ):
            continue
        cleaned[name] = value
    return _safe._ORIGINAL_BUILD_HEADERS(self, cleaned, provider)


# Replace the first-pass class/functions. The startup and resolver wrappers in
# proxy_transport_safe resolve these module globals at call time.
_safe.BoundedAsyncClient = BoundedAsyncClient
_safe._safe_http_client_kwargs = _safe_http_client_kwargs
_safe._safe_target_url = _safe_target_url
_safe._safe_build_headers = _safe_build_headers
_proxy._http_client_kwargs = _safe_http_client_kwargs
_proxy.PromptCompilerProxy._build_headers = _safe_build_headers

__all__ = [
    "BoundedAsyncClient",
    "_safe_build_headers",
    "_safe_http_client_kwargs",
    "_safe_target_url",
]
