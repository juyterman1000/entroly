"""Trust-hardened transport boundary for the Entroly provider proxy.

The core proxy intentionally contains the optimization and provider semantics.
This module patches only its network boundary after the proxy module has loaded:

- provider credentials never follow redirects;
- ambient HTTP(S)_PROXY variables are ignored unless explicitly trusted;
- request and non-streaming response bodies are bounded;
- query strings are preserved without becoming alternate-origin URLs;
- upstream base URLs reject credentials, unsafe schemes, and metadata targets;
- the circuit breaker truly blocks open-state requests and allows one probe;
- Retry-After and environment floating-point values reject NaN/Infinity;
- connection-scoped headers and control bytes are removed;
- transport errors never echo credential-bearing query strings.
"""

from __future__ import annotations

import asyncio
import contextvars
import ipaddress
import json
import logging
import math
import os
import re
import threading
import time
from email.utils import parsedate_to_datetime
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse

from . import proxy as _proxy
from . import proxy_config as _proxy_config

logger = logging.getLogger("entroly.proxy")

_DEFAULT_MAX_REQUEST_BYTES = 32 * 1024 * 1024
_DEFAULT_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_MAX_CONFIGURED_BODY_BYTES = 1024 * 1024 * 1024
_MAX_URL_CHARS = 8192
_MAX_QUERY_CHARS = 16384
_MAX_HEADER_COUNT = 256
_MAX_HEADER_VALUE_CHARS = 16384
_MAX_RETRY_AFTER_SECONDS = 86400.0
_METADATA_HOSTS = frozenset(
    {
        "metadata.google.internal",
        "metadata.azure.internal",
        "instance-data.ec2.internal",
    }
)
_CURRENT_QUERY: contextvars.ContextVar[str] = contextvars.ContextVar(
    "entroly_proxy_query", default=""
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _bounded_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default
    if value <= 0 or value > _MAX_CONFIGURED_BODY_BYTES:
        logger.warning("Out-of-range %s=%r; using default %d", name, raw, default)
        return default
    return value


def _finite_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default
    if not math.isfinite(value):
        logger.warning("Non-finite %s=%r; using default %s", name, raw, default)
        return default
    return value


def _finite_env_optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; ignoring override", name, raw)
        return None
    if not math.isfinite(value):
        logger.warning("Non-finite %s=%r; ignoring override", name, raw)
        return None
    return value


def _safe_retry_after(raw: str | None) -> float | None:
    """Parse a bounded finite Retry-After value."""
    if raw is None:
        return None
    value = raw.strip()
    if not value or len(value) > 256:
        return None
    try:
        seconds = float(value)
        if not math.isfinite(seconds):
            return None
        return min(_MAX_RETRY_AFTER_SECONDS, max(0.0, seconds))
    except ValueError:
        pass
    try:
        target = parsedate_to_datetime(value)
        if target is None:
            return None
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc) if target.tzinfo else datetime.utcnow()
        seconds = (target - now).total_seconds()
        if not math.isfinite(seconds):
            return None
        return min(_MAX_RETRY_AFTER_SECONDS, max(0.0, seconds))
    except (TypeError, ValueError, OverflowError):
        return None


def _validate_upstream_base(base_url: str) -> tuple[str, str, int | None, str, str]:
    if not isinstance(base_url, str):
        raise ValueError("upstream base URL must be text")
    if (
        not base_url
        or len(base_url) > _MAX_URL_CHARS
        or any(ord(character) < 32 for character in base_url)
        or any(character.isspace() for character in base_url)
    ):
        raise ValueError("upstream base URL is not a safe bounded URL")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("upstream base URL must use http or https")
    if not parsed.hostname:
        raise ValueError("upstream base URL must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("upstream base URL must not embed credentials")
    if parsed.fragment:
        raise ValueError("upstream base URL must not contain a fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("upstream base URL contains an invalid port") from exc

    host = parsed.hostname.casefold().rstrip(".")
    if host in _METADATA_HOSTS:
        raise ValueError("cloud metadata hosts are never valid provider targets")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        if (
            address.is_link_local
            or address.is_multicast
            or address.is_unspecified
            or address.is_reserved
        ):
            raise ValueError("unsafe special-purpose upstream address")
        if address.is_private and not address.is_loopback:
            if not _env_flag("ENTROLY_ALLOW_PRIVATE_UPSTREAM"):
                raise ValueError(
                    "private upstream addresses require ENTROLY_ALLOW_PRIVATE_UPSTREAM=1"
                )

    loopback = host == "localhost" or bool(address and address.is_loopback)
    if parsed.scheme == "http" and not loopback:
        if not _env_flag("ENTROLY_ALLOW_INSECURE_UPSTREAM"):
            raise ValueError(
                "non-loopback HTTP upstreams require ENTROLY_ALLOW_INSECURE_UPSTREAM=1"
            )

    netloc = parsed.netloc
    return parsed.scheme, netloc, port, parsed.path.rstrip("/"), parsed.query


def _validate_relative_provider_path(path: str) -> str:
    if not isinstance(path, str) or not path or len(path) > _MAX_URL_CHARS:
        raise ValueError("provider request path is not bounded text")
    decoded = unquote(path)
    if (
        any(ord(character) < 32 for character in decoded)
        or "\\" in decoded
        or decoded.startswith("//")
        or "://" in decoded
    ):
        raise ValueError("provider request path cannot select another origin")
    clean_path = path.split("?", 1)[0]
    if not clean_path.startswith("/"):
        clean_path = "/" + clean_path
    segments = [segment for segment in unquote(clean_path).split("/") if segment]
    if any(segment in {".", ".."} for segment in segments):
        raise ValueError("provider request path contains traversal segments")
    return clean_path


def _safe_target_url(base_url: str, path: str, query: str = "") -> str:
    scheme, netloc, _port, base_path, base_query = _validate_upstream_base(base_url)
    request_path = _validate_relative_provider_path(path)
    raw_query = query or ""
    if (
        len(raw_query) > _MAX_QUERY_CHARS
        or any(ord(character) < 32 for character in raw_query)
    ):
        raise ValueError("provider query string is not safe bounded text")
    combined_path = f"{base_path}{request_path}" if base_path else request_path
    combined_query = "&".join(part for part in (base_query, raw_query) if part)
    return urlunsplit((scheme, netloc, combined_path, combined_query, ""))


def _safe_http_client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        "follow_redirects": False,
        "trust_env": _env_flag("ENTROLY_TRUST_PROXY_ENV"),
        "limits": httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        ),
    }
    ca_bundle = _proxy._resolve_ca_bundle_from_env()
    if ca_bundle:
        kwargs["verify"] = ca_bundle
    return kwargs


def _generic_transport_exception(exc: Exception, request: httpx.Request) -> Exception:
    if isinstance(exc, httpx.TimeoutException):
        return httpx.TimeoutException("upstream request timed out", request=request)
    if isinstance(exc, httpx.ConnectError):
        return httpx.ConnectError("upstream connection failed", request=request)
    if isinstance(exc, httpx.ReadError):
        return httpx.ReadError("upstream connection was interrupted", request=request)
    return exc


class BoundedAsyncClient(httpx.AsyncClient):
    """HTTPX client with redirect refusal and bounded non-streaming bodies."""

    def __init__(self, *args, max_response_bytes: int | None = None, **kwargs):
        self._entroly_max_response_bytes = max_response_bytes or _bounded_positive_int(
            "ENTROLY_PROXY_MAX_RESPONSE_BYTES", _DEFAULT_MAX_RESPONSE_BYTES
        )
        super().__init__(*args, **kwargs)

    async def send(self, request: httpx.Request, *args, **kwargs) -> httpx.Response:
        caller_streaming = bool(kwargs.get("stream", False))
        if caller_streaming:
            try:
                return await super().send(request, *args, **kwargs)
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
                raise _generic_transport_exception(exc, request) from exc

        kwargs["stream"] = True
        try:
            response = await super().send(request, *args, **kwargs)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
            raise _generic_transport_exception(exc, request) from exc

        payload = bytearray()
        too_large = False
        try:
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
                    "detail": "Upstream response exceeded the configured proxy safety limit.",
                    "max_bytes": self._entroly_max_response_bytes,
                },
                request=request,
            )
        if 300 <= response.status_code < 400:
            return httpx.Response(
                424,
                headers={"content-type": "application/json"},
                json={
                    "error": "upstream_redirect_blocked",
                    "detail": "Provider redirects are blocked so credentials cannot change origin.",
                    "upstream_status": response.status_code,
                },
                request=request,
            )
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=bytes(payload),
            request=request,
            extensions=response.extensions,
        )


def _owner_identity() -> tuple[str, int]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    if task is not None:
        return ("task", id(task))
    return ("thread", threading.get_ident())


class CircuitBreaker:
    """Monotonic circuit breaker with exactly one half-open probe."""

    def __init__(self, failure_threshold: int = 3, cooldown_s: float = 30.0):
        if isinstance(failure_threshold, bool) or int(failure_threshold) < 1:
            raise ValueError("failure_threshold must be a positive integer")
        cooldown = float(cooldown_s)
        if not math.isfinite(cooldown) or cooldown <= 0:
            raise ValueError("cooldown_s must be a finite positive number")
        self.failure_threshold = int(failure_threshold)
        self.cooldown_s = cooldown
        self._consecutive_failures = 0
        self._last_failure_time = 0.0
        self._state = "closed"
        self._probe_owner: tuple[str, int] | None = None
        self._probe_started = 0.0
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        owner = _owner_identity()
        now = time.monotonic()
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if now - self._last_failure_time < self.cooldown_s:
                    return False
                self._state = "half_open"
                self._probe_owner = owner
                self._probe_started = now
                return True
            if self._probe_owner == owner:
                return True
            probe_timeout = max(120.0, self.cooldown_s * 4.0)
            if now - self._probe_started > probe_timeout:
                self._probe_owner = owner
                self._probe_started = now
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._state = "closed"
            self._probe_owner = None
            self._probe_started = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._last_failure_time = time.monotonic()
            if self._state == "half_open":
                self._consecutive_failures = self.failure_threshold
                self._state = "open"
                self._probe_owner = None
                self._probe_started = 0.0
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._state = "open"
                self._probe_owner = None
                self._probe_started = 0.0

    def force_open(self) -> None:
        with self._lock:
            self._consecutive_failures = self.failure_threshold
            self._last_failure_time = time.monotonic()
            self._state = "open"
            self._probe_owner = None
            self._probe_started = 0.0

    def retry_after_seconds(self) -> int:
        with self._lock:
            if self._state != "open":
                return 0
            remaining = self.cooldown_s - (time.monotonic() - self._last_failure_time)
            return max(1, int(math.ceil(max(0.0, remaining))))

    @property
    def state(self) -> str:
        with self._lock:
            return self._state


def _circuit_open_response(breaker: CircuitBreaker) -> JSONResponse:
    retry_after = breaker.retry_after_seconds()
    return JSONResponse(
        {
            "error": "circuit_breaker_open",
            "message": "Upstream API is temporarily unavailable; retry after cooldown.",
        },
        status_code=503,
        headers={"Retry-After": str(retry_after)},
    )


def _dynamic_connection_headers(original: Mapping[str, str]) -> set[str]:
    raw = ""
    for name, value in original.items():
        if str(name).casefold() == "connection":
            raw = str(value)
            break
    return {
        token.strip().casefold()
        for token in raw.split(",")
        if token.strip()
    }


def _safe_build_headers(
    self, original: dict[str, str], provider: str
) -> dict[str, str]:
    if not isinstance(original, dict) or len(original) > _MAX_HEADER_COUNT:
        raise ValueError("request headers exceed the proxy safety limit")
    dynamic_hop = _dynamic_connection_headers(original)
    cleaned: dict[str, str] = {}
    for raw_name, raw_value in original.items():
        name = str(raw_name)
        value = str(raw_value)
        lower = name.casefold()
        if (
            not name
            or len(name) > 256
            or len(value) > _MAX_HEADER_VALUE_CHARS
            or any(ord(character) < 32 and character != "\t" for character in name + value)
            or lower in dynamic_hop
            or lower.startswith("proxy-")
            or lower == "forwarded"
            or lower.startswith("x-forwarded-")
            or lower in {"cookie", "set-cookie"}
        ):
            continue
        cleaned[name] = value
    return _ORIGINAL_BUILD_HEADERS(self, cleaned, provider)


async def _read_request_body_bounded(request: Request, limit: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        if declared < 0:
            raise ValueError("Content-Length must not be negative")
        if declared > limit:
            raise OverflowError("request body exceeds configured proxy limit")

    chunks: list[bytes] = []
    total = 0
    try:
        async for chunk in request.stream():
            total += len(chunk)
            if total > limit:
                raise OverflowError("request body exceeds configured proxy limit")
            chunks.append(chunk)
    except ClientDisconnect as exc:
        raise ConnectionError("client disconnected before request body completed") from exc
    body = b"".join(chunks)
    request._body = body  # Starlette's documented body() cache contract is private.
    return body


async def _safe_handle_proxy(self, request: Request):
    limit = _bounded_positive_int(
        "ENTROLY_PROXY_MAX_REQUEST_BYTES", _DEFAULT_MAX_REQUEST_BYTES
    )
    try:
        body = await _read_request_body_bounded(request, limit)
    except OverflowError:
        return JSONResponse(
            {
                "error": "request_body_too_large",
                "detail": "Request exceeded the configured Entroly proxy safety limit.",
                "max_bytes": limit,
            },
            status_code=413,
        )
    except (ConnectionError, ValueError) as exc:
        return JSONResponse(
            {"error": "invalid_request_body", "detail": str(exc)[:300]},
            status_code=400,
        )

    if body:
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            parsed = None
        if parsed is not None and not isinstance(parsed, dict):
            return JSONResponse(
                {
                    "error": "invalid_json_shape",
                    "detail": "Provider requests must use a JSON object root.",
                },
                status_code=400,
            )

    raw_query = request.scope.get("query_string", b"")
    if not isinstance(raw_query, (bytes, bytearray)):
        return JSONResponse({"error": "invalid_query_string"}, status_code=400)
    if len(raw_query) > _MAX_QUERY_CHARS or any(byte < 32 for byte in raw_query):
        return JSONResponse({"error": "invalid_query_string"}, status_code=400)
    try:
        query = bytes(raw_query).decode("ascii")
    except UnicodeDecodeError:
        return JSONResponse({"error": "invalid_query_string"}, status_code=400)

    token = _CURRENT_QUERY.set(query)
    try:
        return await _ORIGINAL_HANDLE_PROXY(self, request)
    finally:
        _CURRENT_QUERY.reset(token)


def _safe_resolve_target(self, provider: str, path: str) -> str:
    if provider == "anthropic":
        base = self.config.anthropic_base_url
    elif provider == "gemini":
        base = self.config.gemini_base_url
    else:
        base = self.config.openai_base_url
    return _safe_target_url(base, path, _CURRENT_QUERY.get())


async def _safe_warmup_connection(self, target_url: str) -> None:
    # An unsolicited /health call is an observable remote side effect and can
    # confuse strict gateways. Keep it explicit rather than surprising.
    if not _env_flag("ENTROLY_PROXY_WARMUP"):
        return
    if self._client is None:
        return
    parsed = urlsplit(target_url)
    warmup_url = urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))
    try:
        response = await self._client.head(warmup_url, timeout=2.0)
        await response.aclose()
    except Exception:
        return


async def _safe_startup(self) -> None:
    self._client = BoundedAsyncClient(**_safe_http_client_kwargs())
    logger.info("Prompt compiler proxy ready with hardened transport")


async def _safe_ensure_client(self) -> BoundedAsyncClient:
    if self._client is None or self._client.is_closed:
        logger.info("Reconnecting hardened HTTP client")
        self._client = BoundedAsyncClient(**_safe_http_client_kwargs())
    return self._client


async def _safe_forward_response(self, *args, **kwargs):
    breaker = self._breaker
    if not breaker.allow_request():
        return _circuit_open_response(breaker)
    was_probe = breaker.state == "half_open"
    response = await _ORIGINAL_FORWARD_RESPONSE(self, *args, **kwargs)
    if was_probe and response.status_code in {429, 500, 502, 503, 504}:
        if breaker.state != "open":
            breaker.force_open()
    elif response.status_code == 429 or response.status_code >= 500:
        try:
            payload = json.loads(bytes(response.body))
        except Exception:
            payload = {}
        if payload.get("error") not in {"upstream_unavailable"}:
            breaker.record_failure()
    return response


async def _safe_stream_response(self, *args, **kwargs):
    breaker = self._breaker
    if not breaker.allow_request():
        return _circuit_open_response(breaker)
    return await _ORIGINAL_STREAM_RESPONSE(self, *args, **kwargs)


def _sanitize_proxy_config(config: Any) -> Any:
    defaults = _proxy_config.ProxyConfig()
    bounded_ranges: dict[str, tuple[float, float]] = {
        "context_fraction": (0.0, 1.0),
        "ecdb_max_fraction": (0.0, 1.0),
        "ios_skeleton_info_factor": (0.0, 10.0),
        "ios_reference_info_factor": (0.0, 10.0),
        "ios_diversity_floor": (0.0, 1.0),
        "fisher_scale": (0.0, 100.0),
        "trajectory_c_min": (0.0, 100.0),
        "trajectory_lambda": (0.0, 100.0),
        "egtc_alpha": (0.0, 100.0),
        "egtc_gamma": (0.0, 100.0),
        "egtc_epsilon": (0.0, 100.0),
    }
    for name, (minimum, maximum) in bounded_ranges.items():
        value = getattr(config, name, getattr(defaults, name))
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float(getattr(defaults, name))
        if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
            logger.warning("Invalid proxy config %s=%r; restoring default", name, value)
            setattr(config, name, getattr(defaults, name))
    if config.quality is not None:
        try:
            quality = float(config.quality)
        except (TypeError, ValueError):
            quality = math.nan
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            logger.warning("Invalid proxy quality=%r; disabling quality override", config.quality)
            config.quality = None
    for name in ("context_window", "ecdb_min_budget", "scaffold_max_tokens"):
        value = getattr(config, name, getattr(defaults, name))
        if isinstance(value, bool):
            setattr(config, name, getattr(defaults, name))
            continue
        try:
            integer = int(value)
        except (TypeError, ValueError):
            integer = int(getattr(defaults, name))
        if integer <= 0:
            integer = int(getattr(defaults, name))
        setattr(config, name, integer)
    for name in ("openai_base_url", "anthropic_base_url", "gemini_base_url"):
        _validate_upstream_base(getattr(config, name))
    return config


def _safe_proxy_init(self, engine: Any, config: Any = None) -> None:
    selected = _sanitize_proxy_config(config or _proxy_config.ProxyConfig())
    _ORIGINAL_PROXY_INIT(self, engine, selected)


def _safe_config_from_env(cls):
    config = _ORIGINAL_CONFIG_FROM_ENV(cls)
    return _sanitize_proxy_config(config)


_ORIGINAL_BUILD_HEADERS = _proxy.PromptCompilerProxy._build_headers
_ORIGINAL_HANDLE_PROXY = _proxy.PromptCompilerProxy.handle_proxy
_ORIGINAL_FORWARD_RESPONSE = _proxy.PromptCompilerProxy._forward_response
_ORIGINAL_STREAM_RESPONSE = _proxy.PromptCompilerProxy._stream_response
_ORIGINAL_PROXY_INIT = _proxy.PromptCompilerProxy.__init__
_ORIGINAL_CONFIG_FROM_ENV = _proxy_config.ProxyConfig.from_env.__func__

_proxy._env_float = _finite_env_float
_proxy._parse_retry_after = _safe_retry_after
_proxy._http_client_kwargs = _safe_http_client_kwargs
_proxy._CircuitBreaker = CircuitBreaker
_proxy.PromptCompilerProxy.__init__ = _safe_proxy_init
_proxy.PromptCompilerProxy.startup = _safe_startup
_proxy.PromptCompilerProxy._ensure_client = _safe_ensure_client
_proxy.PromptCompilerProxy.handle_proxy = _safe_handle_proxy
_proxy.PromptCompilerProxy._resolve_target = _safe_resolve_target
_proxy.PromptCompilerProxy._build_headers = _safe_build_headers
_proxy.PromptCompilerProxy._warmup_connection = _safe_warmup_connection
_proxy.PromptCompilerProxy._forward_response = _safe_forward_response
_proxy.PromptCompilerProxy._stream_response = _safe_stream_response
_proxy_config._env_float = _finite_env_float
_proxy_config._env_optional_float = _finite_env_optional_float
_proxy_config.ProxyConfig.from_env = classmethod(_safe_config_from_env)

__all__ = [
    "BoundedAsyncClient",
    "CircuitBreaker",
    "_read_request_body_bounded",
    "_safe_http_client_kwargs",
    "_safe_retry_after",
    "_safe_target_url",
    "_sanitize_proxy_config",
]
