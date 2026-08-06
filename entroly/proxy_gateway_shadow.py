"""Non-authoritative gateway planning and unified proxy attempt observability.

This module patches the live ``PromptCompilerProxy`` at the same explicit
hardening seam used by the transport layers.  It observes provider requests,
runs ``GatewayControlPlane`` in shadow mode, and records bounded metadata only.
It never mutates the request body, target URL, model, provider, credentials,
retry behavior, or response body.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Mapping

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .cache_routing import CacheAwareRouter, CachePrice, ModelCandidate
from .gateway_control_plane import GatewayControlPlane
from .provider_adapters import (
    ProviderRequestAdapterResult,
    canonical_request_from_provider_body,
)
from .provider_policy import Capability, ProviderTarget
from .proxy_transform import detect_provider
from .stable_prefix import CanonicalPrefixBuilder, StablePrompt

logger = logging.getLogger("entroly.gateway_shadow")

_SCHEMA_VERSION = "entroly.gateway-shadow.v1"
_DEFAULT_MAX_RECORDS = 100
_MAX_RECORDS_LIMIT = 1_000
_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_CORRELATION_SALT = os.urandom(32)
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid %s=%r; using default %s", name, raw, default)
    return default


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except (TypeError, ValueError, OverflowError):
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        value = default
    return max(minimum, min(value, maximum))


def _bounded_identifier(value: object, *, fallback: str) -> str:
    text = str(value or "").strip()
    if _SAFE_ID_RE.fullmatch(text):
        return text
    return fallback


def _correlation_digest(value: str) -> str:
    return hashlib.sha256(
        _CORRELATION_SALT + value.encode("utf-8", errors="replace")
    ).hexdigest()[:16]


def _last_user_content(adapter: ProviderRequestAdapterResult) -> Any:
    for message in reversed(adapter.canonical.messages):
        if str(message.get("role", "")) == "user":
            return message.get("content", "")
    return ""


def _stable_prompt(adapter: ProviderRequestAdapterResult) -> StablePrompt:
    builder = CanonicalPrefixBuilder(
        namespace="entroly-gateway-shadow",
        version="1",
    )
    for message in adapter.canonical.messages:
        if str(message.get("role", "")) == "system":
            builder.add("system", message.get("content", ""), priority=10)
            break
    if adapter.canonical.tools:
        builder.add_tools(adapter.canonical.tools, priority=20)
    return builder.build(dynamic_tail=_last_user_content(adapter))


def _catalog_price(proxy: Any, provider: str, model: str) -> CachePrice | None:
    catalog = getattr(proxy, "_pricing_catalog", None)
    pricing = catalog.resolve(provider, model) if catalog is not None else None
    if pricing is None:
        return None
    return CachePrice(
        input_per_million=float(pricing.input_per_million),
        cache_read_per_million=float(pricing.cache_read_per_million),
        output_per_million=float(pricing.output_per_million),
        cache_write_per_million=float(pricing.cache_write_rate),
    )


def _provider_known_capabilities(provider: str) -> frozenset[Capability]:
    # Provider-wide declarations are intentionally conservative. Unknown
    # schema, reasoning, and cache-control parity keeps alternate models out.
    from .proxy_config import provider_capability

    profile = provider_capability(provider)
    capabilities = {Capability.CHAT, Capability.STREAMING}
    if profile.supports_tools:
        capabilities.add(Capability.TOOLS)
    if profile.supports_vision:
        capabilities.add(Capability.VISION)
    return frozenset(capabilities)


def _shadow_candidates_and_targets(
    proxy: Any,
    adapter: ProviderRequestAdapterResult,
) -> tuple[tuple[ModelCandidate, ...], tuple[ProviderTarget, ...]]:
    provider = adapter.provider
    current_model = adapter.canonical.model
    required = adapter.canonical.required_capabilities()
    current_price = _catalog_price(proxy, provider, current_model)

    candidates = [
        ModelCandidate(
            model=current_model,
            provider=provider,
            price=(
                current_price
                if current_price is not None
                else CachePrice(0.0, 0.0, 0.0, cache_write_per_million=0.0)
            ),
            quality=1.0,
        )
    ]
    targets = [
        ProviderTarget(
            provider=provider,
            model=current_model,
            capabilities=required,
            priority=0,
        )
    ]

    ladder = getattr(proxy, "_escalation_ladder", {})
    recommendation = ladder.get(current_model) if isinstance(ladder, Mapping) else None
    alternate_model = (
        recommendation[0]
        if isinstance(recommendation, tuple) and recommendation
        else ""
    )
    alternate_price = (
        _catalog_price(proxy, provider, alternate_model)
        if isinstance(alternate_model, str) and alternate_model
        else None
    )
    if (
        isinstance(alternate_model, str)
        and alternate_model
        and alternate_model != current_model
        and _SAFE_ID_RE.fullmatch(alternate_model)
        and current_price is not None
        and alternate_price is not None
        and required.issubset(_provider_known_capabilities(provider))
    ):
        candidates.append(
            ModelCandidate(
                model=alternate_model,
                provider=provider,
                price=alternate_price,
                quality=1.0,
            )
        )
        targets.append(
            ProviderTarget(
                provider=provider,
                model=alternate_model,
                capabilities=_provider_known_capabilities(provider),
                priority=100,
            )
        )

    return tuple(candidates), tuple(targets)


@dataclass(frozen=True, slots=True)
class GatewayShadowTicket:
    run_id: str
    attempt_id: str
    plan_id: str
    request_correlation: str
    source_provider: str
    source_model: str
    planned_provider: str
    planned_model: str
    route_reason: str
    boundary: str
    streaming: bool
    excluded_targets: int
    cross_provider_excluded: int
    planning_ms: float
    started_at: float


@dataclass(frozen=True, slots=True)
class GatewayShadowRecord:
    schema_version: str
    run_id: str
    attempt_id: str
    plan_id: str
    request_correlation: str
    source_provider: str
    source_model: str
    planned_provider: str
    planned_model: str
    route_reason: str
    boundary: str
    authoritative_path: str
    streaming: bool
    excluded_targets: int
    cross_provider_excluded: int
    planning_ms: float
    proxy_admission_ms: float
    response_status: int | None
    outcome: str
    observed_at: float


class GatewayShadowObserver:
    """Thread-safe bounded store for non-authoritative gateway evidence."""

    def __init__(
        self,
        *,
        control_plane: GatewayControlPlane,
        enabled: bool = True,
        emit_headers: bool = True,
        max_records: int = _DEFAULT_MAX_RECORDS,
    ) -> None:
        self.control_plane = control_plane
        self.enabled = bool(enabled)
        self.emit_headers = bool(emit_headers)
        self.max_records = max(1, min(int(max_records), _MAX_RECORDS_LIMIT))
        self._lock = threading.RLock()
        self._records: deque[GatewayShadowRecord] = deque(maxlen=self.max_records)
        self._planned = 0
        self._completed = 0
        self._skipped = 0
        self._failures = 0
        self._would_switch = 0
        self._last_error_type = ""

    def observe_request(
        self,
        proxy: Any,
        *,
        body: Mapping[str, Any],
        headers: Mapping[str, str],
        path: str,
        request_id: str,
    ) -> GatewayShadowTicket | None:
        if not self.enabled:
            return None

        started = time.perf_counter()
        provider = detect_provider(path, dict(headers), dict(body))
        adapter = canonical_request_from_provider_body(
            provider,
            body,
            headers=headers,
            path=path,
        )
        if not _SAFE_ID_RE.fullmatch(adapter.provider):
            raise ValueError("shadow provider identifier is not safe bounded text")
        if not _SAFE_ID_RE.fullmatch(adapter.canonical.model):
            raise ValueError("shadow model identifier is not safe bounded text")
        candidates, targets = _shadow_candidates_and_targets(proxy, adapter)
        prompt = _stable_prompt(adapter)
        plan = self.control_plane.plan(
            adapter.canonical,
            stable_prompt=prompt,
            source_provider=adapter.provider,
            current_model=adapter.canonical.model,
            candidates=candidates,
            targets=targets,
            prefix_tokens=adapter.prefix_tokens_estimate,
            new_input_tokens=adapter.new_input_tokens_estimate,
            expected_output_tokens=adapter.expected_output_tokens,
        )
        planning_ms = (time.perf_counter() - started) * 1000.0
        run_id = f"gw_{uuid.uuid4().hex[:16]}"
        attempt_id = f"{run_id}:0"
        excluded = dict(plan.failover.excluded)
        cross_provider_excluded = sum(
            1 for reason in excluded.values() if reason == "cross_provider_disabled"
        )
        plan_material = "\0".join(
            (
                plan.conversation_id,
                plan.source_provider,
                plan.routing.selected_provider,
                plan.routing.selected_model,
                plan.routing.reason,
                plan.stable_prompt.prefix_hash,
            )
        )
        plan_id = hashlib.sha256(plan_material.encode("utf-8")).hexdigest()[:20]
        ticket = GatewayShadowTicket(
            run_id=run_id,
            attempt_id=attempt_id,
            plan_id=plan_id,
            request_correlation=_correlation_digest(request_id),
            source_provider=plan.source_provider,
            source_model=adapter.canonical.model,
            planned_provider=plan.routing.selected_provider,
            planned_model=plan.routing.selected_model,
            route_reason=plan.routing.reason[:160],
            boundary="same-provider",
            streaming=adapter.canonical.stream,
            excluded_targets=len(excluded),
            cross_provider_excluded=cross_provider_excluded,
            planning_ms=round(planning_ms, 3),
            started_at=time.perf_counter(),
        )
        with self._lock:
            self._planned += 1
            if ticket.planned_model != ticket.source_model:
                self._would_switch += 1
        return ticket

    def complete(
        self,
        ticket: GatewayShadowTicket,
        *,
        response_status: int | None,
        outcome: str,
    ) -> GatewayShadowRecord:
        elapsed_ms = (time.perf_counter() - ticket.started_at) * 1000.0
        record = GatewayShadowRecord(
            schema_version=_SCHEMA_VERSION,
            run_id=ticket.run_id,
            attempt_id=ticket.attempt_id,
            plan_id=ticket.plan_id,
            request_correlation=ticket.request_correlation,
            source_provider=ticket.source_provider,
            source_model=ticket.source_model,
            planned_provider=ticket.planned_provider,
            planned_model=ticket.planned_model,
            route_reason=ticket.route_reason,
            boundary=ticket.boundary,
            authoritative_path="legacy-proxy",
            streaming=ticket.streaming,
            excluded_targets=ticket.excluded_targets,
            cross_provider_excluded=ticket.cross_provider_excluded,
            planning_ms=ticket.planning_ms,
            proxy_admission_ms=round(elapsed_ms, 3),
            response_status=response_status,
            outcome=outcome[:64],
            observed_at=time.time(),
        )
        with self._lock:
            self._records.append(record)
            self._completed += 1
        return record

    def skip(self) -> None:
        with self._lock:
            self._skipped += 1

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            self._failures += 1
            self._last_error_type = type(exc).__name__[:80]
        logger.debug("Gateway shadow observation skipped: %s", type(exc).__name__)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            records = [asdict(record) for record in self._records]
            return {
                "schema_version": _SCHEMA_VERSION,
                "mode": "shadow",
                "enabled": self.enabled,
                "authoritative_path": "legacy-proxy",
                "request_mutation": False,
                "provider_switch_execution": False,
                "automatic_retry_execution": False,
                "records_contain_prompt_content": False,
                "planned": self._planned,
                "completed": self._completed,
                "skipped": self._skipped,
                "failures": self._failures,
                "would_switch": self._would_switch,
                "last_error_type": self._last_error_type,
                "max_records": self.max_records,
                "recent": records,
            }


def _observer_for_proxy(proxy: Any) -> GatewayShadowObserver:
    observer = getattr(proxy, "_gateway_shadow_observer", None)
    if isinstance(observer, GatewayShadowObserver):
        return observer

    cache_router = getattr(proxy, "_cache_router", None)
    if not isinstance(cache_router, CacheAwareRouter):
        cache_router = CacheAwareRouter()
    redaction_policy = getattr(proxy, "_gateway_redaction", None)
    control_plane = GatewayControlPlane(
        cache_router=cache_router,
        redaction_policy=redaction_policy,
    )
    observer = GatewayShadowObserver(
        control_plane=control_plane,
        enabled=_env_flag("ENTROLY_GATEWAY_SHADOW", default=False),
        emit_headers=_env_flag("ENTROLY_GATEWAY_SHADOW_HEADERS", default=True),
        max_records=_env_int(
            "ENTROLY_GATEWAY_SHADOW_MAX_RECORDS",
            _DEFAULT_MAX_RECORDS,
            minimum=1,
            maximum=_MAX_RECORDS_LIMIT,
        ),
    )
    proxy._gateway_shadow_observer = observer
    return observer


def _shadow_response_headers(ticket: GatewayShadowTicket) -> dict[str, str]:
    decision = "would-switch" if ticket.planned_model != ticket.source_model else "stay"
    return {
        "X-Entroly-Gateway-Shadow": "observed",
        "X-Entroly-Gateway-Run-Id": ticket.run_id,
        "X-Entroly-Gateway-Attempt-Id": ticket.attempt_id,
        "X-Entroly-Gateway-Plan-Id": ticket.plan_id,
        "X-Entroly-Gateway-Boundary": ticket.boundary,
        "X-Entroly-Gateway-Decision": decision,
        "X-Entroly-Gateway-Authoritative": "legacy-proxy",
    }


async def _run_shadow_handle_proxy(
    proxy: Any,
    request: Request,
    original: Callable[[Any, Request], Awaitable[Response]],
) -> Response:
    observer = _observer_for_proxy(proxy)
    ticket: GatewayShadowTicket | None = None
    if observer.enabled:
        try:
            body_bytes = await request.body()
            if len(body_bytes) > _MAX_REQUEST_BYTES:
                observer.skip()
            else:
                decoded = json.loads(body_bytes)
                if isinstance(decoded, dict):
                    headers = {key: value for key, value in request.headers.items()}
                    request_id = _bounded_identifier(
                        headers.get("x-request-id"),
                        fallback=uuid.uuid4().hex[:12],
                    )
                    ticket = observer.observe_request(
                        proxy,
                        body=decoded,
                        headers=headers,
                        path=request.url.path,
                        request_id=request_id,
                    )
                else:
                    observer.skip()
        except (json.JSONDecodeError, UnicodeDecodeError):
            observer.skip()
        except Exception as exc:
            observer.fail(exc)

    try:
        response = await original(proxy, request)
    except Exception as exc:
        if ticket is not None:
            observer.complete(
                ticket,
                response_status=None,
                outcome=f"proxy_exception:{type(exc).__name__}",
            )
        raise

    if ticket is not None:
        observer.complete(
            ticket,
            response_status=int(getattr(response, "status_code", 0) or 0),
            outcome="response_admitted",
        )
        if observer.emit_headers:
            for name, value in _shadow_response_headers(ticket).items():
                response.headers.setdefault(name, value)
    return response


async def _gateway_shadow_endpoint(request: Request) -> Response:
    proxy = request.app.state.proxy
    observer = _observer_for_proxy(proxy)
    return JSONResponse(
        observer.snapshot(),
        headers={
            "Cache-Control": "no-store, max-age=0",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _install_route(app: Any) -> None:
    routes = getattr(getattr(app, "router", None), "routes", None)
    if not isinstance(routes, list):
        return
    if any(getattr(route, "path", None) == "/gateway-shadow" for route in routes):
        return
    from . import proxy as _proxy

    endpoint = _proxy._sidecar_guard(_gateway_shadow_endpoint)
    routes.insert(
        0,
        Route(
            "/gateway-shadow",
            endpoint=endpoint,
            methods=["GET"],
            name="gateway-shadow",
        ),
    )


def install_gateway_shadow() -> None:
    """Install shadow planning behind the existing bounded transport gate."""
    from . import proxy as _proxy
    from . import proxy_transport_safe as _transport

    # Keep ``_safe_handle_proxy`` as the class entrypoint. It validates and
    # bounds the body before calling this module through its runtime-resolved
    # core-handler global, so shadow mode can never bypass the memory limit.
    current_core = _transport._ORIGINAL_HANDLE_PROXY
    original_core = getattr(
        current_core,
        "__entroly_gateway_shadow_original__",
        current_core,
    )

    async def shadow_core_handle(self: Any, request: Request) -> Response:
        return await _run_shadow_handle_proxy(self, request, original_core)

    shadow_core_handle.__entroly_gateway_shadow_original__ = original_core
    _transport._ORIGINAL_HANDLE_PROXY = shadow_core_handle

    current_factory = _proxy.create_proxy_app
    original_factory = getattr(
        current_factory,
        "__entroly_gateway_shadow_original__",
        current_factory,
    )

    def shadow_create_proxy_app(*args: Any, **kwargs: Any):
        app = original_factory(*args, **kwargs)
        _install_route(app)
        return app

    shadow_create_proxy_app.__entroly_gateway_shadow_original__ = original_factory
    _proxy.create_proxy_app = shadow_create_proxy_app


install_gateway_shadow()


__all__ = [
    "GatewayShadowObserver",
    "GatewayShadowRecord",
    "GatewayShadowTicket",
    "_gateway_shadow_endpoint",
    "_run_shadow_handle_proxy",
    "install_gateway_shadow",
]
