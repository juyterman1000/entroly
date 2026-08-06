"""Single-authority same-provider routing workflow for the live proxy.

Entroly already has several valuable routing components:

* RAVS proposes evidence-gated model changes.
* Cache economics rejects changes that destroy more value than they save.
* Provider adapters preserve provider-specific request semantics.
* The gateway policy fixes the data-recipient boundary.
* Usage ledgers account for provider-reported spend.

This module does not add another router. It coordinates the final execution
seam so at most one same-provider model rewrite can occur for a request. The
workflow is opt-in and compatibility preserving: when disabled, the existing
proxy behavior is untouched.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any, Awaitable, Callable, Mapping
from urllib.parse import urlparse

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .cache_routing import CachePrice, ModelCandidate
from .gateway_control_plane import GatewayControlPlane
from .model_decision_receipt import build_model_decision_receipt
from .models.registry import RegistryTrust, resolve_model
from .provider_adapters import (
    ProviderRequestAdapterResult,
    apply_target_same_provider as _adapter_apply_target_same_provider,
    canonical_request_from_provider_body,
)
from .provider_policy import Capability, ProviderTarget
from .proxy_transform import detect_provider
from .stable_prefix import CanonicalPrefixBuilder, StablePrompt

logger = logging.getLogger("entroly.routing_authority")

_SCHEMA_VERSION = "entroly.routing-authority.v1"
_DEFAULT_MAX_RECORDS = 100
_MAX_RECORDS_LIMIT = 1_000
_CORRELATION_SALT = os.urandom(32)
_AUTH_PRESENCE_HEADERS = frozenset(
    {
        "authorization",
        "api-key",
        "x-api-key",
        "x-goog-api-key",
    }
)
_PORTABLE_HEADERS = frozenset(
    {
        "x-request-id",
        "x-entroly-team",
        "x-entroly-project",
        "x-entroly-tool",
    }
)
_STRONG_TRUST = frozenset(
    {
        RegistryTrust.VERIFIED,
        RegistryTrust.DISCOVERED,
        RegistryTrust.USER,
    }
)
_CURRENT_CONTEXT: contextvars.ContextVar["RoutingRequestContext | None"] = (
    contextvars.ContextVar("entroly_routing_authority_context", default=None)
)


class RoutingAuthorityDenied(RuntimeError):
    """Internal fail-original signal consumed by the existing RAVS guard."""


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
        parsed = default if raw is None else int(raw)
    except (TypeError, ValueError, OverflowError):
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        parsed = default
    return max(minimum, min(parsed, maximum))


def _mode_from_env() -> str:
    mode = os.environ.get("ENTROLY_ROUTING_AUTHORITY_MODE", "observe")
    normalized = mode.strip().casefold()
    if normalized not in {"observe", "execute"}:
        logger.warning(
            "Invalid ENTROLY_ROUTING_AUTHORITY_MODE=%r; using observe",
            mode,
        )
        return "observe"
    return normalized


def _bounded(value: object, *, limit: int = 160) -> str:
    normalized = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return normalized[:limit]


def _correlation_digest(value: str) -> str:
    return hashlib.sha256(
        _CORRELATION_SALT + value.encode("utf-8", errors="replace")
    ).hexdigest()[:16]


def _safe_request_headers(request: Request) -> dict[str, str]:
    """Keep only provider-presence signals and bounded non-secret metadata."""
    result: dict[str, str] = {}
    for name, value in request.headers.items():
        lower = name.lower()
        if lower in _AUTH_PRESENCE_HEADERS:
            result[lower] = "present"
        elif lower in _PORTABLE_HEADERS:
            result[lower] = _bounded(value, limit=256)
    return result


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _estimate_cost_micro_usd(
    pricing: Any,
    adapter: ProviderRequestAdapterResult,
) -> int:
    input_tokens = (
        adapter.prefix_tokens_estimate + adapter.new_input_tokens_estimate
    )
    raw = (
        Decimal(input_tokens) * pricing.input_per_million
        + Decimal(adapter.expected_output_tokens) * pricing.output_per_million
    ) / Decimal(1_000_000)
    return int((raw * Decimal(1_000_000)).quantize(Decimal("1")))


def _cache_price(pricing: Any) -> CachePrice:
    return CachePrice(
        input_per_million=float(pricing.input_per_million),
        cache_read_per_million=float(pricing.cache_read_per_million),
        output_per_million=float(pricing.output_per_million),
        cache_write_per_million=float(pricing.cache_write_rate),
    )


def _last_user_content(adapter: ProviderRequestAdapterResult) -> Any:
    for message in reversed(adapter.canonical.messages):
        if str(message.get("role", "")) == "user":
            return message.get("content", "")
    return ""


def _stable_prompt(adapter: ProviderRequestAdapterResult) -> StablePrompt:
    builder = CanonicalPrefixBuilder(
        namespace="entroly-routing-authority",
        version="1",
    )
    for message in adapter.canonical.messages:
        if str(message.get("role", "")) == "system":
            builder.add("system", message.get("content", ""), priority=10)
            break
    if adapter.canonical.tools:
        builder.add_tools(adapter.canonical.tools, priority=20)
    return builder.build(dynamic_tail=_last_user_content(adapter))


def _gateway_control_plane(proxy: Any) -> GatewayControlPlane:
    observer = getattr(proxy, "_gateway_shadow_observer", None)
    existing = getattr(observer, "control_plane", None)
    if isinstance(existing, GatewayControlPlane):
        return existing
    current = getattr(proxy, "_routing_gateway_control_plane", None)
    if isinstance(current, GatewayControlPlane):
        return current
    control_plane = GatewayControlPlane(
        cache_router=getattr(proxy, "_cache_router", None),
        redaction_policy=getattr(proxy, "_gateway_redaction", None),
        usage_ledger=getattr(proxy, "_usage_ledger", None),
    )
    proxy._routing_gateway_control_plane = control_plane
    return control_plane


def _target_capabilities(
    provider: str,
    model: str,
) -> tuple[frozenset[Capability], str, str]:
    resolution = resolve_model(model)
    capability = resolution.capability
    if capability is None:
        return frozenset(), resolution.trust.value, "unknown_target_model"
    if not resolution.exact:
        return frozenset(), resolution.trust.value, "target_model_not_exact"
    if resolution.trust == RegistryTrust.ANNOUNCED:
        # Announced records may authorize only the provider's basic text chat
        # surface, and only after the separate explicit-pricing gate passes.
        # Advanced capabilities remain unproven until verified/discovered/user
        # metadata exists.
        return (
            frozenset({Capability.CHAT, Capability.STREAMING}),
            resolution.trust.value,
            "",
        )
    if resolution.trust not in _STRONG_TRUST:
        return frozenset(), resolution.trust.value, "target_metadata_not_execution_trusted"

    known = {Capability.CHAT, Capability.STREAMING}
    if capability.supports_tools is True:
        known.add(Capability.TOOLS)
    if capability.supports_vision is True:
        known.add(Capability.VISION)
    if capability.supports_reasoning is True:
        known.add(Capability.REASONING)

    # The model registry currently has no model-level JSON-schema or
    # cache-control parity fields. Those capabilities therefore remain
    # observe-only instead of being inferred from provider-wide behavior.
    return frozenset(known), resolution.trust.value, ""


def _registry_provider_pair_is_safe(
    source_provider: str,
    source_model: str,
    target_model: str,
) -> tuple[bool, str]:
    source = resolve_model(source_model)
    target = resolve_model(target_model)
    if target.capability is None:
        return False, "unknown_target_model"
    if source.capability is None:
        return False, "unknown_source_model"
    source_registry_provider = source.capability.provider
    target_registry_provider = target.capability.provider
    if source_registry_provider != target_registry_provider:
        return False, "registry_provider_mismatch"
    if (
        source_provider != "openai"
        and target_registry_provider != source_provider
    ):
        return False, "transport_registry_provider_mismatch"
    return True, ""


@dataclass(slots=True)
class RoutingRequestContext:
    proxy: Any
    run_id: str
    request_correlation: str
    path: str
    headers: Mapping[str, str]
    mode: str
    require_pricing: bool
    emit_headers: bool
    started_at: float = field(default_factory=time.perf_counter)
    proposal_count: int = 0
    mutation_count: int = 0
    source_provider: str = ""
    source_model: str = ""
    proposed_provider: str = ""
    proposed_model: str = ""
    executed_provider: str = ""
    executed_model: str = ""
    decision: str = "no_proposal"
    reason: str = ""
    conflict_reason: str = ""
    required_capabilities: tuple[str, ...] = ()
    source_pricing: str = ""
    target_pricing: str = ""
    estimated_source_micro_usd: int | None = None
    estimated_target_micro_usd: int | None = None
    source_model_receipt: str = ""
    target_model_receipt: str = ""
    target_model_trust: str = ""
    gateway_plan_id: str = ""
    gateway_route_reason: str = ""
    gateway_redaction_changed: bool = False
    escalation_mode: str = "observe"


@dataclass(frozen=True, slots=True)
class RoutingAuthorityReceipt:
    schema_version: str
    run_id: str
    request_correlation: str
    mode: str
    proposer: str
    source_provider: str
    source_model: str
    proposed_provider: str
    proposed_model: str
    executed_provider: str
    executed_model: str
    decision: str
    reason: str
    conflict_reason: str
    required_capabilities: tuple[str, ...]
    source_pricing: str
    target_pricing: str
    estimated_source_micro_usd: int | None
    estimated_target_micro_usd: int | None
    source_model_receipt: str
    target_model_receipt: str
    target_model_trust: str
    gateway_plan_id: str
    gateway_route_reason: str
    gateway_redaction_changed: bool
    escalation_mode: str
    mutation_count: int
    response_status: int | None
    outcome: str
    elapsed_ms: float
    observed_at: float
    receipt_digest: str

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("receipt_digest", None)
        return value

    def verify(self) -> bool:
        return (
            hashlib.sha256(_canonical_json(self.payload())).hexdigest()
            == self.receipt_digest
        )


class RoutingAuthorityLedger:
    """Thread-safe bounded evidence store with no request content."""

    def __init__(self, *, max_records: int = _DEFAULT_MAX_RECORDS) -> None:
        self.max_records = max(1, min(int(max_records), _MAX_RECORDS_LIMIT))
        self._records: deque[RoutingAuthorityReceipt] = deque(
            maxlen=self.max_records
        )
        self._lock = threading.RLock()
        self._requests = 0
        self._proposals = 0
        self._executed = 0
        self._denied = 0
        self._conflicts = 0
        self._failures = 0

    def register_request(self) -> None:
        with self._lock:
            self._requests += 1

    def register_proposal(self) -> None:
        with self._lock:
            self._proposals += 1

    def register_conflict(self) -> None:
        with self._lock:
            self._conflicts += 1

    def register_failure(self) -> None:
        with self._lock:
            self._failures += 1

    def complete(
        self,
        context: RoutingRequestContext,
        *,
        response_status: int | None,
        outcome: str,
    ) -> RoutingAuthorityReceipt | None:
        if context.proposal_count == 0:
            return None
        unsigned: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": context.run_id,
            "request_correlation": context.request_correlation,
            "mode": context.mode,
            "proposer": "ravs",
            "source_provider": context.source_provider,
            "source_model": context.source_model,
            "proposed_provider": context.proposed_provider,
            "proposed_model": context.proposed_model,
            "executed_provider": context.executed_provider,
            "executed_model": context.executed_model,
            "decision": context.decision,
            "reason": context.reason,
            "conflict_reason": context.conflict_reason,
            "required_capabilities": context.required_capabilities,
            "source_pricing": context.source_pricing,
            "target_pricing": context.target_pricing,
            "estimated_source_micro_usd": context.estimated_source_micro_usd,
            "estimated_target_micro_usd": context.estimated_target_micro_usd,
            "source_model_receipt": context.source_model_receipt,
            "target_model_receipt": context.target_model_receipt,
            "target_model_trust": context.target_model_trust,
            "gateway_plan_id": context.gateway_plan_id,
            "gateway_route_reason": context.gateway_route_reason,
            "gateway_redaction_changed": context.gateway_redaction_changed,
            "escalation_mode": context.escalation_mode,
            "mutation_count": context.mutation_count,
            "response_status": response_status,
            "outcome": _bounded(outcome, limit=80),
            "elapsed_ms": round(
                (time.perf_counter() - context.started_at) * 1000.0,
                3,
            ),
            "observed_at": time.time(),
        }
        digest = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        receipt = RoutingAuthorityReceipt(
            receipt_digest=digest,
            **unsigned,
        )
        with self._lock:
            self._records.append(receipt)
            if context.decision == "executed":
                self._executed += 1
            else:
                self._denied += 1
        return receipt

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": _SCHEMA_VERSION,
                "enabled": _env_flag(
                    "ENTROLY_ROUTING_AUTHORITY",
                    default=False,
                ),
                "mode": _mode_from_env(),
                "authoritative_proposer": "ravs",
                "same_provider_only": True,
                "cross_provider_execution": False,
                "automatic_retry_execution": False,
                "records_contain_prompt_content": False,
                "records_contain_credentials": False,
                "requests": self._requests,
                "proposals": self._proposals,
                "executed": self._executed,
                "denied": self._denied,
                "conflicts": self._conflicts,
                "failures": self._failures,
                "max_records": self.max_records,
                "recent": [asdict(record) for record in self._records],
            }


class RoutingAuthorityCoordinator:
    """Validate the existing RAVS proposal at the final rewrite seam."""

    def __init__(self, ledger: RoutingAuthorityLedger) -> None:
        self.ledger = ledger

    def _deny(
        self,
        context: RoutingRequestContext,
        reason: str,
        *,
        conflict: bool = False,
    ) -> None:
        if context.mutation_count > 0:
            context.conflict_reason = reason
        else:
            context.decision = "denied"
            context.reason = reason
        if conflict:
            self.ledger.register_conflict()
        raise RoutingAuthorityDenied(reason)

    def authorize(
        self,
        context: RoutingRequestContext,
        *,
        provider: str,
        target: ProviderTarget,
        body: Mapping[str, Any],
        url: str,
        apply_fn: Callable[..., tuple[dict[str, Any], str]],
    ) -> tuple[dict[str, Any], str]:
        context.proposal_count += 1
        self.ledger.register_proposal()
        if context.proposal_count > 1 or context.mutation_count > 0:
            self._deny(
                context,
                "multiple_model_mutation_attempts",
                conflict=True,
            )

        source_provider = detect_provider(
            context.path,
            dict(context.headers),
            dict(body),
        )
        adapter = canonical_request_from_provider_body(
            source_provider,
            body,
            headers=context.headers,
            path=context.path,
        )
        source_model = adapter.canonical.model
        context.source_provider = source_provider
        context.source_model = source_model
        context.proposed_provider = target.provider
        context.proposed_model = target.model
        context.required_capabilities = tuple(
            sorted(capability.value for capability in adapter.canonical.required_capabilities())
        )
        context.escalation_mode = str(
            getattr(context.proxy, "_escalation_mode", "observe")
        ).strip().casefold()

        if provider.strip().casefold() != source_provider:
            self._deny(context, "proxy_provider_mismatch")
        if target.provider.strip().casefold() != source_provider:
            self._deny(context, "cross_provider_target_disabled")
        if target.model == source_model:
            self._deny(context, "target_model_unchanged")

        if context.escalation_mode in {"active", "shadow"}:
            self._deny(
                context,
                "competing_escalation_authority",
                conflict=True,
            )

        pair_safe, pair_reason = _registry_provider_pair_is_safe(
            source_provider,
            source_model,
            target.model,
        )
        if not pair_safe:
            self._deny(context, pair_reason)

        known_capabilities, target_trust, capability_reason = (
            _target_capabilities(source_provider, target.model)
        )
        context.target_model_trust = target_trust
        if capability_reason:
            self._deny(context, capability_reason)
        required = adapter.canonical.required_capabilities()
        if not required.issubset(known_capabilities):
            missing = ",".join(
                sorted(capability.value for capability in required - known_capabilities)
            )
            self._deny(context, f"target_capability_unproven:{missing}")

        target_resolution = resolve_model(target.model)
        input_tokens = (
            adapter.prefix_tokens_estimate + adapter.new_input_tokens_estimate
        )
        if input_tokens > target_resolution.effective_input_budget(
            requested_output_tokens=adapter.expected_output_tokens
        ):
            self._deny(context, "target_context_budget_exceeded")

        catalog = getattr(context.proxy, "_pricing_catalog", None)
        source_pricing = (
            catalog.resolve(source_provider, source_model)
            if catalog is not None
            else None
        )
        target_pricing = (
            catalog.resolve(source_provider, target.model)
            if catalog is not None
            else None
        )
        if context.require_pricing and (
            source_pricing is None or target_pricing is None
        ):
            self._deny(context, "auditable_pricing_required")

        if source_pricing is not None and target_pricing is not None:
            context.source_pricing = _bounded(source_pricing.source)
            context.target_pricing = _bounded(target_pricing.source)
            context.estimated_source_micro_usd = _estimate_cost_micro_usd(
                source_pricing,
                adapter,
            )
            context.estimated_target_micro_usd = _estimate_cost_micro_usd(
                target_pricing,
                adapter,
            )
            if (
                context.estimated_target_micro_usd
                >= context.estimated_source_micro_usd
            ):
                self._deny(context, "target_not_cheaper")

        if source_pricing is None or target_pricing is None:
            source_cache_price = CachePrice(0.0, 0.0, 0.0, 0.0)
            target_cache_price = CachePrice(0.0, 0.0, 0.0, 0.0)
        else:
            source_cache_price = _cache_price(source_pricing)
            target_cache_price = _cache_price(target_pricing)

        current_candidate = ModelCandidate(
            model=source_model,
            provider=source_provider,
            price=source_cache_price,
            quality=1.0,
        )
        target_candidate = ModelCandidate(
            model=target.model,
            provider=source_provider,
            price=target_cache_price,
            quality=1.0,
        )
        current_target = ProviderTarget(
            provider=source_provider,
            model=source_model,
            capabilities=adapter.canonical.required_capabilities(),
            priority=0,
            expected_input_cost_per_million=source_cache_price.input_per_million,
        )
        authorized_target = ProviderTarget(
            provider=source_provider,
            model=target.model,
            capabilities=known_capabilities,
            priority=100,
            expected_input_cost_per_million=target_cache_price.input_per_million,
        )
        gateway_plan = _gateway_control_plane(context.proxy).plan(
            adapter.canonical,
            stable_prompt=_stable_prompt(adapter),
            source_provider=source_provider,
            current_model=source_model,
            candidates=(current_candidate, target_candidate),
            targets=(current_target, authorized_target),
            prefix_tokens=adapter.prefix_tokens_estimate,
            new_input_tokens=adapter.new_input_tokens_estimate,
            expected_output_tokens=adapter.expected_output_tokens,
            force_model=target.model,
        )
        if (
            gateway_plan.routing.selected_provider != source_provider
            or gateway_plan.routing.selected_model != target.model
            or gateway_plan.failover.primary.key != f"{source_provider}:{target.model}"
        ):
            self._deny(context, "gateway_plan_target_mismatch")
        plan_material = "\0".join(
            (
                gateway_plan.source_provider,
                gateway_plan.routing.selected_provider,
                gateway_plan.routing.selected_model,
                gateway_plan.routing.reason,
                gateway_plan.stable_prompt.prefix_hash,
            )
        )
        context.gateway_plan_id = hashlib.sha256(
            plan_material.encode("utf-8")
        ).hexdigest()[:20]
        context.gateway_route_reason = _bounded(
            gateway_plan.routing.reason,
            limit=120,
        )
        context.gateway_redaction_changed = gateway_plan.redaction.changed

        source_receipt = build_model_decision_receipt(
            body,
            provider=source_provider,
            path=context.path,
        )
        context.source_model_receipt = (
            source_receipt.receipt_digest
            if source_receipt is not None and source_receipt.verify()
            else ""
        )

        if context.mode != "execute":
            self._deny(context, "observe_only_would_execute")

        rewritten_body, rewritten_url = apply_fn(
            provider=provider,
            target=target,
            body=body,
            url=url,
        )
        before = urlparse(url)
        after = urlparse(rewritten_url)
        if (
            before.scheme,
            before.netloc,
        ) != (
            after.scheme,
            after.netloc,
        ):
            self._deny(context, "target_origin_changed")

        target_receipt = build_model_decision_receipt(
            rewritten_body,
            provider=source_provider,
            path=after.path,
        )
        if target_receipt is None or not target_receipt.verify():
            self._deny(context, "target_model_receipt_unavailable")

        context.target_model_receipt = target_receipt.receipt_digest
        context.mutation_count = 1
        context.executed_provider = source_provider
        context.executed_model = target.model
        context.decision = "executed"
        context.reason = "same_provider_evidence_gates_passed"
        return rewritten_body, rewritten_url


def _ledger_for_proxy(proxy: Any) -> RoutingAuthorityLedger:
    ledger = getattr(proxy, "_routing_authority_ledger", None)
    if isinstance(ledger, RoutingAuthorityLedger):
        return ledger
    ledger = RoutingAuthorityLedger(
        max_records=_env_int(
            "ENTROLY_ROUTING_AUTHORITY_MAX_RECORDS",
            _DEFAULT_MAX_RECORDS,
            minimum=1,
            maximum=_MAX_RECORDS_LIMIT,
        )
    )
    proxy._routing_authority_ledger = ledger
    proxy._routing_authority_coordinator = RoutingAuthorityCoordinator(ledger)
    return ledger


def _coordinator_for_proxy(proxy: Any) -> RoutingAuthorityCoordinator:
    coordinator = getattr(proxy, "_routing_authority_coordinator", None)
    if isinstance(coordinator, RoutingAuthorityCoordinator):
        return coordinator
    _ledger_for_proxy(proxy)
    return proxy._routing_authority_coordinator


def _response_headers(receipt: RoutingAuthorityReceipt) -> dict[str, str]:
    return {
        "X-Entroly-Routing-Authority": "same-provider",
        "X-Entroly-Routing-Run-Id": receipt.run_id,
        "X-Entroly-Routing-Decision": receipt.decision,
        "X-Entroly-Routing-Reason": _bounded(receipt.reason, limit=120),
        "X-Entroly-Routing-Receipt": receipt.receipt_digest,
    }


async def _run_authority_handle_proxy(
    proxy: Any,
    request: Request,
    original: Callable[[Any, Request], Awaitable[Response]],
) -> Response:
    if not _env_flag("ENTROLY_ROUTING_AUTHORITY", default=False):
        return await original(proxy, request)

    safe_headers = _safe_request_headers(request)
    request_id = safe_headers.get("x-request-id") or uuid.uuid4().hex[:12]
    context = RoutingRequestContext(
        proxy=proxy,
        run_id=f"route_{uuid.uuid4().hex[:16]}",
        request_correlation=_correlation_digest(request_id),
        path=request.url.path,
        headers=safe_headers,
        mode=_mode_from_env(),
        require_pricing=_env_flag(
            "ENTROLY_ROUTING_AUTHORITY_REQUIRE_PRICING",
            default=True,
        ),
        emit_headers=_env_flag(
            "ENTROLY_ROUTING_AUTHORITY_HEADERS",
            default=True,
        ),
    )
    ledger = _ledger_for_proxy(proxy)
    ledger.register_request()
    token = _CURRENT_CONTEXT.set(context)
    response: Response | None = None
    try:
        response = await original(proxy, request)
    except Exception:
        ledger.register_failure()
        if context.proposal_count:
            try:
                ledger.complete(
                    context,
                    response_status=None,
                    outcome="proxy_exception",
                )
            except Exception:
                logger.warning(
                    "Routing-authority exception receipt failed",
                    exc_info=True,
                )
        raise
    finally:
        _CURRENT_CONTEXT.reset(token)

    try:
        receipt = ledger.complete(
            context,
            response_status=int(getattr(response, "status_code", 0) or 0),
            outcome="response_admitted",
        )
        if (
            receipt is not None
            and context.emit_headers
            and response is not None
        ):
            for name, value in _response_headers(receipt).items():
                response.headers.setdefault(name, value)
    except Exception:
        ledger.register_failure()
        logger.warning(
            "Routing-authority response receipt failed",
            exc_info=True,
        )
    return response


def _coordinated_apply_target_same_provider(
    *,
    provider: str,
    target: ProviderTarget,
    body: Mapping[str, Any],
    url: str,
) -> tuple[dict[str, Any], str]:
    context = _CURRENT_CONTEXT.get()
    if context is None:
        return _adapter_apply_target_same_provider(
            provider=provider,
            target=target,
            body=body,
            url=url,
        )
    coordinator = _coordinator_for_proxy(context.proxy)
    return coordinator.authorize(
        context,
        provider=provider,
        target=target,
        body=body,
        url=url,
        apply_fn=_adapter_apply_target_same_provider,
    )


async def _routing_authority_endpoint(request: Request) -> Response:
    proxy = request.app.state.proxy
    return JSONResponse(
        _ledger_for_proxy(proxy).snapshot(),
        headers={
            "Cache-Control": "no-store, max-age=0",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _install_route(app: Any) -> None:
    routes = getattr(getattr(app, "router", None), "routes", None)
    if not isinstance(routes, list):
        return
    if any(
        getattr(route, "path", None) == "/routing-authority"
        for route in routes
    ):
        return
    from . import proxy as _proxy

    endpoint = _proxy._sidecar_guard(_routing_authority_endpoint)
    routes.insert(
        0,
        Route(
            "/routing-authority",
            endpoint=endpoint,
            methods=["GET"],
            name="routing-authority",
        ),
    )


def install_routing_authority() -> None:
    """Install the coordinator behind the bounded transport handler."""
    from . import proxy as _proxy
    from . import proxy_transport_safe as _transport

    current_core = _transport._ORIGINAL_HANDLE_PROXY
    if hasattr(current_core, "__entroly_routing_authority_original__"):
        return
    original_core = current_core

    async def authority_core_handle(self: Any, request: Request) -> Response:
        return await _run_authority_handle_proxy(
            self,
            request,
            original_core,
        )

    authority_core_handle.__entroly_routing_authority_original__ = original_core
    if hasattr(original_core, "__entroly_gateway_shadow_original__"):
        authority_core_handle.__entroly_gateway_shadow_original__ = getattr(
            original_core,
            "__entroly_gateway_shadow_original__",
        )
    _transport._ORIGINAL_HANDLE_PROXY = authority_core_handle

    current_apply = _proxy.apply_target_same_provider
    if not hasattr(current_apply, "__entroly_routing_authority_original__"):
        _coordinated_apply_target_same_provider.__entroly_routing_authority_original__ = (
            current_apply
        )
        _proxy.apply_target_same_provider = (
            _coordinated_apply_target_same_provider
        )


install_routing_authority()


__all__ = [
    "RoutingAuthorityCoordinator",
    "RoutingAuthorityDenied",
    "RoutingAuthorityLedger",
    "RoutingAuthorityReceipt",
    "RoutingRequestContext",
    "_coordinated_apply_target_same_provider",
    "_install_route",
    "_run_authority_handle_proxy",
    "install_routing_authority",
]
