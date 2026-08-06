"""Read-only gateway planning for proxy shadow evaluation.

This module deliberately cannot return a provider request body.  It converts an
inbound provider request into the canonical gateway representation, asks the
same-provider ``GatewayControlPlane`` for a plan, and returns only a bounded
receipt suitable for logs, metrics, or response headers.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .cache_routing import CachePrice, ModelCandidate
from .gateway_control_plane import GatewayControlPlane
from .provider_adapters import canonical_request_from_provider_body
from .provider_policy import ProviderTarget
from .stable_prefix import CanonicalPrefixBuilder


@dataclass(frozen=True, slots=True)
class GatewayShadowReceipt:
    """Bounded, content-free result of one shadow planning attempt."""

    request_id: str
    provider: str
    current_model: str
    planned_model: str
    conversation_id: str
    route_reason: str
    required_capabilities: tuple[str, ...]
    executable_targets: tuple[str, ...]
    excluded_targets: tuple[tuple[str, str], ...]
    redaction_changed: bool
    body_unchanged: bool
    duration_ms: float
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "entroly.gateway-shadow.v1",
            "request_id": self.request_id,
            "provider": self.provider,
            "current_model": self.current_model,
            "planned_model": self.planned_model,
            "conversation_id": self.conversation_id,
            "route_reason": self.route_reason,
            "required_capabilities": list(self.required_capabilities),
            "executable_targets": list(self.executable_targets),
            "excluded_targets": [
                {"target": target, "reason": reason}
                for target, reason in self.excluded_targets
            ],
            "redaction_changed": self.redaction_changed,
            "body_unchanged": self.body_unchanged,
            "duration_ms": round(self.duration_ms, 3),
            "error": self.error,
        }

    def headers(self) -> dict[str, str]:
        """Return bounded headers; never expose prompts, tools, or credentials."""
        status = "ok" if self.succeeded else "error"
        return {
            "X-Entroly-Gateway-Shadow": status,
            "X-Entroly-Gateway-Shadow-Provider": self.provider[:64],
            "X-Entroly-Gateway-Shadow-Model": self.planned_model[:128],
            "X-Entroly-Gateway-Shadow-Reason": self.route_reason[:256],
            "X-Entroly-Gateway-Shadow-Unchanged": (
                "1" if self.body_unchanged else "0"
            ),
        }


class GatewayShadowObserver:
    """Evaluate gateway plans without changing the live request.

    The observer intentionally supplies only the current provider/model as an
    executable candidate.  This validates canonicalization, capability checks,
    provider-boundary enforcement, stable identity, redaction planning, and
    receipt generation without creating a second routing authority.
    """

    def __init__(self, control_plane: GatewayControlPlane | None = None) -> None:
        self._control_plane = control_plane or GatewayControlPlane()

    @staticmethod
    def _body_digest(body: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _stable_prompt(canonical) -> Any:
        builder = CanonicalPrefixBuilder(namespace="entroly-gateway-shadow")
        system_messages = [
            message
            for message in canonical.messages
            if str(message.get("role", "")) == "system"
        ]
        if system_messages:
            builder.add("system", system_messages, priority=10)
        if canonical.tools:
            builder.add_tools(canonical.tools, priority=20)
        dynamic_tail = next(
            (
                message
                for message in reversed(canonical.messages)
                if str(message.get("role", "")) == "user"
            ),
            "",
        )
        return builder.build(dynamic_tail=dynamic_tail)

    def observe(
        self,
        *,
        provider: str,
        body: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
        path: str = "",
        request_id: str = "",
    ) -> GatewayShadowReceipt:
        started = time.perf_counter()
        before = self._body_digest(body)
        safe_request_id = str(request_id)[:128]
        provider_name = str(provider).lower()[:64]
        current_model = ""
        try:
            adapter = canonical_request_from_provider_body(
                provider_name,
                body,
                headers=headers,
                path=path,
            )
            canonical = adapter.canonical
            current_model = canonical.model
            required = canonical.required_capabilities()
            target = ProviderTarget(
                provider=provider_name,
                model=current_model,
                capabilities=required,
            )
            candidate = ModelCandidate(
                model=current_model,
                provider=provider_name,
                price=CachePrice(0.0, 0.0, 0.0),
                quality=1.0,
            )
            plan = self._control_plane.plan(
                canonical,
                stable_prompt=self._stable_prompt(canonical),
                source_provider=provider_name,
                current_model=current_model,
                candidates=(candidate,),
                targets=(target,),
                prefix_tokens=adapter.prefix_tokens_estimate,
                new_input_tokens=adapter.new_input_tokens_estimate,
                expected_output_tokens=adapter.expected_output_tokens,
            )
            after = self._body_digest(body)
            return GatewayShadowReceipt(
                request_id=safe_request_id,
                provider=provider_name,
                current_model=current_model,
                planned_model=plan.routing.selected_model,
                conversation_id=plan.conversation_id,
                route_reason=plan.routing.reason,
                required_capabilities=tuple(
                    sorted(capability.value for capability in required)
                ),
                executable_targets=tuple(
                    target.key for target in plan.failover.attempts
                ),
                excluded_targets=tuple(sorted(plan.failover.excluded.items())),
                redaction_changed=plan.redaction.changed,
                body_unchanged=before == after,
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            after = self._body_digest(body)
            return GatewayShadowReceipt(
                request_id=safe_request_id,
                provider=provider_name,
                current_model=current_model,
                planned_model=current_model,
                conversation_id="",
                route_reason="shadow_error",
                required_capabilities=(),
                executable_targets=(),
                excluded_targets=(),
                redaction_changed=False,
                body_unchanged=before == after,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                error=type(exc).__name__,
            )


__all__ = ["GatewayShadowObserver", "GatewayShadowReceipt"]
