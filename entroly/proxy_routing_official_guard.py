"""Final official-API execution guard for safe model routing.

This guard sits outside the generic deployment-safety layer. It closes three
execution-specific gaps that matter when Entroly is allowed to mutate a live
provider request:

* exactly one provider authority is permitted per proxy process;
* an official provider-specific API credential header must be present;
* model-registry provider identity must match the pinned official provider,
  even when another service uses an OpenAI-compatible wire format.

It also guarantees that a preflight denial becomes a bounded routing proposal
receipt. The inner routing coordinator normally increments the proposal counter,
but a deployment guard may deny before the coordinator is entered. This wrapper
registers that denied proposal exactly once and fills only non-secret context
fields needed for auditability.
"""

from __future__ import annotations

from typing import Any, Mapping

from .models.registry import resolve_model
from .provider_adapters import canonical_request_from_provider_body
from .proxy_routing_authority import (
    RoutingAuthorityCoordinator,
    RoutingAuthorityDenied,
    RoutingRequestContext,
)
from .proxy_routing_safety import (
    RoutingSafetyConfig,
    RoutingSafetyConfigurationError,
    install_routing_safety,
)
from .proxy_transform import detect_provider

_REQUIRED_AUTH_HEADERS = {
    "openai": frozenset({"authorization"}),
    "anthropic": frozenset({"x-api-key", "authorization"}),
    "gemini": frozenset({"x-goog-api-key", "authorization"}),
}
_INSTALLED = False
_ORIGINAL_AUTHORIZE = None


def validate_official_routing_boundary(
    config: RoutingSafetyConfig,
) -> RoutingSafetyConfig:
    """Apply the single-provider production boundary before proxy startup."""
    if config.enabled and len(config.allowed_providers) != 1:
        raise RoutingSafetyConfigurationError(
            "safe routing requires exactly one provider authority per proxy process"
        )
    return config


def _prepare_audit_context(
    context: RoutingRequestContext,
    *,
    target: Any,
    body: Mapping[str, Any],
) -> tuple[str, str]:
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
    context.proposed_provider = str(target.provider).strip().casefold()
    context.proposed_model = str(target.model).strip()
    context.required_capabilities = tuple(
        sorted(
            capability.value
            for capability in adapter.canonical.required_capabilities()
        )
    )
    return source_provider, source_model


def _registry_provider(model: str) -> tuple[str, bool]:
    resolution = resolve_model(model)
    capability = resolution.capability
    if capability is None:
        return "", bool(resolution.exact)
    return str(capability.provider).strip().casefold(), bool(resolution.exact)


def _enforce_official_execution(
    coordinator: RoutingAuthorityCoordinator,
    context: RoutingRequestContext,
    *,
    provider: str,
    target: Any,
    body: Mapping[str, Any],
    url: str,
) -> None:
    del url  # The inner deployment-safety layer enforces the exact pinned origin.
    config = getattr(context.proxy, "_routing_safety_config", None)
    if not isinstance(config, RoutingSafetyConfig) or config.mode != "execute":
        return

    source_provider, source_model = _prepare_audit_context(
        context,
        target=target,
        body=body,
    )
    selected_provider = next(iter(config.allowed_providers))
    if source_provider != selected_provider:
        coordinator._deny(context, "source_provider_not_selected_authority")
    if provider.strip().casefold() != selected_provider:
        coordinator._deny(context, "proxy_provider_not_selected_authority")
    if str(target.provider).strip().casefold() != selected_provider:
        coordinator._deny(context, "target_provider_not_selected_authority")

    required_headers = _REQUIRED_AUTH_HEADERS[selected_provider]
    if not any(name in context.headers for name in required_headers):
        coordinator._deny(context, "official_api_credential_missing")

    source_registry_provider, source_exact = _registry_provider(source_model)
    target_registry_provider, target_exact = _registry_provider(str(target.model))
    if not source_exact:
        coordinator._deny(context, "source_model_not_exact")
    if not target_exact:
        coordinator._deny(context, "target_model_not_exact")
    if source_registry_provider != selected_provider:
        coordinator._deny(
            context,
            "source_registry_provider_not_selected_authority",
        )
    if target_registry_provider != selected_provider:
        coordinator._deny(
            context,
            "target_registry_provider_not_selected_authority",
        )


def install_official_routing_guard() -> None:
    """Install the outer guard after generic routing safety, idempotently."""
    global _INSTALLED, _ORIGINAL_AUTHORIZE
    if _INSTALLED:
        return

    install_routing_safety()
    _ORIGINAL_AUTHORIZE = RoutingAuthorityCoordinator.authorize

    def official_authorize(
        self: RoutingAuthorityCoordinator,
        context: RoutingRequestContext,
        **kwargs: Any,
    ):
        provider = kwargs["provider"]
        target = kwargs["target"]
        body = kwargs["body"]
        url = kwargs["url"]
        config = getattr(context.proxy, "_routing_safety_config", None)
        if isinstance(config, RoutingSafetyConfig) and config.enabled:
            _prepare_audit_context(context, target=target, body=body)
        try:
            _enforce_official_execution(
                self,
                context,
                provider=provider,
                target=target,
                body=body,
                url=url,
            )
            return _ORIGINAL_AUTHORIZE(self, context, **kwargs)
        except RoutingAuthorityDenied:
            # Inner authority increments before its own denials. Preflight guards
            # do not, so register the proposal exactly once before re-raising.
            if context.proposal_count == 0:
                context.proposal_count = 1
                self.ledger.register_proposal()
            raise

    official_authorize.__entroly_official_routing_guard_original__ = (
        _ORIGINAL_AUTHORIZE
    )
    RoutingAuthorityCoordinator.authorize = official_authorize
    _INSTALLED = True


__all__ = [
    "install_official_routing_guard",
    "validate_official_routing_boundary",
]
