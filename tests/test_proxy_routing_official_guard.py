from __future__ import annotations

from types import SimpleNamespace

import pytest

import entroly.proxy_routing_official_guard as official_guard
from entroly.provider_policy import Capability, ProviderTarget
from entroly.proxy_routing_authority import (
    RoutingAuthorityCoordinator,
    RoutingAuthorityDenied,
    RoutingAuthorityLedger,
    RoutingRequestContext,
)
from entroly.proxy_routing_official_guard import (
    install_official_routing_guard,
    validate_official_routing_boundary,
)
from entroly.proxy_routing_safety import (
    RoutingSafetyConfig,
    RoutingSafetyConfigurationError,
)


def _config(
    *,
    providers: frozenset[str] = frozenset({"openai"}),
    models: frozenset[str] = frozenset(
        {"openai:gpt-4o", "openai:gpt-4o-mini"}
    ),
    mode: str = "execute",
) -> RoutingSafetyConfig:
    origins = tuple(
        sorted(
            (provider, f"https://api.{provider}.com")
            for provider in providers
        )
    )
    if providers == frozenset({"openai"}):
        origins = (("openai", "https://api.openai.com"),)
    return RoutingSafetyConfig(
        enabled=True,
        mode=mode,
        allowed_providers=providers,
        allowed_models=models,
        allowed_origins=origins,
        require_pricing=True,
        pricing_catalog_name="pricing.json",
        pricing_catalog_sha256="a" * 64,
        operator_acknowledged=True,
        ravs_enabled=True,
        escalation_mode="observe",
        loopback_only=mode == "execute",
    )


def _context(
    ledger: RoutingAuthorityLedger,
    *,
    config: RoutingSafetyConfig | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[RoutingAuthorityCoordinator, RoutingRequestContext]:
    coordinator = RoutingAuthorityCoordinator(ledger)
    proxy = SimpleNamespace(
        _routing_safety_config=config or _config(),
        _routing_authority_coordinator=coordinator,
    )
    context = RoutingRequestContext(
        proxy=proxy,
        run_id="route_official_guard",
        request_correlation="correlation",
        path="/v1/chat/completions",
        headers=headers or {},
        mode="execute",
        require_pricing=True,
        emit_headers=False,
    )
    return coordinator, context


def _target(model: str = "gpt-4o-mini") -> ProviderTarget:
    return ProviderTarget(
        provider="openai",
        model=model,
        capabilities=frozenset({Capability.CHAT, Capability.STREAMING}),
    )


def _body(model: str = "gpt-4o") -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
    }


def _apply(**kwargs):
    body = dict(kwargs["body"])
    body["model"] = kwargs["target"].model
    return body, kwargs["url"]


def test_startup_requires_exactly_one_provider_authority() -> None:
    with pytest.raises(
        RoutingSafetyConfigurationError,
        match="exactly one provider authority",
    ):
        validate_official_routing_boundary(
            _config(
                providers=frozenset({"openai", "anthropic"}),
                models=frozenset(
                    {
                        "openai:gpt-4o",
                        "openai:gpt-4o-mini",
                        "anthropic:claude-opus-4-6",
                    }
                ),
            )
        )


def test_execute_requires_provider_specific_api_auth_header() -> None:
    install_official_routing_guard()
    ledger = RoutingAuthorityLedger()
    coordinator, context = _context(ledger, headers={})

    with pytest.raises(RoutingAuthorityDenied, match="official_api_credential_missing"):
        coordinator.authorize(
            context,
            provider="openai",
            target=_target(),
            body=_body(),
            url="https://api.openai.com/v1/chat/completions",
            apply_fn=_apply,
        )

    assert context.proposal_count == 1
    assert ledger.snapshot()["proposals"] == 1
    receipt = ledger.complete(
        context,
        response_status=200,
        outcome="response_admitted",
    )
    assert receipt is not None
    assert receipt.reason == "official_api_credential_missing"
    assert receipt.source_provider == "openai"
    assert receipt.source_model == "gpt-4o"
    assert receipt.proposed_model == "gpt-4o-mini"


def test_openai_wire_format_cannot_authorize_non_openai_registry_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_official_routing_guard()
    ledger = RoutingAuthorityLedger()
    coordinator, context = _context(
        ledger,
        headers={"authorization": "present"},
    )

    def fake_resolve(model: str):
        provider = "deepseek" if model == "gpt-4o" else "openai"
        return SimpleNamespace(
            exact=True,
            capability=SimpleNamespace(provider=provider),
        )

    monkeypatch.setattr(official_guard, "resolve_model", fake_resolve)

    with pytest.raises(
        RoutingAuthorityDenied,
        match="source_registry_provider_not_selected_authority",
    ):
        coordinator.authorize(
            context,
            provider="openai",
            target=_target(),
            body=_body(),
            url="https://api.openai.com/v1/chat/completions",
            apply_fn=_apply,
        )

    assert context.proposal_count == 1
    assert ledger.snapshot()["proposals"] == 1


def test_inner_preflight_denial_is_registered_exactly_once() -> None:
    install_official_routing_guard()
    ledger = RoutingAuthorityLedger()
    config = _config(models=frozenset({"openai:gpt-4o"}))
    coordinator, context = _context(
        ledger,
        config=config,
        headers={"authorization": "present"},
    )

    with pytest.raises(RoutingAuthorityDenied, match="target_model_not_allowlisted"):
        coordinator.authorize(
            context,
            provider="openai",
            target=_target(),
            body=_body(),
            url="https://api.openai.com/v1/chat/completions",
            apply_fn=_apply,
        )

    assert context.proposal_count == 1
    assert ledger.snapshot()["proposals"] == 1
    receipt = ledger.complete(
        context,
        response_status=200,
        outcome="response_admitted",
    )
    assert receipt is not None
    assert receipt.reason == "target_model_not_allowlisted"


def test_observe_mode_does_not_require_api_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_official_routing_guard()
    ledger = RoutingAuthorityLedger()
    coordinator, context = _context(
        ledger,
        config=_config(mode="observe"),
        headers={},
    )

    sentinel = ({"model": "gpt-4o"}, "https://api.openai.com/v1/chat/completions")
    monkeypatch.setattr(
        official_guard,
        "_ORIGINAL_AUTHORIZE",
        lambda *_args, **_kwargs: sentinel,
    )

    assert coordinator.authorize(
        context,
        provider="openai",
        target=_target(),
        body=_body(),
        url="https://api.openai.com/v1/chat/completions",
        apply_fn=_apply,
    ) == sentinel
