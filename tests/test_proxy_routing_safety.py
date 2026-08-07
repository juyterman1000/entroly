from __future__ import annotations

import json
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from entroly.docker_launcher_safe import _apply_proxy_cli_overrides
from entroly.provider_policy import Capability, ProviderTarget
from entroly.proxy_config import ProxyConfig
from entroly.proxy_routing_authority import (
    RoutingAuthorityCoordinator,
    RoutingAuthorityDenied,
    RoutingAuthorityLedger,
    RoutingRequestContext,
)
from entroly.proxy_routing_safety import (
    RoutingSafetyConfigurationError,
    configure_proxy_routing_safety,
    install_routing_safety,
    validate_routing_environment,
)


def _pricing_file(tmp_path) -> str:
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {
                "source": "operator-verified-test",
                "models": {
                    "openai:gpt-4o": {
                        "input_per_million": 10,
                        "output_per_million": 30,
                        "cache_read_per_million": 5,
                    },
                    "openai:gpt-4o-mini": {
                        "input_per_million": 0.15,
                        "output_per_million": 0.6,
                        "cache_read_per_million": 0.075,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _configure_execute(monkeypatch: pytest.MonkeyPatch, pricing_path: str) -> None:
    monkeypatch.setenv("ENTROLY_ROUTING_AUTHORITY", "1")
    monkeypatch.setenv("ENTROLY_ROUTING_AUTHORITY_MODE", "execute")
    monkeypatch.setenv("ENTROLY_ROUTING_AUTHORITY_ALLOWED_PROVIDERS", "openai")
    monkeypatch.setenv(
        "ENTROLY_ROUTING_AUTHORITY_ALLOWED_MODELS",
        "openai:gpt-4o,openai:gpt-4o-mini",
    )
    monkeypatch.setenv(
        "ENTROLY_ROUTING_AUTHORITY_ALLOWED_ORIGINS",
        "openai=https://api.openai.com",
    )
    monkeypatch.setenv("ENTROLY_ROUTING_AUTHORITY_REQUIRE_PRICING", "1")
    monkeypatch.setenv("ENTROLY_PRICING_CATALOG", pricing_path)
    monkeypatch.setenv("ENTROLY_ROUTING_AUTHORITY_ACK", "authorized-official-api")
    monkeypatch.setenv("ENTROLY_RAVS_ROUTER", "1")
    monkeypatch.setenv("ENTROLY_ESCALATION_MODE", "observe")
    monkeypatch.delenv("ENTROLY_OPENAI_BASE", raising=False)


def test_execute_environment_validates_every_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    pricing_path = _pricing_file(tmp_path)
    _configure_execute(monkeypatch, pricing_path)

    proxy_config = ProxyConfig.from_env()
    config = validate_routing_environment(
        proxy_config=proxy_config,
        host="127.0.0.1",
    )

    assert config.enabled is True
    assert config.mode == "execute"
    assert config.allowed_providers == frozenset({"openai"})
    assert config.allowed_models == frozenset(
        {"openai:gpt-4o", "openai:gpt-4o-mini"}
    )
    assert config.origin_map == {"openai": "https://api.openai.com"}
    assert config.operator_acknowledged is True
    assert config.pricing_catalog_name == "pricing.json"
    assert len(config.pricing_catalog_sha256) == 64


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_ack", "requires ENTROLY_ROUTING_AUTHORITY_ACK"),
        ("remote_host", "loopback-only"),
        ("custom_origin", "official provider API origins"),
        ("pricing_override", "cannot disable auditable pricing"),
    ],
)
def test_execute_environment_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    mutation: str,
    expected: str,
) -> None:
    pricing_path = _pricing_file(tmp_path)
    _configure_execute(monkeypatch, pricing_path)
    host = "127.0.0.1"
    if mutation == "missing_ack":
        monkeypatch.delenv("ENTROLY_ROUTING_AUTHORITY_ACK")
    elif mutation == "remote_host":
        host = "0.0.0.0"
    elif mutation == "custom_origin":
        monkeypatch.setenv(
            "ENTROLY_ROUTING_AUTHORITY_ALLOWED_ORIGINS",
            "openai=https://example.invalid",
        )
        monkeypatch.setenv("ENTROLY_OPENAI_BASE", "https://example.invalid")
    else:
        monkeypatch.setenv("ENTROLY_ROUTING_AUTHORITY_REQUIRE_PRICING", "0")

    with pytest.raises(RoutingSafetyConfigurationError, match=expected):
        validate_routing_environment(
            proxy_config=ProxyConfig.from_env(),
            host=host,
        )


def test_disabled_authority_ignores_stale_routing_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ENTROLY_ROUTING_AUTHORITY", raising=False)
    monkeypatch.setenv("ENTROLY_ROUTING_AUTHORITY_MODE", "broken")
    monkeypatch.setenv("ENTROLY_ROUTING_AUTHORITY_ALLOWED_MODELS", "not-valid")
    monkeypatch.setenv("ENTROLY_PRICING_CATALOG", "/missing/pricing.json")

    config = validate_routing_environment(host="0.0.0.0")

    assert config.enabled is False
    assert config.mode == "observe"


def test_observe_mode_requires_provider_and_origin_but_not_execution_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENTROLY_ROUTING_AUTHORITY", "1")
    monkeypatch.setenv("ENTROLY_ROUTING_AUTHORITY_MODE", "observe")
    monkeypatch.setenv("ENTROLY_ROUTING_AUTHORITY_ALLOWED_PROVIDERS", "anthropic")
    monkeypatch.setenv(
        "ENTROLY_ROUTING_AUTHORITY_ALLOWED_ORIGINS",
        "anthropic=https://api.anthropic.com",
    )
    monkeypatch.delenv("ENTROLY_ROUTING_AUTHORITY_ALLOWED_MODELS", raising=False)
    monkeypatch.delenv("ENTROLY_PRICING_CATALOG", raising=False)
    monkeypatch.delenv("ENTROLY_ROUTING_AUTHORITY_ACK", raising=False)

    config = validate_routing_environment(proxy_config=ProxyConfig.from_env())

    assert config.mode == "observe"
    assert config.operator_acknowledged is False
    assert config.allowed_models == frozenset()


def test_proxy_scoped_policy_denies_unallowlisted_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    pricing_path = _pricing_file(tmp_path)
    _configure_execute(monkeypatch, pricing_path)
    config = validate_routing_environment(proxy_config=ProxyConfig.from_env())
    # Startup validation requires both exact source and target models. Narrow the
    # already-validated per-request policy here so this test reaches the inner
    # allowlist denial while the outer official-model identity guard still passes.
    config = replace(config, allowed_models=frozenset({"openai:gpt-4o"}))
    install_routing_safety()

    ledger = RoutingAuthorityLedger()
    coordinator = RoutingAuthorityCoordinator(ledger)
    proxy = SimpleNamespace(_routing_authority_coordinator=coordinator)
    configure_proxy_routing_safety(proxy, config)
    context = RoutingRequestContext(
        proxy=proxy,
        run_id="route_test",
        request_correlation="correlation",
        path="/v1/chat/completions",
        headers={"authorization": "present"},
        mode="execute",
        require_pricing=True,
        emit_headers=False,
    )
    target = ProviderTarget(
        provider="openai",
        model="gpt-4o-mini",
        capabilities=frozenset({Capability.CHAT}),
    )

    with pytest.raises(RoutingAuthorityDenied, match="target_model_not_allowlisted"):
        coordinator.authorize(
            context,
            provider="openai",
            target=target,
            body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
            },
            url="https://api.openai.com/v1/chat/completions",
            apply_fn=lambda **_kwargs: ({}, ""),
        )

    assert context.reason == "target_model_not_allowlisted"


def test_proxy_scoped_policy_denies_unpinned_origin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    pricing_path = _pricing_file(tmp_path)
    _configure_execute(monkeypatch, pricing_path)
    config = validate_routing_environment(proxy_config=ProxyConfig.from_env())
    install_routing_safety()

    ledger = RoutingAuthorityLedger()
    coordinator = RoutingAuthorityCoordinator(ledger)
    proxy = SimpleNamespace(_routing_authority_coordinator=coordinator)
    configure_proxy_routing_safety(proxy, config)
    context = RoutingRequestContext(
        proxy=proxy,
        run_id="route_test",
        request_correlation="correlation",
        path="/v1/chat/completions",
        headers={"authorization": "present"},
        mode="execute",
        require_pricing=True,
        emit_headers=False,
    )
    target = ProviderTarget(
        provider="openai",
        model="gpt-4o-mini",
        capabilities=frozenset({Capability.CHAT}),
    )

    with pytest.raises(RoutingAuthorityDenied, match="upstream_origin_not_pinned"):
        coordinator.authorize(
            context,
            provider="openai",
            target=target,
            body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
            },
            url="https://example.invalid/v1/chat/completions",
            apply_fn=lambda **_kwargs: ({}, ""),
        )

    assert context.reason == "upstream_origin_not_pinned"


def test_public_summary_never_exposes_full_pricing_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    pricing_path = _pricing_file(tmp_path)
    _configure_execute(monkeypatch, pricing_path)
    config = validate_routing_environment(proxy_config=ProxyConfig.from_env())

    rendered = json.dumps(config.public_summary(), sort_keys=True)

    assert pricing_path not in rendered
    assert "pricing.json" in rendered
    assert "credentials_stored" in rendered


def test_cli_translates_safe_execute_flags_to_strict_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    pricing_path = _pricing_file(tmp_path)
    isolated = os.environ.copy()
    monkeypatch.setattr(os, "environ", isolated)

    _apply_proxy_cli_overrides(
        [
            "proxy",
            "--routing",
            "execute",
            "--provider",
            "openai",
            "--allow-model",
            "gpt-4o",
            "--allow-model",
            "gpt-4o-mini",
            "--pricing-catalog",
            pricing_path,
            "--ack-authorized-api",
        ]
    )

    assert os.environ["ENTROLY_ROUTING_AUTHORITY_MODE"] == "execute"
    assert os.environ["ENTROLY_ROUTING_AUTHORITY_ALLOWED_PROVIDERS"] == "openai"
    assert os.environ["ENTROLY_ROUTING_AUTHORITY_ALLOWED_MODELS"] == (
        "openai:gpt-4o,openai:gpt-4o-mini"
    )
    assert os.environ["ENTROLY_ROUTING_AUTHORITY_ALLOWED_ORIGINS"] == (
        "openai=https://api.openai.com"
    )
    assert os.environ["ENTROLY_ROUTING_AUTHORITY_ACK"] == "authorized-official-api"


def test_cli_refuses_execute_without_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    pricing_path = _pricing_file(tmp_path)
    monkeypatch.setattr(os, "environ", os.environ.copy())

    with pytest.raises(ValueError, match="--ack-authorized-api"):
        _apply_proxy_cli_overrides(
            [
                "proxy",
                "--routing",
                "execute",
                "--provider",
                "openai",
                "--allow-model",
                "gpt-4o",
                "--allow-model",
                "gpt-4o-mini",
                "--pricing-catalog",
                pricing_path,
            ]
        )
