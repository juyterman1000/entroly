from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from entroly.copilot_subscription import (
    CopilotSubscriptionError,
    is_subscription_wrap,
    prepare_subscription_wrap,
    token_exchange_url_for_origin,
    validate_copilot_api_origin,
)
from entroly.copilot_subscription_transport import (
    CopilotSubscriptionAuthError,
    CopilotTokenManager,
    _resolve_github_credential,
    _same_trust_partition,
    _token_from_exchange_payload,
    _validate_exchange_url,
)


def test_subscription_flag_is_wrapper_owned_only_before_separator() -> None:
    assert is_subscription_wrap(
        ["wrap", "copilot", "--subscription", "--", "-p", "hello"]
    )
    assert not is_subscription_wrap(["wrap", "copilot", "--", "--subscription"])
    assert not is_subscription_wrap(["wrap", "cursor", "--subscription"])


def test_prepare_owns_routing_facts_not_copilot_provider_environment(monkeypatch) -> None:
    monkeypatch.setattr(
        "entroly.copilot_subscription._reserve_loopback_port",
        lambda: 19477,
    )
    env = {
        "NO_PROXY": "corp.internal",
        "COPILOT_PROVIDER_API_KEY": "third-party-key",
        "COPILOT_PROVIDER_BEARER_TOKEN": "third-party-bearer",
        "COPILOT_PROVIDER_WIRE_API": "existing-wire-value",
    }
    plan = prepare_subscription_wrap(
        [
            "wrap",
            "copilot",
            "--subscription",
            "--wire-api",
            "responses",
            "--",
            "--model",
            "gpt-5",
            "--prompt=hello",
        ],
        environ=env,
    )

    assert plan.cleaned_argv == (
        "wrap",
        "copilot",
        "--port",
        "19477",
        "--",
        "--model",
        "gpt-5",
        "--prompt=hello",
    )
    assert plan.wire_api == "responses"
    assert env["ENTROLY_COPILOT_SUBSCRIPTION"] == "1"
    assert env["ENTROLY_OPENAI_BASE"] == "https://api.githubcopilot.com"
    assert env["ENTROLY_CLIENT_ROUTE"] == "github-copilot-subscription"
    assert env["COPILOT_MODEL"] == "gpt-5"

    # Provider authentication/configuration has exactly one owner:
    # copilot_cli_provider_contract. Planning must leave pre-existing values
    # untouched rather than partially configuring or clearing them.
    assert env["COPILOT_PROVIDER_API_KEY"] == "third-party-key"
    assert env["COPILOT_PROVIDER_BEARER_TOKEN"] == "third-party-bearer"
    assert env["COPILOT_PROVIDER_WIRE_API"] == "existing-wire-value"

    assert "corp.internal" in env["NO_PROXY"]
    assert "127.0.0.1" in env["NO_PROXY"]
    assert "::1" in env["no_proxy"]
    assert plan.public_summary()["secrets_persisted"] is False


def test_prepare_preserves_explicit_wrapper_port_and_env_model() -> None:
    env = {"COPILOT_MODEL": "gpt-5.4"}
    plan = prepare_subscription_wrap(
        [
            "wrap",
            "copilot",
            "--subscription",
            "--port",
            "19888",
            "--",
            "-p",
            "hi",
        ],
        environ=env,
    )
    assert plan.proxy_port == 19888
    assert plan.model == "gpt-5.4"
    assert plan.cleaned_argv == (
        "wrap",
        "copilot",
        "--port",
        "19888",
        "--",
        "-p",
        "hi",
    )


def test_prepare_requires_explicit_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "entroly.copilot_subscription._reserve_loopback_port",
        lambda: 19477,
    )
    with pytest.raises(CopilotSubscriptionError, match="explicit Copilot model"):
        prepare_subscription_wrap(
            ["wrap", "copilot", "--subscription"],
            environ={},
        )


@pytest.mark.parametrize(
    "value",
    [
        "http://api.githubcopilot.com",
        "https://api.githubcopilot.com/evil",
        "https://api.githubcopilot.com?token=x",
        "https://user:pass@api.githubcopilot.com",
        "https://api.githubcopilot.com.evil.test",
        "https://copilot-api.-bad.ghe.com",
    ],
)
def test_copilot_origin_rejects_credential_exfiltration_shapes(value: str) -> None:
    with pytest.raises(CopilotSubscriptionError):
        validate_copilot_api_origin(value)


def test_token_exchange_endpoint_is_derived_not_arbitrary() -> None:
    assert (
        token_exchange_url_for_origin("https://api.business.githubcopilot.com")
        == "https://api.github.com/copilot_internal/v2/token"
    )
    assert (
        token_exchange_url_for_origin("https://copilot-api.acme.ghe.com")
        == "https://api.acme.ghe.com/copilot_internal/v2/token"
    )
    _validate_exchange_url("https://api.github.com/copilot_internal/v2/token")
    _validate_exchange_url("https://api.acme.ghe.com/copilot_internal/v2/token")
    with pytest.raises(CopilotSubscriptionAuthError):
        _validate_exchange_url("https://api.evil.test/copilot_internal/v2/token")


def test_standalone_token_manager_uses_same_official_identity_as_cli_contract() -> None:
    manager = CopilotTokenManager(
        api_origin="https://api.githubcopilot.com",
        environ={},
        clock=lambda: 1_000,
        exchange=lambda *_args: {"token": "tid", "expires_at": 2_000},
        credential_resolver=lambda: "github-oauth",
    )
    assert manager.integration_id == "copilot-developer-cli"


def test_generic_public_bootstrap_may_adopt_github_advertised_sku_host() -> None:
    assert _same_trust_partition(
        "https://api.githubcopilot.com",
        "https://api.individual.githubcopilot.com",
    )
    token = _token_from_exchange_payload(
        {
            "token": "short-lived",
            "expires_at": 2_000,
            "endpoints": {"api": "https://api.business.githubcopilot.com"},
        },
        requested_origin="https://api.githubcopilot.com",
        now=1_000,
    )
    assert token.api_origin == "https://api.business.githubcopilot.com"


@pytest.mark.parametrize(
    "advertised",
    [
        "https://api.individual.githubcopilot.com",
        "https://api.enterprise.githubcopilot.com",
        "https://api.githubcopilot.com",
    ],
)
def test_explicit_public_sku_host_cannot_silently_change(advertised: str) -> None:
    assert not _same_trust_partition(
        "https://api.business.githubcopilot.com",
        advertised,
    )
    with pytest.raises(CopilotSubscriptionAuthError, match="tenant boundary"):
        _token_from_exchange_payload(
            {
                "token": "short-lived",
                "expires_at": 2_000,
                "endpoints": {"api": advertised},
            },
            requested_origin="https://api.business.githubcopilot.com",
            now=1_000,
        )


def test_explicit_public_sku_host_accepts_exact_same_host() -> None:
    assert _same_trust_partition(
        "https://api.business.githubcopilot.com",
        "https://api.business.githubcopilot.com",
    )


def test_token_manager_pins_first_advertised_api_origin() -> None:
    now = [1_000.0]
    payloads = iter(
        [
            {
                "token": "short-lived-one",
                "expires_at": 2_000,
                "refresh_in": 100,
                "endpoints": {"api": "https://api.individual.githubcopilot.com"},
            },
            {
                "token": "short-lived-two",
                "expires_at": 3_000,
                "refresh_in": 100,
                "endpoints": {"api": "https://api.individual.githubcopilot.com"},
            },
        ]
    )
    manager = CopilotTokenManager(
        api_origin="https://api.githubcopilot.com",
        environ={},
        clock=lambda: now[0],
        exchange=lambda *_args: next(payloads),
        credential_resolver=lambda: "github-oauth",
    )

    first = manager._refresh(force=True)
    assert first.token == "short-lived-one"
    assert manager.api_origin == "https://api.individual.githubcopilot.com"
    now[0] = first.refresh_at + 1
    assert manager._refresh(force=False).token == "short-lived-two"


def test_token_manager_rejects_origin_change_after_pin() -> None:
    payloads = iter(
        [
            {
                "token": "one",
                "expires_at": 2_000,
                "endpoints": {"api": "https://api.individual.githubcopilot.com"},
            },
            {
                "token": "two",
                "expires_at": 3_000,
                "endpoints": {"api": "https://api.business.githubcopilot.com"},
            },
        ]
    )
    manager = CopilotTokenManager(
        api_origin="https://api.githubcopilot.com",
        environ={},
        clock=lambda: 1_000,
        exchange=lambda *_args: next(payloads),
        credential_resolver=lambda: "github-oauth",
    )
    manager._refresh(force=True)
    with pytest.raises(
        CopilotSubscriptionAuthError,
        match="changed the Copilot API origin",
    ):
        manager._refresh(force=True)


def test_enterprise_token_response_cannot_cross_tenant_boundary() -> None:
    with pytest.raises(CopilotSubscriptionAuthError, match="tenant boundary"):
        _token_from_exchange_payload(
            {
                "token": "short-lived",
                "expires_at": time.time() + 600,
                "endpoints": {"api": "https://api.githubcopilot.com"},
            },
            requested_origin="https://copilot-api.acme.ghe.com",
            now=time.time(),
        )


def test_github_credential_precedence_uses_documented_client_token() -> None:
    env = {
        "GITHUB_TOKEN": "github-token",
        "GH_TOKEN": "gh-token",
        "COPILOT_GITHUB_TOKEN": "copilot-token",
    }
    assert _resolve_github_credential(env) == "copilot-token"


def test_github_credential_falls_back_to_gh_without_echoing_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "entroly.copilot_subscription_transport.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="gho-secret\n"),
    )
    assert _resolve_github_credential({}) == "gho-secret"


def test_public_manager_summary_contains_no_token() -> None:
    manager = CopilotTokenManager(
        api_origin="https://api.githubcopilot.com",
        environ={},
        clock=lambda: 1_000,
        exchange=lambda *_args: {
            "token": "do-not-disclose",
            "expires_at": 2_000,
        },
        credential_resolver=lambda: "also-secret",
    )
    manager._refresh(force=True)
    rendered = repr(manager.public_summary())
    assert "do-not-disclose" not in rendered
    assert "also-secret" not in rendered
    assert manager.public_summary()["token_persisted"] is False


def test_container_proxy_installs_subscription_layer_after_final_transport() -> None:
    source = Path("entroly/container_proxy.py").read_text(encoding="utf-8")
    final_pos = source.index("proxy_transport_final")
    subscription_pos = source.index("install_copilot_subscription_transport")
    access_pos = source.index("proxy_access_security")
    assert final_pos < subscription_pos < access_pos


def test_python_module_entry_uses_same_safe_launcher_as_console_script() -> None:
    source = Path("entroly/__main__.py").read_text(encoding="utf-8")
    assert "docker_launcher_safe" in source
    assert "_docker_launcher import launch" not in source
