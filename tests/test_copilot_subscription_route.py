from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from entroly.copilot_subscription import (
    CopilotSubscriptionError,
    is_subscription_wrap,
    prepare_subscription_wrap,
    user_info_url_for_origin,
    validate_copilot_api_origin,
)
from entroly.copilot_subscription_transport import (
    CopilotSubscriptionAuthError,
    CopilotTokenManager,
    _credential_from_user_payload,
    _resolve_github_credential,
    _same_trust_partition,
    _validate_user_info_url,
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

    # Planning has no provider-auth authority. The dedicated CLI contract owns it.
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


def test_user_preflight_endpoint_is_derived_not_arbitrary() -> None:
    assert (
        user_info_url_for_origin("https://api.business.githubcopilot.com")
        == "https://api.github.com/copilot_internal/user"
    )
    assert (
        user_info_url_for_origin("https://copilot-api.acme.ghe.com")
        == "https://api.acme.ghe.com/copilot_internal/user"
    )
    _validate_user_info_url("https://api.github.com/copilot_internal/user")
    _validate_user_info_url("https://api.acme.ghe.com/copilot_internal/user")
    with pytest.raises(CopilotSubscriptionAuthError):
        _validate_user_info_url("https://api.evil.test/copilot_internal/user")


def test_standalone_manager_uses_same_official_identity_as_cli_contract() -> None:
    manager = CopilotTokenManager(
        api_origin="https://api.githubcopilot.com",
        environ={},
        user_info_fetch=lambda *_args: {"chat_enabled": True},
        credential_resolver=lambda: "gho_user-token",
    )
    assert manager.integration_id == "copilot-developer-cli"


def test_generic_public_bootstrap_may_adopt_github_advertised_sku_host() -> None:
    assert _same_trust_partition(
        "https://api.githubcopilot.com",
        "https://api.individual.githubcopilot.com",
    )
    resolved = _credential_from_user_payload(
        "gho_user-token",
        {
            "chat_enabled": True,
            "endpoints": {"api": "https://api.business.githubcopilot.com"},
        },
        requested_origin="https://api.githubcopilot.com",
    )
    assert resolved.token == "gho_user-token"
    assert resolved.api_origin == "https://api.business.githubcopilot.com"


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
        _credential_from_user_payload(
            "gho_user-token",
            {"chat_enabled": True, "endpoints": {"api": advertised}},
            requested_origin="https://api.business.githubcopilot.com",
        )


def test_explicit_public_sku_host_accepts_exact_same_host() -> None:
    assert _same_trust_partition(
        "https://api.business.githubcopilot.com",
        "https://api.business.githubcopilot.com",
    )


def test_manager_pins_first_advertised_api_origin() -> None:
    payloads = iter(
        [
            {
                "chat_enabled": True,
                "endpoints": {"api": "https://api.individual.githubcopilot.com"},
            },
            {
                "chat_enabled": True,
                "endpoints": {"api": "https://api.individual.githubcopilot.com"},
            },
        ]
    )
    manager = CopilotTokenManager(
        api_origin="https://api.githubcopilot.com",
        environ={},
        user_info_fetch=lambda *_args: next(payloads),
        credential_resolver=lambda: "gho_user-token",
    )

    first = manager._refresh(force=True)
    assert first.token == "gho_user-token"
    assert manager.api_origin == "https://api.individual.githubcopilot.com"
    second = manager._refresh(force=True)
    assert second.token == "gho_user-token"
    assert manager.api_origin == "https://api.individual.githubcopilot.com"


def test_manager_rejects_origin_change_after_pin() -> None:
    payloads = iter(
        [
            {
                "chat_enabled": True,
                "endpoints": {"api": "https://api.individual.githubcopilot.com"},
            },
            {
                "chat_enabled": True,
                "endpoints": {"api": "https://api.business.githubcopilot.com"},
            },
        ]
    )
    manager = CopilotTokenManager(
        api_origin="https://api.githubcopilot.com",
        environ={},
        user_info_fetch=lambda *_args: next(payloads),
        credential_resolver=lambda: "gho_user-token",
    )
    manager._refresh(force=True)
    with pytest.raises(
        CopilotSubscriptionAuthError,
        match="changed the Copilot API origin",
    ):
        manager._refresh(force=True)


def test_enterprise_user_response_cannot_cross_tenant_boundary() -> None:
    with pytest.raises(CopilotSubscriptionAuthError, match="tenant boundary"):
        _credential_from_user_payload(
            "gho_user-token",
            {
                "chat_enabled": True,
                "endpoints": {"api": "https://api.githubcopilot.com"},
            },
            requested_origin="https://copilot-api.acme.ghe.com",
        )


def test_disabled_chat_fails_closed() -> None:
    with pytest.raises(CopilotSubscriptionAuthError, match="not enabled"):
        _credential_from_user_payload(
            "gho_user-token",
            {"chat_enabled": False},
            requested_origin="https://api.githubcopilot.com",
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


def test_public_manager_summary_contains_no_credential() -> None:
    manager = CopilotTokenManager(
        api_origin="https://api.githubcopilot.com",
        environ={},
        user_info_fetch=lambda *_args: {"chat_enabled": True},
        credential_resolver=lambda: "gho-do-not-disclose",
    )
    manager._refresh(force=True)
    summary = manager.public_summary()
    rendered = repr(summary)
    assert "gho-do-not-disclose" not in rendered
    assert summary["credential_persisted"] is False
    assert summary["background_refresh"] is False
    assert summary["automatic_auth_replay"] is False


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
