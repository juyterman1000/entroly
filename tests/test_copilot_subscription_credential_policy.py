from __future__ import annotations

import pytest

import entroly.copilot_subscription_transport as transport
from entroly.copilot_subscription_credential_policy import (
    install_copilot_subscription_credential_policy,
    is_direct_copilot_github_credential,
    is_unsupported_classic_pat,
)


@pytest.mark.parametrize(
    "token",
    [
        "gho_user-token",
        "ghu_user-token",
        "github_pat_fine-grained-token",
        "ghs_installation-token",
    ],
)
def test_supported_runtime_github_credentials_are_direct_candidates(token: str) -> None:
    assert is_direct_copilot_github_credential(token) is True


@pytest.mark.parametrize(
    "token",
    ["", "tid_short-lived", "ghp_classic-token", "bearer-anything", "github_pat_bad\nvalue"],
)
def test_unsupported_or_non_github_shapes_do_not_gain_direct_fallback(token: str) -> None:
    assert is_direct_copilot_github_credential(token) is False


def test_classic_pat_is_explicitly_classified_as_unsupported() -> None:
    assert is_unsupported_classic_pat("ghp_classic-token") is True
    assert is_unsupported_classic_pat("github_pat_fine-grained-token") is False


def _manager(
    *,
    credential: str,
    exchange,
    now: float = 1_000.0,
    environ: dict[str, str] | None = None,
    integration_id: str | None = None,
) -> transport.CopilotTokenManager:
    install_copilot_subscription_credential_policy()
    return transport.CopilotTokenManager(
        api_origin="https://api.githubcopilot.com",
        environ={} if environ is None else environ,
        integration_id=integration_id,
        clock=lambda: now,
        exchange=exchange,
        credential_resolver=lambda: credential,
    )


def test_manager_defaults_to_official_copilot_runtime_identity() -> None:
    manager = _manager(
        credential="opaque",
        exchange=lambda *_args: {"token": "tid", "expires_at": 2_000},
    )
    assert manager.integration_id == "copilot-developer-cli"


def test_manager_uses_configured_runtime_identity() -> None:
    manager = _manager(
        credential="opaque",
        exchange=lambda *_args: {"token": "tid", "expires_at": 2_000},
        environ={"GITHUB_COPILOT_INTEGRATION_ID": "my-product-agent"},
    )
    assert manager.integration_id == "my-product-agent"


def test_manager_explicit_identity_conflict_fails_closed() -> None:
    with pytest.raises(transport.CopilotSubscriptionAuthError, match="conflicts"):
        _manager(
            credential="opaque",
            exchange=lambda *_args: {"token": "tid", "expires_at": 2_000},
            environ={"GITHUB_COPILOT_INTEGRATION_ID": "runtime-agent"},
            integration_id="different-agent",
        )


def test_successful_exchange_discovers_origin_without_replacing_direct_pat() -> None:
    def exchange(url: str, credential: str, integration_id: str):
        assert url == "https://api.github.com/copilot_internal/v2/token"
        assert credential == "github_pat_direct-entitled"
        assert integration_id == "copilot-developer-cli"
        return {
            "token": "tid_exchanged",
            "expires_at": 2_000,
            "refresh_in": 300,
            "endpoints": {"api": "https://api.business.githubcopilot.com"},
        }

    manager = _manager(
        credential="github_pat_direct-entitled",
        exchange=exchange,
    )
    resolved = manager._refresh(force=True)

    assert resolved.token == "github_pat_direct-entitled"
    assert resolved.api_origin == "https://api.business.githubcopilot.com"
    assert manager.api_origin == resolved.api_origin
    assert resolved.expires_at > resolved.refresh_at > 1_000.0


def test_exchange_unavailability_falls_back_only_for_supported_direct_token() -> None:
    message = "GitHub Copilot token exchange failed with HTTP 404"

    def unavailable(_url: str, _credential: str, _integration_id: str):
        raise transport.CopilotSubscriptionAuthError(message)

    direct = _manager(credential="gho_direct-user-token", exchange=unavailable)
    resolved = direct._refresh(force=True)
    assert resolved.token == "gho_direct-user-token"
    assert resolved.api_origin == "https://api.githubcopilot.com"
    assert resolved.expires_at > resolved.refresh_at > 1_000.0

    opaque = _manager(credential="opaque-token", exchange=unavailable)
    with pytest.raises(transport.CopilotSubscriptionAuthError, match="HTTP 404"):
        opaque._refresh(force=True)


@pytest.mark.parametrize(
    "message",
    [
        "GitHub Copilot token exchange redirected; refusing to forward credentials",
        "GitHub Copilot token exchange returned invalid JSON",
        "GitHub Copilot token response advertised an untrusted API origin",
        "GitHub Copilot token response crossed the configured tenant boundary",
        "GitHub Copilot token response exceeded the safety limit",
    ],
)
def test_trust_failures_never_fall_back_to_direct_token(message: str) -> None:
    def fail_trust(_url: str, _credential: str, _integration_id: str):
        raise transport.CopilotSubscriptionAuthError(message)

    manager = _manager(
        credential="github_pat_direct-entitled",
        exchange=fail_trust,
    )
    with pytest.raises(transport.CopilotSubscriptionAuthError, match="GitHub Copilot"):
        manager._refresh(force=True)


def test_classic_pat_fails_before_exchange() -> None:
    called = False

    def exchange(_url: str, _credential: str, _integration_id: str):
        nonlocal called
        called = True
        return {"token": "should-not-happen", "expires_at": 2_000}

    manager = _manager(credential="ghp_classic-token", exchange=exchange)
    with pytest.raises(transport.CopilotSubscriptionAuthError, match="classic `ghp_`"):
        manager._refresh(force=True)
    assert called is False


def test_non_direct_credential_keeps_normal_exchange_token() -> None:
    manager = _manager(
        credential="opaque-oauth-shape",
        exchange=lambda *_args: {
            "token": "tid_exchanged",
            "expires_at": 2_000,
            "endpoints": {"api": "https://api.githubcopilot.com"},
        },
    )
    resolved = manager._refresh(force=True)
    assert resolved.token == "tid_exchanged"
