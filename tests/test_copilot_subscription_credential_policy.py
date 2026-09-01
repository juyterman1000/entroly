from __future__ import annotations

import time

import pytest

import entroly.copilot_subscription_transport as transport
from entroly.copilot_subscription_credential_policy import (
    install_copilot_subscription_credential_policy,
    is_direct_copilot_github_credential,
)


@pytest.mark.parametrize(
    "token",
    [
        "gho_user-token",
        "ghu_user-token",
        "ghp_classic-token",
        "github_pat_fine-grained-token",
    ],
)
def test_known_github_user_and_pat_shapes_are_direct_candidates(token: str) -> None:
    assert is_direct_copilot_github_credential(token) is True


@pytest.mark.parametrize(
    "token",
    ["", "tid_short-lived", "ghs_installation", "bearer-anything", "github_pat_bad\nvalue"],
)
def test_arbitrary_or_non_user_token_shapes_do_not_gain_direct_fallback(token: str) -> None:
    assert is_direct_copilot_github_credential(token) is False


def test_successful_exchange_discovers_origin_without_replacing_direct_pat(
    monkeypatch,
) -> None:
    now = time.time()

    def fake_exchange(_token: str, *, requested_origin: str, integration_id: str):
        assert requested_origin == "https://api.githubcopilot.com"
        assert integration_id == "copilot-cli-chat"
        return transport.CopilotAPIToken(
            token="tid_exchanged",
            api_origin="https://api.business.githubcopilot.com",
            expires_at=now + 1800,
            refresh_at=now + 1500,
        )

    monkeypatch.setattr(transport, "_exchange_github_token", fake_exchange)
    install_copilot_subscription_credential_policy()

    resolved = transport._exchange_github_token(
        "github_pat_direct-entitled",
        requested_origin="https://api.githubcopilot.com",
        integration_id="copilot-cli-chat",
    )

    assert resolved.token == "github_pat_direct-entitled"
    assert resolved.api_origin == "https://api.business.githubcopilot.com"
    assert resolved.expires_at > resolved.refresh_at > now


def test_exchange_unavailability_falls_back_only_for_recognized_direct_github_token(
    monkeypatch,
) -> None:
    message = "GitHub Copilot token exchange failed with HTTP 404"

    def fake_exchange(_token: str, *, requested_origin: str, integration_id: str):
        raise transport.CopilotSubscriptionAuthError(message)

    monkeypatch.setattr(transport, "_exchange_github_token", fake_exchange)
    install_copilot_subscription_credential_policy()

    direct = transport._exchange_github_token(
        "gho_direct-user-token",
        requested_origin="https://api.githubcopilot.com",
        integration_id="copilot-cli-chat",
    )
    assert direct.token == "gho_direct-user-token"
    assert direct.api_origin == "https://api.githubcopilot.com"

    with pytest.raises(transport.CopilotSubscriptionAuthError, match="HTTP 404"):
        transport._exchange_github_token(
            "opaque-token",
            requested_origin="https://api.githubcopilot.com",
            integration_id="copilot-cli-chat",
        )


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
def test_trust_failures_never_fall_back_to_direct_token(monkeypatch, message: str) -> None:
    def fake_exchange(_token: str, *, requested_origin: str, integration_id: str):
        raise transport.CopilotSubscriptionAuthError(message)

    monkeypatch.setattr(transport, "_exchange_github_token", fake_exchange)
    install_copilot_subscription_credential_policy()

    with pytest.raises(transport.CopilotSubscriptionAuthError, match="GitHub Copilot"):
        transport._exchange_github_token(
            "github_pat_direct-entitled",
            requested_origin="https://api.githubcopilot.com",
            integration_id="copilot-cli-chat",
        )
