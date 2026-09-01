from __future__ import annotations

import pytest

import entroly.copilot_subscription_transport as transport
from entroly.copilot_subscription_credential_policy import (
    install_copilot_subscription_credential_policy,
    is_direct_copilot_github_credential,
    is_supported_copilot_github_credential,
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
def test_supported_runtime_github_credentials_are_classified(token: str) -> None:
    assert is_supported_copilot_github_credential(token) is True
    assert is_direct_copilot_github_credential(token) is True


@pytest.mark.parametrize(
    "token",
    ["", "tid_short-lived", "ghp_classic-token", "bearer-anything", "github_pat_bad\nvalue"],
)
def test_unsupported_or_non_github_shapes_are_not_classified(token: str) -> None:
    assert is_supported_copilot_github_credential(token) is False
    assert is_direct_copilot_github_credential(token) is False


def test_classic_pat_is_explicitly_classified_as_unsupported() -> None:
    assert is_unsupported_classic_pat("ghp_classic-token") is True
    assert is_unsupported_classic_pat("github_pat_fine-grained-token") is False


def test_policy_installer_is_idempotent_and_does_not_patch_manager_constructor() -> None:
    original_init = transport.CopilotTokenManager.__init__
    assert install_copilot_subscription_credential_policy() is True
    assert install_copilot_subscription_credential_policy() is True
    assert transport.CopilotTokenManager.__init__ is original_init


def _manager(
    *,
    credential: str,
    payload: dict[str, object] | None = None,
    environ: dict[str, str] | None = None,
    integration_id: str | None = None,
) -> transport.CopilotTokenManager:
    return transport.CopilotTokenManager(
        api_origin="https://api.githubcopilot.com",
        environ={} if environ is None else environ,
        integration_id=integration_id,
        user_info_fetch=lambda *_args: (
            {"chat_enabled": True} if payload is None else payload
        ),
        credential_resolver=lambda: credential,
    )


def test_manager_defaults_to_official_copilot_runtime_identity() -> None:
    manager = _manager(credential="gho_user-token")
    assert manager.integration_id == "copilot-developer-cli"


def test_manager_uses_configured_runtime_identity() -> None:
    manager = _manager(
        credential="gho_user-token",
        environ={"GITHUB_COPILOT_INTEGRATION_ID": "my-product-agent"},
    )
    assert manager.integration_id == "my-product-agent"


def test_manager_explicit_identity_conflict_fails_closed() -> None:
    with pytest.raises(transport.CopilotSubscriptionAuthError, match="conflicts"):
        _manager(
            credential="gho_user-token",
            environ={"GITHUB_COPILOT_INTEGRATION_ID": "runtime-agent"},
            integration_id="different-agent",
        )


@pytest.mark.parametrize(
    "credential",
    [
        "gho_user-token",
        "ghu_user-token",
        "github_pat_fine-grained-token",
        "ghs_installation-token",
    ],
)
def test_manager_preserves_supported_runtime_credential_as_capi_bearer(
    credential: str,
) -> None:
    manager = _manager(
        credential=credential,
        payload={
            "chat_enabled": True,
            "endpoints": {"api": "https://api.business.githubcopilot.com"},
        },
    )
    resolved = manager._refresh(force=True)
    assert resolved.token == credential
    assert resolved.api_origin == "https://api.business.githubcopilot.com"
    assert manager.current_token() == credential


def test_classic_pat_fails_before_user_preflight() -> None:
    called = False

    def user_info_fetch(*_args):
        nonlocal called
        called = True
        return {"chat_enabled": True}

    manager = transport.CopilotTokenManager(
        api_origin="https://api.githubcopilot.com",
        environ={},
        user_info_fetch=user_info_fetch,
        credential_resolver=lambda: "ghp_classic-token",
    )
    with pytest.raises(transport.CopilotSubscriptionAuthError, match="classic `ghp_`"):
        manager._refresh(force=True)
    assert called is False


def test_unknown_credential_shape_fails_before_user_preflight() -> None:
    called = False

    def user_info_fetch(*_args):
        nonlocal called
        called = True
        return {"chat_enabled": True}

    manager = transport.CopilotTokenManager(
        api_origin="https://api.githubcopilot.com",
        environ={},
        user_info_fetch=user_info_fetch,
        credential_resolver=lambda: "opaque-token",
    )
    with pytest.raises(transport.CopilotSubscriptionAuthError, match="unsupported"):
        manager._refresh(force=True)
    assert called is False


def test_manager_does_not_accept_legacy_exchange_hook() -> None:
    with pytest.raises(transport.CopilotSubscriptionAuthError, match="no longer accepts"):
        transport.CopilotTokenManager(
            api_origin="https://api.githubcopilot.com",
            environ={},
            exchange=lambda *_args: {},
            credential_resolver=lambda: "gho_user-token",
        )


def test_user_preflight_failure_does_not_create_provider_credential() -> None:
    def fail(*_args):
        raise transport.CopilotSubscriptionAuthError(
            "GitHub rejected the credential for Copilot subscription access"
        )

    manager = transport.CopilotTokenManager(
        api_origin="https://api.githubcopilot.com",
        environ={},
        user_info_fetch=fail,
        credential_resolver=lambda: "gho_user-token",
    )
    with pytest.raises(transport.CopilotSubscriptionAuthError, match="rejected"):
        manager._refresh(force=True)
    summary = manager.public_summary()
    assert summary["credential_cached"] is False
    assert summary["automatic_auth_replay"] is False
