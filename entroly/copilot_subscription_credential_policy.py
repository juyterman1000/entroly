"""Credential policy for GitHub Copilot subscription inference.

Entroly's subscription route reuses the existing CopilotTokenManager. The
manager normally exchanges a GitHub credential for a short-lived Copilot token.
Current GitHub Copilot runtimes also accept supported GitHub credentials through
the runtime authentication path, and preserving that credential can avoid
changing the caller's entitlement lane.

This module wraps the manager's existing exchange callable rather than inventing
a second token manager or transport. It also resolves the manager integration ID
through the same contract used by the Copilot CLI wrapper, so credential
acquisition and outbound ``Copilot-Integration-Id`` can never silently disagree.

Exchange is still attempted first so GitHub can advertise the authoritative CAPI
endpoint. For recognized direct credentials, the advertised endpoint is retained
while the original credential is kept as the inference bearer. If exchange is
merely unavailable/rejected, a recognized direct credential can fall back to the
already-validated configured CAPI origin. Redirects, malformed payloads,
untrusted advertised origins, tenant-boundary failures, and other trust errors
remain hard failures.

GitHub's current Copilot CLI/SDK documentation supports OAuth user tokens,
GitHub App user tokens, and fine-grained PATs. GitHub Actions additionally has a
runtime path for installation/GITHUB_TOKEN credentials. Classic ``ghp_`` PATs
are explicitly unsupported and are rejected before any provider call.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from typing import Any

from . import copilot_subscription_transport as _transport
from .copilot_cli_provider_contract import (
    CopilotCLIProviderContractError,
    configure_copilot_integration_identity,
)

_DIRECT_GITHUB_PREFIXES = ("gho_", "ghu_", "github_pat_", "ghs_")
_CLASSIC_PAT_PREFIX = "ghp_"
_DIRECT_FALLBACK_REFRESH_S = 300.0
_DIRECT_FALLBACK_ERRORS = (
    "GitHub rejected the credential for Copilot subscription access",
    "GitHub Copilot token exchange failed with HTTP ",
    "unable to reach GitHub's Copilot token exchange endpoint",
)


def is_direct_copilot_github_credential(token: object) -> bool:
    """Return whether *token* is a supported direct Copilot credential shape."""
    value = _validated_token_text(token)
    return bool(value and value.startswith(_DIRECT_GITHUB_PREFIXES))


def is_unsupported_classic_pat(token: object) -> bool:
    """Return True for GitHub classic PATs, which Copilot CLI does not support."""
    value = _validated_token_text(token)
    return bool(value and value.startswith(_CLASSIC_PAT_PREFIX))


def _validated_token_text(token: object) -> str:
    value = str(token or "").strip()
    if not value or len(value) > 16_384:
        return ""
    if any(ord(char) < 33 or ord(char) == 127 for char in value):
        return ""
    return value


def _exchange_failure_allows_direct_fallback(exc: Exception) -> bool:
    """Allow fallback only for exchange availability/credential-class failures."""
    message = str(exc)
    return any(message.startswith(prefix) for prefix in _DIRECT_FALLBACK_ERRORS)


def _direct_fallback_payload(
    token: str,
    *,
    now: float,
) -> dict[str, object]:
    """Build a synthetic cache horizon without pretending it is token expiry."""
    return {
        "token": token,
        "expires_at": now + _DIRECT_FALLBACK_REFRESH_S + 60.0,
        "refresh_in": _DIRECT_FALLBACK_REFRESH_S,
    }


def _preserve_direct_token(
    payload: Mapping[str, Any],
    *,
    token: str,
) -> dict[str, Any]:
    """Keep GitHub's endpoint/refresh metadata while retaining caller identity."""
    result = dict(payload)
    result["token"] = token
    return result


def _wrap_exchange(
    exchange: Callable[[str, str, str], Mapping[str, Any]],
    *,
    clock: Callable[[], float],
) -> Callable[[str, str, str], Mapping[str, Any]]:
    if getattr(exchange, "__entroly_direct_credential_policy__", False):
        return exchange

    def exchange_with_policy(
        exchange_url: str,
        github_credential: str,
        integration_id: str,
    ) -> Mapping[str, Any]:
        credential = _validated_token_text(github_credential)
        if is_unsupported_classic_pat(credential):
            raise _transport.CopilotSubscriptionAuthError(
                "GitHub Copilot CLI does not support classic `ghp_` personal access tokens; "
                "use an OAuth token or a fine-grained PAT with Copilot Requests permission"
            )

        direct = is_direct_copilot_github_credential(credential)
        try:
            payload = exchange(exchange_url, github_credential, integration_id)
        except _transport.CopilotSubscriptionAuthError as exc:
            if not direct or not _exchange_failure_allows_direct_fallback(exc):
                raise
            # The manager already validated and pinned its requested CAPI origin.
            # Omitting endpoints here intentionally keeps that trusted origin.
            return _direct_fallback_payload(credential, now=float(clock()))

        if not isinstance(payload, Mapping):
            # Preserve the transport's existing failure semantics. The manager's
            # normal parser will reject this shape; do not turn it into fallback.
            return payload
        if not direct:
            return payload
        return _preserve_direct_token(payload, token=credential)

    exchange_with_policy.__entroly_direct_credential_policy__ = True
    exchange_with_policy.__entroly_direct_credential_policy_original__ = exchange
    return exchange_with_policy


def _resolved_manager_integration_id(kwargs: dict[str, Any]) -> str:
    """Resolve explicit/configured identity using the one shared contract."""
    supplied = kwargs.get("integration_id")
    supplied_text = str(supplied or "").strip()
    configured = kwargs.get("environ")
    identity_env = dict(os.environ if configured is None else configured)

    # An explicit constructor identity is a preference only when the environment
    # has not already stated the provider identity. Feed it into the shared
    # resolver so validation and conflict rules are never reimplemented here.
    if supplied_text and not identity_env.get("ENTROLY_COPILOT_INTEGRATION_ID"):
        identity_env["ENTROLY_COPILOT_INTEGRATION_ID"] = supplied_text

    try:
        resolved = configure_copilot_integration_identity(identity_env)
    except CopilotCLIProviderContractError as exc:
        raise _transport.CopilotSubscriptionAuthError(str(exc)) from exc

    if supplied_text and supplied_text != resolved:
        raise _transport.CopilotSubscriptionAuthError(
            "explicit Copilot integration ID conflicts with the configured runtime identity"
        )
    return resolved


def install_copilot_subscription_credential_policy() -> bool:
    """Install identity and direct-credential policy on the real manager seam."""
    current_init = _transport.CopilotTokenManager.__init__
    if getattr(current_init, "__entroly_direct_credential_policy__", False):
        return True

    def manager_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # CopilotTokenManager's constructor is keyword-only. Keep *args for
        # wrapper compatibility but never synthesize a positional identity.
        kwargs = dict(kwargs)
        kwargs["integration_id"] = _resolved_manager_integration_id(kwargs)
        current_init(self, *args, **kwargs)
        self._exchange = _wrap_exchange(self._exchange, clock=self._clock)

    manager_init.__entroly_direct_credential_policy__ = True
    manager_init.__entroly_direct_credential_policy_original__ = current_init
    _transport.CopilotTokenManager.__init__ = manager_init
    return True


__all__ = [
    "install_copilot_subscription_credential_policy",
    "is_direct_copilot_github_credential",
    "is_unsupported_classic_pat",
]
