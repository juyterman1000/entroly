"""Credential policy for GitHub Copilot subscription inference.

Entroly's provider-bound Copilot route has two distinct credential classes and
must never confuse them:

* a reusable GitHub credential authenticates the caller to GitHub and is used
  only to acquire Copilot entitlement/API state;
* the credential returned by GitHub's Copilot token exchange authenticates the
  actual provider-bound CAPI request.

GitHub's SDK accepts OAuth user tokens, GitHub App user tokens, fine-grained
PATs, and a separate runtime path for installation tokens. That means those
credentials are valid *runtime/acquisition* inputs; it does not make them CAPI
bearer tokens. This module therefore never replaces an exchanged Copilot token
with the original GitHub credential and never falls back to a long-lived GitHub
credential when exchange fails.

The module also resolves the manager integration ID through the same contract
used by the Copilot CLI wrapper, so credential acquisition and outbound
``Copilot-Integration-Id`` cannot silently disagree.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from . import copilot_subscription_transport as _transport
from .copilot_cli_provider_contract import (
    CopilotCLIProviderContractError,
    configure_copilot_integration_identity,
)

_SUPPORTED_GITHUB_PREFIXES = ("gho_", "ghu_", "github_pat_", "ghs_")
_CLASSIC_PAT_PREFIX = "ghp_"


def is_supported_copilot_github_credential(token: object) -> bool:
    """Return whether *token* is a recognized Copilot runtime credential shape."""
    value = _validated_token_text(token)
    return bool(value and value.startswith(_SUPPORTED_GITHUB_PREFIXES))


def is_direct_copilot_github_credential(token: object) -> bool:
    """Compatibility alias for the old classifier name.

    ``True`` means the shape is a supported GitHub credential for acquisition;
    it deliberately does **not** mean Entroly may forward the token to CAPI.
    """
    return is_supported_copilot_github_credential(token)


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


def _wrap_exchange(
    exchange: Callable[[str, str, str], Mapping[str, Any]],
) -> Callable[[str, str, str], Mapping[str, Any]]:
    """Guard acquisition credentials without changing exchange semantics.

    The returned payload is passed through verbatim. In particular, its
    ``token`` field remains the provider credential; the input GitHub credential
    is never substituted back into the payload.
    """
    if getattr(exchange, "__entroly_credential_policy__", False):
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
        if not credential:
            raise _transport.CopilotSubscriptionAuthError(
                "GitHub credential is empty or malformed"
            )
        return exchange(exchange_url, credential, integration_id)

    exchange_with_policy.__entroly_credential_policy__ = True
    exchange_with_policy.__entroly_credential_policy_original__ = exchange
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
    """Install identity/acquisition policy on the real token-manager seam."""
    current_init = _transport.CopilotTokenManager.__init__
    if getattr(current_init, "__entroly_credential_policy__", False):
        return True

    def manager_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # CopilotTokenManager's constructor is keyword-only. Keep *args for
        # wrapper compatibility but never synthesize a positional identity.
        kwargs = dict(kwargs)
        kwargs["integration_id"] = _resolved_manager_integration_id(kwargs)
        current_init(self, *args, **kwargs)
        self._exchange = _wrap_exchange(self._exchange)

    manager_init.__entroly_credential_policy__ = True
    manager_init.__entroly_credential_policy_original__ = current_init
    _transport.CopilotTokenManager.__init__ = manager_init
    return True


__all__ = [
    "install_copilot_subscription_credential_policy",
    "is_direct_copilot_github_credential",
    "is_supported_copilot_github_credential",
    "is_unsupported_classic_pat",
]
