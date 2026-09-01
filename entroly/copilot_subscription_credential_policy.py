"""Credential-mode policy for GitHub Copilot subscription inference.

GitHub Copilot accepts more than one legitimate credential lifecycle. OAuth
credentials can often be exchanged for a short-lived Copilot entitlement token,
but GitHub's own CLI also documents fine-grained PATs with the ``Copilot
Requests`` permission as inference credentials. Public Copilot integrations
likewise report that some GitHub user/PAT token shapes are valid directly even
when ``/copilot_internal/v2/token`` is unavailable for that account.

Entroly therefore uses token exchange as *endpoint discovery* when possible,
without unnecessarily replacing a recognized direct GitHub credential. Direct
fallback is deliberately narrower than a generic fail-open: only recognized
GitHub user/PAT token shapes, only an already-validated configured CAPI origin,
and only exchange-unavailable/rejected failures qualify. Redirects, malformed
payloads, unsafe advertised origins, tenant-boundary violations, and other trust
failures remain hard failures.
"""

from __future__ import annotations

import time

from . import copilot_subscription_transport as _transport

_DIRECT_GITHUB_PREFIXES = ("gho_", "ghu_", "ghp_", "github_pat_")
_DIRECT_FALLBACK_REFRESH_S = 300.0
_DIRECT_FALLBACK_ERRORS = (
    "GitHub rejected the credential for Copilot subscription access",
    "GitHub Copilot token exchange failed with HTTP ",
    "unable to reach GitHub's Copilot token exchange endpoint",
)


def is_direct_copilot_github_credential(token: object) -> bool:
    """Return whether *token* is a recognized GitHub user/PAT bearer shape."""
    value = str(token or "").strip()
    if not value or len(value) > 16_384:
        return False
    if any(ord(char) < 33 or ord(char) == 127 for char in value):
        return False
    return value.startswith(_DIRECT_GITHUB_PREFIXES)


def _exchange_failure_allows_direct_fallback(exc: Exception) -> bool:
    """Allow fallback only for exchange availability/credential-class failures."""
    message = str(exc)
    return any(message.startswith(prefix) for prefix in _DIRECT_FALLBACK_ERRORS)


def _direct_token(
    token: str,
    *,
    api_origin: str,
    now: float | None = None,
    refresh_after_s: float = _DIRECT_FALLBACK_REFRESH_S,
) -> _transport.CopilotAPIToken:
    current = time.time() if now is None else float(now)
    refresh_after = max(30.0, float(refresh_after_s))
    # The GitHub credential itself may be long-lived or externally refreshed.
    # This synthetic horizon controls only how often Entroly re-evaluates origin
    # discovery; it is not presented as the credential's actual expiry.
    return _transport.CopilotAPIToken(
        token=token,
        api_origin=api_origin,
        expires_at=current + refresh_after + 60.0,
        refresh_at=current + refresh_after,
    )


def install_copilot_subscription_credential_policy() -> bool:
    """Wrap token acquisition with direct-GitHub-credential preservation."""
    current_exchange = _transport._exchange_github_token
    if getattr(current_exchange, "__entroly_direct_credential_policy__", False):
        return True

    def exchange_with_policy(
        github_token: str,
        *,
        requested_origin: str,
        integration_id: str,
    ) -> _transport.CopilotAPIToken:
        direct = is_direct_copilot_github_credential(github_token)
        try:
            exchanged = current_exchange(
                github_token,
                requested_origin=requested_origin,
                integration_id=integration_id,
            )
        except _transport.CopilotSubscriptionAuthError as exc:
            if not direct or not _exchange_failure_allows_direct_fallback(exc):
                raise
            # ``requested_origin`` has already passed strict GitHub CAPI origin
            # validation in CopilotTokenManager.__init__. No arbitrary host can
            # enter through this fallback.
            return _direct_token(github_token, api_origin=requested_origin)

        if not direct:
            return exchanged

        # Keep GitHub's advertised/pinned API origin, but preserve the caller's
        # legitimate GitHub credential for inference. This avoids reducing model
        # entitlement by substituting an independently minted token while still
        # benefiting from GitHub's endpoint discovery when available.
        now = time.time()
        remaining = max(30.0, exchanged.refresh_at - now)
        return _direct_token(
            github_token,
            api_origin=exchanged.api_origin,
            now=now,
            refresh_after_s=remaining,
        )

    exchange_with_policy.__entroly_direct_credential_policy__ = True
    exchange_with_policy.__entroly_direct_credential_policy_original__ = current_exchange
    _transport._exchange_github_token = exchange_with_policy
    return True


__all__ = [
    "install_copilot_subscription_credential_policy",
    "is_direct_copilot_github_credential",
]
