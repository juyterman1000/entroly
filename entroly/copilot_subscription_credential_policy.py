"""Compatibility credential classifiers for Copilot subscription routing.

The live authentication policy is owned by :class:`CopilotTokenManager` in
``copilot_subscription_transport``.  That manager validates supported GitHub
credential classes, rejects classic PATs, resolves the shared Copilot integration
identity, performs the bounded user/entitlement preflight, and pins the CAPI
origin.

An earlier draft wrapped ``CopilotTokenManager.__init__`` to intercept a token-
exchange callable.  The current runtime contract deliberately has no exchange
callable, so retaining that monkey patch would create a second auth authority and
can break proxy startup.  This module therefore keeps only stable compatibility
classifiers used by callers/tests; installation is intentionally idempotent and
side-effect free.
"""

from __future__ import annotations

_SUPPORTED_GITHUB_PREFIXES = ("gho_", "ghu_", "github_pat_", "ghs_")
_CLASSIC_PAT_PREFIX = "ghp_"
_MAX_TOKEN_CHARS = 16_384


def _validated_token_text(token: object) -> str:
    value = str(token or "").strip()
    if not value or len(value) > _MAX_TOKEN_CHARS:
        return ""
    if any(ord(char) < 33 or ord(char) == 127 for char in value):
        return ""
    return value


def is_supported_copilot_github_credential(token: object) -> bool:
    """Return whether *token* has a supported GitHub runtime credential shape."""
    value = _validated_token_text(token)
    return bool(value and value.startswith(_SUPPORTED_GITHUB_PREFIXES))


def is_direct_copilot_github_credential(token: object) -> bool:
    """Backward-compatible name for the supported-runtime classifier.

    ``True`` means the credential shape is supported by the subscription runtime;
    it does not create a separate forwarding or fallback policy.
    """
    return is_supported_copilot_github_credential(token)


def is_unsupported_classic_pat(token: object) -> bool:
    """Return True for classic GitHub PATs, unsupported by this Copilot route."""
    value = _validated_token_text(token)
    return bool(value and value.startswith(_CLASSIC_PAT_PREFIX))


def install_copilot_subscription_credential_policy() -> bool:
    """Compatibility installer; live policy is already owned by the token manager.

    Kept so ``container_proxy`` import order remains stable while avoiding a
    second mutable authentication seam.
    """
    return True


__all__ = [
    "install_copilot_subscription_credential_policy",
    "is_direct_copilot_github_credential",
    "is_supported_copilot_github_credential",
    "is_unsupported_classic_pat",
]
