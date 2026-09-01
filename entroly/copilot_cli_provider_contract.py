"""Normalize Copilot CLI custom-provider settings onto the documented surface.

GitHub documents ``COPILOT_PROVIDER_API_KEY`` for Copilot CLI BYOK endpoints.
Entroly only needs a non-secret local placeholder because the hardened proxy
replaces it with the active GitHub Copilot credential before forwarding.

GitHub's current Copilot SDK documents ``GITHUB_COPILOT_INTEGRATION_ID`` as the
runtime identity used for Copilot routing and attribution, with
``copilot-developer-cli`` as the default. Entroly must use exactly the same
identity when it owns the provider-bound credential route; a token and a
conflicting integration ID can be rejected by Copilot's authorization layer.
This module therefore makes the client and Entroly identities one explicit
contract and fails closed if an operator configured contradictory values.

The SDK also supports selecting the Responses wire API, but that setting is not
part of the public Copilot CLI BYOK environment-variable table. Therefore the
production default stays on Chat Completions without setting a private wire
selector. An explicit ``responses`` request remains supported as a narrowly
scoped compatibility feature and is surfaced as such in the returned summary.
"""

from __future__ import annotations

from collections.abc import MutableMapping

_LOCAL_PROVIDER_KEY = "entroly-local-provider-route"
_DEFAULT_INTEGRATION_ID = "copilot-developer-cli"
_MAX_INTEGRATION_ID_CHARS = 128


class CopilotCLIProviderContractError(ValueError):
    """Invalid Copilot CLI custom-provider configuration."""


def _validated_integration_id(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if (
        len(text) > _MAX_INTEGRATION_ID_CHARS
        or any(
            not (char.isascii() and (char.isalnum() or char in "._-"))
            for char in text
        )
    ):
        raise CopilotCLIProviderContractError(
            "Copilot integration ID must contain only ASCII letters, digits, '.', '_', or '-'"
        )
    return text


def configure_copilot_integration_identity(
    environ: MutableMapping[str, str],
) -> str:
    """Make the Copilot runtime and Entroly provider identities identical.

    The function is intentionally reusable by both the wrapper and the proxy's
    token-manager construction path so direct ``container_proxy`` launches and
    normal ``entroly wrap copilot --subscription`` launches obey one identity
    rule.
    """
    entroly_id = _validated_integration_id(
        environ.get("ENTROLY_COPILOT_INTEGRATION_ID")
    )
    runtime_id = _validated_integration_id(
        environ.get("GITHUB_COPILOT_INTEGRATION_ID")
    )
    if entroly_id and runtime_id and entroly_id != runtime_id:
        raise CopilotCLIProviderContractError(
            "ENTROLY_COPILOT_INTEGRATION_ID and GITHUB_COPILOT_INTEGRATION_ID "
            "must match for a provider-bound Copilot session"
        )

    selected = entroly_id or runtime_id or _DEFAULT_INTEGRATION_ID
    environ["ENTROLY_COPILOT_INTEGRATION_ID"] = selected
    environ["GITHUB_COPILOT_INTEGRATION_ID"] = selected
    return selected


def apply_copilot_cli_provider_contract(
    environ: MutableMapping[str, str],
    *,
    wire_api: str,
) -> dict[str, object]:
    """Apply the smallest documented Copilot CLI provider contract.

    The placeholder key is process-local and intentionally carries no provider
    authority. The Entroly proxy must replace it before any request reaches a
    GitHub-operated origin.
    """
    normalized_wire = str(wire_api or "").strip().casefold()
    if normalized_wire not in {"completions", "responses"}:
        raise CopilotCLIProviderContractError(
            "Copilot wire API must be 'completions' or 'responses'"
        )

    integration_id = configure_copilot_integration_identity(environ)
    environ["COPILOT_PROVIDER_API_KEY"] = _LOCAL_PROVIDER_KEY
    environ.pop("COPILOT_PROVIDER_BEARER_TOKEN", None)

    experimental_wire_selector = normalized_wire == "responses"
    if experimental_wire_selector:
        environ["COPILOT_PROVIDER_WIRE_API"] = "responses"
    else:
        # Chat Completions is the documented/default OpenAI-compatible CLI path;
        # avoid depending on an additional private environment variable.
        environ.pop("COPILOT_PROVIDER_WIRE_API", None)

    return {
        "provider_auth_surface": "documented-api-key-placeholder",
        "wire_api": normalized_wire,
        "wire_selector_experimental": experimental_wire_selector,
        "provider_secret_in_cli_env": False,
        "integration_id": integration_id,
    }


__all__ = [
    "CopilotCLIProviderContractError",
    "apply_copilot_cli_provider_contract",
    "configure_copilot_integration_identity",
]
