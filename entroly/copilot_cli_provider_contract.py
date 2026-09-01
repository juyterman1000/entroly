"""Normalize Copilot CLI custom-provider settings onto the documented surface.

GitHub documents ``COPILOT_PROVIDER_API_KEY`` for Copilot CLI BYOK endpoints.
Entroly only needs a non-secret local placeholder because the hardened proxy
replaces it with the short-lived GitHub Copilot credential before forwarding.

The SDK also supports selecting the Responses wire API, but that setting is not
part of the public Copilot CLI BYOK environment-variable table. Therefore the
production default stays on Chat Completions without setting a private wire
selector. An explicit ``responses`` request remains supported as a narrowly
scoped compatibility feature and is surfaced as such in the returned summary.
"""

from __future__ import annotations

from collections.abc import MutableMapping

_LOCAL_PROVIDER_KEY = "entroly-local-provider-route"


class CopilotCLIProviderContractError(ValueError):
    """Invalid Copilot CLI custom-provider configuration."""


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
    }


__all__ = [
    "CopilotCLIProviderContractError",
    "apply_copilot_cli_provider_contract",
]
