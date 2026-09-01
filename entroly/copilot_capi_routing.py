"""GitHub Copilot CAPI path normalization for Entroly subscription mode.

Copilot CLI's OpenAI-compatible custom-provider surface addresses a local base
URL ending in ``/v1``. GitHub Copilot CAPI itself serves the corresponding chat,
Responses, and model-discovery endpoints without that ``/v1`` prefix. Entroly's
normal proxy intentionally preserves request paths for ordinary providers, so
subscription mode needs one narrow path-normalization seam before the existing
hardened target resolver builds the upstream URL.

This is not a second router: provider detection, origin validation, query
handling, redirects, retries, and transport remain owned by the existing proxy.
Only three exact OpenAI-compatible paths are translated, and only when a live
Copilot subscription token manager is attached to the proxy instance.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from .copilot_subscription_transport import CopilotTokenManager

_CAPI_PATHS = {
    "/v1/chat/completions": "/chat/completions",
    "/v1/responses": "/responses",
    "/v1/models": "/models",
}


def normalize_copilot_capi_path(path: str) -> str:
    """Translate only the known OpenAI-compatible ``/v1`` CAPI endpoints."""
    return _CAPI_PATHS.get(path, path)


def _env_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get("ENTROLY_COPILOT_SUBSCRIPTION", "")).strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def install_copilot_capi_routing() -> bool:
    """Wrap Entroly's hardened target resolver with exact CAPI path translation."""
    if not _env_enabled():
        return False

    from . import proxy as proxy_module

    current_resolve = proxy_module.PromptCompilerProxy._resolve_target
    if getattr(current_resolve, "__entroly_copilot_capi_routing__", False):
        return True

    def resolve_target(self: Any, provider: str, path: str) -> str:
        if provider == "openai" and isinstance(
            getattr(self, "_copilot_subscription_token_manager", None),
            CopilotTokenManager,
        ):
            path = normalize_copilot_capi_path(path)
        return current_resolve(self, provider, path)

    resolve_target.__entroly_copilot_capi_routing__ = True
    resolve_target.__entroly_copilot_capi_routing_original__ = current_resolve
    proxy_module.PromptCompilerProxy._resolve_target = resolve_target
    return True


__all__ = ["install_copilot_capi_routing", "normalize_copilot_capi_path"]
