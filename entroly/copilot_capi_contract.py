"""Trusted GitHub Copilot CAPI request metadata for subscription proxy mode.

Authentication and token refresh live in :mod:`copilot_subscription_transport`.
This module owns only the small request-metadata contract that GitHub's current
Copilot SDK expects on CAPI traffic. It deliberately wraps Entroly's existing
provider-header builder rather than adding another HTTP client or router.

The boundary is fail-closed for token-bound identity and fail-safe for optional
client metadata:

* ``Copilot-Integration-Id`` always comes from the same trusted manager that
  minted the short-lived Copilot credential;
* ``X-GitHub-Api-Version`` is a validated ISO date, preferring a valid client
  value, then an explicit Entroly override, then a conservative default;
* ``X-Interaction-Id`` preserves a valid client correlation id or generates a
  fresh UUID4 locally;
* the outbound user-agent must identify Copilot traffic without pretending to
  be a particular VS Code/Copilot build;
* optional initiator/intent metadata is copied only through a bounded allowlist.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import date
from typing import Any, Mapping

from . import __version__
from .copilot_subscription_transport import CopilotTokenManager

_DEFAULT_API_VERSION = "2025-04-01"
_MAX_METADATA_CHARS = 256
_INTERACTION_ID_RE = re.compile(r"^[0-9A-Fa-f-]{8,64}$")


class CopilotCAPIContractError(ValueError):
    """Invalid trusted Copilot CAPI configuration."""


def _header_value(headers: Mapping[str, Any], name: str) -> str:
    target = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == target:
            return str(value)
    return ""


def _visible_ascii(value: object, *, limit: int = _MAX_METADATA_CHARS) -> str:
    text = str(value or "").strip()
    if not text or len(text) > limit:
        return ""
    if any(ord(char) < 32 or ord(char) == 127 or ord(char) > 126 for char in text):
        return ""
    return text


def _validated_api_version(value: object) -> str:
    text = _visible_ascii(value, limit=10)
    if len(text) != 10:
        return ""
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        return ""
    if parsed.year < 2022 or parsed.year > 2100:
        return ""
    return text


def _api_version(
    original: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> str:
    env = os.environ if environ is None else environ
    client = _validated_api_version(_header_value(original, "X-GitHub-Api-Version"))
    if client:
        return client
    configured_raw = env.get("ENTROLY_COPILOT_API_VERSION")
    if configured_raw is not None:
        configured = _validated_api_version(configured_raw)
        if not configured:
            raise CopilotCAPIContractError(
                "ENTROLY_COPILOT_API_VERSION must be an ISO date such as 2025-04-01"
            )
        return configured
    return _DEFAULT_API_VERSION


def _interaction_id(original: Mapping[str, Any]) -> str:
    supplied = _visible_ascii(_header_value(original, "X-Interaction-Id"), limit=64)
    if supplied and _INTERACTION_ID_RE.fullmatch(supplied):
        return supplied
    return str(uuid.uuid4())


def _drop_case_insensitive(headers: dict[str, str], names: set[str]) -> None:
    lowered = {name.casefold() for name in names}
    for key in tuple(headers):
        if key.casefold() in lowered:
            headers.pop(key, None)


def build_copilot_capi_headers(
    *,
    original: Mapping[str, Any],
    forwarded: Mapping[str, Any],
    integration_id: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the trusted outbound CAPI header set for one request.

    ``forwarded`` must already contain the short-lived Copilot Authorization
    credential installed by ``copilot_subscription_transport``. This function
    does not read or synthesize credentials.
    """
    trusted_integration_id = _visible_ascii(integration_id, limit=128)
    if not trusted_integration_id or any(
        not (char.isascii() and (char.isalnum() or char in "._-"))
        for char in trusted_integration_id
    ):
        raise CopilotCAPIContractError("invalid trusted Copilot integration ID")

    out = {str(key): str(value) for key, value in forwarded.items()}
    _drop_case_insensitive(
        out,
        {
            "Copilot-Integration-Id",
            "X-GitHub-Api-Version",
            "X-Interaction-Id",
            "X-Initiator",
            "OpenAI-Intent",
        },
    )

    out["Copilot-Integration-Id"] = trusted_integration_id
    out["X-GitHub-Api-Version"] = _api_version(original, environ)
    out["X-Interaction-Id"] = _interaction_id(original)

    current_ua = _visible_ascii(_header_value(out, "User-Agent"), limit=256)
    if not current_ua or "copilot" not in current_ua.casefold():
        _drop_case_insensitive(out, {"User-Agent"})
        out["User-Agent"] = f"Entroly-Copilot/{__version__}"

    initiator = _visible_ascii(_header_value(original, "X-Initiator"), limit=16)
    if initiator.casefold() in {"user", "agent"}:
        out["X-Initiator"] = initiator.casefold()

    intent = _visible_ascii(_header_value(original, "OpenAI-Intent"), limit=128)
    if intent:
        out["OpenAI-Intent"] = intent

    return out


def _env_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get("ENTROLY_COPILOT_SUBSCRIPTION", "")).strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def install_copilot_capi_contract() -> bool:
    """Wrap the existing provider-header seam with the trusted CAPI contract."""
    if not _env_enabled():
        return False

    from . import proxy as proxy_module

    current_headers = proxy_module.PromptCompilerProxy._build_headers
    if getattr(current_headers, "__entroly_copilot_capi_contract__", False):
        return True

    def build_headers(
        self: Any,
        original: dict[str, str],
        provider: str,
    ) -> dict[str, str]:
        headers = current_headers(self, original, provider)
        if provider != "openai":
            return headers
        manager = getattr(self, "_copilot_subscription_token_manager", None)
        if not isinstance(manager, CopilotTokenManager):
            return headers
        return build_copilot_capi_headers(
            original=original,
            forwarded=headers,
            integration_id=manager.integration_id,
        )

    build_headers.__entroly_copilot_capi_contract__ = True
    build_headers.__entroly_copilot_capi_contract_original__ = current_headers
    proxy_module.PromptCompilerProxy._build_headers = build_headers
    return True


__all__ = [
    "CopilotCAPIContractError",
    "build_copilot_capi_headers",
    "install_copilot_capi_contract",
]
