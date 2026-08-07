"""Strict operator safety boundary for live model-routing execution.

The routing authority coordinates evidence and performs at most one same-provider
model rewrite. This module adds the deployment controls that turn that mechanism
into a safe operator-facing product surface:

* explicit provider, model, and upstream-origin allowlists;
* an auditable pricing catalog that is validated before startup;
* an explicit acknowledgement that official API credentials are authorized;
* loopback-only execution for the first production boundary;
* fail-closed startup and per-request enforcement;
* bounded status metadata with no prompts, credentials, or full filesystem paths.

Observe mode remains non-mutating. Execute mode is unavailable until every
required control is configured and verified.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .provider_adapters import canonical_request_from_provider_body
from .proxy_routing_authority import (
    RoutingAuthorityCoordinator,
    RoutingAuthorityLedger,
    RoutingRequestContext,
)
from .proxy_transform import detect_provider

_ACK_VALUE = "authorized-official-api"
_MAX_PRICING_BYTES = 2 * 1024 * 1024
_SUPPORTED_PROVIDERS = frozenset({"openai", "anthropic", "gemini"})
_OFFICIAL_ORIGINS = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
}
_INSTALLED = False
_ORIGINAL_AUTHORIZE = None
_ORIGINAL_SNAPSHOT = None
_ORIGINAL_LEDGER_FOR_PROXY = None


class RoutingSafetyConfigurationError(RuntimeError):
    """Raised before proxy startup when execute-mode controls are incomplete."""


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RoutingSafetyConfigurationError(f"{name} must be a boolean flag")


def _csv(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    values = []
    seen = set()
    for item in raw.split(","):
        value = item.strip()
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    return tuple(values)


def _canonical_origin(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise RoutingSafetyConfigurationError(
            "routing upstream origins must be absolute HTTPS origins"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RoutingSafetyConfigurationError(
            "routing upstream origins must not contain credentials, query, or fragment"
        )
    if parsed.path not in {"", "/"}:
        raise RoutingSafetyConfigurationError(
            "routing upstream allowlists accept origins only, not URL paths"
        )
    host = parsed.hostname.casefold()
    port = parsed.port
    default_port = 443
    suffix = "" if port in {None, default_port} else f":{port}"
    return f"https://{host}{suffix}"


def _origin_of_url(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.hostname:
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    suffix = "" if port in {None, 443} else f":{port}"
    return f"{parsed.scheme.casefold()}://{parsed.hostname.casefold()}{suffix}"


def _parse_provider_models(values: tuple[str, ...]) -> frozenset[str]:
    result: set[str] = set()
    for value in values:
        provider, separator, model = value.partition(":")
        provider = provider.strip().casefold()
        model = model.strip()
        if not separator or provider not in _SUPPORTED_PROVIDERS or not model:
            raise RoutingSafetyConfigurationError(
                "ENTROLY_ROUTING_AUTHORITY_ALLOWED_MODELS entries must use provider:model"
            )
        result.add(f"{provider}:{model}")
    return frozenset(result)


def _parse_provider_origins(values: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    result: dict[str, str] = {}
    for value in values:
        provider, separator, origin = value.partition("=")
        provider = provider.strip().casefold()
        if not separator or provider not in _SUPPORTED_PROVIDERS:
            raise RoutingSafetyConfigurationError(
                "ENTROLY_ROUTING_AUTHORITY_ALLOWED_ORIGINS entries must use provider=https://host"
            )
        canonical = _canonical_origin(origin)
        previous = result.get(provider)
        if previous is not None and previous != canonical:
            raise RoutingSafetyConfigurationError(
                f"multiple routing origins configured for provider {provider!r}"
            )
        result[provider] = canonical
    return tuple(sorted(result.items()))


def _is_loopback_host(host: str) -> bool:
    normalized = (host or "").strip().casefold()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _nonnegative_number(value: object, *, field: str, model: str) -> None:
    from decimal import Decimal, InvalidOperation

    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RoutingSafetyConfigurationError(
            f"pricing catalog field {field!r} for {model!r} must be numeric"
        ) from exc
    if not parsed.is_finite() or parsed < 0:
        raise RoutingSafetyConfigurationError(
            f"pricing catalog field {field!r} for {model!r} must be finite and non-negative"
        )


def _pricing_catalog_evidence(
    path_value: str,
    *,
    allowed_models: frozenset[str],
) -> tuple[str, str]:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise RoutingSafetyConfigurationError(
            "ENTROLY_PRICING_CATALOG must be an absolute path in execute mode"
        )
    if not path.is_file():
        raise RoutingSafetyConfigurationError(
            "ENTROLY_PRICING_CATALOG must point to an existing JSON file"
        )
    metadata = path.stat()
    if metadata.st_size > _MAX_PRICING_BYTES:
        raise RoutingSafetyConfigurationError("pricing catalog exceeds the 2 MiB safety limit")
    if os.name == "posix" and metadata.st_mode & stat.S_IWOTH:
        raise RoutingSafetyConfigurationError("pricing catalog must not be world-writable")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RoutingSafetyConfigurationError("pricing catalog must contain valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise RoutingSafetyConfigurationError(
            "pricing catalog must contain a top-level models object"
        )
    models = payload["models"]
    missing = sorted(model for model in allowed_models if model not in models)
    if missing:
        rendered = ", ".join(missing[:5])
        raise RoutingSafetyConfigurationError(
            f"pricing catalog is missing allowlisted model entries: {rendered}"
        )
    for model in sorted(allowed_models):
        entry = models.get(model)
        if not isinstance(entry, dict):
            raise RoutingSafetyConfigurationError(
                f"pricing catalog entry for {model!r} must be an object"
            )
        for field in ("input_per_million", "output_per_million"):
            if field not in entry:
                raise RoutingSafetyConfigurationError(
                    f"pricing catalog entry for {model!r} is missing {field!r}"
                )
            _nonnegative_number(entry[field], field=field, model=model)
        for field in ("cache_read_per_million", "cache_write_per_million"):
            if field in entry:
                _nonnegative_number(entry[field], field=field, model=model)
    return path.name, hashlib.sha256(raw).hexdigest()


def _configured_base_origin(config: Any, provider: str) -> str:
    attribute = {
        "openai": "openai_base_url",
        "anthropic": "anthropic_base_url",
        "gemini": "gemini_base_url",
    }[provider]
    return _origin_of_url(str(getattr(config, attribute, "")))


@dataclass(frozen=True, slots=True)
class RoutingSafetyConfig:
    enabled: bool
    mode: str
    allowed_providers: frozenset[str]
    allowed_models: frozenset[str]
    allowed_origins: tuple[tuple[str, str], ...]
    require_pricing: bool
    pricing_catalog_name: str
    pricing_catalog_sha256: str
    operator_acknowledged: bool
    ravs_enabled: bool
    escalation_mode: str
    loopback_only: bool

    @property
    def origin_map(self) -> dict[str, str]:
        return dict(self.allowed_origins)

    def public_summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "strict_execution": self.mode == "execute",
            "same_provider_only": True,
            "loopback_only": self.loopback_only,
            "allowed_providers": sorted(self.allowed_providers),
            "allowed_models": sorted(self.allowed_models),
            "allowed_origins": {
                provider: origin for provider, origin in self.allowed_origins
            },
            "pricing_required": self.require_pricing,
            "pricing_catalog_name": self.pricing_catalog_name,
            "pricing_catalog_sha256": self.pricing_catalog_sha256,
            "operator_acknowledged": self.operator_acknowledged,
            "ravs_enabled": self.ravs_enabled,
            "escalation_mode": self.escalation_mode,
            "credentials_stored": False,
            "consumer_sessions_supported": False,
            "cross_provider_execution": False,
            "automatic_retries": False,
        }


def validate_routing_environment(
    *,
    proxy_config: Any | None = None,
    host: str | None = None,
) -> RoutingSafetyConfig:
    """Validate routing deployment controls before the proxy starts."""
    enabled = _env_flag("ENTROLY_ROUTING_AUTHORITY", default=False)
    if not enabled:
        return RoutingSafetyConfig(
            enabled=False,
            mode="observe",
            allowed_providers=frozenset(),
            allowed_models=frozenset(),
            allowed_origins=(),
            require_pricing=True,
            pricing_catalog_name="",
            pricing_catalog_sha256="",
            operator_acknowledged=False,
            ravs_enabled=False,
            escalation_mode="observe",
            loopback_only=False,
        )

    mode = os.environ.get("ENTROLY_ROUTING_AUTHORITY_MODE", "observe").strip().casefold()
    if mode not in {"observe", "execute"}:
        raise RoutingSafetyConfigurationError(
            "ENTROLY_ROUTING_AUTHORITY_MODE must be observe or execute"
        )

    provider_values = tuple(
        value.casefold()
        for value in _csv("ENTROLY_ROUTING_AUTHORITY_ALLOWED_PROVIDERS")
    )
    unknown = sorted(set(provider_values) - _SUPPORTED_PROVIDERS)
    if unknown:
        raise RoutingSafetyConfigurationError(
            f"unsupported routing providers: {', '.join(unknown)}"
        )
    allowed_providers = frozenset(provider_values)
    allowed_models = _parse_provider_models(
        _csv("ENTROLY_ROUTING_AUTHORITY_ALLOWED_MODELS")
    )
    allowed_origins = _parse_provider_origins(
        _csv("ENTROLY_ROUTING_AUTHORITY_ALLOWED_ORIGINS")
    )
    origin_map = dict(allowed_origins)
    require_pricing = _env_flag(
        "ENTROLY_ROUTING_AUTHORITY_REQUIRE_PRICING",
        default=True,
    )
    ravs_enabled = _env_flag("ENTROLY_RAVS_ROUTER", default=False)
    acknowledged = os.environ.get("ENTROLY_ROUTING_AUTHORITY_ACK", "") == _ACK_VALUE
    escalation_mode = os.environ.get("ENTROLY_ESCALATION_MODE", "observe").strip().casefold()
    effective_host = host or str(getattr(proxy_config, "host", "127.0.0.1"))

    catalog_name = ""
    catalog_digest = ""
    pricing_path = os.environ.get("ENTROLY_PRICING_CATALOG", "").strip()
    if pricing_path:
        catalog_name, catalog_digest = _pricing_catalog_evidence(
            pricing_path,
            allowed_models=allowed_models,
        )

    if enabled and not allowed_providers:
        raise RoutingSafetyConfigurationError(
            "routing authority requires an explicit provider allowlist"
        )
    if any(model.split(":", 1)[0] not in allowed_providers for model in allowed_models):
        raise RoutingSafetyConfigurationError(
            "every allowlisted model must belong to an allowlisted provider"
        )
    if any(provider not in allowed_providers for provider in origin_map):
        raise RoutingSafetyConfigurationError(
            "every pinned origin must belong to an allowlisted provider"
        )

    if enabled:
        missing_origins = sorted(allowed_providers - origin_map.keys())
        if missing_origins:
            raise RoutingSafetyConfigurationError(
                "missing pinned upstream origins for: " + ", ".join(missing_origins)
            )
        for provider in allowed_providers:
            if proxy_config is not None:
                configured = _configured_base_origin(proxy_config, provider)
                if configured != origin_map[provider]:
                    raise RoutingSafetyConfigurationError(
                        f"configured {provider} base origin does not match its routing pin"
                    )

    if enabled and mode == "execute":
        non_official = sorted(
            provider
            for provider in allowed_providers
            if origin_map.get(provider) != _OFFICIAL_ORIGINS[provider]
        )
        if non_official:
            raise RoutingSafetyConfigurationError(
                "routing execute mode currently permits only official provider API origins: "
                + ", ".join(non_official)
            )
        if not _is_loopback_host(effective_host):
            raise RoutingSafetyConfigurationError(
                "routing execute mode is loopback-only in this release"
            )
        if not ravs_enabled:
            raise RoutingSafetyConfigurationError(
                "routing execute mode requires ENTROLY_RAVS_ROUTER=1"
            )
        if not acknowledged:
            raise RoutingSafetyConfigurationError(
                f"routing execute mode requires ENTROLY_ROUTING_AUTHORITY_ACK={_ACK_VALUE}"
            )
        if not require_pricing:
            raise RoutingSafetyConfigurationError(
                "routing execute mode cannot disable auditable pricing"
            )
        if len(allowed_models) < 2:
            raise RoutingSafetyConfigurationError(
                "routing execute mode requires at least two exact allowlisted models"
            )
        if not pricing_path or not catalog_digest:
            raise RoutingSafetyConfigurationError(
                "routing execute mode requires a validated pricing catalog"
            )
        if escalation_mode in {"active", "shadow"}:
            raise RoutingSafetyConfigurationError(
                "routing execute mode conflicts with active or shadow escalation"
            )

    return RoutingSafetyConfig(
        enabled=enabled,
        mode=mode,
        allowed_providers=allowed_providers,
        allowed_models=allowed_models,
        allowed_origins=allowed_origins,
        require_pricing=require_pricing,
        pricing_catalog_name=catalog_name,
        pricing_catalog_sha256=catalog_digest,
        operator_acknowledged=acknowledged,
        ravs_enabled=ravs_enabled,
        escalation_mode=escalation_mode,
        loopback_only=mode == "execute",
    )


def configure_proxy_routing_safety(proxy: Any, config: RoutingSafetyConfig) -> None:
    """Attach immutable validated policy to one proxy instance."""
    proxy._routing_safety_config = config
    ledger = getattr(proxy, "_routing_authority_ledger", None)
    if isinstance(ledger, RoutingAuthorityLedger):
        ledger._routing_safety_config = config


def _enforce_request_policy(
    coordinator: RoutingAuthorityCoordinator,
    context: RoutingRequestContext,
    *,
    provider: str,
    target: Any,
    body: Mapping[str, Any],
    url: str,
) -> None:
    config = getattr(context.proxy, "_routing_safety_config", None)
    if not isinstance(config, RoutingSafetyConfig) or not config.enabled:
        return

    source_provider = detect_provider(context.path, dict(context.headers), dict(body))
    adapter = canonical_request_from_provider_body(
        source_provider,
        body,
        headers=context.headers,
        path=context.path,
    )
    source_model = adapter.canonical.model
    target_provider = str(target.provider).strip().casefold()
    target_model = str(target.model).strip()

    if source_provider not in config.allowed_providers:
        coordinator._deny(context, "source_provider_not_allowlisted")
    if provider.strip().casefold() not in config.allowed_providers:
        coordinator._deny(context, "proxy_provider_not_allowlisted")
    if target_provider not in config.allowed_providers:
        coordinator._deny(context, "target_provider_not_allowlisted")

    pinned_origin = config.origin_map.get(source_provider, "")
    if _origin_of_url(url) != pinned_origin:
        coordinator._deny(context, "upstream_origin_not_pinned")

    if config.allowed_models:
        if f"{source_provider}:{source_model}" not in config.allowed_models:
            coordinator._deny(context, "source_model_not_allowlisted")
        if f"{target_provider}:{target_model}" not in config.allowed_models:
            coordinator._deny(context, "target_model_not_allowlisted")


def install_routing_safety() -> None:
    """Install idempotent proxy-scoped enforcement around routing authority."""
    global _INSTALLED, _ORIGINAL_AUTHORIZE, _ORIGINAL_SNAPSHOT, _ORIGINAL_LEDGER_FOR_PROXY
    if _INSTALLED:
        return

    from . import proxy_routing_authority as authority

    _ORIGINAL_AUTHORIZE = RoutingAuthorityCoordinator.authorize
    _ORIGINAL_SNAPSHOT = RoutingAuthorityLedger.snapshot
    _ORIGINAL_LEDGER_FOR_PROXY = authority._ledger_for_proxy

    def safe_authorize(
        self: RoutingAuthorityCoordinator,
        context: RoutingRequestContext,
        **kwargs: Any,
    ):
        _enforce_request_policy(
            self,
            context,
            provider=kwargs["provider"],
            target=kwargs["target"],
            body=kwargs["body"],
            url=kwargs["url"],
        )
        return _ORIGINAL_AUTHORIZE(self, context, **kwargs)

    def safe_snapshot(self: RoutingAuthorityLedger) -> dict[str, Any]:
        value = _ORIGINAL_SNAPSHOT(self)
        config = getattr(self, "_routing_safety_config", None)
        if isinstance(config, RoutingSafetyConfig):
            value["safety"] = config.public_summary()
        return value

    def safe_ledger_for_proxy(proxy: Any) -> RoutingAuthorityLedger:
        ledger = _ORIGINAL_LEDGER_FOR_PROXY(proxy)
        config = getattr(proxy, "_routing_safety_config", None)
        if isinstance(config, RoutingSafetyConfig):
            ledger._routing_safety_config = config
        return ledger

    safe_authorize.__entroly_routing_safety_original__ = _ORIGINAL_AUTHORIZE
    safe_snapshot.__entroly_routing_safety_original__ = _ORIGINAL_SNAPSHOT
    safe_ledger_for_proxy.__entroly_routing_safety_original__ = _ORIGINAL_LEDGER_FOR_PROXY
    RoutingAuthorityCoordinator.authorize = safe_authorize
    RoutingAuthorityLedger.snapshot = safe_snapshot
    authority._ledger_for_proxy = safe_ledger_for_proxy
    _INSTALLED = True


def official_origin(provider: str) -> str:
    """Return the official default API origin for a supported provider."""
    normalized = provider.strip().casefold()
    try:
        return _OFFICIAL_ORIGINS[normalized]
    except KeyError as exc:
        raise RoutingSafetyConfigurationError(
            f"unsupported routing provider: {provider!r}"
        ) from exc


__all__ = [
    "RoutingSafetyConfig",
    "RoutingSafetyConfigurationError",
    "configure_proxy_routing_safety",
    "install_routing_safety",
    "official_origin",
    "validate_routing_environment",
]
