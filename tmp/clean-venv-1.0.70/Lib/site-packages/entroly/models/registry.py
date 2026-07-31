from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


class RegistryTrust(str, Enum):
    """How strongly Entroly can rely on a model capability record."""

    VERIFIED = "verified"
    DISCOVERED = "discovered"
    USER = "user"
    ANNOUNCED = "announced"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class ModelCapability:
    id: str
    provider: str
    aliases: tuple[str, ...]
    context_window: int | None
    max_output_tokens: int | None
    supports_tools: bool | None
    supports_vision: bool | None
    supports_reasoning: bool | None
    reasoning_levels: tuple[str, ...]
    input_price_per_million: float | None
    output_price_per_million: float | None
    trust: RegistryTrust
    source: str
    verified_at: str | None
    observed_at: str | None

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
        *,
        default_trust: RegistryTrust,
    ) -> "ModelCapability":
        model_id = _normalise_name(value.get("id"))
        provider = _normalise_name(value.get("provider"))
        if not model_id:
            raise ValueError("model capability id must be non-empty")
        if not provider:
            raise ValueError(f"provider must be non-empty for {model_id!r}")

        context = value.get("context_window")
        if context is not None and (not isinstance(context, int) or context <= 0):
            raise ValueError(f"invalid context_window for {model_id!r}: {context!r}")

        output = value.get("max_output_tokens")
        if output is not None and (not isinstance(output, int) or output <= 0):
            raise ValueError(f"invalid max_output_tokens for {model_id!r}: {output!r}")
        if context is not None and output is not None and output >= context:
            raise ValueError(
                f"max_output_tokens must be smaller than context_window for {model_id!r}"
            )

        aliases = tuple(
            alias
            for alias in (_normalise_name(item) for item in value.get("aliases", ()))
            if alias
        )
        return cls(
            id=model_id,
            provider=provider,
            aliases=aliases,
            context_window=context,
            max_output_tokens=output,
            supports_tools=_optional_bool(value.get("supports_tools")),
            supports_vision=_optional_bool(value.get("supports_vision")),
            supports_reasoning=_optional_bool(value.get("supports_reasoning")),
            reasoning_levels=tuple(str(item) for item in value.get("reasoning_levels", ())),
            input_price_per_million=_optional_float(value.get("input_price_per_million")),
            output_price_per_million=_optional_float(value.get("output_price_per_million")),
            trust=RegistryTrust(value.get("trust", default_trust.value)),
            source=str(value.get("source", "unknown")).strip() or "unknown",
            verified_at=_optional_iso_date(value.get("verified_at"), field="verified_at"),
            observed_at=_optional_iso_date(value.get("observed_at"), field="observed_at"),
        )

    def estimated_cost_usd(self, input_tokens: int, output_tokens: int) -> float | None:
        """Estimate cost only when both price dimensions are known."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts cannot be negative")
        if self.input_price_per_million is None or self.output_price_per_million is None:
            return None
        return (
            input_tokens * self.input_price_per_million
            + output_tokens * self.output_price_per_million
        ) / 1_000_000.0

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "aliases": sorted(self.aliases),
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "supports_reasoning": self.supports_reasoning,
            "reasoning_levels": list(self.reasoning_levels),
            "input_price_per_million": self.input_price_per_million,
            "output_price_per_million": self.output_price_per_million,
            "trust": self.trust.value,
            "source": self.source,
            "verified_at": self.verified_at,
            "observed_at": self.observed_at,
        }


@dataclass(frozen=True, slots=True)
class ModelResolution:
    requested_model: str
    capability: ModelCapability | None
    context_window: int
    exact: bool
    trust: RegistryTrust
    warning: str | None
    registry_digest: str
    base_registry_digest: str

    @property
    def model_id(self) -> str:
        return self.capability.id if self.capability else self.requested_model

    def effective_input_budget(
        self,
        *,
        requested_output_tokens: int | None = None,
        safety_fraction: float = 0.05,
        minimum_safety_tokens: int = 512,
    ) -> int:
        """Return a safe input ceiling after reserving output and uncertainty room.

        Context windows are shared by input and output. Budgeting against the raw
        window can therefore overfill a request even when the model metadata is
        otherwise correct. Entroly reserves the greater of a percentage margin
        and a small absolute floor, plus the requested or model output allowance.
        """
        if not 0.0 <= safety_fraction < 1.0:
            raise ValueError("safety_fraction must be in [0, 1)")
        if minimum_safety_tokens < 0:
            raise ValueError("minimum_safety_tokens cannot be negative")
        if requested_output_tokens is not None and requested_output_tokens < 0:
            raise ValueError("requested_output_tokens cannot be negative")

        model_output = self.capability.max_output_tokens if self.capability else None
        output_reserve = requested_output_tokens
        if output_reserve is None:
            output_reserve = model_output or 0
        elif model_output is not None:
            output_reserve = min(output_reserve, model_output)

        safety = max(minimum_safety_tokens, int(self.context_window * safety_fraction))
        return max(1, self.context_window - output_reserve - safety)


@dataclass(frozen=True, slots=True)
class DiscoveryReport:
    models: tuple[ModelCapability, ...]
    warnings: tuple[str, ...]


class ModelRegistry:
    """Deterministic layered model registry.

    Precedence is user overrides > discovered local models > bundled registry.
    Unknown models fail visibly through a warning-bearing conservative fallback.
    """

    def __init__(
        self,
        bundled: Iterable[ModelCapability],
        *,
        overrides: Iterable[ModelCapability] = (),
        discovered: Iterable[ModelCapability] = (),
        fallback_context_window: int = 128_000,
        registry_digest: str = "unknown",
        discovery_warnings: Iterable[str] = (),
    ) -> None:
        if fallback_context_window <= 0:
            raise ValueError("fallback_context_window must be positive")
        self._fallback_context_window = fallback_context_window
        self._base_registry_digest = registry_digest
        self._discovery_warnings = tuple(discovery_warnings)
        self._by_id: dict[str, ModelCapability] = {}
        self._aliases: dict[str, str] = {}
        self._aliases_by_id: dict[str, set[str]] = {}
        for capability in bundled:
            self._install(capability)
        for capability in discovered:
            # Runtime discovery refines an existing route. Preserve its known
            # bundled aliases so enabling discovery cannot break callers that
            # use the canonical provider alias. Explicit user overrides retain
            # replacement semantics and may intentionally retire aliases.
            self._install(capability, preserve_aliases=True)
        for capability in overrides:
            self._install(capability)
        self._registry_digest = _effective_registry_digest(
            self._by_id.values(),
            fallback_context_window=self._fallback_context_window,
        )

    @property
    def registry_digest(self) -> str:
        """Fingerprint of the effective bundled + discovered + override registry."""
        return self._registry_digest

    @property
    def base_registry_digest(self) -> str:
        """Fingerprint of the immutable bundled snapshot before runtime layers."""
        return self._base_registry_digest

    @property
    def discovery_warnings(self) -> tuple[str, ...]:
        return self._discovery_warnings

    def _install(
        self,
        capability: ModelCapability,
        *,
        preserve_aliases: bool = False,
    ) -> None:
        # Replacing a record must remove aliases no longer present; otherwise an
        # older layer can remain reachable through a stale alias.
        old_aliases = self._aliases_by_id.get(capability.id, set())
        if preserve_aliases and old_aliases:
            capability = replace(
                capability,
                aliases=tuple(sorted((old_aliases | set(capability.aliases)) - {capability.id})),
            )
        for old_alias in old_aliases:
            if self._aliases.get(old_alias) == capability.id:
                self._aliases.pop(old_alias, None)

        aliases = {capability.id, *capability.aliases}
        self._by_id[capability.id] = capability
        self._aliases_by_id[capability.id] = aliases
        for alias in aliases:
            self._aliases[alias] = capability.id

    def all(self) -> tuple[ModelCapability, ...]:
        return tuple(sorted(self._by_id.values(), key=lambda item: item.id))

    def diagnostics(self) -> dict[str, Any]:
        counts = {trust.value: 0 for trust in RegistryTrust}
        for capability in self._by_id.values():
            counts[capability.trust.value] += 1
        return {
            "registry_digest": self._registry_digest,
            "base_registry_digest": self._base_registry_digest,
            "models": len(self._by_id),
            "trust_counts": counts,
            "discovery_warnings": list(self._discovery_warnings),
        }

    def resolve(self, model: str) -> ModelResolution:
        requested = _normalise_name(model)
        canonical = self._aliases.get(requested)
        exact = canonical is not None
        if canonical is None:
            matches = [
                alias
                for alias in self._aliases
                if _is_prefix_alias(alias) and requested.startswith(alias)
            ]
            if matches:
                longest = max(len(alias) for alias in matches)
                candidate_ids = {
                    self._aliases[alias]
                    for alias in matches
                    if len(alias) == longest
                }
                if len(candidate_ids) == 1:
                    canonical = candidate_ids.pop()
                else:
                    return self._fallback_resolution(
                        model,
                        warning=(
                            f"Ambiguous model prefix {model!r}; matched "
                            f"{', '.join(sorted(candidate_ids))}. Add an exact alias override."
                        ),
                    )

        if canonical is None:
            return self._fallback_resolution(
                model,
                warning=(
                    f"Unknown model {model!r}; using conservative "
                    f"{self._fallback_context_window:,}-token fallback. "
                    "Add verified metadata or an ENTROLY_MODEL_REGISTRY override."
                ),
            )

        capability = self._by_id[canonical]
        if capability.context_window is None:
            return ModelResolution(
                requested_model=model,
                capability=capability,
                context_window=self._fallback_context_window,
                exact=exact,
                trust=capability.trust,
                warning=(
                    f"Model {capability.id!r} is recognized but its context window is unverified; "
                    f"using conservative {self._fallback_context_window:,}-token fallback."
                ),
                registry_digest=self._registry_digest,
                base_registry_digest=self._base_registry_digest,
            )
        return ModelResolution(
            requested_model=model,
            capability=capability,
            context_window=capability.context_window,
            exact=exact,
            trust=capability.trust,
            warning=None,
            registry_digest=self._registry_digest,
            base_registry_digest=self._base_registry_digest,
        )

    def _fallback_resolution(self, model: str, *, warning: str) -> ModelResolution:
        return ModelResolution(
            requested_model=model,
            capability=None,
            context_window=self._fallback_context_window,
            exact=False,
            trust=RegistryTrust.FALLBACK,
            warning=warning,
            registry_digest=self._registry_digest,
            base_registry_digest=self._base_registry_digest,
        )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise HTTPError(req.full_url, code, "redirects disabled", headers, fp)


def _normalise_name(value: object) -> str:
    return str(value or "").strip().lower()


def _is_prefix_alias(alias: str) -> bool:
    # Prefix behavior must be explicit. Bare aliases are exact-only so a model
    # such as "gpt-4xyz" cannot accidentally inherit the gpt-4 context limit.
    return alias.endswith(("-", "/", ":", "."))


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"capability flags must be boolean or null, got {value!r}")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    if result < 0:
        raise ValueError("price metadata cannot be negative")
    return result


def _optional_iso_date(value: object, *, field: str) -> str | None:
    if value in {None, ""}:
        return None
    result = str(value)
    try:
        date.fromisoformat(result)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date, got {result!r}") from exc
    return result


def _effective_registry_digest(
    capabilities: Iterable[ModelCapability],
    *,
    fallback_context_window: int,
) -> str:
    payload = {
        "fallback_context_window": fallback_context_window,
        "models": [
            capability.fingerprint_payload()
            for capability in sorted(capabilities, key=lambda item: item.id)
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_json(path: Path, *, default_trust: RegistryTrust) -> list[ModelCapability]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("models", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"model registry {path} must contain a list of models")
    capabilities = [
        ModelCapability.from_mapping(item, default_trust=default_trust)
        for item in records
    ]
    ids = [capability.id for capability in capabilities]
    if len(ids) != len(set(ids)):
        raise ValueError(f"model registry {path} contains duplicate model ids")
    return capabilities


def _bundled_models() -> tuple[list[ModelCapability], str]:
    path = Path(str(files("entroly.models").joinpath("registry.json")))
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    return _load_json(path, default_trust=RegistryTrust.VERIFIED), digest


def _override_models() -> list[ModelCapability]:
    raw = os.environ.get("ENTROLY_MODEL_REGISTRY", "").strip()
    if not raw:
        return []
    models: list[ModelCapability] = []
    for item in raw.split(os.pathsep):
        path = Path(item).expanduser()
        models.extend(_load_json(path, default_trust=RegistryTrust.USER))
    return models


def _loopback_base_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid local model endpoint: {value!r}")
    if parsed.username or parsed.password:
        raise ValueError("credentials are not permitted in local discovery URLs")
    hostname = parsed.hostname.lower()
    if hostname != "localhost":
        try:
            if not ipaddress.ip_address(hostname).is_loopback:
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                f"local model discovery is restricted to loopback addresses, got {hostname!r}"
            ) from exc
    return value.rstrip("/")


def _json_request(
    url: str,
    *,
    timeout: float,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    max_bytes: int = 2 * 1024 * 1024,
) -> Any:
    if timeout <= 0:
        raise ValueError("discovery timeout must be positive")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers=request_headers,
    )
    opener = build_opener(_NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"local model discovery response exceeded {max_bytes} bytes")
    return json.loads(raw.decode("utf-8"))


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result > 0 else None


def _openrouter_price_per_million(value: object) -> float | None:
    """Convert OpenRouter's dollars-per-token strings to dollars per million."""
    if value in {None, ""}:
        return None
    try:
        result = float(value) * 1_000_000
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _ollama_context_length(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    model_info = payload.get("model_info")
    if not isinstance(model_info, dict):
        return None
    candidates = [
        value
        for key, value in model_info.items()
        if str(key).endswith(".context_length") and isinstance(value, int) and value > 0
    ]
    return max(candidates) if candidates else None


def discover_ollama_models(
    base_url: str = "http://127.0.0.1:11434",
    *,
    timeout: float = 0.2,
    max_models: int = 64,
    inspect_context: bool = False,
) -> DiscoveryReport:
    if max_models <= 0:
        raise ValueError("max_models must be positive")
    base = _loopback_base_url(base_url)
    warnings: list[str] = []
    try:
        payload = _json_request(f"{base}/api/tags", timeout=timeout)
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        return DiscoveryReport((), (f"Ollama discovery unavailable at {base}: {exc}",))

    records = payload.get("models", []) if isinstance(payload, dict) else []
    models: list[ModelCapability] = []
    for record in records[:max_models] if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        name = _normalise_name(record.get("name") or record.get("model"))
        if not name:
            continue
        context: int | None = None
        if inspect_context:
            try:
                details = _json_request(
                    f"{base}/api/show",
                    timeout=timeout,
                    payload={"model": name},
                )
                context = _ollama_context_length(details)
            except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
                warnings.append(f"Could not inspect Ollama model {name!r}: {exc}")
        models.append(
            ModelCapability(
                id=f"ollama/{name}",
                provider="ollama",
                aliases=(name,),
                context_window=context,
                max_output_tokens=None,
                supports_tools=None,
                supports_vision=None,
                supports_reasoning=None,
                reasoning_levels=(),
                input_price_per_million=None,
                output_price_per_million=None,
                trust=RegistryTrust.DISCOVERED,
                source=f"{base}/api/tags",
                verified_at=None,
                observed_at=date.today().isoformat(),
            )
        )
    return DiscoveryReport(tuple(models), tuple(warnings))


def discover_openai_compatible_models(
    base_url: str = "http://127.0.0.1:1234",
    *,
    timeout: float = 0.2,
    max_models: int = 64,
    provider: str = "lmstudio",
) -> DiscoveryReport:
    if max_models <= 0:
        raise ValueError("max_models must be positive")
    base = _loopback_base_url(base_url)
    try:
        payload = _json_request(f"{base}/v1/models", timeout=timeout)
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        return DiscoveryReport((), (f"Local OpenAI-compatible discovery unavailable at {base}: {exc}",))

    records = payload.get("data", []) if isinstance(payload, dict) else []
    models: list[ModelCapability] = []
    for record in records[:max_models] if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        name = _normalise_name(record.get("id"))
        if not name:
            continue
        models.append(
            ModelCapability(
                id=f"{provider}/{name}",
                provider=provider,
                aliases=(name,),
                context_window=None,
                max_output_tokens=None,
                supports_tools=None,
                supports_vision=None,
                supports_reasoning=None,
                reasoning_levels=(),
                input_price_per_million=None,
                output_price_per_million=None,
                trust=RegistryTrust.DISCOVERED,
                source=f"{base}/v1/models",
                verified_at=None,
                observed_at=date.today().isoformat(),
            )
        )
    return DiscoveryReport(tuple(models), ())


def discover_openrouter_models(
    api_key: str | None = None,
    *,
    timeout: float = 3.0,
    max_models: int = 512,
) -> DiscoveryReport:
    """Discover current OpenRouter metadata without trusting it as a bundled fact.

    The endpoint is intentionally fixed: unlike local discovery, callers cannot
    redirect the bearer credential to an arbitrary host. The API key is never
    persisted and is excluded from diagnostics and error messages.
    """
    if max_models <= 0:
        raise ValueError("max_models must be positive")
    key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
    if not key:
        return DiscoveryReport((), ("OpenRouter discovery skipped: OPENROUTER_API_KEY is unset",))

    endpoint = "https://openrouter.ai/api/v1/models"
    try:
        payload = _json_request(
            endpoint,
            timeout=timeout,
            headers={"Authorization": f"Bearer {key}"},
        )
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        return DiscoveryReport((), (f"OpenRouter discovery unavailable: {exc}",))

    records = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        return DiscoveryReport((), ("OpenRouter discovery returned an invalid model list",))

    observed_at = date.today().isoformat()
    models: list[ModelCapability] = []
    warnings: list[str] = []
    for record in records[:max_models]:
        if not isinstance(record, dict):
            continue
        model_id = _normalise_name(record.get("id"))
        if not model_id or "/" not in model_id:
            continue

        parameters = record.get("supported_parameters")
        has_parameters = isinstance(parameters, list)
        supported = {
            _normalise_name(item)
            for item in parameters
            if isinstance(item, str)
        } if isinstance(parameters, list) else set()
        architecture = record.get("architecture")
        input_modalities = architecture.get("input_modalities") \
            if isinstance(architecture, dict) else None
        has_modalities = isinstance(input_modalities, list)
        modalities = {
            _normalise_name(item)
            for item in input_modalities
            if isinstance(item, str)
        } if isinstance(input_modalities, list) else set()
        top_provider = record.get("top_provider")
        context_window = _positive_int(record.get("context_length"))
        max_output = _positive_int(top_provider.get("max_completion_tokens")) \
            if isinstance(top_provider, dict) else None
        if context_window is not None and max_output is not None and max_output >= context_window:
            warnings.append(
                f"OpenRouter model {model_id!r} reported max completion tokens "
                "outside its context window; output limit ignored"
            )
            max_output = None
        pricing = record.get("pricing")
        pricing = pricing if isinstance(pricing, dict) else {}

        models.append(
            ModelCapability(
                id=model_id,
                provider="openrouter",
                aliases=(f"openrouter/{model_id}",),
                context_window=context_window,
                max_output_tokens=max_output,
                supports_tools=(
                    bool(supported & {"tools", "tool_choice"}) if has_parameters else None
                ),
                supports_vision=("image" in modalities if has_modalities else None),
                supports_reasoning=(
                    bool(supported & {"reasoning", "reasoning_effort", "include_reasoning"})
                    if has_parameters
                    else None
                ),
                reasoning_levels=(),
                input_price_per_million=_openrouter_price_per_million(pricing.get("prompt")),
                output_price_per_million=_openrouter_price_per_million(
                    pricing.get("completion")
                ),
                trust=RegistryTrust.DISCOVERED,
                source=endpoint,
                verified_at=None,
                observed_at=observed_at,
            )
        )
    return DiscoveryReport(tuple(models), tuple(warnings))


def discover_local_models() -> DiscoveryReport:
    raw = os.environ.get("ENTROLY_DISCOVER_LOCAL_MODELS", "").strip().lower()
    if not raw:
        return DiscoveryReport((), ())
    requested = {item.strip() for item in raw.split(",") if item.strip()}
    if requested & {"1", "true", "yes", "all"}:
        requested = {"ollama", "lmstudio"}

    timeout = float(os.environ.get("ENTROLY_MODEL_DISCOVERY_TIMEOUT", "0.2"))
    max_models = int(os.environ.get("ENTROLY_MODEL_DISCOVERY_MAX", "64"))
    models: list[ModelCapability] = []
    warnings: list[str] = []

    if "ollama" in requested:
        report = discover_ollama_models(
            os.environ.get("ENTROLY_OLLAMA_BASE", "http://127.0.0.1:11434"),
            timeout=timeout,
            max_models=max_models,
            inspect_context=os.environ.get("ENTROLY_OLLAMA_INSPECT_CONTEXT", "").lower()
            in {"1", "true", "yes"},
        )
        models.extend(report.models)
        warnings.extend(report.warnings)
    if "lmstudio" in requested:
        report = discover_openai_compatible_models(
            os.environ.get("ENTROLY_LMSTUDIO_BASE", "http://127.0.0.1:1234"),
            timeout=timeout,
            max_models=max_models,
        )
        models.extend(report.models)
        warnings.extend(report.warnings)

    unknown = requested - {"ollama", "lmstudio"}
    if unknown:
        warnings.append(f"Unknown local model discovery providers: {', '.join(sorted(unknown))}")
    return DiscoveryReport(tuple(models), tuple(warnings))


def discover_remote_models() -> DiscoveryReport:
    raw = os.environ.get("ENTROLY_DISCOVER_REMOTE_MODELS", "").strip().lower()
    if not raw:
        return DiscoveryReport((), ())
    requested = {item.strip() for item in raw.split(",") if item.strip()}
    if requested & {"1", "true", "yes", "all"}:
        requested = {"openrouter"}

    timeout = float(os.environ.get("ENTROLY_REMOTE_MODEL_DISCOVERY_TIMEOUT", "3.0"))
    max_models = int(os.environ.get("ENTROLY_MODEL_DISCOVERY_MAX", "512"))
    models: list[ModelCapability] = []
    warnings: list[str] = []
    if "openrouter" in requested:
        report = discover_openrouter_models(timeout=timeout, max_models=max_models)
        models.extend(report.models)
        warnings.extend(report.warnings)

    unknown = requested - {"openrouter"}
    if unknown:
        warnings.append(f"Unknown remote model discovery providers: {', '.join(sorted(unknown))}")
    return DiscoveryReport(tuple(models), tuple(warnings))


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    fallback = int(os.environ.get("ENTROLY_UNKNOWN_MODEL_CONTEXT", "128000"))
    bundled, digest = _bundled_models()
    local_discovery = discover_local_models()
    remote_discovery = discover_remote_models()
    return ModelRegistry(
        bundled,
        overrides=_override_models(),
        discovered=(*local_discovery.models, *remote_discovery.models),
        fallback_context_window=fallback,
        registry_digest=digest,
        discovery_warnings=(*local_discovery.warnings, *remote_discovery.warnings),
    )


def resolve_model(model: str) -> ModelResolution:
    return get_model_registry().resolve(model)
