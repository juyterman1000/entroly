from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from dataclasses import dataclass
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
    supports_tools: bool
    supports_vision: bool
    supports_reasoning: bool
    reasoning_levels: tuple[str, ...]
    input_price_per_million: float | None
    output_price_per_million: float | None
    trust: RegistryTrust
    source: str
    verified_at: str | None

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
            supports_tools=bool(value.get("supports_tools", True)),
            supports_vision=bool(value.get("supports_vision", False)),
            supports_reasoning=bool(value.get("supports_reasoning", False)),
            reasoning_levels=tuple(str(item) for item in value.get("reasoning_levels", ())),
            input_price_per_million=_optional_float(value.get("input_price_per_million")),
            output_price_per_million=_optional_float(value.get("output_price_per_million")),
            trust=RegistryTrust(value.get("trust", default_trust.value)),
            source=str(value.get("source", "unknown")),
            verified_at=value.get("verified_at"),
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


@dataclass(frozen=True, slots=True)
class ModelResolution:
    requested_model: str
    capability: ModelCapability | None
    context_window: int
    exact: bool
    trust: RegistryTrust
    warning: str | None
    registry_digest: str

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
        self._registry_digest = registry_digest
        self._discovery_warnings = tuple(discovery_warnings)
        self._by_id: dict[str, ModelCapability] = {}
        self._aliases: dict[str, str] = {}
        self._aliases_by_id: dict[str, set[str]] = {}
        for collection in (bundled, discovered, overrides):
            for capability in collection:
                self._install(capability)

    @property
    def registry_digest(self) -> str:
        return self._registry_digest

    @property
    def discovery_warnings(self) -> tuple[str, ...]:
        return self._discovery_warnings

    def _install(self, capability: ModelCapability) -> None:
        # Replacing a record must remove aliases no longer present; otherwise an
        # older layer can remain reachable through a stale alias.
        for old_alias in self._aliases_by_id.get(capability.id, set()):
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
            "models": len(self._by_id),
            "trust_counts": counts,
            "discovery_warnings": list(self._discovery_warnings),
        }

    def resolve(self, model: str) -> ModelResolution:
        requested = _normalise_name(model)
        canonical = self._aliases.get(requested)
        exact = canonical is not None
        if canonical is None:
            matches = [alias for alias in self._aliases if requested.startswith(alias)]
            if matches:
                longest = max(len(alias) for alias in matches)
                candidate_ids = {self._aliases[alias] for alias in matches if len(alias) == longest}
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
            )
        return ModelResolution(
            requested_model=model,
            capability=capability,
            context_window=capability.context_window,
            exact=exact,
            trust=capability.trust,
            warning=None,
            registry_digest=self._registry_digest,
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
        )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise HTTPError(req.full_url, code, "redirects disabled", headers, fp)


def _normalise_name(value: object) -> str:
    return str(value or "").strip().lower()


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    if result < 0:
        raise ValueError("price metadata cannot be negative")
    return result


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
    max_bytes: int = 2 * 1024 * 1024,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    opener = build_opener(_NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"local model discovery response exceeded {max_bytes} bytes")
    return json.loads(raw.decode("utf-8"))


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
                supports_tools=False,
                supports_vision=False,
                supports_reasoning=False,
                reasoning_levels=(),
                input_price_per_million=0.0,
                output_price_per_million=0.0,
                trust=RegistryTrust.DISCOVERED,
                source=f"{base}/api/tags",
                verified_at=date.today().isoformat(),
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
                supports_tools=True,
                supports_vision=False,
                supports_reasoning=False,
                reasoning_levels=(),
                input_price_per_million=0.0,
                output_price_per_million=0.0,
                trust=RegistryTrust.DISCOVERED,
                source=f"{base}/v1/models",
                verified_at=date.today().isoformat(),
            )
        )
    return DiscoveryReport(tuple(models), ())


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


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    fallback = int(os.environ.get("ENTROLY_UNKNOWN_MODEL_CONTEXT", "128000"))
    bundled, digest = _bundled_models()
    discovery = discover_local_models()
    return ModelRegistry(
        bundled,
        overrides=_override_models(),
        discovered=discovery.models,
        fallback_context_window=fallback,
        registry_digest=digest,
        discovery_warnings=discovery.warnings,
    )


def resolve_model(model: str) -> ModelResolution:
    return get_model_registry().resolve(model)
