from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable


class RegistryTrust(StrEnum):
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
    def from_mapping(cls, value: dict[str, Any], *, default_trust: RegistryTrust) -> "ModelCapability":
        context = value.get("context_window")
        if context is not None and (not isinstance(context, int) or context <= 0):
            raise ValueError(f"invalid context_window for {value.get('id')!r}: {context!r}")
        output = value.get("max_output_tokens")
        if output is not None and (not isinstance(output, int) or output <= 0):
            raise ValueError(f"invalid max_output_tokens for {value.get('id')!r}: {output!r}")
        return cls(
            id=str(value["id"]).strip().lower(),
            provider=str(value["provider"]).strip().lower(),
            aliases=tuple(str(item).strip().lower() for item in value.get("aliases", ())),
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


@dataclass(frozen=True, slots=True)
class ModelResolution:
    requested_model: str
    capability: ModelCapability | None
    context_window: int
    exact: bool
    trust: RegistryTrust
    warning: str | None

    @property
    def model_id(self) -> str:
        return self.capability.id if self.capability else self.requested_model


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
    ) -> None:
        if fallback_context_window <= 0:
            raise ValueError("fallback_context_window must be positive")
        self._fallback_context_window = fallback_context_window
        self._by_id: dict[str, ModelCapability] = {}
        self._aliases: dict[str, str] = {}
        for collection in (bundled, discovered, overrides):
            for capability in collection:
                self._install(capability)

    def _install(self, capability: ModelCapability) -> None:
        self._by_id[capability.id] = capability
        for alias in (capability.id, *capability.aliases):
            self._aliases[alias.lower()] = capability.id

    def all(self) -> tuple[ModelCapability, ...]:
        return tuple(sorted(self._by_id.values(), key=lambda item: item.id))

    def resolve(self, model: str) -> ModelResolution:
        requested = model.strip().lower()
        canonical = self._aliases.get(requested)
        exact = canonical is not None
        if canonical is None:
            candidates = sorted(self._aliases, key=len, reverse=True)
            canonical = next((self._aliases[prefix] for prefix in candidates if requested.startswith(prefix)), None)
        if canonical is None:
            return ModelResolution(
                requested_model=model,
                capability=None,
                context_window=self._fallback_context_window,
                exact=False,
                trust=RegistryTrust.FALLBACK,
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
            )
        return ModelResolution(
            requested_model=model,
            capability=capability,
            context_window=capability.context_window,
            exact=exact,
            trust=capability.trust,
            warning=None,
        )


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
    return [ModelCapability.from_mapping(item, default_trust=default_trust) for item in records]


def _bundled_models() -> list[ModelCapability]:
    path = Path(str(files("entroly.models").joinpath("registry.json")))
    return _load_json(path, default_trust=RegistryTrust.VERIFIED)


def _override_models() -> list[ModelCapability]:
    raw = os.environ.get("ENTROLY_MODEL_REGISTRY", "").strip()
    if not raw:
        return []
    path = Path(raw).expanduser()
    return _load_json(path, default_trust=RegistryTrust.USER)


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    fallback = int(os.environ.get("ENTROLY_UNKNOWN_MODEL_CONTEXT", "128000"))
    return ModelRegistry(
        _bundled_models(),
        overrides=_override_models(),
        fallback_context_window=fallback,
    )


def resolve_model(model: str) -> ModelResolution:
    return get_model_registry().resolve(model)
