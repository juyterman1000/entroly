"""Deterministic, provenance-aware model intelligence for Entroly."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger("entroly.model_registry")

SCHEMA_VERSION = 1
DEFAULT_UNKNOWN_CONTEXT_WINDOW = 32_768
_POLICIES = frozenset({"warn", "error", "legacy", "silent"})
_TRUST_RANK = {
    "unknown": 0,
    "inferred": 10,
    "bundled": 20,
    "signed": 30,
    "discovered": 40,
    "user": 50,
}


class ModelRegistryError(ValueError):
    """Malformed, ambiguous, or untrusted model metadata."""


class UnknownModelError(LookupError):
    """Strict resolution could not establish a context window."""


def _norm(value: str) -> str:
    value = value.strip().lower()
    return value[7:] if value.startswith("models/") else value


def _positive_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ModelRegistryError(f"{field} must be an integer or null")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelRegistryError(f"{field} must be an integer or null") from exc
    if value <= 0:
        raise ModelRegistryError(f"{field} must be positive")
    return value


def _optional_bool(value: Any, field: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise ModelRegistryError(f"{field} must be true, false, or null")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    provider: str
    aliases: tuple[str, ...] = ()
    match_prefixes: tuple[str, ...] = ()
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_tools: bool | None = None
    supports_vision: bool | None = None
    supports_reasoning: bool | None = None
    reasoning_levels: tuple[str, ...] = ()
    pricing: Mapping[str, Any] | None = None
    source: str = "unspecified"
    verified_at: str | None = None
    confidence: float = 0.0
    status: str = "active"
    trust: str = "bundled"
    local: bool = False

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        default_source: str,
        trust: str,
    ) -> "ModelSpec":
        model_id = str(raw.get("id") or "").strip()
        provider = str(raw.get("provider") or "").strip().lower()
        if not model_id or not provider:
            raise ModelRegistryError("each model requires id and provider")
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError) as exc:
            raise ModelRegistryError(f"invalid confidence for {model_id!r}") from exc
        if not 0 <= confidence <= 1:
            raise ModelRegistryError(f"confidence for {model_id!r} must be 0..1")

        def strings(name: str) -> tuple[str, ...]:
            values = raw.get(name, ())
            if not isinstance(values, (list, tuple)):
                raise ModelRegistryError(f"{model_id}.{name} must be an array")
            return tuple(dict.fromkeys(str(v).strip() for v in values if str(v).strip()))

        pricing = raw.get("pricing")
        if pricing is not None and not isinstance(pricing, Mapping):
            raise ModelRegistryError(f"{model_id}.pricing must be an object or null")
        return cls(
            id=model_id,
            provider=provider,
            aliases=strings("aliases"),
            match_prefixes=strings("match_prefixes"),
            context_window=_positive_int(raw.get("context_window"), f"{model_id}.context_window"),
            max_output_tokens=_positive_int(raw.get("max_output_tokens"), f"{model_id}.max_output_tokens"),
            supports_tools=_optional_bool(raw.get("supports_tools"), f"{model_id}.supports_tools"),
            supports_vision=_optional_bool(raw.get("supports_vision"), f"{model_id}.supports_vision"),
            supports_reasoning=_optional_bool(
                raw.get("supports_reasoning"), f"{model_id}.supports_reasoning"
            ),
            reasoning_levels=tuple(v.lower() for v in strings("reasoning_levels")),
            pricing=dict(pricing) if pricing is not None else None,
            source=str(raw.get("source") or default_source),
            verified_at=str(raw["verified_at"]) if raw.get("verified_at") else None,
            confidence=confidence,
            status=str(raw.get("status") or "active").lower(),
            trust=trust,
            local=bool(raw.get("local", False)),
        )

    @property
    def key(self) -> str:
        return f"{self.provider}:{_norm(self.id)}"

    @property
    def verified(self) -> bool:
        return self.context_window is not None and self.confidence >= 0.9


@dataclass(frozen=True, slots=True)
class ModelResolution:
    requested: str
    provider: str | None
    spec: ModelSpec | None
    match_kind: str
    effective_context_window: int
    warnings: tuple[str, ...] = ()

    @property
    def known(self) -> bool:
        return self.spec is not None

    @property
    def verified(self) -> bool:
        return bool(self.spec and self.spec.verified)

    @property
    def canonical_id(self) -> str | None:
        return self.spec.id if self.spec else None

    @property
    def provenance(self) -> dict[str, Any]:
        if self.spec is None:
            return {"source": "unknown", "trust": "unknown", "confidence": 0.0}
        return {
            "source": self.spec.source,
            "trust": self.spec.trust,
            "confidence": self.spec.confidence,
            "verified_at": self.spec.verified_at,
        }

    def to_dict(self) -> dict[str, Any]:
        spec = self.spec
        return {
            "requested": self.requested,
            "provider": self.provider,
            "canonical_id": self.canonical_id,
            "match_kind": self.match_kind,
            "known": self.known,
            "verified": self.verified,
            "context_window": self.effective_context_window,
            "max_output_tokens": spec.max_output_tokens if spec else None,
            "supports_tools": spec.supports_tools if spec else None,
            "supports_vision": spec.supports_vision if spec else None,
            "supports_reasoning": spec.supports_reasoning if spec else None,
            "reasoning_levels": list(spec.reasoning_levels) if spec else [],
            "status": spec.status if spec else "unknown",
            "warnings": list(self.warnings),
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class RegistryMetadata:
    registry_version: str
    source: str
    trust: str
    digest: str
    generated_at: str | None = None


class ModelRegistry:
    """Immutable catalog with exact > alias > longest-prefix resolution."""

    def __init__(self, models: Iterable[ModelSpec], *, metadata: RegistryMetadata):
        selected: dict[str, ModelSpec] = {}
        for spec in models:
            previous = selected.get(spec.key)
            if previous is None or self._rank(spec) >= self._rank(previous):
                selected[spec.key] = spec
        self._models = tuple(
            sorted(selected.values(), key=lambda s: (s.provider, _norm(s.id)))
        )
        self.metadata = metadata
        self._exact: dict[str, list[ModelSpec]] = {}
        self._aliases: dict[str, list[ModelSpec]] = {}
        self._prefixes: list[tuple[str, ModelSpec]] = []
        self._index()

    @staticmethod
    def _rank(spec: ModelSpec) -> tuple[int, float, str]:
        return (_TRUST_RANK.get(spec.trust, 0), spec.confidence, spec.source)

    @staticmethod
    def _append(index: dict[str, list[ModelSpec]], key: str, spec: ModelSpec) -> None:
        index.setdefault(_norm(key), []).append(spec)

    def _index(self) -> None:
        alias_owners: dict[tuple[str, str], str] = {}
        for spec in self._models:
            for key in (spec.id, f"{spec.provider}/{spec.id}", f"{spec.provider}:{spec.id}"):
                self._append(self._exact, key, spec)
            for alias in spec.aliases:
                owner_key = (spec.provider, _norm(alias))
                owner = alias_owners.get(owner_key)
                if owner and owner != spec.key:
                    raise ModelRegistryError(
                        f"duplicate alias {alias!r} for provider {spec.provider!r}"
                    )
                alias_owners[owner_key] = spec.key
                for key in (alias, f"{spec.provider}/{alias}", f"{spec.provider}:{alias}"):
                    self._append(self._aliases, key, spec)
            self._prefixes.extend(
                (_norm(prefix), spec) for prefix in spec.match_prefixes if _norm(prefix)
            )
        for values in (*self._exact.values(), *self._aliases.values()):
            values.sort(key=self._rank, reverse=True)
        self._prefixes.sort(
            key=lambda pair: (len(pair[0]), self._rank(pair[1])), reverse=True
        )

    @property
    def models(self) -> tuple[ModelSpec, ...]:
        return self._models

    @property
    def fingerprint(self) -> str:
        rows = [
            {
                "provider": s.provider,
                "id": s.id,
                "aliases": s.aliases,
                "prefixes": s.match_prefixes,
                "context": s.context_window,
                "output": s.max_output_tokens,
                "tools": s.supports_tools,
                "vision": s.supports_vision,
                "reasoning": s.supports_reasoning,
                "levels": s.reasoning_levels,
                "source": s.source,
                "verified_at": s.verified_at,
                "confidence": s.confidence,
                "status": s.status,
                "trust": s.trust,
                "local": s.local,
            }
            for s in self._models
        ]
        encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def overlay(
        self,
        models: Iterable[ModelSpec],
        *,
        source: str,
        trust: str,
    ) -> "ModelRegistry":
        additions = [replace(s, source=s.source or source, trust=trust) for s in models]
        metadata = replace(self.metadata, source=f"{self.metadata.source}+{source}", trust=trust)
        return ModelRegistry((*self._models, *additions), metadata=metadata)

    def _pick(self, values: Iterable[ModelSpec], provider: str | None) -> ModelSpec | None:
        matches = [s for s in values if provider is None or s.provider == provider]
        return max(matches, key=self._rank) if matches else None

    def resolve(
        self,
        model: str,
        *,
        provider: str | None = None,
        unknown_policy: str | None = None,
        fallback_context_window: int | None = None,
    ) -> ModelResolution:
        requested = model.strip()
        key = _norm(requested)
        provider = provider.lower().strip() if provider else None
        if not key:
            return self._unknown(requested, provider, unknown_policy, fallback_context_window, "empty model id")

        spec = self._pick(self._exact.get(key, ()), provider)
        kind = "exact"
        if spec is None:
            spec = self._pick(self._aliases.get(key, ()), provider)
            kind = "alias"
        if spec is None:
            for prefix, candidate in self._prefixes:
                if key.startswith(prefix) and (provider is None or candidate.provider == provider):
                    spec, kind = candidate, "prefix"
                    break
        if spec is None:
            return self._unknown(
                requested, provider, unknown_policy, fallback_context_window,
                "model is not present in the active registry",
            )
        if spec.context_window is None:
            return self._unknown(
                requested, provider or spec.provider, unknown_policy, fallback_context_window,
                f"registry entry {spec.id!r} has no verified context window",
                spec=spec, kind=kind,
            )
        warnings = ()
        if spec.confidence < 0.9:
            warnings = (
                f"model metadata for {requested!r} is not fully verified "
                f"(confidence={spec.confidence:.2f}, source={spec.source})",
            )
        return ModelResolution(
            requested, provider or spec.provider, spec, kind, spec.context_window, warnings
        )

    def _unknown(
        self,
        requested: str,
        provider: str | None,
        policy: str | None,
        fallback: int | None,
        reason: str,
        *,
        spec: ModelSpec | None = None,
        kind: str = "unknown",
    ) -> ModelResolution:
        policy = (policy or os.environ.get("ENTROLY_UNKNOWN_MODEL_POLICY", "warn")).lower()
        if policy not in _POLICIES:
            raise ModelRegistryError(
                "ENTROLY_UNKNOWN_MODEL_POLICY must be warn, error, legacy, or silent"
            )
        if policy == "error":
            raise UnknownModelError(
                f"cannot budget model {requested!r}: {reason}; add a registry override"
            )
        if fallback is None:
            raw = os.environ.get("ENTROLY_UNKNOWN_CONTEXT_WINDOW", "").strip()
            fallback = _positive_int(raw, "ENTROLY_UNKNOWN_CONTEXT_WINDOW") if raw else None
        fallback = fallback or (128_000 if policy == "legacy" else DEFAULT_UNKNOWN_CONTEXT_WINDOW)
        warning = (
            f"unverified model budget for {requested!r}: {reason}; "
            f"using conservative {fallback:,}-token window"
        )
        if policy == "warn":
            logger.warning(warning)
        warnings = () if policy == "silent" else (warning,)
        return ModelResolution(requested, provider, spec, kind, fallback, warnings)

    def pricing_catalog_payload(self) -> dict[str, Any]:
        rows = {
            f"{s.provider}:{s.id}": dict(s.pricing)
            for s in self._models
            if s.pricing
            and all(
                key in s.pricing
                for key in (
                    "input_per_million",
                    "output_per_million",
                    "cache_read_per_million",
                )
            )
        }
        return {"source": f"model-registry:{self.fingerprint}", "models": rows}


def _canonical(payload: Mapping[str, Any]) -> bytes:
    unsigned = {k: v for k, v in payload.items() if k != "integrity"}
    return json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _verify_integrity(
    payload: Mapping[str, Any], *, require_signature: bool
) -> tuple[str, str]:
    canonical = _canonical(payload)
    digest = hashlib.sha256(canonical).hexdigest()
    integrity = payload.get("integrity")
    if integrity is None:
        if require_signature:
            raise ModelRegistryError("registry signature is required")
        return digest, "unsigned"
    if not isinstance(integrity, Mapping):
        raise ModelRegistryError("integrity must be an object")
    if str(integrity.get("payload_sha256") or "").lower() != digest:
        raise ModelRegistryError("registry payload_sha256 does not match content")
    signature = integrity.get("signature")
    if signature is None:
        if require_signature:
            raise ModelRegistryError("registry signature is required")
        return digest, "digest"
    if not isinstance(signature, Mapping) or signature.get("algorithm") != "ed25519":
        raise ModelRegistryError("only ed25519 registry signatures are supported")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        public_key = base64.b64decode(str(signature["public_key_base64"]), validate=True)
        signed = base64.b64decode(str(signature["signature_base64"]), validate=True)
        Ed25519PublicKey.from_public_bytes(public_key).verify(signed, canonical)
    except ImportError as exc:
        raise ModelRegistryError(
            "signed registry requires the optional cryptography dependency"
        ) from exc
    except Exception as exc:
        raise ModelRegistryError("registry ed25519 signature verification failed") from exc
    return digest, "signed"


def registry_from_payload(
    payload: Mapping[str, Any],
    *,
    default_source: str,
    trust: str,
    require_signature: bool = False,
) -> ModelRegistry:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ModelRegistryError(
            f"unsupported model registry schema {payload.get('schema_version')!r}"
        )
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ModelRegistryError("registry requires a non-empty models array")
    if not all(isinstance(item, Mapping) for item in raw_models):
        raise ModelRegistryError("every models entry must be an object")
    digest, integrity_trust = _verify_integrity(
        payload, require_signature=require_signature
    )
    trust = "signed" if integrity_trust == "signed" else trust
    source = str(payload.get("source") or default_source)
    models = [
        ModelSpec.from_mapping(item, default_source=source, trust=trust)
        for item in raw_models
    ]
    return ModelRegistry(
        models,
        metadata=RegistryMetadata(
            registry_version=str(payload.get("registry_version") or "unknown"),
            source=source,
            trust=trust,
            digest=digest,
            generated_at=str(payload["generated_at"]) if payload.get("generated_at") else None,
        ),
    )


def load_registry_file(
    path: str | Path,
    *,
    trust: str = "user",
    require_signature: bool = False,
) -> ModelRegistry:
    path = Path(path).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRegistryError(f"cannot load model registry {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ModelRegistryError("registry root must be an object")
    return registry_from_payload(
        payload,
        default_source=f"file:{path}",
        trust=trust,
        require_signature=require_signature,
    )


def _bundled_registry() -> ModelRegistry:
    resource = resources.files("entroly").joinpath("models", "registry.json")
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRegistryError(f"cannot load bundled model registry: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ModelRegistryError("bundled registry root must be an object")
    return registry_from_payload(
        payload,
        default_source="package:entroly/models/registry.json",
        trust="bundled",
    )


def _overlay_paths() -> tuple[str, ...]:
    raw = os.environ.get("ENTROLY_MODEL_REGISTRY", "").strip()
    return tuple(path for path in raw.split(os.pathsep) if path)


@lru_cache(maxsize=16)
def _cached_registry(
    paths: tuple[str, ...],
    require_signed: bool,
    discover_ollama: bool,
    ollama_base: str,
) -> ModelRegistry:
    registry = _bundled_registry()
    for path in paths:
        overlay = load_registry_file(
            path, trust="user", require_signature=require_signed
        )
        registry = registry.overlay(
            overlay.models, source=overlay.metadata.source, trust=overlay.metadata.trust
        )
    if discover_ollama:
        from .model_discovery import discover_ollama_models
        discovered = discover_ollama_models(base_url=ollama_base or None)
        if discovered:
            registry = registry.overlay(
                discovered, source=f"ollama:{ollama_base or 'default'}", trust="discovered"
            )
    return registry


def default_registry() -> ModelRegistry:
    require_signed = os.environ.get(
        "ENTROLY_REQUIRE_SIGNED_MODEL_REGISTRY", "0"
    ).lower() in {"1", "true", "yes", "on"}
    discover = os.environ.get("ENTROLY_OLLAMA_DISCOVERY", "0").lower() in {
        "1", "true", "yes", "on"
    }
    return _cached_registry(
        _overlay_paths(),
        require_signed,
        discover,
        os.environ.get("ENTROLY_OLLAMA_BASE", "").strip(),
    )


def reset_registry_cache() -> None:
    _cached_registry.cache_clear()


def resolve_model(
    model: str,
    *,
    provider: str | None = None,
    unknown_policy: str | None = None,
    fallback_context_window: int | None = None,
) -> ModelResolution:
    return default_registry().resolve(
        model,
        provider=provider,
        unknown_policy=unknown_policy,
        fallback_context_window=fallback_context_window,
    )


def context_window_for_model(
    model: str,
    *,
    provider: str | None = None,
    unknown_policy: str | None = None,
) -> int:
    return resolve_model(
        model, provider=provider, unknown_policy=unknown_policy
    ).effective_context_window


def bundled_context_windows() -> dict[str, int]:
    return {
        spec.id: spec.context_window
        for spec in default_registry().models
        if spec.context_window is not None
    }
