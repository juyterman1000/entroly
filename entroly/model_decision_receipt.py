"""Deterministic, privacy-safe receipts for model resolution and budgeting.

A model decision receipt proves which registry snapshot and capability record
Entroly used to budget a provider request.  It deliberately excludes raw
registry sources, aliases, prompts, credentials, and private override content.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .models.registry import ModelResolution, resolve_model

MODEL_DECISION_SCHEMA = "entroly.model-decision.v1"
_SAFE_IDENTIFIER = re.compile(r"[^A-Za-z0-9._:/@+-]+")


def _safe_identifier(value: object, *, limit: int = 160) -> str:
    """Return a bounded header-safe identifier without control characters."""
    normalized = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    normalized = _SAFE_IDENTIFIER.sub("_", normalized)
    return normalized[:limit]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _extract_model(body: Mapping[str, Any], path: str) -> str:
    direct = body.get("model")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    # Gemini encodes the model in paths such as
    # /v1beta/models/gemini-2.5-pro:generateContent.
    marker = "/models/"
    if marker in path:
        tail = path.split(marker, 1)[1]
        candidate = tail.split(":", 1)[0].split("?", 1)[0].strip("/")
        if candidate:
            return candidate
    return ""


def _non_negative_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _requested_output_tokens(body: Mapping[str, Any]) -> int | None:
    for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        parsed = _non_negative_int(body.get(key))
        if parsed is not None:
            return parsed

    generation = body.get("generationConfig")
    if isinstance(generation, Mapping):
        parsed = _non_negative_int(generation.get("maxOutputTokens"))
        if parsed is not None:
            return parsed
    return None


def _warning_code(resolution: ModelResolution) -> str:
    if resolution.capability is None:
        return "unknown_model"
    if resolution.capability.context_window is None:
        return "unverified_context_window"
    if resolution.warning:
        return "registry_warning"
    return "none"


@dataclass(frozen=True, slots=True)
class ModelDecisionReceipt:
    """Canonical proof of the model metadata used for a request budget."""

    schema: str
    requested_model: str
    resolved_model: str
    provider: str
    trust: str
    exact: bool
    fallback_used: bool
    context_window: int
    safe_input_budget: int
    requested_output_tokens: int | None
    warning_code: str
    registry_digest: str
    base_registry_digest: str
    receipt_digest: str

    def payload(self) -> dict[str, Any]:
        """Return the canonical payload covered by ``receipt_digest``."""
        return {
            "schema": self.schema,
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "provider": self.provider,
            "trust": self.trust,
            "exact": self.exact,
            "fallback_used": self.fallback_used,
            "context_window": self.context_window,
            "safe_input_budget": self.safe_input_budget,
            "requested_output_tokens": self.requested_output_tokens,
            "warning_code": self.warning_code,
            "registry_digest": self.registry_digest,
            "base_registry_digest": self.base_registry_digest,
        }

    def verify(self) -> bool:
        expected = hashlib.sha256(_canonical_json(self.payload())).hexdigest()
        return expected == self.receipt_digest

    def to_tags(self) -> dict[str, str]:
        """Return bounded control-plane tags converted to response headers later."""
        return {
            "entroly_model_receipt_schema": self.schema,
            "entroly_model_receipt": self.receipt_digest,
            "entroly_model_requested": _safe_identifier(self.requested_model),
            "entroly_model_resolved": _safe_identifier(self.resolved_model),
            "entroly_model_provider": _safe_identifier(self.provider),
            "entroly_model_trust": self.trust,
            "entroly_model_exact": "true" if self.exact else "false",
            "entroly_model_fallback": "true" if self.fallback_used else "false",
            "entroly_model_context_window": str(self.context_window),
            "entroly_model_input_budget": str(self.safe_input_budget),
            "entroly_model_output_reserve": str(self.requested_output_tokens or 0),
            "entroly_model_warning": self.warning_code,
            "entroly_registry_digest": self.registry_digest,
            "entroly_registry_base_digest": self.base_registry_digest,
        }


def build_model_decision_receipt(
    body: Mapping[str, Any] | None,
    *,
    provider: str = "unknown",
    path: str = "",
    safety_fraction: float = 0.05,
) -> ModelDecisionReceipt | None:
    """Resolve a request model and produce a reproducible decision receipt.

    ``None`` means the request has no model-bearing surface.  Resolution errors
    are handled by :func:`model_decision_tags` so proxy traffic remains fail-open.
    """
    payload = body or {}
    requested_model = _extract_model(payload, path)
    if not requested_model:
        return None

    requested_output = _requested_output_tokens(payload)
    resolution = resolve_model(requested_model)
    safe_input_budget = resolution.effective_input_budget(
        requested_output_tokens=requested_output,
        safety_fraction=safety_fraction,
    )
    fallback_used = (
        resolution.capability is None
        or resolution.capability.context_window is None
    )
    unsigned = {
        "schema": MODEL_DECISION_SCHEMA,
        "requested_model": requested_model,
        "resolved_model": resolution.model_id,
        "provider": provider,
        "trust": resolution.trust.value,
        "exact": resolution.exact,
        "fallback_used": fallback_used,
        "context_window": resolution.context_window,
        "safe_input_budget": safe_input_budget,
        "requested_output_tokens": requested_output,
        "warning_code": _warning_code(resolution),
        "registry_digest": resolution.registry_digest,
        "base_registry_digest": resolution.base_registry_digest,
    }
    digest = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    return ModelDecisionReceipt(receipt_digest=digest, **unsigned)


def model_decision_tags(
    body: Mapping[str, Any] | None,
    *,
    provider: str = "unknown",
    path: str = "",
) -> dict[str, str]:
    """Return fail-open model receipt tags for the proxy control plane."""
    try:
        receipt = build_model_decision_receipt(
            body,
            provider=provider,
            path=path,
        )
    except Exception:
        return {"entroly_model_receipt_status": "unavailable"}
    if receipt is None:
        return {"entroly_model_receipt_status": "not_applicable"}
    tags = receipt.to_tags()
    tags["entroly_model_receipt_status"] = "verified" if receipt.verify() else "invalid"
    return tags
