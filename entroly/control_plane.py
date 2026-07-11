"""Entroly context control plane.

This module is intentionally read-only: it plans and audits request handling
without mutating provider payloads. The goal is to make compression decisions
explainable and to prove that request transforms did not rewrite provider-owned
controls such as temperature, thinking, generationConfig, model, or tool
contracts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .model_decision_receipt import model_decision_tags

Provider = Literal["openai", "anthropic", "gemini", "unknown"]
Action = Literal["compress", "observe", "passthrough"]

_MISSING = object()


_PROVIDER_CONTROL_PATHS: dict[Provider, tuple[tuple[str, ...], ...]] = {
    "openai": (
        ("model",),
        ("temperature",),
        ("top_p",),
        ("max_tokens",),
        ("max_completion_tokens",),
        ("max_output_tokens",),
        ("stop",),
        ("seed",),
        ("presence_penalty",),
        ("frequency_penalty",),
        ("logit_bias",),
        ("response_format",),
        ("reasoning",),
        ("stream",),
    ),
    "anthropic": (
        ("model",),
        ("temperature",),
        ("top_p",),
        ("top_k",),
        ("max_tokens",),
        ("stop_sequences",),
        ("thinking",),
        ("stream",),
    ),
    "gemini": (
        ("model",),
        ("generationConfig",),
        ("safetySettings",),
        ("stream",),
    ),
    "unknown": (
        ("model",),
        ("temperature",),
        ("top_p",),
        ("max_tokens",),
        ("max_output_tokens",),
        ("stream",),
    ),
}


_TOOL_CONTRACT_PATHS: dict[Provider, tuple[tuple[str, ...], ...]] = {
    "openai": (("tools",), ("tool_choice",), ("parallel_tool_calls",)),
    "anthropic": (("tools",), ("tool_choice",)),
    "gemini": (("tools",), ("toolConfig",)),
    "unknown": (("tools",), ("tool_choice",), ("toolConfig",)),
}


@dataclass(frozen=True)
class ContextSurface:
    """A token-bearing or modality-bearing part of a provider request."""

    source: str
    role: str
    kind: str
    modality: str
    estimated_tokens: int
    digest: str


@dataclass(frozen=True)
class ControlPlaneDecision:
    """Read-only plan for how Entroly should treat a request."""

    provider: Provider
    action: Action
    reasons: tuple[str, ...]
    surfaces: tuple[ContextSurface, ...]
    estimated_text_tokens: int
    token_budget: int | None
    budget_pressure: float | None
    cache_policy: str
    invariants: tuple[str, ...]

    @property
    def should_compress(self) -> bool:
        return self.action == "compress"

    def to_tags(self) -> dict[str, str]:
        tags = {
            "entroly_provider": self.provider,
            "entroly_action": self.action,
            "entroly_reasons": ",".join(self.reasons),
            "entroly_text_tokens": str(self.estimated_text_tokens),
            "entroly_surface_count": str(len(self.surfaces)),
            "entroly_cache_policy": self.cache_policy,
        }
        if self.budget_pressure is not None:
            tags["entroly_budget_pressure"] = f"{self.budget_pressure:.3f}"
        return tags


# Public product name for the same read-only decision object. Keep the
# implementation name stable for existing imports while exposing the clearer
# compression-control vocabulary at the SDK/proxy boundary.
EntrolyCompressionDecision = ControlPlaneDecision


@dataclass(frozen=True)
class ControlViolation:
    """A compliance-sensitive transform difference without raw payload values."""

    path: str
    kind: Literal["added", "removed", "changed"]
    before_type: str
    after_type: str
    before_fingerprint: str
    after_fingerprint: str
    message: str


@dataclass(frozen=True)
class ControlAudit:
    """Result of auditing a provider payload before/after transformation."""

    provider: Provider
    provider_control_violations: tuple[ControlViolation, ...]
    tool_contract_violations: tuple[ControlViolation, ...]

    @property
    def compliant(self) -> bool:
        return not self.provider_control_violations and not self.tool_contract_violations

    def to_tags(self) -> dict[str, str]:
        return {
            "entroly_provider": self.provider,
            "entroly_transform_compliant": "true" if self.compliant else "false",
            "entroly_provider_control_violations": str(len(self.provider_control_violations)),
            "entroly_tool_contract_violations": str(len(self.tool_contract_violations)),
        }


def plan_request(
    body: Mapping[str, Any] | None,
    *,
    headers: Mapping[str, str] | None = None,
    provider: Provider | None = None,
    path: str = "",
    token_budget: int | None = None,
    compression_enabled: bool = True,
    cache_alignment_enabled: bool = True,
) -> ControlPlaneDecision:
    """Return a deterministic, read-only request plan.

    The planner never edits ``body``. It only computes reason codes and a
    coarse surface inventory that downstream proxy, SDK, or dashboard code can
    use for explainability.
    """

    normalized_headers = _normalize_headers(headers or {})
    payload: Mapping[str, Any] = body or {}
    detected_provider = provider or infer_provider(path=path, headers=normalized_headers, body=payload)
    surfaces = extract_context_surfaces(payload, provider=detected_provider)
    estimated_tokens = sum(s.estimated_tokens for s in surfaces)

    reasons: list[str] = []
    if _bypass_enabled(normalized_headers):
        reasons.append("bypass_header")
    if not compression_enabled:
        reasons.append("compression_disabled")
    if not surfaces:
        reasons.append("no_context_surface")
    if any(s.kind == "tool_output" for s in surfaces):
        reasons.append("tool_output_present")
    if any(s.modality == "image" for s in surfaces):
        reasons.append("image_present")
    if _extract_values(payload, _PROVIDER_CONTROL_PATHS[detected_provider]):
        reasons.append("provider_controls_present")

    pressure: float | None = None
    if token_budget is not None and token_budget > 0:
        pressure = estimated_tokens / token_budget
        if estimated_tokens <= token_budget:
            reasons.append("under_budget")
        else:
            reasons.append("budget_pressure")

    if "bypass_header" in reasons or "compression_disabled" in reasons or not surfaces:
        action: Action = "passthrough"
    elif token_budget is not None and estimated_tokens <= token_budget:
        action = "observe"
    else:
        action = "compress"

    if action == "passthrough":
        cache_policy = "preserve_bytes"
    elif cache_alignment_enabled:
        cache_policy = "align_context_prefix"
        reasons.append("cache_alignment_candidate")
    else:
        cache_policy = "none"

    return ControlPlaneDecision(
        provider=detected_provider,
        action=action,
        reasons=tuple(dict.fromkeys(reasons)),
        surfaces=surfaces,
        estimated_text_tokens=estimated_tokens,
        token_budget=token_budget,
        budget_pressure=pressure,
        cache_policy=cache_policy,
        invariants=(
            "planner_read_only",
            "provider_controls_read_only",
            "tool_contract_read_only_unless_explicit",
        ),
    )


def audit_request_transform(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
    *,
    provider: Provider | None = None,
    path: str = "",
    headers: Mapping[str, str] | None = None,
    allow_tool_contract_changes: bool = False,
    allow_model_change: bool = False,
) -> ControlAudit:
    """Audit that a transform preserved provider controls and tool contracts."""

    normalized_headers = _normalize_headers(headers or {})
    left: Mapping[str, Any] = before or {}
    right: Mapping[str, Any] = after or {}
    detected_provider = provider or infer_provider(path=path, headers=normalized_headers, body=left)

    provider_controls_before = _extract_values(left, _PROVIDER_CONTROL_PATHS[detected_provider])
    provider_controls_after = _extract_values(right, _PROVIDER_CONTROL_PATHS[detected_provider])
    provider_violations = _diff_values(
        provider_controls_before,
        provider_controls_after,
        label="provider control",
    )
    if allow_model_change:
        provider_violations = tuple(
            violation
            for violation in provider_violations
            if violation.path != "model"
        )

    tool_violations: tuple[ControlViolation, ...] = ()
    if not allow_tool_contract_changes:
        tool_before = _extract_values(left, _TOOL_CONTRACT_PATHS[detected_provider])
        tool_after = _extract_values(right, _TOOL_CONTRACT_PATHS[detected_provider])
        tool_violations = _diff_values(tool_before, tool_after, label="tool contract")

    return ControlAudit(
        provider=detected_provider,
        provider_control_violations=provider_violations,
        tool_contract_violations=tool_violations,
    )


def canonical_json_dumps(value: Any) -> str:
    """Serialize provider-adjacent data in a cache-stable JSON form.

    Object keys are sorted and insignificant whitespace is removed. Array order
    is preserved because provider tool order and message order can be semantic.
    """

    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_request_fingerprint(
    body: Mapping[str, Any] | None,
    *,
    headers: Mapping[str, str] | None = None,
    provider: Provider | None = None,
    path: str = "",
) -> dict[str, str]:
    """Return privacy-safe fingerprints for sticky cache/protocol contracts.

    The fingerprints deliberately avoid raw payload values. They are stable for
    semantically identical dict key ordering while still changing when sticky
    provider headers or tool contracts change.
    """

    normalized_headers = _normalize_headers(headers or {})
    payload = body or {}
    detected_provider = provider or infer_provider(path=path, headers=normalized_headers, body=payload)
    sticky_headers = {
        k: v
        for k, v in normalized_headers.items()
        if k
        in {
            "anthropic-beta",
            "openai-beta",
            "x-goog-api-client",
            "x-goog-api-key",
            "x-api-key",
            "authorization",
        }
    }
    tool_contract = _extract_values(payload, _TOOL_CONTRACT_PATHS[detected_provider])
    provider_controls = _extract_values(payload, _PROVIDER_CONTROL_PATHS[detected_provider])
    protocol = {
        "provider": detected_provider,
        "headers": sticky_headers,
        "provider_controls": provider_controls,
        "tool_contract": tool_contract,
    }
    tags = {
        "entroly_request_fingerprint": _fingerprint(payload),
        "entroly_protocol_fingerprint": _fingerprint(protocol),
        "entroly_tool_fingerprint": _fingerprint(tool_contract),
        "entroly_header_fingerprint": _fingerprint(sticky_headers),
    }
    tags.update(
        model_decision_tags(
            payload,
            provider=detected_provider,
            path=path,
        )
    )
    return tags


def infer_provider(
    *,
    path: str = "",
    headers: Mapping[str, str] | None = None,
    body: Mapping[str, Any] | None = None,
) -> Provider:
    """Infer provider shape without importing proxy code."""

    normalized_headers = _normalize_headers(headers or {})
    payload = body or {}
    if "/v1/messages" in path:
        return "anthropic"
    if "generateContent" in path or "streamGenerateContent" in path:
        return "gemini"
    if "x-goog-api-key" in normalized_headers:
        return "gemini"
    if "x-api-key" in normalized_headers and "authorization" not in normalized_headers:
        return "anthropic"
    if "contents" in payload and "messages" not in payload:
        return "gemini"
    if "messages" in payload or "input" in payload:
        return "openai"
    return "unknown"


def extract_context_surfaces(
    body: Mapping[str, Any] | None,
    *,
    provider: Provider = "unknown",
) -> tuple[ContextSurface, ...]:
    """Extract read-only context surfaces from common provider request shapes."""

    payload = body or {}
    surfaces: list[ContextSurface] = []

    if isinstance(payload.get("messages"), Sequence) and not isinstance(payload.get("messages"), (str, bytes)):
        for i, msg in enumerate(payload.get("messages", [])):
            if not isinstance(msg, Mapping):
                continue
            role = str(msg.get("role", "unknown"))
            _collect_content_surfaces(msg.get("content"), f"messages[{i}].content", role, surfaces, provider=provider)

    if "input" in payload and "messages" not in payload:
        _collect_content_surfaces(payload.get("input"), "input", "user", surfaces, provider=provider)

    if provider == "gemini" or "contents" in payload:
        contents = payload.get("contents", [])
        if isinstance(contents, Sequence) and not isinstance(contents, (str, bytes)):
            for i, item in enumerate(contents):
                if not isinstance(item, Mapping):
                    continue
                role = str(item.get("role", "user"))
                parts = item.get("parts", [])
                _collect_content_surfaces(parts, f"contents[{i}].parts", role, surfaces, provider=provider)

    if isinstance(payload.get("system"), str):
        _add_text_surface(payload["system"], "system", "system", surfaces)

    return tuple(surfaces)


def _collect_content_surfaces(
    content: Any,
    source: str,
    role: str,
    surfaces: list[ContextSurface],
    *,
    forced_kind: str | None = None,
    provider: Provider = "unknown",
) -> None:
    if isinstance(content, str):
        _add_text_surface(content, source, role, surfaces, forced_kind=forced_kind)
        return
    if isinstance(content, Mapping):
        block_type = str(content.get("type", ""))
        if _is_image_block(content):
            _add_modal_surface(
                source,
                role,
                "image",
                surfaces,
                estimated_tokens=_estimate_image_block_tokens(content, provider),
            )
            return
        if block_type == "tool_result":
            _collect_content_surfaces(
                content.get("content", content.get("text", "")),
                source,
                role,
                surfaces,
                forced_kind="tool_output",
                provider=provider,
            )
            return
        if "text" in content:
            _add_text_surface(str(content.get("text", "")), source, role, surfaces, forced_kind=forced_kind)
            return
        if "content" in content:
            _collect_content_surfaces(content.get("content"), source, role, surfaces, forced_kind=forced_kind, provider=provider)
            return
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        for i, item in enumerate(content):
            _collect_content_surfaces(item, f"{source}[{i}]", role, surfaces, forced_kind=forced_kind, provider=provider)


def _add_text_surface(
    text: str,
    source: str,
    role: str,
    surfaces: list[ContextSurface],
    *,
    forced_kind: str | None = None,
) -> None:
    if not text:
        return
    kind = forced_kind or _kind_from_role(role)
    surfaces.append(
        ContextSurface(
            source=source,
            role=role,
            kind=kind,
            modality=_text_modality(text),
            estimated_tokens=max(1, (len(text) + 3) // 4),
            digest=_fingerprint(text),
        )
    )


def _add_modal_surface(
    source: str,
    role: str,
    modality: str,
    surfaces: list[ContextSurface],
    *,
    estimated_tokens: int = 0,
) -> None:
    surfaces.append(
        ContextSurface(
            source=source,
            role=role,
            kind=_kind_from_role(role),
            modality=modality,
            estimated_tokens=estimated_tokens,
            digest=_fingerprint({"source": source, "modality": modality}),
        )
    )


def _kind_from_role(role: str) -> str:
    if role == "tool":
        return "tool_output"
    if role == "system":
        return "system_prompt"
    if role == "assistant":
        return "assistant_context"
    return "user_context"


def _text_modality(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if stripped[0] in "[{":
        try:
            json.loads(stripped)
            return "json"
        except Exception:  # noqa: BLE001 - modality detection is best effort
            pass
    lower = stripped.lower()
    if "traceback " in lower or "error:" in lower or "fatal" in lower or "[error]" in lower:
        return "log"
    code_markers = ("def ", "class ", "function ", "import ", "from ", "const ", "let ", "fn ", "pub ")
    if sum(1 for marker in code_markers if marker in stripped) >= 2:
        return "code"
    return "text"


def _is_image_block(block: Mapping[str, Any]) -> bool:
    block_type = str(block.get("type", ""))
    return block_type in {"image", "image_url", "input_image"} or "inlineData" in block


def _estimate_image_block_tokens(block: Mapping[str, Any], provider: Provider) -> int:
    try:
        from .image_optimizer import decode_base64_image, estimate_image_tokens

        data: Any = None
        detail = str(block.get("detail", "high")).lower()
        if "inlineData" in block and isinstance(block["inlineData"], Mapping):
            data = block["inlineData"].get("data")
        elif isinstance(block.get("source"), Mapping):
            data = block["source"].get("data")
        elif isinstance(block.get("image_url"), Mapping):
            url = str(block["image_url"].get("url", ""))
            if url.startswith("data:"):
                data = url
        if not isinstance(data, str) or not data:
            return 0
        estimate = estimate_image_tokens(
            decode_base64_image(data),
            provider=provider,
            detail="low" if detail == "low" else "high",
        )
        return estimate.estimated_tokens
    except Exception:
        return 0


def _normalize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in headers.items()}


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonicalize(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, tuple):
        return [_canonicalize(v) for v in value]
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    return value


def _bypass_enabled(headers: Mapping[str, str]) -> bool:
    return (
        headers.get("x-entroly-bypass", "").strip().lower() == "true"
        or headers.get("x-entroly-mode", "").strip().lower() == "passthrough"
    )


def _extract_values(
    body: Mapping[str, Any],
    paths: Sequence[tuple[str, ...]],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for path in paths:
        value = _get_path(body, path)
        if value is not _MISSING:
            values[".".join(path)] = value
    return values


def _get_path(body: Mapping[str, Any], path: Sequence[str]) -> Any:
    cur: Any = body
    for part in path:
        if not isinstance(cur, Mapping) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _diff_values(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    label: str,
) -> tuple[ControlViolation, ...]:
    violations: list[ControlViolation] = []
    for path in sorted(set(before) | set(after)):
        left = before.get(path, _MISSING)
        right = after.get(path, _MISSING)
        if left is _MISSING and right is not _MISSING:
            kind: Literal["added", "removed", "changed"] = "added"
        elif left is not _MISSING and right is _MISSING:
            kind = "removed"
        elif left != right:
            kind = "changed"
        else:
            continue
        violations.append(
            ControlViolation(
                path=path,
                kind=kind,
                before_type=_type_name(left),
                after_type=_type_name(right),
                before_fingerprint=_fingerprint(left),
                after_fingerprint=_fingerprint(right),
                message=f"{label} {path} was {kind} by the transform",
            )
        )
    return tuple(violations)


def _type_name(value: Any) -> str:
    if value is _MISSING:
        return "missing"
    return type(value).__name__


def _fingerprint(value: Any) -> str:
    if value is _MISSING:
        return "missing"
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    except TypeError:
        raw = repr(value)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
