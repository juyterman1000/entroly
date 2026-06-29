"""Provider adapters for the production gateway control plane.

The cache-aware control plane is provider-neutral; the live proxy is not.  This
module is the narrow seam between both worlds.  It converts OpenAI, Anthropic,
and Gemini request bodies into one canonical shape, and can render a canonical
request back to a provider body when an enterprise gateway explicitly elects to
execute a capability-safe failover target.

The adapter is deliberately conservative:
- it preserves generation controls from the original body only for the same
  provider;
- cross-provider rendering only emits portable fields that have equivalent
  semantics;
- model identifiers embedded in Gemini URLs are validated before rewrite;
- opaque provider-specific fields are never silently translated.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .provider_policy import CanonicalGatewayRequest, ProviderTarget

_SAFE_MODEL_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


@dataclass(frozen=True, slots=True)
class ProviderRequestAdapterResult:
    """Canonical view plus token estimates needed by the router."""

    provider: str
    canonical: CanonicalGatewayRequest
    prefix_tokens_estimate: int
    new_input_tokens_estimate: int
    expected_output_tokens: int


def _text_from_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(part for part in (_text_from_content(v) for v in value) if part)
    if isinstance(value, Mapping):
        for key in ("text", "content", "input"):
            nested = value.get(key)
            if isinstance(nested, (str, list, Mapping)):
                text = _text_from_content(nested)
                if text:
                    return text
        parts = value.get("parts")
        if isinstance(parts, list):
            return _text_from_content(parts)
        return ""
    return str(value)


def _token_estimate(value: Any) -> int:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return max(1, len(raw.encode("utf-8")) // 4)


def _has_vision(value: Any) -> bool:
    if isinstance(value, Mapping):
        if any(key in value for key in ("image_url", "image", "inline_data", "file_data")):
            return True
        mime = value.get("mime_type") or value.get("mimeType")
        if isinstance(mime, str) and mime.startswith("image/"):
            return True
        return any(_has_vision(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_vision(child) for child in value)
    return False


def _portable_metadata(headers: Mapping[str, str] | None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not headers:
        return metadata
    for key, value in headers.items():
        lower = key.lower()
        if lower in {"x-request-id", "x-entroly-team", "x-entroly-project", "x-entroly-tool"}:
            metadata[lower] = str(value)[:256]
    return metadata


def _last_user_text(messages: tuple[Mapping[str, Any], ...]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _text_from_content(message.get("content"))
    return ""


def _normalize_openai_messages(body: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if isinstance(body.get("messages"), list):
        return tuple(dict(m) for m in body["messages"] if isinstance(m, Mapping))

    # Responses API: preserve a system-like instruction separately and convert
    # input items into user messages so conversation identity remains stable.
    messages: list[Mapping[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})
    raw_input = body.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if isinstance(item, Mapping):
                role = str(item.get("role") or "user")
                content = item.get("content", item.get("text", ""))
                messages.append({"role": role, "content": content})
            elif isinstance(item, str):
                messages.append({"role": "user", "content": item})
    return tuple(messages)


def _normalize_anthropic_messages(body: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    messages: list[Mapping[str, Any]] = []
    system = body.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})
    raw_messages = body.get("messages")
    if isinstance(raw_messages, list):
        messages.extend(dict(m) for m in raw_messages if isinstance(m, Mapping))
    return tuple(messages)


def _normalize_gemini_messages(body: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    messages: list[Mapping[str, Any]] = []
    system = body.get("systemInstruction")
    if isinstance(system, Mapping):
        messages.append({"role": "system", "content": _text_from_content(system)})
    elif isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})

    contents = body.get("contents")
    if isinstance(contents, list):
        for item in contents:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role") or "user")
            if role == "model":
                role = "assistant"
            messages.append({"role": role, "content": item.get("parts", item.get("content", ""))})
    return tuple(messages)


def canonical_request_from_provider_body(
    provider: str,
    body: Mapping[str, Any],
    *,
    headers: Mapping[str, str] | None = None,
    path: str = "",
) -> ProviderRequestAdapterResult:
    """Convert a provider-specific request body into canonical gateway form."""

    provider = provider.lower()
    if provider in {"openai", "responses"}:
        messages = _normalize_openai_messages(body)
        tools = tuple(dict(t) for t in body.get("tools", ()) if isinstance(t, Mapping)) if isinstance(body.get("tools"), list) else ()
        schema = None
        response_format = body.get("response_format")
        if isinstance(response_format, Mapping) and response_format.get("type") in {"json_schema", "json_object"}:
            schema = response_format
        model = str(body.get("model") or "")
        stream = bool(body.get("stream", False))
    elif provider == "anthropic":
        messages = _normalize_anthropic_messages(body)
        tools = tuple(dict(t) for t in body.get("tools", ()) if isinstance(t, Mapping)) if isinstance(body.get("tools"), list) else ()
        schema = body.get("response_schema") if isinstance(body.get("response_schema"), Mapping) else None
        model = str(body.get("model") or "")
        stream = bool(body.get("stream", False))
    elif provider == "gemini":
        messages = _normalize_gemini_messages(body)
        tools = tuple(dict(t) for t in body.get("tools", ()) if isinstance(t, Mapping)) if isinstance(body.get("tools"), list) else ()
        generation = body.get("generationConfig") if isinstance(body.get("generationConfig"), Mapping) else {}
        schema = generation.get("responseSchema") if isinstance(generation, Mapping) else None
        model = str(body.get("model") or _model_from_gemini_path(path) or "")
        stream = bool(body.get("stream", False) or "streamGenerateContent" in path)
    else:
        raise ValueError(f"unsupported provider: {provider}")

    if not model:
        raise ValueError("provider request does not identify a model")

    portable_input = {
        key: body[key]
        for key in ("system", "systemInstruction", "instructions", "messages", "contents", "input", "tools")
        if key in body
    }
    total_input = _token_estimate(portable_input)
    new_input = max(1, len(_last_user_text(messages).encode("utf-8")) // 4)
    prefix = max(0, total_input - new_input)

    expected_output: Any = body.get("max_completion_tokens")
    if expected_output is None:
        expected_output = body.get("max_tokens")
    generation = body.get("generationConfig")
    if expected_output is None and isinstance(generation, Mapping):
        expected_output = generation.get("maxOutputTokens")
    try:
        output_tokens = max(0, min(int(expected_output or 1024), 1_000_000))
    except (TypeError, ValueError):
        output_tokens = 1024

    canonical = CanonicalGatewayRequest(
        model=model,
        messages=messages,
        tools=tools,
        stream=stream,
        response_schema=schema if isinstance(schema, Mapping) else None,
        metadata=_portable_metadata(headers),
        requires_vision=_has_vision(messages),
        requires_reasoning=any(key in body for key in ("thinking", "reasoning_effort", "effort")),
    )
    return ProviderRequestAdapterResult(
        provider=provider,
        canonical=canonical,
        prefix_tokens_estimate=prefix,
        new_input_tokens_estimate=new_input,
        expected_output_tokens=output_tokens,
    )


def _model_from_gemini_path(path: str) -> str:
    match = re.search(r"/models/([^/:?]+)", path or "")
    return match.group(1) if match else ""


def rewrite_gemini_model_in_url(url: str, model: str) -> str:
    if not _SAFE_MODEL_RE.fullmatch(model):
        raise ValueError("invalid URL-embedded model identifier")
    rewritten, count = re.subn(r"(/models/)[^/:?]+", rf"\g<1>{model}", url, count=1)
    if count != 1:
        raise ValueError("Gemini target URL does not contain a model")
    return rewritten


def apply_target_same_provider(
    *,
    provider: str,
    target: ProviderTarget,
    body: Mapping[str, Any],
    url: str,
) -> tuple[dict[str, Any], str]:
    """Apply a selected target to an existing provider request body.

    This is the safe path for the current live proxy: it can change model within
    the same provider while preserving all provider-specific generation controls.
    Cross-provider execution must use ``render_canonical_request`` because it
    cannot safely reuse opaque provider-specific fields.
    """

    provider = provider.lower()
    if target.provider.lower() != provider:
        raise ValueError("same-provider rewrite received a cross-provider target")
    if not _SAFE_MODEL_RE.fullmatch(target.model):
        raise ValueError("invalid model identifier")
    output = copy.deepcopy(dict(body))
    if provider == "gemini":
        output.pop("model", None)
        return output, rewrite_gemini_model_in_url(url, target.model)
    output["model"] = target.model
    return output, url


def render_canonical_request(
    request: CanonicalGatewayRequest,
    target: ProviderTarget,
) -> dict[str, Any]:
    """Render canonical request to a provider body using only portable fields."""

    if not _SAFE_MODEL_RE.fullmatch(target.model):
        raise ValueError("invalid model identifier")
    provider = target.provider.lower()
    messages = [dict(message) for message in request.messages]

    if provider == "openai":
        body: dict[str, Any] = {
            "model": target.model,
            "messages": messages,
        }
        if request.tools:
            body["tools"] = [dict(tool) for tool in request.tools]
        if request.stream:
            body["stream"] = True
        if request.response_schema is not None:
            body["response_format"] = request.response_schema
        return body

    if provider == "anthropic":
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "system":
                system_parts.append(_text_from_content(message.get("content")))
            else:
                anthropic_messages.append(message)
        body = {"model": target.model, "messages": anthropic_messages}
        if system_parts:
            body["system"] = "\n\n".join(part for part in system_parts if part)
        if request.tools:
            body["tools"] = [dict(tool) for tool in request.tools]
        if request.stream:
            body["stream"] = True
        return body

    if provider == "gemini":
        contents: list[dict[str, Any]] = []
        system_text = ""
        for message in messages:
            role = str(message.get("role") or "user")
            if role == "system":
                system_text = (system_text + "\n\n" + _text_from_content(message.get("content"))).strip()
                continue
            gemini_role = "model" if role == "assistant" else "user"
            content = message.get("content", "")
            parts = content if isinstance(content, list) else [{"text": _text_from_content(content)}]
            contents.append({"role": gemini_role, "parts": parts})
        body = {"contents": contents}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        if request.tools:
            body["tools"] = [dict(tool) for tool in request.tools]
        return body

    raise ValueError(f"unsupported provider target: {target.provider}")


__all__ = [
    "ProviderRequestAdapterResult",
    "apply_target_same_provider",
    "canonical_request_from_provider_body",
    "render_canonical_request",
    "rewrite_gemini_model_in_url",
]
