"""Secure local-model discovery adapters for Entroly model intelligence."""
from __future__ import annotations

import ipaddress
import json
import logging
import os
from typing import Any, Callable, Mapping
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from .model_registry import ModelRegistryError, ModelSpec

logger = logging.getLogger("entroly.model_discovery")

_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_MODELS = 128


def _is_loopback_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _bounded_context_lengths(value: Any, *, depth: int = 0) -> list[int]:
    if depth > 5:
        return []
    found: list[int] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = str(key).lower()
            if normalized_key.endswith("context_length"):
                try:
                    parsed = int(child)
                except (TypeError, ValueError):
                    parsed = 0
                if 1_024 <= parsed <= 100_000_000:
                    found.append(parsed)
            else:
                found.extend(_bounded_context_lengths(child, depth=depth + 1))
    elif isinstance(value, list):
        for child in value[:512]:
            found.extend(_bounded_context_lengths(child, depth=depth + 1))
    return found


def models_from_ollama_payload(
    tags_payload: Mapping[str, Any],
    *,
    show_payloads: Mapping[str, Mapping[str, Any]] | None = None,
    source: str = "ollama",
) -> tuple[ModelSpec, ...]:
    raw_models = tags_payload.get("models")
    if not isinstance(raw_models, list):
        raise ModelRegistryError("Ollama /api/tags response requires models[]")
    show_payloads = show_payloads or {}
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for item in raw_models[:512]:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        normalized = name.lower()
        if not name or normalized in seen:
            continue
        seen.add(normalized)
        show = show_payloads.get(name, {})
        lengths = _bounded_context_lengths(show)
        capabilities = (
            {
                str(value).strip().lower()
                for value in show.get("capabilities", ())
                if str(value).strip()
            }
            if isinstance(show, Mapping)
            else set()
        )
        context_window = max(lengths) if lengths else None
        specs.append(
            ModelSpec(
                id=name,
                provider="ollama",
                aliases=(name.removesuffix(":latest"),)
                if name.endswith(":latest")
                else (),
                context_window=context_window,
                supports_tools=("tools" in capabilities) if capabilities else None,
                supports_vision=(
                    bool(capabilities & {"vision", "images"})
                    if capabilities
                    else None
                ),
                supports_reasoning=(
                    bool(capabilities & {"thinking", "reasoning"})
                    if capabilities
                    else None
                ),
                source=source,
                confidence=1.0 if context_window is not None else 0.65,
                status="local",
                trust="discovered",
                local=True,
            )
        )
    return tuple(specs)


def _default_json_transport(
    request: Request,
    *,
    timeout: float,
) -> Mapping[str, Any]:
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - guarded URL
        if getattr(response, "status", 200) >= 400:
            raise ModelRegistryError(f"Ollama returned HTTP {response.status}")
        raw = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise ModelRegistryError("Ollama metadata response exceeded 4 MiB")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ModelRegistryError("Ollama response must be a JSON object")
    return payload


def discover_ollama_models(
    *,
    base_url: str | None = None,
    timeout: float = 0.75,
    transport: Callable[[Request], Mapping[str, Any]] | None = None,
) -> tuple[ModelSpec, ...]:
    """Discover Ollama models without allowing implicit remote SSRF.

    Discovery is loopback-only unless ENTROLY_OLLAMA_ALLOW_REMOTE=1. Responses
    are bounded, model count is capped, and /api/show failures are fail-open.
    """
    base = (
        base_url
        or os.environ.get("ENTROLY_OLLAMA_BASE")
        or os.environ.get("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    ).rstrip("/") + "/"
    allow_remote = os.environ.get("ENTROLY_OLLAMA_ALLOW_REMOTE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not allow_remote and not _is_loopback_base_url(base):
        raise ModelRegistryError(
            "Ollama discovery is loopback-only; set ENTROLY_OLLAMA_ALLOW_REMOTE=1 "
            "to explicitly allow a remote endpoint"
        )

    def send(request: Request) -> Mapping[str, Any]:
        if transport is not None:
            return transport(request)
        return _default_json_transport(request, timeout=timeout)

    tags_request = Request(urljoin(base, "api/tags"), method="GET")
    tags_payload = send(tags_request)
    raw_models = tags_payload.get("models")
    if not isinstance(raw_models, list):
        raise ModelRegistryError("Ollama /api/tags response requires models[]")

    show_payloads: dict[str, Mapping[str, Any]] = {}
    for item in raw_models[:_MAX_MODELS]:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if not name:
            continue
        body = json.dumps({"model": name, "verbose": False}).encode("utf-8")
        request = Request(
            urljoin(base, "api/show"),
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            show_payloads[name] = send(request)
        except Exception as exc:
            logger.debug("Ollama /api/show failed for %s: %s", name, exc)
    return models_from_ollama_payload(
        tags_payload,
        show_payloads=show_payloads,
        source=f"ollama:{base}",
    )
