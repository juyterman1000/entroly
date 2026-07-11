from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import Request

import pytest

from entroly.model_discovery import discover_ollama_models, models_from_ollama_payload
from entroly.model_registry import (
    ModelRegistryError,
    ModelSpec,
    UnknownModelError,
    context_window_for_model,
    default_registry,
    load_registry_file,
    registry_from_payload,
    reset_registry_cache,
)


def test_bundled_registry_resolves_exact_alias_and_longest_prefix(monkeypatch):
    monkeypatch.delenv("ENTROLY_MODEL_REGISTRY", raising=False)
    monkeypatch.delenv("ENTROLY_OLLAMA_DISCOVERY", raising=False)
    reset_registry_cache()
    registry = default_registry()

    assert registry.resolve("gpt-4o").effective_context_window == 128_000
    assert registry.resolve("gpt-4o-2024-08-06").canonical_id == "gpt-4o"
    assert registry.resolve("gpt-4o-mini-2024-07-18").canonical_id == "gpt-4o-mini"
    assert registry.resolve("models/gemini-2.5-pro-preview").canonical_id == "gemini-2.5-pro"


def test_unknown_model_is_fail_visible_and_conservative(monkeypatch, caplog):
    monkeypatch.delenv("ENTROLY_UNKNOWN_CONTEXT_WINDOW", raising=False)
    reset_registry_cache()

    resolution = default_registry().resolve("future-model-not-yet-known")

    assert resolution.known is False
    assert resolution.effective_context_window == 32_768
    assert resolution.warnings
    assert "unverified model budget" in resolution.warnings[0]


def test_unknown_model_strict_mode_raises():
    with pytest.raises(UnknownModelError):
        default_registry().resolve("future-model-not-yet-known", unknown_policy="error")


def test_declared_but_unverified_model_does_not_fake_context():
    resolution = default_registry().resolve("gpt-5.6-sol", unknown_policy="silent")

    assert resolution.known is True
    assert resolution.verified is False
    assert resolution.spec is not None
    assert resolution.spec.context_window is None
    assert resolution.effective_context_window == 32_768


def test_user_overlay_wins_deterministically(tmp_path, monkeypatch):
    overlay = {
        "schema_version": 1,
        "registry_version": "test",
        "source": "unit-test",
        "models": [
            {
                "id": "gpt-4o",
                "provider": "openai",
                "context_window": 999_999,
                "confidence": 1.0,
            }
        ],
    }
    canonical = json.dumps(
        overlay, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    overlay["integrity"] = {
        "payload_sha256": hashlib.sha256(canonical).hexdigest()
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(overlay), encoding="utf-8")
    monkeypatch.setenv("ENTROLY_MODEL_REGISTRY", str(path))
    reset_registry_cache()

    resolution = default_registry().resolve("gpt-4o")
    assert resolution.effective_context_window == 999_999
    assert resolution.spec is not None
    assert resolution.spec.trust == "user"


def test_registry_digest_mismatch_is_rejected(tmp_path):
    payload = {
        "schema_version": 1,
        "registry_version": "test",
        "source": "unit-test",
        "models": [{"id": "m", "provider": "p", "context_window": 4096}],
        "integrity": {"payload_sha256": "0" * 64},
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelRegistryError, match="payload_sha256"):
        load_registry_file(path)


def test_ollama_payload_extracts_context_and_capabilities():
    specs = models_from_ollama_payload(
        {"models": [{"name": "qwen3:latest"}]},
        show_payloads={
            "qwen3:latest": {
                "model_info": {"qwen3.context_length": 131_072},
                "capabilities": ["completion", "tools", "thinking"],
            }
        },
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.id == "qwen3:latest"
    assert spec.aliases == ("qwen3",)
    assert spec.context_window == 131_072
    assert spec.supports_tools is True
    assert spec.supports_reasoning is True
    assert spec.local is True


def test_ollama_discovery_is_loopback_only(monkeypatch):
    monkeypatch.delenv("ENTROLY_OLLAMA_ALLOW_REMOTE", raising=False)
    with pytest.raises(ModelRegistryError, match="loopback-only"):
        discover_ollama_models(base_url="https://example.com")


def test_ollama_discovery_uses_bounded_local_api():
    calls: list[tuple[str, str]] = []

    def transport(request: Request):
        calls.append((request.method, request.full_url))
        if request.full_url.endswith("/api/tags"):
            return {"models": [{"name": "local-model:latest"}]}
        return {
            "model_info": {"architecture.context_length": 65_536},
            "capabilities": ["tools", "vision"],
        }

    specs = discover_ollama_models(
        base_url="http://127.0.0.1:11434",
        transport=transport,
    )

    assert [method for method, _ in calls] == ["GET", "POST"]
    assert specs[0].context_window == 65_536
    assert specs[0].supports_vision is True


def test_registry_fingerprint_is_order_independent():
    payload_a = {
        "schema_version": 1,
        "registry_version": "x",
        "source": "test",
        "models": [
            {"id": "a", "provider": "p", "context_window": 4096},
            {"id": "b", "provider": "p", "context_window": 8192},
        ],
    }
    payload_b = {**payload_a, "models": list(reversed(payload_a["models"]))}

    registry_a = registry_from_payload(payload_a, default_source="x", trust="user")
    registry_b = registry_from_payload(payload_b, default_source="x", trust="user")

    assert registry_a.fingerprint == registry_b.fingerprint


def test_duplicate_provider_alias_is_rejected():
    payload = {
        "schema_version": 1,
        "registry_version": "x",
        "source": "test",
        "models": [
            {"id": "a", "provider": "p", "aliases": ["shared"]},
            {"id": "b", "provider": "p", "aliases": ["shared"]},
        ],
    }

    with pytest.raises(ModelRegistryError, match="duplicate alias"):
        registry_from_payload(payload, default_source="x", trust="user")


def test_compatibility_context_helper_uses_registry(monkeypatch):
    monkeypatch.setenv("ENTROLY_UNKNOWN_MODEL_POLICY", "silent")
    reset_registry_cache()

    assert context_window_for_model("claude-opus-4-6") == 200_000
    assert context_window_for_model("not-known") == 32_768
