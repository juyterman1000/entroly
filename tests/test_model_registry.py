from __future__ import annotations

import json

from entroly.models.registry import (
    ModelCapability,
    ModelRegistry,
    RegistryTrust,
    get_model_registry,
)


def test_bundled_registry_resolves_exact_and_prefixed_models():
    registry = get_model_registry()

    exact = registry.resolve("gemini-2.5-pro")
    assert exact.exact is True
    assert exact.context_window == 1_048_576
    assert exact.warning is None
    assert exact.trust is RegistryTrust.VERIFIED

    dated = registry.resolve("gpt-4o-2024-08-06")
    assert dated.context_window == 128_000
    assert dated.capability is not None
    assert dated.capability.id == "openai/gpt-4o"


def test_announced_model_is_recognized_without_inventing_context_metadata():
    result = get_model_registry().resolve("gpt-5.6-sol")

    assert result.capability is not None
    assert result.capability.id == "openai/gpt-5.6-sol"
    assert result.trust is RegistryTrust.ANNOUNCED
    assert result.context_window == 128_000
    assert "unverified" in (result.warning or "")


def test_unknown_model_fails_visibly_with_conservative_fallback():
    result = get_model_registry().resolve("private-lab/model-x")

    assert result.capability is None
    assert result.trust is RegistryTrust.FALLBACK
    assert result.context_window == 128_000
    assert "Unknown model" in (result.warning or "")


def test_override_precedence_allows_private_and_new_models(tmp_path, monkeypatch):
    override = tmp_path / "models.json"
    override.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "private/model-x",
                        "provider": "openai",
                        "aliases": ["model-x"],
                        "context_window": 262144,
                        "supports_tools": True,
                        "trust": "user",
                        "source": "local-test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENTROLY_MODEL_REGISTRY", str(override))
    get_model_registry.cache_clear()
    try:
        result = get_model_registry().resolve("model-x")
        assert result.context_window == 262_144
        assert result.trust is RegistryTrust.USER
        assert result.warning is None
    finally:
        get_model_registry.cache_clear()


def test_later_layers_override_bundled_capabilities():
    base = ModelCapability.from_mapping(
        {
            "id": "vendor/model",
            "provider": "openai",
            "aliases": ["model"],
            "context_window": 100,
            "source": "base",
        },
        default_trust=RegistryTrust.VERIFIED,
    )
    override = ModelCapability.from_mapping(
        {
            "id": "vendor/model",
            "provider": "openai",
            "aliases": ["model"],
            "context_window": 200,
            "trust": "user",
            "source": "override",
        },
        default_trust=RegistryTrust.USER,
    )

    result = ModelRegistry([base], overrides=[override]).resolve("model")
    assert result.context_window == 200
    assert result.trust is RegistryTrust.USER
