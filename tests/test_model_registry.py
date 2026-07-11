from __future__ import annotations

import json

import pytest

from entroly.models import registry as registry_module
from entroly.models.registry import (
    ModelCapability,
    ModelRegistry,
    RegistryTrust,
    discover_ollama_models,
    get_model_registry,
)


def test_bundled_registry_resolves_exact_and_prefixed_models():
    registry = get_model_registry()

    exact = registry.resolve("gemini-2.5-pro")
    assert exact.exact is True
    assert exact.context_window == 1_048_576
    assert exact.warning is None
    assert exact.trust is RegistryTrust.VERIFIED
    assert len(exact.registry_digest) == 64

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


def test_later_layers_override_bundled_capabilities_and_remove_stale_aliases():
    base = ModelCapability.from_mapping(
        {
            "id": "vendor/model",
            "provider": "openai",
            "aliases": ["model", "retired-name"],
            "context_window": 100,
            "source": "base",
        },
        default_trust=RegistryTrust.VERIFIED,
    )
    override = ModelCapability.from_mapping(
        {
            "id": "vendor/model",
            "provider": "openai",
            "aliases": ["model", "new-name"],
            "context_window": 200,
            "trust": "user",
            "source": "override",
        },
        default_trust=RegistryTrust.USER,
    )

    registry = ModelRegistry([base], overrides=[override])
    result = registry.resolve("model")
    assert result.context_window == 200
    assert result.trust is RegistryTrust.USER
    assert registry.resolve("retired-name").trust is RegistryTrust.FALLBACK
    assert registry.resolve("new-name").context_window == 200


def test_effective_input_budget_reserves_output_and_uncertainty_margin():
    result = get_model_registry().resolve("openai/o1")

    # 200K window - 5% safety - explicit 20K output reserve.
    assert result.effective_input_budget(requested_output_tokens=20_000) == 170_000


def test_cost_estimate_requires_complete_price_metadata():
    capability = ModelCapability.from_mapping(
        {
            "id": "vendor/priced",
            "provider": "vendor",
            "context_window": 100_000,
            "input_price_per_million": 2.0,
            "output_price_per_million": 8.0,
        },
        default_trust=RegistryTrust.USER,
    )
    assert capability.estimated_cost_usd(500_000, 100_000) == pytest.approx(1.8)

    unpriced = get_model_registry().resolve("gpt-4o").capability
    assert unpriced is not None
    assert unpriced.estimated_cost_usd(1_000, 1_000) is None


def test_ollama_discovery_is_loopback_only():
    with pytest.raises(ValueError, match="loopback"):
        discover_ollama_models("https://models.example.com")


def test_ollama_discovery_can_inspect_context_without_external_dependencies(monkeypatch):
    def fake_request(url, *, timeout, payload=None, max_bytes=2 * 1024 * 1024):
        assert timeout == 0.1
        assert max_bytes > 0
        if url.endswith("/api/tags"):
            return {"models": [{"name": "llama3.2:latest"}]}
        assert url.endswith("/api/show")
        assert payload == {"model": "llama3.2:latest"}
        return {"model_info": {"llama.context_length": 131072}}

    monkeypatch.setattr(registry_module, "_json_request", fake_request)
    report = discover_ollama_models(timeout=0.1, inspect_context=True)

    assert report.warnings == ()
    assert len(report.models) == 1
    capability = report.models[0]
    assert capability.id == "ollama/llama3.2:latest"
    assert capability.context_window == 131_072
    assert capability.trust is RegistryTrust.DISCOVERED


def test_registry_diagnostics_report_provenance_counts():
    diagnostics = get_model_registry().diagnostics()

    assert diagnostics["models"] > 0
    assert diagnostics["trust_counts"]["verified"] > 0
    assert diagnostics["trust_counts"]["announced"] > 0
    assert len(diagnostics["registry_digest"]) == 64
