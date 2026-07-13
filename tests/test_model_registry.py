from __future__ import annotations

import json

import pytest

from entroly.models import registry as registry_module
from entroly.models.registry import (
    ModelCapability,
    ModelRegistry,
    RegistryTrust,
    discover_ollama_models,
    discover_openrouter_models,
    get_model_registry,
)


def test_bundled_registry_resolves_exact_and_explicit_prefix_models():
    registry = get_model_registry()

    exact = registry.resolve("gemini-2.5-pro")
    assert exact.exact is True
    assert exact.context_window == 1_048_576
    assert exact.warning is None
    assert exact.trust is RegistryTrust.VERIFIED
    assert len(exact.registry_digest) == 64
    assert len(exact.base_registry_digest) == 64

    dated = registry.resolve("gpt-4o-2024-08-06")
    assert dated.context_window == 128_000
    assert dated.capability is not None
    assert dated.capability.id == "openai/gpt-4o"

    unrelated = registry.resolve("gpt-4xyz")
    assert unrelated.capability is None
    assert unrelated.trust is RegistryTrust.FALLBACK


def test_gpt_5_6_uses_verified_context_pricing_and_reasoning_metadata():
    result = get_model_registry().resolve("gpt-5.6-sol")

    assert result.capability is not None
    assert result.capability.id == "openai/gpt-5.6-sol"
    assert result.trust is RegistryTrust.VERIFIED
    assert result.context_window == 1_050_000
    assert result.warning is None
    assert result.capability.max_output_tokens == 128_000
    assert result.capability.input_price_per_million == 5.0
    assert result.capability.output_price_per_million == 30.0
    assert result.capability.supports_reasoning is True


def test_unknown_model_fails_visibly_with_conservative_fallback():
    result = get_model_registry().resolve("private-lab/model-x")

    assert result.capability is None
    assert result.trust is RegistryTrust.FALLBACK
    assert result.context_window == 128_000
    assert "Unknown model" in (result.warning or "")


def test_override_precedence_changes_effective_registry_digest(tmp_path, monkeypatch):
    base_digest = get_model_registry().registry_digest
    base_snapshot_digest = get_model_registry().base_registry_digest
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
        registry = get_model_registry()
        result = registry.resolve("model-x")
        assert result.context_window == 262_144
        assert result.trust is RegistryTrust.USER
        assert result.warning is None
        assert registry.registry_digest != base_digest
        assert registry.base_registry_digest == base_snapshot_digest
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


def test_discovery_overlay_preserves_bundled_aliases():
    base = ModelCapability.from_mapping(
        {
            "id": "vendor/model",
            "provider": "vendor",
            "aliases": ["stable-alias"],
            "context_window": 100,
            "source": "bundled",
        },
        default_trust=RegistryTrust.VERIFIED,
    )
    discovered = ModelCapability.from_mapping(
        {
            "id": "vendor/model",
            "provider": "router",
            "aliases": ["router/vendor/model"],
            "context_window": 200,
            "source": "router",
        },
        default_trust=RegistryTrust.DISCOVERED,
    )

    registry = ModelRegistry([base], discovered=[discovered])

    assert registry.resolve("stable-alias").context_window == 200
    assert registry.resolve("router/vendor/model").context_window == 200


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
    def fake_request(
        url, *, timeout, payload=None, headers=None, max_bytes=2 * 1024 * 1024
    ):
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
    assert capability.supports_tools is None
    assert capability.supports_vision is None
    assert capability.input_price_per_million is None
    assert capability.output_price_per_million is None
    assert capability.observed_at is not None
    assert capability.verified_at is None


def test_openrouter_discovery_requires_an_explicit_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    report = discover_openrouter_models()

    assert report.models == ()
    assert "unset" in report.warnings[0]


def test_openrouter_discovery_maps_live_metadata_without_persisting_credentials(monkeypatch):
    seen = {}

    def fake_request(url, *, timeout, payload=None, headers=None, max_bytes=2 * 1024 * 1024):
        seen.update(url=url, timeout=timeout, payload=payload, headers=headers)
        return {
            "data": [
                {
                    "id": "meta/muse-spark-1.1",
                    "context_length": 1_000_000,
                    "pricing": {"prompt": "0.0000015", "completion": "0.000006"},
                    "supported_parameters": ["tools", "reasoning"],
                    "architecture": {"input_modalities": ["text", "image"]},
                    "top_provider": {"max_completion_tokens": 65536},
                }
            ]
        }

    monkeypatch.setattr(registry_module, "_json_request", fake_request)
    report = discover_openrouter_models("super-secret", timeout=0.5)

    assert report.warnings == ()
    assert seen == {
        "url": "https://openrouter.ai/api/v1/models",
        "timeout": 0.5,
        "payload": None,
        "headers": {"Authorization": "Bearer super-secret"},
    }
    capability = report.models[0]
    assert capability.id == "meta/muse-spark-1.1"
    assert capability.provider == "openrouter"
    assert capability.context_window == 1_000_000
    assert capability.max_output_tokens == 65_536
    assert capability.supports_tools is True
    assert capability.supports_vision is True
    assert capability.supports_reasoning is True
    assert capability.input_price_per_million == pytest.approx(1.5)
    assert capability.output_price_per_million == pytest.approx(6.0)
    assert capability.trust is RegistryTrust.DISCOVERED


def test_openrouter_discovery_does_not_invent_missing_or_impossible_metadata(monkeypatch):
    def fake_request(url, *, timeout, payload=None, headers=None, max_bytes=2 * 1024 * 1024):
        return {
            "data": [
                {
                    "id": "vendor/incomplete",
                    "context_length": 4096,
                    "pricing": {"prompt": "NaN", "completion": "Infinity"},
                    "top_provider": {"max_completion_tokens": 8192},
                }
            ]
        }

    monkeypatch.setattr(registry_module, "_json_request", fake_request)
    report = discover_openrouter_models("ephemeral-key")

    assert len(report.models) == 1
    capability = report.models[0]
    assert capability.max_output_tokens is None
    assert capability.supports_tools is None
    assert capability.supports_vision is None
    assert capability.supports_reasoning is None
    assert capability.input_price_per_million is None
    assert capability.output_price_per_million is None
    assert "outside its context window" in report.warnings[0]


def test_nemotron_registry_distinguishes_verified_native_and_unverified_alias():
    registry = get_model_registry()

    official = registry.resolve("nemotron-3-ultra")
    assert official.capability is not None
    assert official.capability.id == "nvidia/nemotron-3-ultra-550b-a55b"
    assert official.context_window == 262_144
    assert official.trust is RegistryTrust.VERIFIED

    provisional = registry.resolve("nemotron-super")
    assert provisional.capability is not None
    assert provisional.context_window == 128_000
    assert provisional.trust is RegistryTrust.ANNOUNCED
    assert "unverified" in (provisional.warning or "")


def test_registry_diagnostics_report_provenance_counts():
    diagnostics = get_model_registry().diagnostics()

    assert diagnostics["models"] > 0
    assert diagnostics["trust_counts"]["verified"] > 0
    assert diagnostics["trust_counts"]["announced"] > 0
    assert len(diagnostics["registry_digest"]) == 64
    assert len(diagnostics["base_registry_digest"]) == 64
