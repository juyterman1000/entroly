"""Tests for Spec 1: Model-Aware Context Shaping via AttentionArchitecture."""

from __future__ import annotations

from entroly.models.registry import (
    AttentionArchitecture,
    ModelCapability,
    RegistryTrust,
    _infer_attention_profile,
)
from entroly.proxy_config import ProxyConfig
from entroly.proxy_transform import (
    _EXPANDED_BUDGET_CEILING,
    _LINEAR_BUDGET_CEILING,
    compute_dynamic_budget,
)


class TestAttentionInference:
    """_infer_attention_profile classifies model families from name patterns."""

    def test_gpt_models_are_full_mha(self):
        assert _infer_attention_profile("openai/gpt-4o") is AttentionArchitecture.FULL_MHA_GQA

    def test_claude_models_are_full_mha(self):
        assert _infer_attention_profile("anthropic/claude-opus-4.6") is AttentionArchitecture.FULL_MHA_GQA

    def test_gemini_models_are_full_mha(self):
        assert _infer_attention_profile("google/gemini-2.5-pro") is AttentionArchitecture.FULL_MHA_GQA

    def test_deepseek_is_latent_mla(self):
        assert _infer_attention_profile("openai-compatible/deepseek") is AttentionArchitecture.LATENT_MLA

    def test_mamba_is_linear_hybrid(self):
        assert _infer_attention_profile("local/mamba-3b") is AttentionArchitecture.LINEAR_HYBRID_GDN

    def test_rwkv_is_linear_hybrid(self):
        assert _infer_attention_profile("local/rwkv-6-world") is AttentionArchitecture.LINEAR_HYBRID_GDN

    def test_mixtral_is_sparse_dsa(self):
        assert _infer_attention_profile("ollama/mixtral-8x7b") is AttentionArchitecture.SPARSE_DSA

    def test_unknown_model_returns_none(self):
        assert _infer_attention_profile("lab/totally-new-thing") is None

    def test_case_insensitive(self):
        assert _infer_attention_profile("DEEPSEEK-V3") is AttentionArchitecture.LATENT_MLA


class TestAttentionOnModelCapability:
    """AttentionArchitecture round-trips through from_mapping."""

    def test_explicit_profile_in_json(self):
        cap = ModelCapability.from_mapping(
            {
                "id": "custom/model",
                "provider": "custom",
                "attention_profile": "latent_mla",
            },
            default_trust=RegistryTrust.ANNOUNCED,
        )
        assert cap.attention_profile is AttentionArchitecture.LATENT_MLA

    def test_inferred_profile_when_absent(self):
        cap = ModelCapability.from_mapping(
            {
                "id": "openai-compatible/deepseek",
                "provider": "openai",
            },
            default_trust=RegistryTrust.ANNOUNCED,
        )
        assert cap.attention_profile is AttentionArchitecture.LATENT_MLA

    def test_unknown_model_gets_none_profile(self):
        cap = ModelCapability.from_mapping(
            {
                "id": "lab/novel-arch",
                "provider": "lab",
            },
            default_trust=RegistryTrust.ANNOUNCED,
        )
        assert cap.attention_profile is None

    def test_fingerprint_includes_profile(self):
        cap = ModelCapability.from_mapping(
            {
                "id": "openai/gpt-4o",
                "provider": "openai",
            },
            default_trust=RegistryTrust.ANNOUNCED,
        )
        payload = cap.fingerprint_payload()
        assert "attention_profile" in payload
        assert payload["attention_profile"] == "full_mha_gqa"

    def test_invalid_profile_raises(self):
        import pytest

        with pytest.raises(ValueError, match="invalid attention_profile"):
            ModelCapability.from_mapping(
                {
                    "id": "x/y",
                    "provider": "x",
                    "attention_profile": "bogus_arch",
                },
                default_trust=RegistryTrust.ANNOUNCED,
            )


class TestBudgetShaping:
    """compute_dynamic_budget applies architecture-specific clamping."""

    def test_standard_model_unchanged(self):
        config = ProxyConfig()
        budget = compute_dynamic_budget("gpt-4o", config, vagueness=0.5, total_fragments=100)
        window = 128_000
        assert config.ecdb_min_budget <= budget <= int(window * config.ecdb_max_fraction)

    def test_deepseek_can_expand_budget(self):
        config = ProxyConfig()
        budget_ds = compute_dynamic_budget(
            "deepseek-v3", config, vagueness=1.0, total_fragments=500
        )
        budget_gpt = compute_dynamic_budget(
            "gpt-4o", config, vagueness=1.0, total_fragments=500
        )
        assert budget_ds >= budget_gpt

    def test_linear_model_budget_ceiling(self):
        config = ProxyConfig(context_fraction=0.99)
        budget = compute_dynamic_budget(
            "mamba-3b", config, vagueness=1.0, total_fragments=10000
        )
        assert budget <= _LINEAR_BUDGET_CEILING

    def test_expanded_ceiling_does_not_exceed_window(self):
        config = ProxyConfig(context_fraction=0.99)
        budget = compute_dynamic_budget(
            "deepseek-v3", config, vagueness=1.0, total_fragments=10000
        )
        window = 128_000
        assert budget <= window
