from __future__ import annotations

from entroly.model_registry import default_registry, reset_registry_cache
from entroly.proxy_config import (
    MODEL_CONTEXT_WINDOWS,
    PROVIDER_CAPABILITIES,
    context_window_for_model,
)


def test_proxy_budgeting_uses_model_registry(monkeypatch):
    monkeypatch.setenv("ENTROLY_UNKNOWN_MODEL_POLICY", "silent")
    reset_registry_cache()

    assert context_window_for_model("gpt-4o-2024-08-06") == 128_000
    assert context_window_for_model("models/gemini-2.5-pro-preview") == 1_048_576
    assert context_window_for_model("gpt-5.6-sol") == 32_768
    assert context_window_for_model("not-yet-catalogued") == 32_768


def test_transport_catalog_remains_consistent_with_registry():
    registry = default_registry()

    for capability in PROVIDER_CAPABILITIES.values():
        for model, expected_window in capability.context_windows.items():
            resolution = registry.resolve(model, unknown_policy="error")
            assert resolution.effective_context_window == expected_window

    assert MODEL_CONTEXT_WINDOWS["claude-opus-4-6"] == 200_000
