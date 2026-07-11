from __future__ import annotations

import logging

from entroly.proxy_config import (
    MODEL_CONTEXT_WINDOWS,
    _warn_model_resolution_once,
    context_window_for_model,
    model_input_budget_for_model,
    model_resolution_for_model,
)


def test_proxy_budget_lookup_preserves_legacy_catalog():
    assert context_window_for_model("gpt-4") == 8_192
    assert context_window_for_model("o1-mini-2024-09-12") == 128_000
    assert context_window_for_model("claude-3-haiku-20240307") == 200_000
    assert MODEL_CONTEXT_WINDOWS["gemini-1.5-pro"] == 2_097_152


def test_proxy_budget_lookup_resolves_frontier_models_with_provenance():
    resolution = model_resolution_for_model("nemotron-super")

    assert resolution.model_id == "nvidia/nemotron-super"
    assert resolution.context_window == 1_000_000
    assert resolution.trust.value == "announced"
    assert len(resolution.registry_digest) == 64


def test_proxy_unknown_model_warning_is_visible_but_deduplicated(caplog):
    _warn_model_resolution_once.cache_clear()
    caplog.set_level(logging.WARNING, logger="entroly.proxy")

    assert context_window_for_model("lab/unknown-model") == 128_000
    assert context_window_for_model("lab/unknown-model") == 128_000

    warnings = [record for record in caplog.records if "Unknown model" in record.message]
    assert len(warnings) == 1


def test_proxy_safe_input_budget_reserves_output_capacity():
    context = context_window_for_model("gemini-2.5-pro")
    input_budget = model_input_budget_for_model(
        "gemini-2.5-pro",
        requested_output_tokens=10_000,
    )

    assert input_budget < context - 10_000
    assert input_budget > 900_000
