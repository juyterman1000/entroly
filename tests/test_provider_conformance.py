from __future__ import annotations

import json
import subprocess
import sys

from entroly.provider_conformance import (
    SCHEMA_VERSION,
    assess_provider_request,
    provider_capability_matrix,
    run_provider_conformance,
)


def test_cross_provider_unknown_fields_fail_closed() -> None:
    result = assess_provider_request(
        "openai",
        "anthropic",
        {
            "model": "gpt",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
        },
    )

    assert result["allowed"] is False
    assert result["nonportable_fields"] == ["temperature"]
    assert "nonportable fields" in result["reason"]


def test_same_provider_unknown_fields_are_preserved() -> None:
    result = assess_provider_request(
        "openai",
        "openai",
        {
            "model": "gpt",
            "messages": [{"role": "user", "content": "hello"}],
            "future_control": {"mode": "strict"},
        },
    )

    assert result["allowed"] is True
    assert result["mode"] == "same_provider_preserve"
    assert "future_control" in result["preserved_top_level_fields"]


def test_cache_control_requires_fail_closed_cross_provider() -> None:
    result = assess_provider_request(
        "anthropic",
        "openai",
        {
            "model": "claude",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}}],
            }],
        },
    )

    assert result["allowed"] is False
    assert "cache_control" in result["required_capabilities"]


def test_conformance_suite_is_offline_and_green() -> None:
    report = run_provider_conformance()

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["healthy"] is True
    assert report["summary"]["failed"] == 0
    assert report["claims"]["provider_connectivity_verified"] is False
    assert report["claims"]["full_semantic_equivalence_implied"] is False


def test_capability_matrix_distinguishes_preservation_from_translation() -> None:
    report = provider_capability_matrix()
    openai = report["protocols"]["openai_chat_completions"]

    assert openai["same_provider"]["opaque_fields_preserved"] is True
    assert openai["cross_provider"]["text_only_messages"] is True
    assert openai["cross_provider"]["tools"] is False
    assert openai["connectivity_verified"] is False


def test_module_cli_emits_stable_json() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "entroly.provider_conformance", "--json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["healthy"] is True
