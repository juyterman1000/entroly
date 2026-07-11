from __future__ import annotations

from entroly.control_plane import stable_request_fingerprint
from entroly.model_decision_receipt import (
    MODEL_DECISION_SCHEMA,
    build_model_decision_receipt,
    model_decision_tags,
)


def test_verified_model_receipt_is_deterministic_and_self_verifying() -> None:
    body = {
        "model": "gpt-4o",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "hello"}],
    }

    left = build_model_decision_receipt(body, provider="openai")
    right = build_model_decision_receipt(body, provider="openai")

    assert left is not None
    assert right is not None
    assert left == right
    assert left.schema == MODEL_DECISION_SCHEMA
    assert left.requested_model == "gpt-4o"
    assert left.resolved_model == "openai/gpt-4o"
    assert left.trust == "verified"
    assert left.exact is True
    assert left.fallback_used is False
    assert left.context_window == 128_000
    assert left.requested_output_tokens == 4096
    assert left.safe_input_budget == 117_504
    assert len(left.receipt_digest) == 64
    assert left.verify() is True


def test_receipt_digest_changes_when_budget_decision_changes() -> None:
    low = build_model_decision_receipt(
        {"model": "gpt-4o", "max_tokens": 1024},
        provider="openai",
    )
    high = build_model_decision_receipt(
        {"model": "gpt-4o", "max_tokens": 8192},
        provider="openai",
    )

    assert low is not None and high is not None
    assert low.safe_input_budget > high.safe_input_budget
    assert low.receipt_digest != high.receipt_digest


def test_unknown_model_is_explicit_conservative_fallback() -> None:
    receipt = build_model_decision_receipt(
        {"model": "private-future-model", "max_output_tokens": 2048},
        provider="openai",
    )

    assert receipt is not None
    assert receipt.trust == "fallback"
    assert receipt.fallback_used is True
    assert receipt.warning_code == "unknown_model"
    assert receipt.context_window == 128_000
    assert receipt.safe_input_budget == 119_552
    assert receipt.verify() is True


def test_recognized_unverified_frontier_model_is_not_misrepresented() -> None:
    receipt = build_model_decision_receipt(
        {"model": "gpt-5.6-terra", "max_output_tokens": 4096},
        provider="openai",
    )

    assert receipt is not None
    assert receipt.resolved_model == "openai/gpt-5.6-terra"
    assert receipt.trust == "announced"
    assert receipt.fallback_used is True
    assert receipt.warning_code == "unverified_context_window"
    assert receipt.context_window == 128_000


def test_gemini_path_model_and_output_reserve_are_detected() -> None:
    receipt = build_model_decision_receipt(
        {"generationConfig": {"maxOutputTokens": 8192}},
        provider="gemini",
        path="/v1beta/models/gemini-2.5-pro:generateContent",
    )

    assert receipt is not None
    assert receipt.requested_model == "gemini-2.5-pro"
    assert receipt.resolved_model == "google/gemini-2.5-pro"
    assert receipt.context_window == 1_048_576
    assert receipt.requested_output_tokens == 8192
    assert receipt.safe_input_budget == 987_955


def test_tags_exclude_registry_sources_and_private_metadata() -> None:
    tags = model_decision_tags(
        {"model": "gpt-4o", "max_tokens": 1024},
        provider="openai",
    )
    rendered = "\n".join(f"{key}={value}" for key, value in sorted(tags.items()))

    assert tags["entroly_model_receipt_status"] == "verified"
    assert tags["entroly_model_receipt_schema"] == MODEL_DECISION_SCHEMA
    assert len(tags["entroly_model_receipt"]) == 64
    assert "platform.openai.com" not in rendered
    assert "source" not in rendered
    assert "aliases" not in rendered
    assert "\r" not in rendered


def test_missing_model_is_not_applicable_and_never_raises() -> None:
    assert model_decision_tags({}, provider="unknown") == {
        "entroly_model_receipt_status": "not_applicable"
    }


def test_control_plane_fingerprint_exports_model_receipt_tags() -> None:
    tags = stable_request_fingerprint(
        {
            "model": "gpt-4o",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": "hello"}],
        },
        provider="openai",
        path="/v1/chat/completions",
    )

    assert tags["entroly_model_receipt_status"] == "verified"
    assert tags["entroly_model_resolved"] == "openai/gpt-4o"
    assert tags["entroly_model_context_window"] == "128000"
    assert tags["entroly_model_output_reserve"] == "2048"
    assert len(tags["entroly_model_receipt"]) == 64
    assert len(tags["entroly_registry_digest"]) == 64
    assert len(tags["entroly_registry_base_digest"]) == 64
