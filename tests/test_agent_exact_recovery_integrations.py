"""Contracts shared by Hermes, OpenClaw, and OpenCode integrations."""

from __future__ import annotations

import json

import pytest

from entroly.ccr import CompressedContextStore
from entroly.integrations.exact_recovery import (
    ExactRecoveryError,
    normalize_exact_handle,
    retrieve_exact,
)


def test_exact_handle_is_case_normalized() -> None:
    assert normalize_exact_handle("CCR:ABCDEF0123456789ABCDEF01") == (
        "ccr:abcdef0123456789abcdef01"
    )
    assert normalize_exact_handle("ABCDEF0123456789ABCDEF01") == (
        "ccr:abcdef0123456789abcdef01"
    )


@pytest.mark.parametrize(
    "value",
    ["", "src/auth.py", "find the auth code", "ccr:abc", "ccr:zzzzzzzzzzzzzzzzzzzzzzzz"],
)
def test_exact_recovery_rejects_non_hash_inputs(value: str) -> None:
    with pytest.raises(ExactRecoveryError):
        normalize_exact_handle(value)


def test_exact_recovery_returns_complete_original_without_search() -> None:
    store = CompressedContextStore(max_entries=4)
    original = "alpha\nneedle\nomega\n"
    handle = store.store(
        source="test:document",
        original_content=original,
        compressed_content="alpha … omega",
        original_tokens=8,
        compressed_tokens=3,
    )

    result = retrieve_exact(handle.upper(), store=store)

    assert result["lookup"] == "hash_only"
    assert result["full_content"] is True
    assert result["original_content"] == original
    assert result["retrieval_handle"] == handle


def _engine(**kwargs):
    from entroly.integrations.hermes_context_engine import EntrolyContextEngine

    defaults = {
        "context_length": 2_000,
        "threshold_percent": 0.75,
        "protect_first_n": 3,
        "protect_last_n": 4,
        "profile": "max",
    }
    defaults.update(kwargs)
    return EntrolyContextEngine(**defaults)


def _messages(count: int = 20, width: int = 400) -> list[dict[str, str]]:
    result = [
        {
            "role": "system",
            "content": (
                "You are a function calling AI model. Function signatures are "
                "inside <tools></tools> XML tags."
            ),
        }
    ]
    for index in range(count):
        result.append(
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"message {index}: " + (chr(97 + index % 20) * width),
            }
        )
    return result


def test_hermes_exposes_hash_only_tool_schema() -> None:
    schema = _engine().get_tool_schemas()
    assert len(schema) == 1
    assert schema[0]["name"] == "entroly_retrieve"
    parameters = schema[0]["parameters"]
    assert parameters["required"] == ["hash"]
    assert parameters["additionalProperties"] is False
    assert set(parameters["properties"]) == {"hash"}
    assert "query" not in json.dumps(schema).lower()


def test_hermes_compression_emits_recoverable_exact_conversation() -> None:
    engine = _engine()
    source = _messages()

    compressed = engine.compress(source, focus_topic="fix authentication")

    markers = [
        message
        for message in compressed
        if message.get("role") == "system"
        and "<entroly_exact_recovery" in str(message.get("content", ""))
    ]
    assert len(markers) == 1
    handle = engine.get_status()["exact_recovery"]["last_handle"]
    assert isinstance(handle, str) and handle.startswith("ccr:")

    recovered = json.loads(engine.handle_tool_call("entroly_retrieve", {"hash": handle}))
    assert recovered["status"] == "recovered"
    assert recovered["lookup"] == "hash_only"
    assert recovered["full_content"] is True
    assert json.loads(recovered["original_content"]) == source


def test_hermes_tool_rejects_query_or_source_arguments() -> None:
    engine = _engine()
    bad_query = json.loads(
        engine.handle_tool_call(
            "entroly_retrieve",
            {"hash": "ccr:abcdef0123456789abcdef01", "query": "auth"},
        )
    )
    bad_source = json.loads(
        engine.handle_tool_call("entroly_retrieve", {"hash": "src/auth.py"})
    )
    assert bad_query["status"] == "error"
    assert bad_source["status"] == "not_found"


def test_hermes_select_context_is_request_only_and_fail_open() -> None:
    engine = _engine()
    messages = _messages(count=2, width=20)
    assert engine.select_context(messages, budget_tokens=10_000) is None
    assert engine.select_context([], budget_tokens=100) is None
    assert messages == _messages(count=2, width=20)


def test_hermes_model_switch_recalculates_threshold_and_status() -> None:
    engine = _engine(context_length=100_000)
    engine.update_model("provider/new-model", 64_000, threshold_percent=0.5)
    status = engine.get_status()
    assert status["model"] == "provider/new-model"
    assert status["context_length"] == 64_000
    assert status["threshold_tokens"] == 32_000
    assert status["exact_recovery"]["lookup"] == "hash_only"
