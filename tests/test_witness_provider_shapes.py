from __future__ import annotations

import json

from entroly.proxy import PromptCompilerProxy, _extract_text_from_sse
from entroly.proxy_config import ProxyConfig


def _proxy() -> PromptCompilerProxy:
    return PromptCompilerProxy(object(), ProxyConfig(witness_mode="strict"))


def test_witness_openai_non_streaming_shape() -> None:
    proxy = _proxy()
    content = {
        "choices": [
            {"message": {"role": "assistant", "content": "Paris is the capital of Germany."}}
        ]
    }

    updated, headers = proxy._apply_witness_gateway(content, "The context mentions Berlin.")

    assert headers["X-Entroly-Witness"] == "flagged"
    assert updated["choices"][0]["message"]["content"] != "Paris is the capital of Germany."


def test_witness_openai_tool_call_response_is_left_alone() -> None:
    proxy = _proxy()
    content = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": "{}"}}],
                }
            }
        ]
    }

    updated, headers = proxy._apply_witness_gateway(content, "The context mentions Berlin.")

    assert updated == content
    assert headers["X-Entroly-Witness"] == "no-text"


def test_witness_openai_structured_output_shape_stays_valid_json() -> None:
    proxy = _proxy()
    structured = json.dumps({"answer": "Paris is the capital of Germany.", "confidence": 0.9})
    content = {"choices": [{"message": {"role": "assistant", "content": structured}}]}

    updated, headers = proxy._apply_witness_gateway(content, "The context mentions Berlin.")

    assert json.loads(updated["choices"][0]["message"]["content"])["answer"] == "Paris is the capital of Germany."
    assert headers["X-Entroly-Witness-Rewrite-Skipped"] == "structured-output"


def test_witness_anthropic_non_streaming_shape() -> None:
    proxy = _proxy()
    content = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Paris is the capital of Germany."}],
    }

    updated, headers = proxy._apply_witness_gateway(content, "The context mentions Berlin.")

    assert headers["X-Entroly-Witness"] == "flagged"
    assert updated["content"][0]["text"] != "Paris is the capital of Germany."


def test_witness_gemini_non_streaming_shape() -> None:
    proxy = _proxy()
    content = {
        "candidates": [
            {"content": {"parts": [{"text": "Paris is the capital of Germany."}]}}
        ]
    }

    updated, headers = proxy._apply_witness_gateway(content, "The context mentions Berlin.")

    assert headers["X-Entroly-Witness"] == "flagged"
    assert updated["candidates"][0]["content"]["parts"][0]["text"] != "Paris is the capital of Germany."


def test_witness_streaming_shapes_are_parseable_for_each_provider() -> None:
    proxy = _proxy()
    for provider in ("openai", "anthropic", "gemini"):
        sse = proxy._format_witness_sse(provider, {"model": "test-model"}, "Verified text.")
        assert _extract_text_from_sse(sse) == "Verified text."

