from __future__ import annotations

import pytest

from entroly.provider_adapters import (
    apply_target_same_provider,
    canonical_request_from_provider_body,
    render_canonical_request,
    rewrite_gemini_model_in_url,
)
from entroly.provider_policy import Capability, ProviderTarget


FULL_CAPS = frozenset(
    {
        Capability.CHAT,
        Capability.STREAMING,
        Capability.TOOLS,
        Capability.JSON_SCHEMA,
        Capability.VISION,
        Capability.REASONING,
    }
)


def _target(provider: str, model: str) -> ProviderTarget:
    return ProviderTarget(provider=provider, model=model, capabilities=FULL_CAPS)


def test_openai_chat_canonicalization_detects_tools_schema_vision_and_reasoning() -> None:
    body = {
        "model": "gpt-4.1",
        "stream": True,
        "reasoning_effort": "medium",
        "max_completion_tokens": 77,
        "messages": [
            {"role": "system", "content": "stable policy"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                ],
            },
        ],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer"}},
    }

    result = canonical_request_from_provider_body(
        "openai",
        body,
        headers={"x-request-id": "req-1", "authorization": "Bearer secret"},
    )

    assert result.canonical.model == "gpt-4.1"
    assert result.canonical.stream is True
    assert result.canonical.tools
    assert result.canonical.response_schema == body["response_format"]
    assert result.canonical.requires_vision is True
    assert result.canonical.requires_reasoning is True
    assert result.expected_output_tokens == 77
    assert result.new_input_tokens_estimate > 0
    assert result.prefix_tokens_estimate >= 0
    assert result.canonical.metadata == {"x-request-id": "req-1"}


def test_openai_responses_api_is_canonicalized_without_losing_instructions() -> None:
    result = canonical_request_from_provider_body(
        "openai",
        {
            "model": "gpt-4.1-mini",
            "instructions": "answer as JSON",
            "input": [
                {"role": "user", "content": "summarize"},
                "include action items",
            ],
        },
    )

    assert [m["role"] for m in result.canonical.messages] == ["system", "user", "user"]
    assert result.canonical.messages[0]["content"] == "answer as JSON"
    assert result.canonical.messages[-1]["content"] == "include action items"


def test_anthropic_system_field_becomes_stable_system_message() -> None:
    result = canonical_request_from_provider_body(
        "anthropic",
        {
            "model": "claude-sonnet-4-5",
            "system": "stable enterprise policy",
            "messages": [{"role": "user", "content": "debug"}],
            "max_tokens": 123,
        },
    )

    assert result.canonical.messages[0] == {
        "role": "system",
        "content": "stable enterprise policy",
    }
    assert result.canonical.messages[1]["role"] == "user"
    assert result.expected_output_tokens == 123


def test_gemini_path_model_and_streaming_are_canonicalized() -> None:
    result = canonical_request_from_provider_body(
        "gemini",
        {
            "systemInstruction": {"parts": [{"text": "stable"}]},
            "contents": [
                {"role": "user", "parts": [{"text": "hello"}]},
                {"role": "model", "parts": [{"text": "hi"}]},
            ],
            "generationConfig": {
                "maxOutputTokens": 321,
                "responseSchema": {"type": "object"},
            },
        },
        path="/v1beta/models/gemini-2.0-flash:streamGenerateContent",
    )

    assert result.canonical.model == "gemini-2.0-flash"
    assert result.canonical.stream is True
    assert result.canonical.messages[0]["role"] == "system"
    assert result.canonical.messages[2]["role"] == "assistant"
    assert result.canonical.response_schema == {"type": "object"}
    assert result.expected_output_tokens == 321


def test_same_provider_openai_rewrite_preserves_provider_specific_controls() -> None:
    body, url = apply_target_same_provider(
        provider="openai",
        target=_target("openai", "gpt-4.1-mini"),
        body={"model": "gpt-4.1", "messages": [], "temperature": 0.2},
        url="https://api.openai.com/v1/chat/completions",
    )

    assert body == {"model": "gpt-4.1-mini", "messages": [], "temperature": 0.2}
    assert url == "https://api.openai.com/v1/chat/completions"


def test_same_provider_gemini_rewrite_validates_url_embedded_model() -> None:
    body, url = apply_target_same_provider(
        provider="gemini",
        target=_target("gemini", "gemini-2.0-flash"),
        body={"contents": []},
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
    )

    assert body == {"contents": []}
    assert "/models/gemini-2.0-flash:generateContent" in url
    with pytest.raises(ValueError):
        rewrite_gemini_model_in_url(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
            "../bad",
        )


def test_cross_provider_same_body_rewrite_is_rejected() -> None:
    with pytest.raises(ValueError):
        apply_target_same_provider(
            provider="openai",
            target=_target("anthropic", "claude-haiku"),
            body={"model": "gpt-4.1", "messages": []},
            url="https://api.openai.com/v1/chat/completions",
        )


def test_render_canonical_request_to_anthropic_uses_only_portable_fields() -> None:
    canonical = canonical_request_from_provider_body(
        "openai",
        {
            "model": "gpt-4.1",
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "question"},
            ],
            "tools": [{"name": "lookup"}],
            "temperature": 0.99,
        },
    ).canonical

    rendered = render_canonical_request(canonical, _target("anthropic", "claude-haiku"))

    assert rendered == {
        "model": "claude-haiku",
        "messages": [{"role": "user", "content": "question"}],
        "system": "policy",
        "tools": [{"name": "lookup"}],
    }
    assert "temperature" not in rendered


def test_render_canonical_request_to_gemini_maps_assistant_to_model() -> None:
    canonical = canonical_request_from_provider_body(
        "openai",
        {
            "model": "gpt-4.1",
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "next"},
            ],
        },
    ).canonical

    rendered = render_canonical_request(canonical, _target("gemini", "gemini-2.0-flash"))

    assert rendered["systemInstruction"] == {"parts": [{"text": "policy"}]}
    assert rendered["contents"][0]["role"] == "model"
    assert rendered["contents"][1]["role"] == "user"


def test_canonicalization_fails_without_model() -> None:
    with pytest.raises(ValueError):
        canonical_request_from_provider_body("openai", {"messages": []})


def test_unsupported_provider_is_rejected() -> None:
    with pytest.raises(ValueError):
        canonical_request_from_provider_body("unknown", {"model": "x"})
