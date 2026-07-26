from __future__ import annotations

from entroly.proxy_transform import inject_context_live_zone


def test_openai_live_zone_leaves_system_and_raw_turn_prefix_untouched() -> None:
    body = {
        "model": "gpt-5",
        "messages": [
            {"role": "system", "content": "stable policy"},
            {"role": "user", "content": "inspect auth"},
        ],
    }

    updated, injected = inject_context_live_zone(body, "def authenticate(): ...", "openai")

    assert injected
    assert updated["messages"][0] == body["messages"][0]
    assert updated["messages"][1]["content"].startswith("inspect auth")
    assert "<entroly_dynamic_context>" in updated["messages"][1]["content"]
    assert body["messages"][1]["content"] == "inspect auth"


def test_anthropic_live_zone_preserves_system_and_tool_result_shape() -> None:
    body = {
        "model": "claude-sonnet-4-6",
        "system": [{"type": "text", "text": "stable policy"}],
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1"}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "raw output",
                    }
                ],
            },
        ],
    }

    updated, injected = inject_context_live_zone(body, "relevant source", "anthropic")

    assert injected
    assert updated["system"] == body["system"]
    assert updated["messages"][1]["content"][0] == body["messages"][1]["content"][0]
    suffix = updated["messages"][1]["content"][1]
    assert suffix["type"] == "text"
    assert "<entroly_dynamic_context>" in suffix["text"]


def test_responses_live_zone_appends_after_function_output() -> None:
    body = {
        "model": "gpt-5",
        "instructions": "stable policy",
        "input": [
            {"type": "function_call_output", "call_id": "c1", "output": "raw"}
        ],
    }

    updated, injected = inject_context_live_zone(body, "source context", "openai")

    assert injected
    assert updated["instructions"] == "stable policy"
    assert updated["input"][0]["output"].startswith("raw")
    assert "<entroly_dynamic_context>" in updated["input"][0]["output"]


def test_gemini_live_zone_does_not_rewrite_system_instruction() -> None:
    body = {
        "systemInstruction": {"parts": [{"text": "stable policy"}]},
        "contents": [{"role": "user", "parts": [{"text": "raw query"}]}],
    }

    updated, injected = inject_context_live_zone(body, "source context", "gemini")

    assert injected
    assert updated["systemInstruction"] == body["systemInstruction"]
    assert updated["contents"][0]["parts"][0]["text"] == "raw query"
    assert "<entroly_dynamic_context>" in updated["contents"][0]["parts"][1]["text"]


def test_live_zone_refuses_ambiguous_assistant_only_shape() -> None:
    body = {"messages": [{"role": "assistant", "content": "waiting"}]}

    updated, injected = inject_context_live_zone(body, "context", "anthropic")

    assert not injected
    assert updated == body
