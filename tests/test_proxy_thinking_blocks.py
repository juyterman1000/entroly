"""Conversation compression must never rewrite a signed thinking block.

Anthropic signs ``thinking`` and ``redacted_thinking`` blocks and requires
them back exactly as generated: they cannot be edited or partly dropped, and a
rebuilt one is rejected with a 400. ``_compress_content_blocks`` compressed
the text of every ``thinking`` block on an assistant turn it rewrote, so a
long conversation crossing the compression threshold became a failed request.

The same stage measured utilisation against a fixed 128K window rather than
the window of the model the request goes to.
"""

from __future__ import annotations

import copy

import pytest

from entroly.proxy import PromptCompilerProxy, compress_conversation_messages
from entroly.proxy_config import ProxyConfig, context_window_for_model

try:  # the progressive scorer driving this stage is native
    import entroly_core  # noqa: F401

    _NATIVE = True
except Exception:  # pragma: no cover - taken by the base-install job
    _NATIVE = False

requires_native = pytest.mark.skipif(
    not _NATIVE, reason="needs py_progressive_thresholds from entroly_core"
)

PROSE = (
    "The assistant explains its reasoning about the failing module in detail, "
    "covering the call path, the state it mutates and the guard it bypasses.\n"
) * 60
THINKING = "Let me trace the call path step by step before editing anything.\n" * 80
SIGNATURE = "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds" * 4


def _assistant_turn(index: int) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": THINKING, "signature": SIGNATURE},
            {"type": "redacted_thinking", "data": f"redacted-{index}-" + "Zm9v" * 64},
            {"type": "text", "text": PROSE},
            {
                "type": "tool_use",
                "id": f"toolu_{index}",
                "name": "read_file",
                "input": {"path": f"/src/m{index}.py"},
            },
        ],
    }


def _conversation() -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": "fix the failing test"}]
    for index in range(10):
        messages.append(_assistant_turn(index))
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"toolu_{index}",
                        "content": "line of tool output\n" * 50,
                    }
                ],
            }
        )
    messages.append({"role": "user", "content": "so what is broken?"})
    return messages


def _blocks(messages: list[dict], kind: str) -> list[dict]:
    return [
        block
        for message in messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == kind
    ]


@requires_native
def test_thinking_blocks_survive_while_neighbouring_text_is_compressed() -> None:
    original = _conversation()
    # A small window drives utilisation past 95%, the tier that rewrites old
    # assistant turns -- the only tier that reached thinking blocks.
    result = compress_conversation_messages(copy.deepcopy(original), context_window=10_000)

    assert _blocks(result, "thinking") == _blocks(original, "thinking")
    assert _blocks(result, "redacted_thinking") == _blocks(original, "redacted_thinking")
    # The stage really ran on those turns: their text blocks got shorter.
    before = sum(len(block["text"]) for block in _blocks(original, "text"))
    after = sum(len(block["text"]) for block in _blocks(result, "text"))
    assert after < before
    # Block order inside each assistant turn is unchanged, which Anthropic
    # also validates for thinking blocks.
    for got, want in zip(result, original):
        if isinstance(want.get("content"), list):
            assert [b.get("type") for b in got["content"]] == [
                b.get("type") for b in want["content"]
            ]


def test_thinking_blocks_survive_inside_tool_results() -> None:
    from entroly.proxy import _compress_content_blocks

    signed = {"type": "thinking", "thinking": THINKING, "signature": SIGNATURE}
    content = [
        {"type": "tool_result", "tool_use_id": "toolu_x", "content": [copy.deepcopy(signed)]},
        copy.deepcopy(signed),
    ]
    result = _compress_content_blocks(content, lambda text: text[:10])

    assert result[1] == signed
    assert result[0]["content"][0] == signed


@pytest.mark.parametrize(
    ("body", "path", "expected"),
    [
        ({"model": "gpt-4o"}, "/v1/chat/completions", context_window_for_model("gpt-4o")),
        (
            {"model": "claude-sonnet-4-5-20250929"},
            "/v1/messages",
            context_window_for_model("claude-sonnet-4-5-20250929"),
        ),
        ({}, "/v1beta/models/gemini-2.5-pro:generateContent", context_window_for_model("gemini-2.5-pro")),
        ({}, "/v1/messages", 64_000),
    ],
    ids=["openai", "anthropic", "gemini-path", "no-model-uses-config"],
)
def test_utilisation_is_measured_against_the_requests_model(body, path, expected) -> None:
    proxy = PromptCompilerProxy(object(), ProxyConfig(context_window=64_000))
    assert proxy._conversation_context_window(body, path) == expected


def test_sonnet_4_5_is_no_longer_sized_at_128k() -> None:
    proxy = PromptCompilerProxy(object(), ProxyConfig())
    window = proxy._conversation_context_window(
        {"model": "claude-sonnet-4-5-20250929"}, "/v1/messages"
    )
    assert window == 200_000
