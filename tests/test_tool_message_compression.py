from __future__ import annotations

from entroly.proxy_transform import compress_tool_messages


def _large_tool_output() -> str:
    return "\n".join(
        f"line {i}: {'x' * 80}"
        for i in range(200)
    )


def test_recent_anthropic_tool_result_string_block_is_compressed():
    content = _large_tool_output()
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will inspect the logs."},
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": content},
            ],
        }
    ]

    out, saved = compress_tool_messages(messages)

    blocks = out[0]["content"]
    assert saved > 0
    assert blocks[0] == messages[0]["content"][0]
    assert len(blocks[1]["content"]) < len(content) * 0.2
    assert messages[0]["content"][1]["content"] == content


def test_recent_anthropic_tool_result_text_items_are_compressed():
    content = _large_tool_output()
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_2",
                    "content": [{"type": "text", "text": content}],
                },
            ],
        }
    ]

    out, saved = compress_tool_messages(messages)

    inner = out[0]["content"][0]["content"][0]
    assert saved > 0
    assert inner["type"] == "text"
    assert len(inner["text"]) < len(content) * 0.2


def test_non_tool_blocks_are_not_rewritten():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": _large_tool_output()},
            ],
        }
    ]

    out, saved = compress_tool_messages(messages)

    assert saved == 0
    assert out == messages
