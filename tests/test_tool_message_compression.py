from __future__ import annotations

import json

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


def test_auto_policy_preserves_parseable_json_tool_payloads():
    content = json.dumps({
        "rows": [
            {"id": i, "payload": "x" * 80}
            for i in range(80)
        ]
    })
    messages = [{"role": "tool", "name": "json_query", "content": content}]

    out, saved = compress_tool_messages(messages, policy="auto")

    assert saved == 0
    assert out == messages


def test_excluded_tool_name_preserves_exact_output_even_in_compress_policy():
    content = _large_tool_output()
    messages = [{"role": "tool", "name": "bash", "content": content}]

    out, saved = compress_tool_messages(
        messages,
        policy="compress",
        excluded_tools="bash",
    )

    assert saved == 0
    assert out == messages


# ── short shell output reaches the agent exactly as printed ─────────────


def _ls_long(files: int = 30) -> str:
    return "".join(
        f"-rw-r--r--  1 user staff  {1000 + i * 37:6d} Oct 31 10:{i % 60:02d} file_{i:03d}.py\n"
        for i in range(files)
    )


def _anthropic_call(name: str, output: str) -> list[dict]:
    return [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_ls", "name": name, "input": {"command": "ls -l"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_ls", "content": output}]},
    ]


def test_short_anthropic_shell_output_is_forwarded_verbatim():
    # Before: sizes rounded to "1KB", modes and timestamps dropped.
    messages = _anthropic_call("Bash", _ls_long())

    out, saved = compress_tool_messages(messages, policy="compress")

    assert saved == 0
    assert out == messages


def test_short_openai_shell_output_is_forwarded_verbatim():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "shell", "arguments": "{\"cmd\": \"ls -l\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": _ls_long()},
    ]

    out, saved = compress_tool_messages(messages, policy="compress")

    assert saved == 0
    assert out == messages


def test_short_output_of_other_tools_is_still_compressed():
    messages = _anthropic_call("list_files", _ls_long())

    out, saved = compress_tool_messages(messages, policy="compress")

    assert saved > 0


def test_long_shell_output_is_still_compressed():
    messages = _anthropic_call("Bash", _ls_long(200))

    out, saved = compress_tool_messages(messages, policy="compress")

    assert saved > 0


def test_shell_floor_is_configurable(monkeypatch):
    monkeypatch.setenv("ENTROLY_SHELL_VERBATIM_CHARS", "0")
    _, saved = compress_tool_messages(_anthropic_call("Bash", _ls_long()), policy="compress")
    assert saved > 0

    monkeypatch.setenv("ENTROLY_SHELL_VERBATIM_CHARS", "4000")
    monkeypatch.setenv("ENTROLY_SHELL_TOOLS", "my_runner")
    messages = _anthropic_call("my_runner", _ls_long())
    out, saved = compress_tool_messages(messages, policy="compress")
    assert saved == 0 and out == messages


def test_excluded_tool_name_now_matches_an_anthropic_tool_result():
    # The result block names only its tool_use_id, so a name in
    # excluded_tools never matched Anthropic traffic before.
    messages = _anthropic_call("db_query", _large_tool_output())

    out, saved = compress_tool_messages(messages, policy="compress", excluded_tools="db_query")

    assert saved == 0
    assert out == messages
