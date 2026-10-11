"""Conversation compression must work on provider block content, not just str.

`compress_conversation_messages` decided a resolution per message from text
flattened out of the content, then threw that decision away: the apply step
re-read the raw `content` and skipped anything that was not a `str`. Anthropic
sends a list of typed blocks, so the stage was a complete no-op for every
Anthropic request while reporting nothing -- unlike the two stages beside it,
it logs only on failure, so a 0% outcome was indistinguishable from success.

A second gate sat behind that one. The progressive scorer keys on role and
holds every `user` turn verbatim, and Anthropic puts `tool_result` blocks
inside a `role="user"` envelope. Almost all of the bulk in a tool-calling
conversation lives there, so even a message that reached the compressor would
have been scored untouchable.

These tests pin both, and pin the structural contract that makes the fix safe:
compressing a tool call's `input` would corrupt the call rather than shorten
it, and losing `tool_use_id` would break the pairing the provider validates.
"""
from __future__ import annotations

import pytest

from entroly.proxy import compress_conversation_messages

BULK = "tool output line carrying real detail\n" * 400
WINDOW = 20_000


def _blocks(_id: str) -> list[dict]:
    return [
        {"role": "assistant", "content": [
            {"type": "text", "text": f"calling a tool for {_id}"},
            {"type": "tool_use", "id": _id, "name": "read_file",
             "input": {"path": f"/src/{_id}.py", "limit": 500}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": _id, "content": BULK},
        ]},
    ]


def _conversation() -> list[dict]:
    messages: list[dict] = []
    for index in range(12):
        messages.extend(_blocks(f"toolu_{index}"))
    messages.append({"role": "user", "content": "so what is broken?"})
    return messages


def _typed(messages: list[dict], kind: str) -> list[dict]:
    return [
        block
        for message in messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == kind
    ]


def _chars(messages: list[dict]) -> int:
    return sum(len(str(message.get("content", ""))) for message in messages)


def test_block_content_is_actually_compressed() -> None:
    messages = _conversation()
    out = compress_conversation_messages(messages, context_window=WINDOW)
    assert _chars(out) < _chars(messages) * 0.5


def test_block_and_string_content_compress_alike() -> None:
    """The content *shape* must not decide whether compression happens."""
    body = "assistant reasoning text that is long enough to matter\n" * 300
    as_string = [
        {"role": "assistant", "content": body} for _ in range(12)
    ] + [{"role": "user", "content": "and now?"}]
    as_blocks = [
        {"role": "assistant", "content": [{"type": "text", "text": body}]}
        for _ in range(12)
    ] + [{"role": "user", "content": "and now?"}]

    string_saved = 1 - _chars(
        compress_conversation_messages(as_string, context_window=WINDOW)
    ) / _chars(as_string)
    block_saved = 1 - _chars(
        compress_conversation_messages(as_blocks, context_window=WINDOW)
    ) / _chars(as_blocks)

    assert block_saved == pytest.approx(string_saved, abs=0.05)


def test_tool_call_arguments_are_never_rewritten() -> None:
    """`input` is the structured argument object the provider parses."""
    messages = _conversation()
    out = compress_conversation_messages(messages, context_window=WINDOW)

    assert [b["input"] for b in _typed(out, "tool_use")] == [
        b["input"] for b in _typed(messages, "tool_use")
    ]
    assert [b["id"] for b in _typed(out, "tool_use")] == [
        b["id"] for b in _typed(messages, "tool_use")
    ]


def test_tool_result_pairing_survives() -> None:
    """A tool_result whose tool_use_id is lost is rejected by the provider."""
    messages = _conversation()
    out = compress_conversation_messages(messages, context_window=WINDOW)

    assert [b["tool_use_id"] for b in _typed(out, "tool_result")] == [
        b["tool_use_id"] for b in _typed(messages, "tool_result")
    ]


def test_block_shape_is_preserved() -> None:
    messages = _conversation()
    out = compress_conversation_messages(messages, context_window=WINDOW)

    assert len(out) == len(messages)
    for before, after in zip(messages, out):
        assert before.get("role") == after.get("role")
        if isinstance(before.get("content"), list):
            assert isinstance(after["content"], list)
            assert len(after["content"]) == len(before["content"])
            assert all(isinstance(b, dict) and "type" in b for b in after["content"])


def test_the_live_user_turn_is_left_alone() -> None:
    messages = _conversation()
    out = compress_conversation_messages(messages, context_window=WINDOW)
    assert out[-1] == messages[-1]


def test_below_threshold_is_untouched() -> None:
    """Compression only runs above the utilization trigger."""
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert compress_conversation_messages(messages, context_window=WINDOW) == messages
