"""Tests for the canonical token-counting and message-trimming module."""

from __future__ import annotations

from entroly.tokens import count_tokens, estimate_tokens, trim_messages


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_nonempty():
    result = count_tokens("hello world")
    assert result >= 1


def test_estimate_tokens_delegates_to_count():
    assert estimate_tokens("hello") == count_tokens("hello")


def test_trim_messages_empty():
    assert trim_messages([], max_tokens=100) == []


def test_trim_messages_fits():
    msgs = [{"role": "user", "content": "hi"}]
    result = trim_messages(msgs, max_tokens=1000)
    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_trim_messages_preserves_system():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    result = trim_messages(msgs, max_tokens=50, strategy="last")
    assert result[0]["role"] == "system"


def test_trim_messages_last_strategy():
    msgs = [
        {"role": "user", "content": "A " * 200},
        {"role": "assistant", "content": "B " * 200},
        {"role": "user", "content": "C"},
    ]
    result = trim_messages(msgs, max_tokens=50, strategy="last")
    assert len(result) < len(msgs)
    assert result[-1]["content"] == "C"


def test_trim_messages_first_strategy():
    msgs = [
        {"role": "user", "content": "C"},
        {"role": "assistant", "content": "B " * 200},
        {"role": "user", "content": "A " * 200},
    ]
    result = trim_messages(msgs, max_tokens=50, strategy="first")
    assert len(result) < len(msgs)
    assert result[0]["content"] == "C"


def test_trim_messages_invalid_strategy():
    import pytest

    with pytest.raises(ValueError, match="Unknown trim strategy"):
        trim_messages([{"role": "user", "content": "hi"}], max_tokens=100, strategy="middle")


def test_trim_messages_system_excluded():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    result = trim_messages(msgs, max_tokens=100, include_system=False)
    assert all(m["role"] != "system" or m in result for m in msgs)


def test_trim_messages_custom_counter():
    msgs = [
        {"role": "user", "content": "hello world"},
    ]
    result = trim_messages(
        msgs, max_tokens=5, token_counter=lambda t: len(t.split())
    )
    assert len(result) <= 1


def test_trim_messages_creates_digest_stubs():
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Old message 1 " * 100},
        {"role": "assistant", "content": "Old response 1 " * 100},
        {"role": "user", "content": "Recent message"},
    ]
    # Without stubs
    result_no_stubs = trim_messages(msgs, max_tokens=100, strategy="last", create_stubs=False)
    assert len(result_no_stubs) < len(msgs)
    assert not any("ENTROLY CONTEXT COMPACTION" in m["content"] for m in result_no_stubs)

    # With stubs
    result_with_stubs = trim_messages(msgs, max_tokens=100, strategy="last", create_stubs=True)
    assert len(result_with_stubs) >= 2
    # System message is preserved
    assert result_with_stubs[0]["role"] == "system"
    assert result_with_stubs[0]["content"] == "You are a helpful assistant."
    # Stub is present
    stub_msg = result_with_stubs[1]
    assert "ENTROLY CONTEXT COMPACTION" in stub_msg["content"]
    assert "sha256:" in stub_msg["content"]
    assert "Recoverable via:" not in stub_msg["content"]
    # Recent message is preserved
    assert result_with_stubs[-1]["content"] == "Recent message"
    from entroly.tokens import count_messages_tokens

    assert count_messages_tokens(result_with_stubs) <= 100


def test_trim_messages_does_not_drop_oversized_latest_request():
    msgs = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "latest task " * 400},
    ]
    assert trim_messages(msgs, max_tokens=30, create_stubs=True) == msgs

