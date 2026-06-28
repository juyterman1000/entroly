from __future__ import annotations

from entroly.compression_proxy import compress_proxy_payload


def test_compression_proxy_compresses_openai_tool_messages() -> None:
    heavy = "\n".join(["compile dependency ok" for _ in range(400)] + ["ERROR final link failed in src/main.rs:91"])
    body = {
        "model": "gpt-test",
        "messages": [
            {"role": "user", "content": "why did the build fail?"},
            {"role": "tool", "content": heavy},
        ],
    }

    result = compress_proxy_payload(
        body,
        provider="openai",
        query="why did the build fail",
        budget_tokens=160,
    )

    assert result.changed
    assert result.receipt.compressed_blocks == 1
    assert result.receipt.tokens_saved > 0
    assert result.receipt.savings_ratio > 0.5
    assert result.body["messages"][0] == body["messages"][0]
    assert "entroly-elc" in result.body["messages"][1]["content"]
    assert "ERROR final link failed" in result.body["messages"][1]["content"]
    assert result.headers()["x-entroly-compression-mode"] == "elc"


def test_compression_proxy_compresses_anthropic_tool_result_blocks() -> None:
    heavy = "\n".join(["pytest test_login ok" for _ in range(300)] + ["FAILED tests/test_auth.py::test_refresh_timeout"])
    body = {
        "model": "claude-test",
        "messages": [
            {"role": "user", "content": "why did auth fail?"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": heavy}
                ],
            },
        ],
    }

    result = compress_proxy_payload(
        body,
        provider="anthropic",
        query="auth refresh timeout",
        budget_tokens=160,
    )

    compressed = result.body["messages"][1]["content"][0]["content"]
    assert result.changed
    assert "entroly-elc" in compressed
    assert "FAILED tests/test_auth.py::test_refresh_timeout" in compressed
    assert result.receipt.receipts[0]["recoverable"] is True


def test_compression_proxy_preserves_user_messages_by_default() -> None:
    huge_user = "\n".join(["background detail" for _ in range(500)] + ["ERROR user pasted failure"])
    body = {"messages": [{"role": "user", "content": huge_user}]}

    result = compress_proxy_payload(body, query="failure", budget_tokens=120)

    assert result.changed is False
    assert result.body == body
    assert result.receipt.compressed_blocks == 0


def test_compression_proxy_can_opt_in_to_user_message_compression() -> None:
    huge_user = "\n".join(["background detail" for _ in range(500)] + ["ERROR user pasted failure"])
    body = {"messages": [{"role": "user", "content": huge_user}]}

    result = compress_proxy_payload(
        body,
        query="failure",
        budget_tokens=120,
        compress_user_messages=True,
    )

    assert result.changed
    assert "ERROR user pasted failure" in result.body["messages"][0]["content"]


def test_compression_proxy_off_mode_is_passthrough() -> None:
    body = {"messages": [{"role": "tool", "content": "x" * 10000}]}

    result = compress_proxy_payload(body, mode="off")

    assert result.changed is False
    assert result.body == body
    assert result.receipt.tokens_saved == 0
