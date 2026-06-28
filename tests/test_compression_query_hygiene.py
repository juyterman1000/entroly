from __future__ import annotations

from entroly.compression_proxy import compress_proxy_payload


def test_query_hygiene_does_not_lock_every_build_line() -> None:
    lines = [f"[build] compiling module_{i} ... ok" for i in range(1200)]
    lines.extend(
        [
            "src/auth/session.rs:184:9",
            "ERROR: refresh timeout after retry window",
            "hint: increase token refresh slack before retry",
            "Build finished with exit code 1",
        ]
    )
    body = {
        "messages": [
            {"role": "user", "content": "why did the auth build fail"},
            {"role": "tool", "content": "\n".join(lines)},
        ]
    }

    result = compress_proxy_payload(body, query="why did the auth build fail", budget_tokens=500)
    compressed = result.body["messages"][1]["content"]

    assert result.changed
    assert result.receipt.savings_ratio >= 0.70
    assert "src/auth/session.rs:184" in compressed
    assert "ERROR: refresh timeout" in compressed
    assert "increase token refresh slack" in compressed
    assert "compiling module_200" not in compressed
