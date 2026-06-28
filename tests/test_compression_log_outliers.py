from __future__ import annotations

from entroly.compression_proxy import compress_proxy_payload


def test_elc_does_not_hard_lock_every_timestamp_heartbeat() -> None:
    lines = []
    for i in range(1200):
        lines.append(f"2026-06-28T08:{i % 60:02d}:00Z INFO worker heartbeat shard={i % 32}")
    lines.extend(
        [
            "2026-06-28T08:44:19Z WARN api latency p99 crossed threshold service=auth",
            "2026-06-28T08:44:22Z ERROR upstream refused connection service=auth region=us-west-2",
            "2026-06-28T08:44:25Z FATAL incident INC-9281 auth unavailable",
            "rollback candidate: deploy 2026.06.28.4",
        ]
    )
    body = {
        "messages": [
            {"role": "user", "content": "which incident caused auth outage"},
            {"role": "tool", "content": "\n".join(lines)},
        ]
    }

    result = compress_proxy_payload(body, query="which incident caused auth outage", budget_tokens=500)
    compressed = result.body["messages"][1]["content"]

    assert result.changed
    assert result.receipt.savings_ratio >= 0.70
    assert "INC-9281" in compressed
    assert "upstream refused connection" in compressed
    assert "rollback candidate" in compressed
    assert "worker heartbeat shard=17" not in compressed
