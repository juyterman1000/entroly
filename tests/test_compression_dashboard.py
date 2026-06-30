from __future__ import annotations

from entroly.compression_dashboard import dashboard_from_proxy_receipts, dashboard_from_store
from entroly.compression_proxy import compress_proxy_payload
from entroly.compression_retrieval_store import CompressionRetrievalStore


def test_dashboard_from_proxy_receipts() -> None:
    receipts = [
        {
            "original_tokens": 1000,
            "compressed_tokens": 250,
            "compression_level": 3,
            "omitted_spans": [{"start_line": 1, "end_line": 10, "line_count": 10, "reason": "budget"}],
            "retrieval": {"receipt_id": "r1", "span_count": 1, "span_ids": ["s1"]},
        }
    ]

    dashboard = dashboard_from_proxy_receipts(receipts)

    assert dashboard.tokens_saved == 750
    assert dashboard.savings_ratio == 0.75
    assert dashboard.compressed_blocks == 1
    assert dashboard.recoverable_spans >= 1
    assert "Tokens saved" in dashboard.to_markdown()


def test_dashboard_from_store(tmp_path) -> None:
    store = CompressionRetrievalStore(tmp_path / "store.json")
    heavy = "\n".join(["background" for _ in range(300)] + ["ERROR final failure"])
    body = {"messages": [{"role": "tool", "content": heavy}]}

    result = compress_proxy_payload(body, query="final failure", budget_tokens=120, retrieval_store=store)
    dashboard = dashboard_from_store(store)

    assert result.changed
    assert dashboard.retrieval_receipts == 1
    assert dashboard.recoverable_spans >= 1
    assert dashboard.tokens_saved > 0
