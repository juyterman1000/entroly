from __future__ import annotations

from entroly.compression_proxy import compress_proxy_payload, compress_proxy_payload_from_env
from entroly.compression_retrieval_store import CompressionRetrievalStore


def test_retrieval_store_saves_and_fetches_omitted_spans(tmp_path) -> None:
    store_path = tmp_path / "compression-store.json"
    store = CompressionRetrievalStore(store_path)
    heavy = "\n".join([f"background line {i}" for i in range(500)] + ["ERROR final failure line"])
    body = {
        "messages": [
            {"role": "user", "content": "why did it fail?"},
            {"role": "tool", "content": heavy},
        ]
    }

    result = compress_proxy_payload(
        body,
        query="final failure",
        budget_tokens=120,
        retrieval_store=store,
    )

    retrieval = result.receipt.receipts[0]["retrieval"]
    receipt_id = retrieval["receipt_id"]
    span_id = retrieval["span_ids"][0]
    span = store.get_span(receipt_id, span_id)

    assert result.changed
    assert retrieval["span_count"] >= 1
    assert span is not None
    assert "background line" in span.content

    restored = CompressionRetrievalStore(store_path)
    restored_span = restored.get_span(receipt_id, span_id)
    assert restored_span is not None
    assert restored_span.content == span.content


def test_retrieval_store_searches_omitted_spans() -> None:
    store = CompressionRetrievalStore()
    heavy = "\n".join(["payment worker heartbeat" for _ in range(400)] + ["ERROR auth outage INC-777"])
    body = {"messages": [{"role": "tool", "content": heavy}]}

    result = compress_proxy_payload(
        body,
        query="auth outage",
        budget_tokens=120,
        retrieval_store=store,
    )

    assert result.changed
    matches = store.search("payment worker")
    assert matches
    assert "payment worker" in matches[0].content


def test_env_proxy_mode_uses_store_when_enabled(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "store.json"
    monkeypatch.setenv("ENTROLY_COMPRESSION_PROXY_MODE", "elc")
    monkeypatch.setenv("ENTROLY_ELC_BUDGET_TOKENS", "120")
    monkeypatch.setenv("ENTROLY_COMPRESSION_STORE", str(store_path))
    heavy = "\n".join(["compile ok" for _ in range(400)] + ["ERROR link failure"])
    body = {"messages": [{"role": "tool", "content": heavy}]}

    result = compress_proxy_payload_from_env(body, query="link failure")

    assert result.changed
    assert store_path.exists()
    assert result.receipt.receipts[0]["retrieval"]["span_count"] >= 1


def test_env_proxy_mode_defaults_to_off(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_COMPRESSION_PROXY_MODE", raising=False)
    heavy = "\n".join(["compile ok" for _ in range(400)] + ["ERROR link failure"])
    body = {"messages": [{"role": "tool", "content": heavy}]}

    result = compress_proxy_payload_from_env(body, query="link failure")

    assert result.changed is False
    assert result.body == body
