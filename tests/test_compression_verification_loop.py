from __future__ import annotations

from entroly.compression_proxy import compress_proxy_payload
from entroly.compression_retrieval_store import CompressionRetrievalStore
from entroly.compression_verification_loop import answer_with_retrieval_verification


def test_verification_loop_retrieves_and_retries(tmp_path) -> None:
    store = CompressionRetrievalStore(tmp_path / "store.json")
    heavy = "\n".join(["payment worker heartbeat" for _ in range(300)] + ["ERROR auth outage INC-4242"])
    body = {
        "messages": [
            {"role": "user", "content": "which incident caused auth outage?"},
            {"role": "tool", "content": heavy},
        ]
    }
    compressed = compress_proxy_payload(
        body,
        query="auth outage incident",
        budget_tokens=120,
        retrieval_store=store,
    )

    calls = {"count": 0}

    def model_call(messages):
        calls["count"] += 1
        joined = "\n".join(str(m.get("content", "")) for m in messages)
        if "Recovered compressed span" in joined and "INC-4242" in joined:
            return "The auth outage was caused by incident INC-4242."
        return "Cannot determine from the compressed context."

    result = answer_with_retrieval_verification(
        compressed.body["messages"],
        model_call=model_call,
        retrieval_store=store,
        receipts=compressed.receipt.receipts,
        query="auth outage incident",
    )

    assert result.retried
    assert calls["count"] == 2
    assert "INC-4242" in result.answer
    assert result.retrieved_spans


def test_verification_loop_no_retry_when_answer_supported() -> None:
    store = CompressionRetrievalStore()

    def model_call(_messages):
        return "The answer is clear."

    result = answer_with_retrieval_verification(
        [{"role": "user", "content": "hello"}],
        model_call=model_call,
        retrieval_store=store,
        receipts=[],
    )

    assert result.retried is False
    assert result.answer == "The answer is clear."
