from __future__ import annotations

from entroly.compression_retrieval_store import CompressionRetrievalStore
from entroly.optimization_ledger import (
    OptimizationAdjustment,
    OptimizationEvent,
    OptimizationLedger,
    SavingsTier,
)


def test_ledger_keeps_realized_estimated_and_opportunity_separate(tmp_path) -> None:
    ledger = OptimizationLedger(tmp_path / "optimization.sqlite3")
    measured = OptimizationEvent("m1", "compression", SavingsTier.MEASURED, 1000)
    assert ledger.record(measured)
    assert not ledger.record(measured)
    ledger.record(OptimizationEvent("e1", "forecast", SavingsTier.ESTIMATED, 400))
    ledger.record(OptimizationEvent("o1", "loop", SavingsTier.OPPORTUNITY, 250))
    adjustment = OptimizationAdjustment("r1", "m1", 125)
    assert ledger.adjust(adjustment)
    assert not ledger.adjust(adjustment)

    summary = ledger.summary()
    assert summary.measured_gross_tokens == 1000
    assert summary.measured_reexpanded_tokens == 125
    assert summary.measured_net_tokens == 875
    assert summary.estimated_tokens == 400
    assert summary.opportunity_tokens == 250


def test_retrieval_debits_measured_compression_savings(tmp_path) -> None:
    ledger = OptimizationLedger(tmp_path / "optimization.sqlite3")
    store = CompressionRetrievalStore(
        tmp_path / "compression.json",
        optimization_ledger=ledger,
    )
    original = "\n".join(f"line {index} payload" for index in range(50))
    stored = store.put(
        original_text=original,
        compressed_text="summary",
        receipt={
            "original_tokens": 300,
            "compressed_tokens": 50,
            "omitted_spans": [{"start_line": 1, "end_line": 10}],
        },
    )
    span = stored.spans[0]
    inspected = store.get_span(stored.receipt_id, span.span_id)
    assert inspected is not None
    assert store.savings_summary()["retrieved_tokens"] == 0

    returned = store.retrieve_span(
        stored.receipt_id,
        span.span_id,
        retrieval_id="request-1",
    )
    assert returned is not None
    summary = store.savings_summary()
    assert summary["retrieved_tokens"] > 0
    event = ledger.event(f"compression:{stored.receipt_id}")
    assert event is not None
    assert event["net_tokens_saved"] == 250 - summary["retrieved_tokens"]
    store.retrieve_span(stored.receipt_id, span.span_id, retrieval_id="request-1")
    assert store.savings_summary() == summary
    assert "retrieval_ids" not in returned.as_dict()

    reloaded = CompressionRetrievalStore(
        tmp_path / "compression.json",
        optimization_ledger=ledger,
    )
    assert reloaded.savings_summary() == summary
    assert ledger.event(f"compression:{stored.receipt_id}")["net_tokens_saved"] == (
        250 - summary["retrieved_tokens"]
    )


def test_old_retrieval_store_schema_loads_with_zero_retrievals(tmp_path) -> None:
    path = tmp_path / "compression.json"
    path.write_text(
        '{"items":[{"receipt_id":"r","original_hash":"h",'
        '"original_tokens":20,"compressed_tokens":5,"created_ns":1,'
        '"metadata":{},"spans":[{"span_id":"s","receipt_id":"r",'
        '"start_line":1,"end_line":1,"content":"old","reason":"budget",'
        '"created_ns":1}]}]}',
        encoding="utf-8",
    )
    store = CompressionRetrievalStore(path)
    assert store.list_receipts()[0]["net_tokens_saved"] == 15
    assert store.get_span("r", "s").retrieval_count == 0
