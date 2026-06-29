from __future__ import annotations

from dataclasses import dataclass

from entroly.compression_retrieval_store import CompressionRetrievalStore
from entroly.session_intelligence import (
    BehavioralWasteDetector,
    CacheRetentionForecaster,
    CheckpointRelevanceScorer,
    RealizedSavingsRecord,
    SavingsConfidence,
    SavingsTierLedger,
    extract_decision_digest,
)


def test_realized_savings_nets_retrieval_and_repeated_expansion() -> None:
    record = RealizedSavingsRecord(
        receipt_id="r1",
        original_tokens=1000,
        compressed_tokens=300,
        retrieved_tokens=120,
        repeated_expansion_tokens=80,
        confidence=SavingsConfidence.MEASURED,
    )

    assert record.gross_saved_tokens == 700
    assert record.net_realized_saved_tokens == 500
    assert record.as_dict()["confidence"] == "measured"


def test_savings_tier_ledger_keeps_measured_estimated_opportunity_separate() -> None:
    ledger = SavingsTierLedger()
    ledger.add(RealizedSavingsRecord("measured", 100, 40, confidence=SavingsConfidence.MEASURED))
    ledger.add(RealizedSavingsRecord("estimated", 100, 50, confidence=SavingsConfidence.ESTIMATED))
    ledger.add(RealizedSavingsRecord("opportunity", 100, 70, confidence=SavingsConfidence.OPPORTUNITY))

    summary = ledger.summary()

    assert summary["records"] == 3
    assert summary["by_confidence"]["measured"]["net_realized_saved_tokens"] == 60
    assert summary["by_confidence"]["estimated"]["net_realized_saved_tokens"] == 50
    assert summary["by_confidence"]["opportunity"]["net_realized_saved_tokens"] == 30


def test_retrieval_store_records_net_realized_savings(tmp_path) -> None:
    store_path = tmp_path / "compression-store.json"
    store = CompressionRetrievalStore(store_path)
    original = "\n".join(f"line {i} important evidence" for i in range(1, 11))
    stored = store.put(
        original_text=original,
        compressed_text="line 1 important evidence\n[omitted]",
        receipt={
            "original_tokens": 100,
            "compressed_tokens": 20,
            "omitted_spans": [{"start_line": 3, "end_line": 5, "reason": "budget"}],
        },
    )
    span_id = stored.spans[0].span_id

    span = store.get_span(stored.receipt_id, span_id)
    assert span is not None
    assert span.retrieval_count == 1
    again = store.get_span(stored.receipt_id, span_id)
    assert again is not None
    assert again.retrieval_count == 2

    realized = store.realized_savings(stored.receipt_id)
    assert realized is not None
    assert realized["gross_saved_tokens"] == 80
    assert realized["retrieved_tokens"] > 0
    assert realized["repeated_expansion_tokens"] > 0
    assert realized["net_realized_saved_tokens"] < 80

    reopened = CompressionRetrievalStore(store_path)
    persisted = reopened.get_span(stored.receipt_id, span_id, record_retrieval=False)
    assert persisted is not None
    assert persisted.retrieval_count == 2
    assert reopened.realized_savings_summary()["receipts"] == 1


def test_search_records_retrieved_spans_once_per_result(tmp_path) -> None:
    store = CompressionRetrievalStore(tmp_path / "store.json")
    stored = store.put(
        original_text="alpha beta gamma\nneedle evidence here\nplain tail",
        compressed_text="alpha beta gamma",
        receipt={
            "original_tokens": 60,
            "compressed_tokens": 15,
            "omitted_spans": [{"start_line": 2, "end_line": 2}],
        },
    )

    hits = store.search("needle", limit=5)

    assert len(hits) == 1
    assert hits[0].retrieval_count == 1
    assert store.realized_savings(stored.receipt_id)["retrieved_tokens"] == hits[0].retrieved_tokens


def test_extract_decision_digest_preserves_decisions_paths_failures_and_next_tasks() -> None:
    digest = extract_decision_digest(
        """
        Decision: use cache-aware routing before model switch
        failed: provider returned timeout
        next: wire ledger into proxy
        modified entroly/proxy.py and tests/test_proxy.py
        """
    )

    assert any("cache-aware routing" in item for item in digest.decisions)
    assert any("provider returned timeout" in item for item in digest.failures)
    assert any("wire ledger" in item for item in digest.remaining_tasks)
    assert "entroly/proxy.py" in digest.modified_paths
    assert "tests/test_proxy.py" in digest.modified_paths


@dataclass
class FakeCheckpoint:
    checkpoint_id: str
    timestamp: float
    fragments: list[dict]
    metadata: dict
    stats: dict


def test_checkpoint_relevance_scores_query_and_continuity_with_trust_fence() -> None:
    scorer = CheckpointRelevanceScorer(half_life_seconds=1000)
    relevant = FakeCheckpoint(
        checkpoint_id="relevant",
        timestamp=1000,
        fragments=[{"source": "entroly/proxy.py", "content": "cache routing ledger"}],
        metadata={
            "continuity": {
                "decisions": ["Decision: use cache-aware routing"],
                "modified_paths": ["entroly/proxy.py"],
            }
        },
        stats={"coverage_pct": 80},
    )
    stale_untrusted = FakeCheckpoint(
        checkpoint_id="stale",
        timestamp=1,
        fragments=[{"source": "notes.txt", "content": "cache routing ledger"}],
        metadata={"untrusted": True},
        stats={},
    )

    ranked = scorer.rank([stale_untrusted, relevant], "cache routing proxy", now=1100)

    assert ranked[0].checkpoint_id == "relevant"
    assert ranked[0].is_trusted is True
    assert ranked[1].is_trusted is False


def test_cache_retention_forecaster_selects_long_when_pauses_fit_long_ttl() -> None:
    decision = CacheRetentionForecaster().decide(
        prefix_tokens=100_000,
        input_price_per_million=10.0,
        pause_samples_seconds=[600, 900, 1200],
        expected_future_turns=2,
    )

    assert decision.selected == "long"
    assert decision.expected_savings_usd["long"] > decision.expected_savings_usd["short"]
    assert decision.reason == "forecast:lowest_expected_cost"


def test_cache_retention_forecaster_returns_none_when_savings_not_material() -> None:
    decision = CacheRetentionForecaster().decide(
        prefix_tokens=10,
        input_price_per_million=0.01,
        pause_samples_seconds=[10, 20],
        min_savings_usd=1.0,
    )

    assert decision.selected == "none"
    assert decision.reason == "forecast:no_material_savings"


def test_behavioral_waste_detector_counts_repeats_and_model_churn() -> None:
    detector = BehavioralWasteDetector(window_seconds=60)
    detector.record("error", "timeout", timestamp=10)
    detector.record("error", "timeout", timestamp=11)
    detector.record("tool_call", "grep:foo", timestamp=12)
    detector.record("tool_call", "grep:foo", timestamp=13)
    detector.record("retry", "request-1", timestamp=14)
    detector.record("retry", "request-1", timestamp=15)
    detector.record("model", "gpt-a", timestamp=16)
    detector.record("model", "gpt-b", timestamp=17)
    report = detector.record("model", "gpt-a", timestamp=18)

    assert report.repeated_errors == 1
    assert report.repeated_tool_calls == 1
    assert report.retry_loops == 1
    assert report.model_switch_churn == 2
    assert report.waste_score > 0


def test_behavioral_waste_detector_evicts_old_events() -> None:
    detector = BehavioralWasteDetector(window_seconds=5)
    detector.record("error", "old", timestamp=1)
    detector.record("error", "old", timestamp=2)
    report = detector.record("error", "new", timestamp=10)

    assert report.repeated_errors == 0
    assert report.total_events == 1
