from __future__ import annotations

from dataclasses import dataclass

from entroly.compression_retrieval_store import CompressionRetrievalStore
from entroly.session_intelligence import (
    BehavioralWasteDetector,
    CacheRetentionForecaster,
    CheckpointRelevanceScorer,
    HallucinationTaintTracker,
    RealizedSavingsRecord,
    SavingsConfidence,
    SavingsTierLedger,
    SessionReceiptChain,
    allocate_session_turn_budget,
    extract_decision_digest,
    extract_taint_entities,
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

    span = store.retrieve_span(stored.receipt_id, span_id)
    assert span is not None
    assert span.retrieval_count == 1
    again = store.retrieve_span(stored.receipt_id, span_id)
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

    hits = store.search("needle", limit=5, record_retrieval=True)

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


def test_session_receipt_chain_links_parent_hashes_and_writes_json(tmp_path) -> None:
    chain = SessionReceiptChain(session_id="agent-session-1")
    decision = allocate_session_turn_budget(
        total_budget=10_000,
        turn_index=0,
        policy="flat",
    )

    first = chain.append(
        {
            "receipt_id": "cr_turn1",
            "reproducibility_hash": "caller_supplied_hash_is_not_trusted",
            "query": "inspect checkout",
            "token_budget": 2048,
            "risk_summary": {"witness": "pass"},
        },
        budget_decision=decision,
        created_at=1.0,
    )
    second = chain.append(
        {
            "receipt_id": "cr_turn2",
            "reproducibility_hash": "hash_turn2",
            "query": "run tests",
            "token_budget": 1024,
        },
        created_at=2.0,
    )

    assert first.parent_receipt_id is None
    assert first.receipt_hash != "caller_supplied_hash_is_not_trusted"
    assert first.receipt_id == f"cr_{first.receipt_hash[:12]}"
    assert first.source_receipt_id == "cr_turn1"
    assert first.budget_decision["policy"] == "flat"
    assert second.parent_receipt_id == first.receipt_id
    assert second.parent_receipt_hash == first.receipt_hash
    assert chain.verify_integrity()["valid"] is True

    output = chain.write_json(tmp_path / "session_chain.json")
    assert output.read_text(encoding="utf-8").count("cr_turn") == 2
    loaded = SessionReceiptChain.read_json(output)
    assert loaded.verify_integrity()["valid"] is True
    assert loaded.chain_hash() == chain.chain_hash()
    assert loaded.links[0].budget_decision["allocated_budget"] > 0


def test_session_receipt_chain_detects_tampered_json() -> None:
    chain = SessionReceiptChain(session_id="agent-session-1")
    chain.append({"receipt_id": "external_1", "query": "inspect"}, created_at=1.0)
    chain.append({"receipt_id": "external_2", "query": "test"}, created_at=2.0)
    payload = chain.as_dict()

    payload["links"][1]["receipt_hash"] = "0" * 64
    tampered = SessionReceiptChain.from_dict(payload)

    result = tampered.verify_integrity()
    assert result["valid"] is False
    assert any("chain_hash mismatch" in issue for issue in result["issues"])
    assert any("receipt_id is not hash-derived" in issue for issue in result["issues"])


def test_session_receipt_chain_hash_survives_write_read_round_trip(tmp_path) -> None:
    chain = SessionReceiptChain(session_id="round-trip")
    chain.append({"receipt_id": "external_1", "query": "inspect"}, created_at=1.0)
    chain.append({"receipt_id": "external_2", "query": "test"}, created_at=2.0)

    output = chain.write_json(tmp_path / "session_chain.json")
    loaded = SessionReceiptChain.read_json(output)

    assert loaded.chain_hash() == chain.chain_hash()
    assert loaded.verify_integrity()["valid"] is True


def test_session_receipt_chain_rejects_duplicate_content_hashes() -> None:
    chain = SessionReceiptChain(session_id="agent-session-1")
    chain.append({"receipt_id": "external_1", "query": "same"}, created_at=1.0)
    chain.append({"receipt_id": "external_2", "query": "same"}, created_at=2.0)

    result = chain.verify_integrity()

    assert result["valid"] is False
    assert any("duplicate receipt_hash" in issue for issue in result["issues"])


def test_allocate_session_turn_budget_reserves_closing_budget() -> None:
    decision = allocate_session_turn_budget(
        total_budget=10_000,
        spent_tokens=0,
        turn_index=0,
        expected_total_turns=5,
        policy="decay",
        closing_reserve_ratio=0.2,
        min_turn_budget=512,
    )

    assert decision.reserved_closing_budget == 2_000
    assert 512 <= decision.allocated_budget < 8_000
    assert decision.remaining_budget == 10_000
    assert decision.expected_remaining_turns == 5

    closing = allocate_session_turn_budget(
        total_budget=10_000,
        spent_tokens=7_900,
        turn_index=4,
        expected_total_turns=5,
        policy="flat",
        closing_reserve_ratio=0.2,
        is_closing_turn=True,
    )

    assert closing.reserved_closing_budget == 0
    assert closing.allocated_budget == 2_100


def test_hallucination_taint_tracker_records_origin_and_propagation() -> None:
    tracker = HallucinationTaintTracker()

    created = tracker.observe_witness(
        turn_index=3,
        receipt_id="cr_turn3",
        witness_result={
            "certificates": [
                {
                    "claim_text": "Call imaginary_api_client before deployment",
                    "label": "unsupported",
                    "risk": 0.92,
                }
            ]
        },
    )
    report = tracker.observe_turn(
        turn_index=4,
        receipt_id="cr_turn4",
        context="Previous plan referenced imaginary_api_client.",
        response="I will wire imaginary_api_client into entroly/proxy.py.",
    )

    assert {item.entity for item in created} >= {"imaginary_api_client"}
    assert "imaginary_api_client" in report.propagated_entities
    assert report.origin_turns["imaginary_api_client"] == 3
    assert report.propagation_pressure == 1
    assert report.risk_level == "high"
    suspect = tracker.suspects["imaginary_api_client"]
    assert suspect.propagated_turns == (4,)


def test_taint_entity_extraction_prefers_concrete_identifiers() -> None:
    entities = extract_taint_entities(
        "The API returned None. Call imaginary_api_client before deployment "
        "and then use EntrolyLoop."
    )

    assert "imaginary_api_client" in entities
    assert "entrolyloop" in entities
    assert "deployment" not in entities
    assert "api" not in entities
    assert "none" not in entities


def test_taint_tracker_persists_and_decays_stale_suspects(tmp_path) -> None:
    tracker = HallucinationTaintTracker(risk_half_life_turns=2.0)
    tracker.observe_witness(
        turn_index=1,
        receipt_id="cr_turn1",
        witness_result={
            "certificates": [
                {
                    "claim_text": "Use phantomIntegrationAdapter for exports",
                    "label": "unsupported",
                    "risk": 0.8,
                }
            ]
        },
    )

    first = tracker.observe_turn(
        turn_index=2,
        receipt_id="cr_turn2",
        response="phantomIntegrationAdapter appears again",
    )
    later = tracker.observe_turn(
        turn_index=6,
        receipt_id="cr_turn6",
        response="phantomIntegrationAdapter appears after several clean turns",
    )

    assert first.taint_score > later.taint_score
    saved = tracker.write_json(tmp_path / "taint.json")
    loaded = HallucinationTaintTracker.read_json(saved)
    assert "phantomintegrationadapter" in loaded.suspects
    assert loaded.suspects["phantomintegrationadapter"].propagated_turns == (2, 6)


def test_taint_score_uses_noisy_or_not_additive_saturation() -> None:
    tracker = HallucinationTaintTracker(risk_half_life_turns=1000.0)
    tracker.observe_witness(
        turn_index=1,
        receipt_id="cr_turn1",
        witness_result={
            "certificates": [
                {
                    "claim_text": f"Use fake_service_{idx} for routing",
                    "label": "unsupported",
                    "risk": 0.3,
                }
                for idx in range(10)
            ]
        },
    )

    report = tracker.observe_turn(
        turn_index=2,
        receipt_id="cr_turn2",
        response=" ".join(f"fake_service_{idx}" for idx in range(10)),
    )

    assert 0.95 < report.taint_score < 1.0
