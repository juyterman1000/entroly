from __future__ import annotations

from entroly.assurance_telemetry import AssuranceLedger


def test_ledger_records_only_decision_metadata_and_summarizes(tmp_path) -> None:
    ledger = AssuranceLedger(tmp_path / "assurance.sqlite3")
    ledger.record_selection(
        {
            "decision": "COMPRESSED_CERTIFIED",
            "requested_budget": 50,
            "raw_tokens": 100,
            "delivered_tokens": 40,
            "exact_identity": False,
            "budget_compliant": True,
            "input_sha256": "a" * 64,
            "output_sha256": "b" * 64,
            "attempts": [
                {
                    "certificate_scope": "candidate_units",
                    "certificate_verdict": "sufficient",
                }
            ],
        },
        latency_ms=4.0,
        metadata={"query": "this value is bounded but raw context is never stored"},
    )
    ledger.record_domain(
        {
            "decision": "BYPASS_INVALID",
            "content_type": "logs",
            "requested_budget": 20,
            "original_tokens": 80,
            "emitted_tokens": 80,
            "exact_identity": True,
            "budget_compliant": False,
            "input_sha256": "c" * 64,
            "output_sha256": "c" * 64,
            "validation": {"valid": False, "reasons": ["critical line missing"]},
        },
        latency_ms=10.0,
    )

    summary = ledger.summary()
    assert summary.events == 2
    assert summary.accepted_events == 1
    assert summary.bypass_events == 1
    assert summary.identity_events == 1
    assert summary.accepted_coverage == 0.5
    assert summary.bypass_rate == 0.5
    assert summary.p95_latency_ms >= summary.p50_latency_ms
    assert summary.decision_counts["COMPRESSED_CERTIFIED"] == 1


def test_ledger_prune(tmp_path) -> None:
    ledger = AssuranceLedger(tmp_path / "assurance.sqlite3")
    ledger.record(
        kind="selection",
        decision="BYPASS_ALREADY_FITS",
        requested_budget=10,
        original_tokens=5,
        delivered_tokens=5,
        exact_identity=True,
        budget_compliant=True,
        latency_ms=0,
        input_sha256="a",
        output_sha256="a",
        created_at=10,
    )
    assert ledger.prune(before=11) == 1
    assert ledger.summary().events == 0
