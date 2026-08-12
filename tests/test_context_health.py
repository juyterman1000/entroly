from __future__ import annotations

import json

from entroly.context_health import build_context_health
from entroly.optimization_ledger import (
    OptimizationAdjustment,
    OptimizationEvent,
    OptimizationLedger,
    SavingsTier,
)


class _Index:
    def list_sessions(self, *, query: str = "", limit: int = 50):
        assert query == ""
        assert limit == 100
        return {
            "total": 2,
            "diagnostics": [{"message": "bounded diagnostic"}],
            "sessions": [
                {"key": "valid"},
                {"key": "invalid"},
            ],
        }

    def get_session(self, key: str):
        if key == "valid":
            return {
                "integrity": {"valid": True},
                "receipts": [
                    {
                        "selected_tokens": 400,
                        "omitted_tokens": 600,
                        "selected": [{"excerpt": "private selected code"}],
                        "omitted": [
                            {
                                "recoverable": True,
                                "excerpt": "private omitted code",
                                "source_path": "private.py",
                            },
                            {"recoverable": False, "excerpt": "secret"},
                        ],
                    }
                ],
            }
        return {
            "integrity": {"valid": False, "issues": ["tampered-private-id"]},
            "receipts": [],
        }


def test_context_health_reports_net_evidence_without_user_content(tmp_path):
    ledger_path = tmp_path / "optimization.sqlite3"
    ledger = OptimizationLedger(ledger_path)
    ledger.record(
        OptimizationEvent(
            event_id="event-private",
            feature="compression",
            tier=SavingsTier.MEASURED,
            gross_tokens_saved=1_000,
            gross_micro_usd=5_000,
            cost_micro_usd=500,
            session_id="session-private",
            metadata={"prompt": "do not leak me"},
        )
    )
    ledger.adjust(
        OptimizationAdjustment(
            adjustment_id="adjust-private",
            event_id="event-private",
            tokens_reexpanded=250,
            cost_micro_usd=1_000,
        )
    )

    report = build_context_health(
        index=_Index(),
        ledger_path=ledger_path,
        value_confidence={
            "lifetime": {
                "tokens_saved": 2_000,
                "cost_saved_usd": 0.012345,
                "hallucinations_blocked": 3,
            }
        },
    )

    assert report["schema_version"] == "entroly.context-health.v1"
    assert report["value"]["measured_gross_tokens"] == 1_000
    assert report["value"]["measured_reexpanded_tokens"] == 250
    assert report["value"]["measured_net_tokens"] == 750
    assert report["value"]["measured_net_usd"] == 0.0035
    assert report["value"]["recovery_tax_pct"] == 25.0
    assert report["evidence"]["integrity_pct"] == 50.0
    assert report["evidence"]["recoverability_pct"] == 50.0
    assert report["protections"]["confusion"]["unsupported_claims_blocked"] == 3
    assert report["protections"]["drift"]["source_freshness"] == "unavailable"
    assert report["privacy"]["content_blind"] is True
    serialized = json.dumps(report, sort_keys=True)
    for private in (
        "private selected code",
        "private omitted code",
        "private.py",
        "tampered-private-id",
        "event-private",
        "session-private",
        "do not leak me",
    ):
        assert private not in serialized


def test_context_health_marks_unobserved_dimensions_honestly(tmp_path):
    report = build_context_health(
        index=_IndexWithoutSessions(),
        ledger_path=tmp_path / "missing.sqlite3",
        value_confidence={"lifetime": {}},
    )

    assert report["value"]["ledger_status"] == "unavailable"
    assert report["value"]["recovery_tax_pct"] is None
    assert report["evidence"]["integrity_pct"] is None
    assert report["protections"]["confusion"]["status"] == "no_events_observed"
    assert report["protections"]["rot"]["status"] == "unavailable"
    assert report["protections"]["drift"]["status"] == "unavailable"
    assert "fixed" not in report["share"]["text"].casefold()


def test_context_health_preserves_token_negative_recovery_outcomes(tmp_path):
    ledger_path = tmp_path / "negative.sqlite3"
    ledger = OptimizationLedger(ledger_path)
    ledger.record(
        OptimizationEvent(
            event_id="negative-event",
            feature="compression",
            tier=SavingsTier.MEASURED,
            gross_tokens_saved=100,
            gross_micro_usd=100,
        )
    )
    ledger.adjust(
        OptimizationAdjustment(
            adjustment_id="negative-adjustment",
            event_id="negative-event",
            tokens_reexpanded=150,
            cost_micro_usd=200,
        )
    )

    report = build_context_health(
        index=_IndexWithoutSessions(),
        ledger_path=ledger_path,
        value_confidence={"lifetime": {}},
    )

    assert report["value"]["measured_net_tokens"] == -50
    assert report["value"]["measured_net_usd"] == -0.0001
    assert report["value"]["recovery_tax_pct"] == 150.0


class _IndexWithoutSessions:
    def list_sessions(self, *, query: str = "", limit: int = 50):
        return {"total": 0, "diagnostics": [], "sessions": []}

    def get_session(self, key: str):
        raise AssertionError("get_session must not be called without sessions")
