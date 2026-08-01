from __future__ import annotations

from entroly import assured_context
from entroly.guarded_selection import GuardDecision


def envelope(verdict: str, scope: str, budget: int):
    return {
        "selected": [
            {
                "source": "a.txt",
                "content": "answer",
                "token_count": min(2, budget),
                "_sufficiency": {"verdict": verdict, "scope": scope, "reasons": []},
            }
        ],
        "candidates": [],
        "metrics": {"verdict": verdict, "scope": scope, "reasons": []},
        "requested_budget": budget,
        "raw_tokens": 100,
        "emitted_tokens": 2,
        "selection_mode": "atomic_audited",
    }


def test_structural_mode_accepts_candidate_unit_certificate(monkeypatch) -> None:
    monkeypatch.setattr(
        assured_context.audited_qccr,
        "select_with_audit",
        lambda _f, budget, _q: envelope("sufficient", "candidate_units", budget),
    )
    result = assured_context.select_structurally_assured(
        [{"source": "a.txt", "content": "x" * 400, "token_count": 100}],
        10,
        "answer",
        max_expansions=0,
    )
    assert result.receipt.decision is GuardDecision.COMPRESSED_CERTIFIED
    assert result.selected[0]["content"] == "answer"
    assert len(result.audits) == 1


def test_semantic_mode_rejects_structural_only_certificate(monkeypatch) -> None:
    monkeypatch.setattr(
        assured_context.audited_qccr,
        "select_with_audit",
        lambda _f, budget, _q: envelope("sufficient", "candidate_units", budget),
    )
    original = [{"source": "a.txt", "content": "x" * 400, "token_count": 100}]
    result = assured_context.select_assured(
        original, 10, "answer", max_expansions=0
    )
    assert result.receipt.decision is GuardDecision.BYPASS_UNCERTIFIED
    assert list(result.selected) == original
