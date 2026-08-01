from __future__ import annotations

from bench import assurance_overhead


def test_overhead_report_uses_same_input_and_checks_determinism(monkeypatch) -> None:
    monkeypatch.setattr(
        assurance_overhead,
        "legacy_select",
        lambda fragments, **_kwargs: list(fragments),
    )
    monkeypatch.setattr(
        assurance_overhead,
        "select_with_audit",
        lambda _fragments, budget, _query: {
            "selected": [{"content": "answer"}],
            "candidates": [],
            "metrics": {"scope": "candidate_units", "verdict": "sufficient"},
            "emitted_tokens": 2,
            "requested_budget": budget,
        },
    )
    report = assurance_overhead.compare_overhead(
        [{"source": "a", "content": "answer", "token_count": 2}],
        query="answer",
        budget=10,
        iterations=3,
        warmup=0,
    )
    assert report.deterministic
    assert report.budget_compliant
    assert report.emitted_tokens == 2
    assert report.receipt_bytes > 0
