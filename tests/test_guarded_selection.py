"""Policy tests for fail-closed guarded QCCR selection."""

from __future__ import annotations

import pytest

from entroly.guarded_selection import (
    SufficiencyNotCertifiedError,
    select_guarded,
)


def test_input_that_already_fits_is_returned_byte_identical() -> None:
    fragments = [
        {"source": "a.txt", "content": "small context", "token_count": 3}
    ]

    output, receipt = select_guarded(
        fragments,
        token_budget=100,
        query="small",
        selector=lambda *_args, **_kwargs: pytest.fail(
            "selector must not run when identity already fits"
        ),
    )

    assert output == fragments
    assert receipt.decision == "bypass_already_fits"
    assert receipt.exact_identity
    assert receipt.input_sha256 == receipt.output_sha256
    assert receipt.budget_compliant


def test_degraded_first_attempt_expands_then_accepts() -> None:
    fragments = [
        {"source": "a.txt", "content": "x" * 1000, "token_count": 250}
    ]
    budgets: list[int] = []

    def selector(_fragments, *, token_budget: int, query: str):
        assert query == "find x"
        budgets.append(token_budget)
        verdict = "sufficient" if token_budget >= 164 else "degraded"
        return [
            {
                "source": "a.txt",
                "content": "x" * min(1000, token_budget * 4),
                "token_count": token_budget,
                "_sufficiency": {
                    "verdict": verdict,
                    "scope": "semantic",
                    "reasons": [] if verdict == "sufficient" else ["budget too tight"],
                },
            }
        ]

    output, receipt = select_guarded(
        fragments,
        token_budget=100,
        query="find x",
        max_expansions=2,
        expansion_factor=1.5,
        min_expansion_tokens=64,
        selector=selector,
    )

    assert budgets == [100, 164]
    assert output
    assert receipt.decision == "expanded"
    assert receipt.final_budget == 164
    assert len(receipt.attempts) == 2
    assert receipt.attempts[0]["certificate_verdict"] == "degraded"
    assert receipt.attempts[1]["certificate_verdict"] == "sufficient"


def test_proxy_scope_cannot_masquerade_as_semantic_certificate() -> None:
    fragments = [
        {"source": "a.txt", "content": "x" * 1000, "token_count": 250}
    ]

    def selector(_fragments, *, token_budget: int, query: str):
        return [
            {
                "source": "a.txt",
                "content": "relevant topic",
                "token_count": 4,
                "_sufficiency": {
                    "verdict": "sufficient",
                    "scope": "optimizer_proxy",
                    "reasons": [],
                },
            }
        ]

    output, receipt = select_guarded(
        fragments,
        token_budget=100,
        query="find answer",
        max_expansions=0,
        required_scope="semantic",
        selector=selector,
    )

    assert output == fragments
    assert receipt.decision == "bypass_uncertified"
    assert receipt.exact_identity
    assert not receipt.budget_compliant
    assert any("does not satisfy" in reason for reason in receipt.reasons)


def test_missing_certificate_fails_closed() -> None:
    fragments = [
        {"source": "a.txt", "content": "x" * 1000, "token_count": 250}
    ]

    output, receipt = select_guarded(
        fragments,
        token_budget=100,
        query="find answer",
        max_expansions=0,
        selector=lambda *_args, **_kwargs: [
            {"source": "a.txt", "content": "maybe", "token_count": 2}
        ],
    )

    assert output == fragments
    assert receipt.decision == "bypass_uncertified"
    assert "no sufficiency certificate" in receipt.reasons[0]


def test_hard_budget_mode_is_explicitly_uncertified() -> None:
    fragments = [
        {"source": "a.txt", "content": "x" * 1000, "token_count": 250}
    ]

    def selector(_fragments, *, token_budget: int, query: str):
        return [
            {
                "source": "a.txt",
                "content": "short",
                "token_count": 2,
                "_sufficiency": {
                    "verdict": "uncertain",
                    "scope": "optimizer_proxy",
                    "reasons": ["post-trim boundary unavailable"],
                },
            }
        ]

    output, receipt = select_guarded(
        fragments,
        token_budget=100,
        query="find answer",
        max_expansions=0,
        fallback="selected",
        selector=selector,
    )

    assert output[0]["content"] == "short"
    assert receipt.decision == "uncertified_budget_enforced"
    assert receipt.budget_compliant
    assert not receipt.exact_identity


def test_raise_mode_surfaces_attempt_receipt() -> None:
    fragments = [
        {"source": "a.txt", "content": "x" * 1000, "token_count": 250}
    ]

    with pytest.raises(SufficiencyNotCertifiedError) as exc:
        select_guarded(
            fragments,
            token_budget=100,
            query="find answer",
            max_expansions=0,
            fallback="raise",
            selector=lambda *_args, **_kwargs: [],
        )

    assert "raise_uncertified" in str(exc.value)
    assert "requested_budget" in str(exc.value)
