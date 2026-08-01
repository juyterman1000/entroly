from __future__ import annotations

import pytest

from entroly.guarded_selection import (
    GuardDecision,
    SufficiencyNotCertifiedError,
    select_guarded,
)
from entroly.sufficiency_contract import (
    CertificateScope,
    CertificateVerdict,
    SufficiencyCertificate,
    scope_satisfies,
)


def _large_input() -> list[dict]:
    return [{"source": "a.txt", "content": "x" * 1000, "token_count": 250}]


def test_input_that_already_fits_is_returned_byte_identical() -> None:
    fragments = [{"source": "a.txt", "content": "small context", "token_count": 3}]
    output, receipt = select_guarded(
        fragments,
        token_budget=100,
        query="small",
        selector=lambda *_args, **_kwargs: pytest.fail("selector must not run"),
    )
    assert output == fragments
    assert receipt.decision is GuardDecision.BYPASS_ALREADY_FITS
    assert receipt.exact_identity
    assert receipt.input_sha256 == receipt.output_sha256
    assert receipt.budget_compliant


def test_degraded_first_attempt_expands_then_accepts() -> None:
    budgets: list[int] = []

    def selector(_fragments, *, token_budget: int, query: str):
        budgets.append(token_budget)
        verdict = "sufficient" if token_budget >= 164 else "degraded"
        return [{
            "source": "a.txt",
            "content": "x" * min(1000, token_budget * 4),
            "token_count": token_budget,
            "_sufficiency": {
                "verdict": verdict,
                "scope": "semantic",
                "reasons": [] if verdict == "sufficient" else ["budget too tight"],
            },
        }]

    output, receipt = select_guarded(
        _large_input(),
        100,
        "find x",
        selector=selector,
        max_expansions=2,
        expansion_factor=1.5,
        min_expansion_tokens=64,
    )
    assert budgets == [100, 164]
    assert output
    assert receipt.decision is GuardDecision.EXPANDED_CERTIFIED
    assert receipt.final_budget == 164
    assert len(receipt.attempts) == 2


def test_proxy_scope_cannot_masquerade_as_semantic_certificate() -> None:
    def selector(_fragments, *, token_budget: int, query: str):
        return [{
            "source": "a.txt",
            "content": "relevant topic",
            "token_count": 4,
            "_sufficiency": {
                "verdict": "sufficient",
                "scope": "optimizer_proxy",
            },
        }]

    output, receipt = select_guarded(
        _large_input(),
        100,
        "find answer",
        max_expansions=0,
        selector=selector,
    )
    assert output == _large_input()
    assert receipt.decision is GuardDecision.BYPASS_UNCERTIFIED
    assert not receipt.budget_compliant
    assert any("does not satisfy" in reason for reason in receipt.reasons)


def test_missing_certificate_is_uncertain_and_fails_closed() -> None:
    output, receipt = select_guarded(
        _large_input(),
        100,
        "find answer",
        max_expansions=0,
        selector=lambda *_args, **_kwargs: [
            {"source": "a.txt", "content": "maybe", "token_count": 2}
        ],
    )
    assert output == _large_input()
    assert receipt.decision is GuardDecision.BYPASS_UNCERTIFIED
    assert receipt.attempts[0]["certificate_verdict"] == "uncertain"
    assert receipt.attempts[0]["certificate_scope"] == "unavailable"


def test_hard_budget_mode_is_explicitly_uncertified() -> None:
    def selector(_fragments, *, token_budget: int, query: str):
        return [{
            "source": "a.txt",
            "content": "short",
            "token_count": 2,
            "_sufficiency": {
                "verdict": "uncertain",
                "scope": "candidate_units",
                "reasons": ["no held-out semantic calibration"],
            },
        }]

    output, receipt = select_guarded(
        _large_input(),
        100,
        "find answer",
        max_expansions=0,
        fallback="selected",
        selector=selector,
    )
    assert output[0]["content"] == "short"
    assert receipt.decision is GuardDecision.UNCERTIFIED_BUDGET_ENFORCED
    assert receipt.budget_compliant
    assert not receipt.exact_identity


def test_raise_mode_surfaces_attempt_receipt() -> None:
    with pytest.raises(SufficiencyNotCertifiedError) as exc:
        select_guarded(
            _large_input(),
            100,
            "find answer",
            max_expansions=0,
            fallback="raise",
            selector=lambda *_args, **_kwargs: [],
        )
    assert "RAISE_UNCERTIFIED" in str(exc.value)
    assert "requested_budget" in str(exc.value)


def test_unknown_certificate_values_fail_closed() -> None:
    certificate = SufficiencyCertificate.from_mapping({
        "verdict": "magic",
        "scope": "infinite",
        "reasons": "bad producer",
    })
    assert certificate.verdict is CertificateVerdict.UNCERTAIN
    assert certificate.scope is CertificateScope.UNAVAILABLE
    assert certificate.reasons == ("bad producer",)


def test_sufficient_without_scope_is_downgraded_to_uncertain() -> None:
    certificate = SufficiencyCertificate.from_mapping({"verdict": "sufficient"})
    assert certificate.verdict is CertificateVerdict.UNCERTAIN
    assert certificate.scope is CertificateScope.UNAVAILABLE
    assert certificate.reasons


def test_scope_order_is_explicit() -> None:
    assert scope_satisfies(
        CertificateScope.SEMANTIC,
        CertificateScope.CANDIDATE_UNITS,
    )
    assert not scope_satisfies(
        CertificateScope.OPTIMIZER_PROXY,
        CertificateScope.SEMANTIC,
    )


def test_unknown_required_scope_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown required_scope"):
        select_guarded(
            _large_input(),
            100,
            "find answer",
            required_scope="magic",
        )


def test_invalid_expansion_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="expansion_factor"):
        select_guarded(
            _large_input(),
            100,
            "find answer",
            max_expansions=1,
            expansion_factor=1.0,
        )
