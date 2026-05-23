"""SDK public-surface regression tests.

These are real pytest tests (the old file was module-level script style
and collected ZERO tests under pytest, which is exactly how the broken
`detect_hallucination` shipped — 3/4 signals silently dead and the
WITNESS signal inverted/mislabelled). Each test below would have failed
on the shipped v0.19.x code.
"""

from __future__ import annotations

import pytest

from entroly import compress, detect_hallucination, optimize, verify

CTX = ("The Eiffel Tower is in Paris, France, completed in 1889 by "
       "Gustave Eiffel.")
HALLUCINATION = ("The Eiffel Tower is in Berlin and was built in 1750 "
                 "by Napoleon Bonaparte.")
GROUNDED = "The Eiffel Tower is in Paris and was completed in 1889."


def test_sdk_functions_importable():
    for fn in (detect_hallucination, optimize, compress, verify):
        assert callable(fn)


def test_detect_hallucination_shape():
    r = detect_hallucination(HALLUCINATION, context=CTX)
    for k in ("fused_risk", "verdict", "recommendation", "primary_signal",
              "auxiliary_abstention", "witness", "ece", "epr", "spectral",
              "flagged_claims"):
        assert k in r, f"missing key {k!r}"
    assert r["verdict"] in ("pass", "warn", "flag")
    assert 0.0 <= r["fused_risk"] <= 1.0


def test_all_four_signals_actually_execute():
    """The shipped P0 bug: ece/epr/spectral threw on every call and were
    silently zeroed with status 'unavailable'. Guard that they run."""
    r = detect_hallucination(HALLUCINATION, context=CTX)
    for sig in ("witness", "ece", "epr", "spectral"):
        assert "risk_score" in r[sig], f"{sig} has no risk_score"
        assert r[sig].get("status") != "unavailable", (
            f"{sig} did not execute (regression: signal API broken)"
        )
    assert r["witness"]["validated"] is True
    for sig in ("ece", "epr", "spectral"):
        assert r[sig]["validated"] is False, (
            f"{sig} must be marked unvalidated, not sold as validated"
        )


def test_score_is_witness_only_never_polluted():
    """fused_risk must equal the validated WITNESS risk exactly — no
    fitted blend (that blend was the falsified Fusion-4 artifact)."""
    r = detect_hallucination(HALLUCINATION, context=CTX)
    assert r["primary_signal"] == "witness"
    assert r["fused_risk"] == pytest.approx(
        r["witness"]["risk_score"], abs=1e-4
    ), "score was blended with unvalidated signals — artifact risk"


def test_hallucination_scores_higher_than_grounded():
    bad = detect_hallucination(HALLUCINATION, context=CTX)
    good = detect_hallucination(GROUNDED, context=CTX)
    assert bad["fused_risk"] > good["fused_risk"] + 0.2, (
        f"detector not discriminative: bad={bad['fused_risk']} "
        f"good={good['fused_risk']}"
    )
    assert bad["verdict"] in ("warn", "flag")
    assert good["verdict"] == "pass"
    # WITNESS must surface the contradicted claim on the hallucination.
    assert bad["flagged_claims"], "no flagged claims on a blatant hallucination"
    assert "risk" in bad["flagged_claims"][0]


def test_auxiliaries_can_only_escalate_action_not_lower():
    """Selective abstention may turn pass→warn but must never lower a
    verdict nor change the number."""
    r = detect_hallucination(GROUNDED, context=CTX)
    assert r["verdict"] in ("pass", "warn")  # never silently downgraded
    if r["auxiliary_abstention"]:
        assert r["verdict"] == "warn"
    # Number is unchanged regardless of abstention.
    assert r["fused_risk"] == pytest.approx(r["witness"]["risk_score"],
                                            abs=1e-4)


def test_optimize_and_compress_still_work():
    frags = [
        {"content": "def login(u,p): validate(u); return token",
         "source": "auth.py", "token_count": 15},
        {"content": "const PI=3.14159; function area(r){return PI*r*r;}",
         "source": "math.js", "token_count": 18},
    ]
    o = optimize(frags, budget=30, query="fix the login function")
    assert o["fragments_total"] == 2
    assert "context_text" in o
    long_text = "The quick brown fox jumped over the lazy dog. " * 100
    assert len(compress(long_text, budget=50)) <= len(long_text)
