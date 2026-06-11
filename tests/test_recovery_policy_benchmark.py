import pytest

# Pulls QCCR (the Rust SSOT) transitively via benchmarks/; self-skip on the
# pure-Python (engine-less) install surface.
pytest.importorskip("entroly_core")

from benchmarks.recovery_policy_benchmark import (  # noqa: E402
    compress_with_exact_replay_candidates,
    evaluate_cases,
)


def test_bounded_exact_replay_recovers_selected_source_detail():
    context = (
        "Authentication uses a signed session cookie. "
        "The cookie is validated before the request reaches the handler. "
        "The exact rotation interval is 17 minutes. "
        "Expired cookies are rejected before authorization. "
    ) * 8

    compressed, replay = compress_with_exact_replay_candidates(
        context,
        "How are authentication cookies validated?",
        token_budget=18,
        recovery_token_budget=500,
    )

    assert compressed
    assert replay
    assert "The exact rotation interval is 17 minutes." in replay


def test_evaluate_cases_separates_first_pass_and_recovered_survival():
    cases = [
        {
            "context": (
                "Authentication uses a signed session cookie. "
                "The cookie is validated before the request reaches the handler. "
                "The exact rotation interval is 17 minutes. "
                "Expired cookies are rejected before authorization. "
            ) * 8,
            "question": "How are authentication cookies validated?",
            "answers": ["17 minutes"],
        },
        {
            "context": "The deployment region is us-west. " * 20,
            "question": "What is the deployment region?",
            "answers": ["us-west"],
        },
    ]

    result = evaluate_cases(
        cases,
        token_budget=18,
        recovery_token_budget=500,
    )

    assert result["metric_scope"] == "offline_extractable_evidence_survival"
    assert result["trigger"] == "oracle_answer_span_miss_upper_bound"
    assert result["total"] == 2
    assert result["first_pass_survived"] >= 1
    assert result["bounded_recovery_survival"] >= result["first_pass_survival"]
    assert result["avg_retry_expansion_tokens"] >= 0


def test_recovery_budget_is_strict():
    context = (
        "Authentication uses a signed session cookie. "
        "The exact rotation interval is 17 minutes. "
    ) * 20

    _compressed, replay = compress_with_exact_replay_candidates(
        context,
        "How are authentication cookies validated?",
        token_budget=12,
        recovery_token_budget=1,
    )

    assert replay == ""
