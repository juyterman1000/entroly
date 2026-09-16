from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from entroly.granularity_frontier import (
    ConflictingGranularityEvidence,
    EvaluatorVerdict,
    GranularityFrontierError,
    GranularityPromotionPolicy,
    GranularityScope,
    PairedGranularityObservation,
    VerifiedGranularityFrontier,
    verify_granularity_receipt,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _scope(**overrides: str) -> GranularityScope:
    fields = {
        "task_class": "code-repair",
        "workload_id": "heldout-repo-repairs-v1",
        "model_id": "test-model-v1",
        "tokenizer_id": "test-tokenizer-v1",
        "evaluator_protocol": "repo-tests-plus-grounding-v1",
    }
    fields.update(overrides)
    return GranularityScope(**fields)


def _observation(
    trial: int,
    *,
    candidate_id: str = "structural-v1",
    scope: GranularityScope | None = None,
    baseline_score: float = 0.90,
    candidate_score: float = 0.90,
    baseline_passed: bool = True,
    candidate_passed: bool = True,
    baseline_tokens: int = 1_000,
    candidate_tokens: int = 600,
    catastrophic_failure: bool = False,
    evaluator_family: str = "deterministic-repo-tests",
    second_evaluator_family: str | None = None,
) -> PairedGranularityObservation:
    source = f"source-{trial}"
    verdicts = [
        EvaluatorVerdict(
            evaluator_id=f"{evaluator_family}-v1",
            evaluator_family=evaluator_family,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            baseline_passed=baseline_passed,
            candidate_passed=candidate_passed,
        )
    ]
    if second_evaluator_family:
        verdicts.append(
            EvaluatorVerdict(
                evaluator_id=f"{second_evaluator_family}-v1",
                evaluator_family=second_evaluator_family,
                baseline_score=baseline_score,
                candidate_score=candidate_score,
                baseline_passed=baseline_passed,
                candidate_passed=candidate_passed,
            )
        )
    return PairedGranularityObservation(
        scope=scope or _scope(),
        trial_id=f"trial-{trial}",
        candidate_id=candidate_id,
        source_sha256=_digest(source),
        baseline_context_sha256=_digest(f"baseline-{source}"),
        candidate_context_sha256=_digest(f"candidate-{candidate_id}-{source}"),
        source_bytes=8_000,
        baseline_tokens=baseline_tokens,
        candidate_tokens=candidate_tokens,
        verdicts=tuple(verdicts),
        catastrophic_failure=catastrophic_failure,
    )


def _policy(**overrides) -> GranularityPromotionPolicy:
    fields = {
        "min_paired_trials": 5,
        "confidence": 0.90,
        "max_mean_quality_regression": 0.02,
        "max_verifier_regression_rate": 0.40,
        "min_token_savings_fraction": 0.10,
    }
    fields.update(overrides)
    return GranularityPromotionPolicy(**fields)


def test_public_api_is_reachable() -> None:
    from entroly import VerifiedGranularityFrontier as PublicFrontier

    assert PublicFrontier is VerifiedGranularityFrontier


def test_cold_start_retains_full_context_and_receipt_verifies() -> None:
    frontier = VerifiedGranularityFrontier(_policy())

    receipt = frontier.decide(_scope())

    assert receipt["selection"]["status"] == "retain_baseline"
    assert receipt["selection"]["selected_id"] == "full-context"
    assert receipt["selection"]["reason"] == "no_candidate_evidence"
    assert verify_granularity_receipt(receipt)


def test_candidate_needs_matched_evidence_before_promotion() -> None:
    frontier = VerifiedGranularityFrontier(_policy(min_paired_trials=5))
    frontier.record_many(_observation(i) for i in range(4))

    assessment = frontier.assess_candidate(_scope(), "structural-v1")

    assert assessment["status"] == "hold"
    assert assessment["hold_reasons"] == ["insufficient_paired_trials"]
    assert frontier.decide(_scope())["selection"]["selected_id"] == "full-context"


def test_sparse_evidence_is_held_not_misreported_as_failure() -> None:
    frontier = VerifiedGranularityFrontier()
    frontier.record(_observation(1))

    assessment = frontier.assess_candidate(_scope(), "structural-v1")

    assert assessment["status"] == "hold"
    assert assessment["hold_reasons"] == ["insufficient_paired_trials"]
    assert assessment["rejection_reasons"] == []


def test_noninferior_cheaper_candidate_earns_promotion() -> None:
    frontier = VerifiedGranularityFrontier(_policy())
    frontier.record_many(_observation(i, candidate_score=0.91) for i in range(5))

    assessment = frontier.assess_candidate(_scope(), "structural-v1")
    receipt = frontier.decide(_scope())

    assert assessment["status"] == "promote"
    assert assessment["quality"]["one_sided_lower_bound"] == pytest.approx(0.01)
    assert assessment["economics"]["mean_token_savings_fraction"] == 0.4
    assert receipt["selection"]["status"] == "promote_candidate"
    assert receipt["selection"]["selected_id"] == "structural-v1"
    assert verify_granularity_receipt(receipt)


def test_default_policy_can_be_earned_with_forty_clean_pairs() -> None:
    frontier = VerifiedGranularityFrontier()
    frontier.record_many(_observation(i) for i in range(40))

    assessment = frontier.assess_candidate(_scope(), "structural-v1")

    assert assessment["status"] == "promote"
    assert assessment["verification"]["wilson_upper_bound"] < 0.10


def test_quality_regression_rejects_candidate() -> None:
    frontier = VerifiedGranularityFrontier(_policy())
    frontier.record_many(_observation(i, candidate_score=0.80) for i in range(5))

    assessment = frontier.assess_candidate(_scope(), "structural-v1")

    assert assessment["status"] == "reject"
    assert "quality_noninferiority_not_demonstrated" in assessment["rejection_reasons"]


def test_verifier_regression_bound_rejects_candidate() -> None:
    frontier = VerifiedGranularityFrontier(
        _policy(max_verifier_regression_rate=0.30)
    )
    observations = [_observation(i) for i in range(5)]
    observations[-1] = _observation(4, candidate_passed=False)
    frontier.record_many(observations)

    assessment = frontier.assess_candidate(_scope(), "structural-v1")

    assert assessment["status"] == "reject"
    assert assessment["verification"]["regression_count"] == 1
    assert "verifier_regression_bound_exceeds_policy" in assessment["rejection_reasons"]


def test_unreliable_baseline_cannot_authorize_candidate() -> None:
    frontier = VerifiedGranularityFrontier(_policy(min_baseline_pass_rate=0.80))
    frontier.record_many(
        _observation(
            i,
            baseline_score=0.20,
            candidate_score=0.20,
            baseline_passed=False,
            candidate_passed=False,
        )
        for i in range(5)
    )

    assessment = frontier.assess_candidate(_scope(), "structural-v1")

    assert assessment["status"] == "hold"
    assert assessment["verification"]["baseline_pass_rate"] == 0.0
    assert "baseline_success_below_policy" in assessment["hold_reasons"]


def test_catastrophic_failure_blocks_even_with_good_average() -> None:
    frontier = VerifiedGranularityFrontier(_policy())
    observations = [_observation(i, candidate_score=0.95) for i in range(5)]
    observations[-1] = _observation(
        4,
        candidate_score=0.95,
        candidate_passed=False,
        catastrophic_failure=True,
    )
    frontier.record_many(observations)

    assessment = frontier.assess_candidate(_scope(), "structural-v1")

    assert assessment["status"] == "reject"
    assert "catastrophic_failure_observed" in assessment["rejection_reasons"]


def test_evaluator_family_gate_does_not_treat_agent_count_as_independence() -> None:
    frontier = VerifiedGranularityFrontier(
        _policy(min_evaluator_families=2, max_verifier_regression_rate=0.50)
    )
    frontier.record_many(_observation(i) for i in range(5))

    assessment = frontier.assess_candidate(_scope(), "structural-v1")

    assert assessment["status"] == "hold"
    assert assessment["evaluator_families"] == ["deterministic-repo-tests"]
    assert "insufficient_evaluator_families" in assessment["hold_reasons"]


def test_evaluator_family_must_cover_every_trial() -> None:
    frontier = VerifiedGranularityFrontier(
        _policy(min_evaluator_families=2, max_verifier_regression_rate=0.50)
    )
    frontier.record_many(
        _observation(i, second_evaluator_family="independent-grounding")
        if i == 0
        else _observation(i)
        for i in range(5)
    )

    assessment = frontier.assess_candidate(_scope(), "structural-v1")

    assert assessment["status"] == "hold"
    assert assessment["evaluator_family_trial_counts"]["independent-grounding"] == 1
    assert assessment["complete_evaluator_families"] == ["deterministic-repo-tests"]


def test_two_complete_evaluator_families_can_satisfy_diversity_gate() -> None:
    frontier = VerifiedGranularityFrontier(
        _policy(min_evaluator_families=2, max_verifier_regression_rate=0.50)
    )
    frontier.record_many(
        _observation(i, second_evaluator_family="independent-grounding")
        for i in range(5)
    )

    assessment = frontier.assess_candidate(_scope(), "structural-v1")

    assert assessment["status"] == "promote"
    assert assessment["complete_evaluator_families"] == [
        "deterministic-repo-tests",
        "independent-grounding",
    ]


def test_candidate_must_survive_the_most_conservative_evaluator() -> None:
    frontier = VerifiedGranularityFrontier(
        _policy(min_evaluator_families=2, max_verifier_regression_rate=0.50)
    )
    observations = []
    for i in range(5):
        base = _observation(i, second_evaluator_family="independent-grounding")
        skeptical = EvaluatorVerdict(
            evaluator_id="independent-grounding-v1",
            evaluator_family="independent-grounding",
            baseline_score=0.90,
            candidate_score=0.80,
            baseline_passed=True,
            candidate_passed=True,
        )
        observations.append(replace(base, verdicts=(base.verdicts[0], skeptical)))
    frontier.record_many(observations)

    assessment = frontier.assess_candidate(_scope(), "structural-v1")

    assert assessment["status"] == "reject"
    assert assessment["quality"]["mean_paired_delta"] == pytest.approx(-0.10)
    assert "quality_noninferiority_not_demonstrated" in assessment["rejection_reasons"]


def test_scope_prevents_cross_model_or_tokenizer_evidence_leakage() -> None:
    frontier = VerifiedGranularityFrontier(_policy())
    first_scope = _scope()
    other_scope = _scope(tokenizer_id="different-tokenizer")
    frontier.record_many(_observation(i, scope=first_scope) for i in range(5))

    assert frontier.decide(first_scope)["selection"]["selected_id"] == "structural-v1"
    assert frontier.decide(other_scope)["selection"]["selected_id"] == "full-context"


def test_frontier_selects_lowest_cost_candidate_that_passes_all_gates() -> None:
    frontier = VerifiedGranularityFrontier(_policy())
    frontier.record_many(
        _observation(i, candidate_id="balanced", candidate_tokens=650)
        for i in range(5)
    )
    frontier.record_many(
        _observation(i, candidate_id="compact", candidate_tokens=400)
        for i in range(5)
    )

    receipt = frontier.decide(_scope())

    assert receipt["selection"]["selected_id"] == "compact"


def test_frontier_compares_cost_as_baseline_normalized_reduction() -> None:
    frontier = VerifiedGranularityFrontier(_policy())
    frontier.record_many(
        _observation(
            i,
            candidate_id="small-tasks-low-absolute-cost",
            baseline_tokens=500,
            candidate_tokens=400,
        )
        for i in range(5)
    )
    frontier.record_many(
        _observation(
            i,
            candidate_id="large-tasks-better-reduction",
            baseline_tokens=2_000,
            candidate_tokens=1_000,
        )
        for i in range(5)
    )

    receipt = frontier.decide(_scope())

    assert receipt["selection"]["selected_id"] == "large-tasks-better-reduction"


def test_candidates_from_different_trial_cohorts_do_not_compete() -> None:
    frontier = VerifiedGranularityFrontier(_policy())
    frontier.record_many(
        _observation(i, candidate_id="cohort-a", candidate_tokens=600)
        for i in range(5)
    )
    frontier.record_many(
        _observation(i + 5, candidate_id="cohort-b", candidate_tokens=400)
        for i in range(5)
    )

    receipt = frontier.decide(_scope())

    assert receipt["selection"]["status"] == "retain_baseline"
    assert receipt["selection"]["reason"] == "candidate_trial_cohorts_not_aligned"


def test_exact_replay_is_idempotent_and_conflicting_rewrite_is_rejected() -> None:
    frontier = VerifiedGranularityFrontier(_policy())
    original = _observation(1)

    assert frontier.record(original) is True
    assert frontier.record(original) is False
    with pytest.raises(ConflictingGranularityEvidence):
        frontier.record(_observation(1, candidate_tokens=500))


def test_state_round_trip_preserves_decision_and_receipt(tmp_path) -> None:
    frontier = VerifiedGranularityFrontier(_policy())
    frontier.record_many(_observation(i) for i in range(5))
    before = frontier.decide(_scope())
    state_path = tmp_path / "frontier.json"

    frontier.save(state_path)
    restored = VerifiedGranularityFrontier.load(state_path)
    after = restored.decide(_scope())

    assert after == before
    assert verify_granularity_receipt(after)


def test_receipt_mutation_is_detected() -> None:
    frontier = VerifiedGranularityFrontier(_policy())
    receipt = frontier.decide(_scope())

    receipt["selection"]["selected_id"] = "forged"

    assert not verify_granularity_receipt(receipt)


def test_receipt_internal_evidence_digest_is_checked() -> None:
    frontier = VerifiedGranularityFrontier(_policy())
    frontier.record_many(_observation(i) for i in range(5))
    receipt = frontier.decide(_scope())
    receipt["candidate_assessments"][0]["observation_digests"].append(_digest("forged"))
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()

    assert not verify_granularity_receipt(receipt)


@pytest.mark.parametrize(
    "mutation, error",
    [
        ({"source_sha256": "not-a-digest"}, "source_sha256"),
        ({"candidate_tokens": 0}, "candidate_tokens"),
        ({"candidate_id": "full-context"}, "candidate_id"),
    ],
)
def test_malformed_evidence_fails_closed(mutation, error) -> None:
    fields = _observation(1).to_dict()
    fields.update(mutation)

    with pytest.raises(GranularityFrontierError, match=error):
        PairedGranularityObservation.from_dict(fields)


def test_non_finite_evaluator_scores_are_rejected() -> None:
    with pytest.raises(GranularityFrontierError, match="candidate_score"):
        EvaluatorVerdict(
            evaluator_id="judge-v1",
            evaluator_family="judge",
            baseline_score=0.9,
            candidate_score=float("nan"),
            baseline_passed=True,
            candidate_passed=True,
        )
