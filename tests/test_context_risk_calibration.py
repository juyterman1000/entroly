"""Statistical and protocol checks for fixed-policy paired divergence bounds."""

from __future__ import annotations

import math
import json
from dataclasses import asdict, replace

import pytest

from entroly.context_assurance import ContextAssuranceError, require_assurance
from entroly.context_receipts.models import stable_hash
from entroly.decision_evaluation import (
    DecisionScope,
    FixedPolicyCalibrationProtocol,
    JsonFieldProjection,
    binomial_upper_bound,
    calibrate_fixed_policy_divergence,
    observe_decisions,
)
from benchmarks.context_risk_calibration import run


PROJECTION = JsonFieldProjection("command-exit.v1", ("exit_code",))
SCOPE = DecisionScope(
    "fixed-model",
    "version-1",
    '{"seed":42,"temperature":0}',
    "independent-coding-tasks",
    "command-exit-evaluator.v1",
    PROJECTION.protocol_id,
    "o200k_base",
    "fixed-selector-budget-4096",
    "a" * 64,
    "b" * 40,
)


def observation(task_id: str, *, diverged: bool = False, self_changed: bool = False):
    return observe_decisions(
        scope=SCOPE,
        trial_id=f"trial-{task_id}",
        task_id=task_id,
        query="run the task",
        full_output={"exit_code": 0},
        selected_output={"exit_code": int(diverged)},
        full_repeat_output={"exit_code": int(self_changed)},
        projection=PROJECTION,
        full_context_tokens=200,
        selected_tokens=100,
        budget=4096,
        assurance_verdict="structurally_valid_risk_unmeasured",
        seeds=(42, 42, 42),
    )


def protocol(calibration_count: int = 5, holdout_count: int = 100):
    return FixedPolicyCalibrationProtocol(
        protocol_id="paired-independent-tasks-v1",
        dataset_id="declared-coding-task-cohort-v1",
        sampling_design="independent tasks sampled by repository before outcomes",
        precommit_reference="external-register:fixture-protocol-v1",
        scope_id=SCOPE.scope_id,
        projection_commitment=stable_hash(asdict(PROJECTION)),
        calibration_task_ids=tuple(f"cal-{i}" for i in range(calibration_count)),
        holdout_task_ids=tuple(f"test-{i}" for i in range(holdout_count)),
        risk_target=0.05,
        alpha=0.05,
    )


def rows(spec):
    return (
        [observation(task_id) for task_id in spec.calibration_task_ids],
        [observation(task_id) for task_id in spec.holdout_task_ids],
    )


def test_exact_binomial_bound_has_known_edge_values():
    assert binomial_upper_bound(0, 1) == pytest.approx(0.95)
    assert binomial_upper_bound(0, 2) == pytest.approx(1 - math.sqrt(0.05))
    assert binomial_upper_bound(1, 2) == pytest.approx(math.sqrt(0.95))
    assert binomial_upper_bound(2, 2) == 1.0
    assert binomial_upper_bound(0, 100) == pytest.approx(0.029513049607, rel=1e-8)


def test_bound_is_monotone_in_failures_for_a_larger_cohort():
    bounds = [binomial_upper_bound(k, 1000) for k in (0, 10, 100, 500, 1000)]
    assert bounds == sorted(bounds)
    assert all(bound >= k / 1000 for bound, k in zip(bounds, (0, 10, 100, 500, 1000)))


@pytest.mark.parametrize("probability", [0.01, 0.1, 0.5, 0.9])
@pytest.mark.parametrize("trials", range(1, 8))
def test_bound_has_at_least_nominal_coverage_on_small_binomial_oracle(
    trials, probability
):
    alpha = 0.1
    covered = sum(
        math.comb(trials, failures)
        * probability**failures
        * (1 - probability) ** (trials - failures)
        for failures in range(trials + 1)
        if binomial_upper_bound(failures, trials, alpha=alpha) + 1e-12
        >= probability
    )
    assert covered >= 1 - alpha - 1e-12


def test_report_uses_untouched_holdout_and_never_grants_production_authority():
    spec = protocol()
    calibration, holdout = rows(spec)
    calibration[0] = observation(spec.calibration_task_ids[0], diverged=True)
    holdout[0] = observation(spec.holdout_task_ids[0], self_changed=True)
    report = calibrate_fixed_policy_divergence(
        spec, calibration=calibration, holdout=holdout
    )
    assert report["calibration"]["divergences"] == 1
    assert report["holdout"]["divergences"] == 0
    assert report["holdout"]["full_repeat_divergences"] == 1
    assert report["holdout"]["upper_rate_bound"] == pytest.approx(
        binomial_upper_bound(0, 100)
    )
    assert report["conditional_target_met"] is True
    assert report["production_authority"] is False
    assert "not causal" in report["limitations"]
    with pytest.raises(ContextAssuranceError):
        require_assurance(
            {
                "verdict": "structurally_valid_risk_unmeasured",
                "decision_risk": report,
            },
            decision_risk=True,
        )


def test_holdout_failures_can_exceed_the_declared_target():
    spec = protocol(2, 20)
    calibration, holdout = rows(spec)
    for index in range(3):
        holdout[index] = observation(spec.holdout_task_ids[index], diverged=True)
    report = calibrate_fixed_policy_divergence(
        spec, calibration=calibration, holdout=holdout
    )
    assert report["conditional_target_met"] is False
    assert report["holdout"]["upper_rate_bound"] > spec.risk_target


def test_duplicate_task_with_distinct_trial_cannot_fake_independent_sample():
    spec = protocol(2, 2)
    calibration, holdout = rows(spec)
    duplicate = dict(holdout[0], trial_id="another-trial")
    duplicate["observation_id"] = stable_hash(
        {key: value for key, value in duplicate.items() if key != "observation_id"}
    )
    holdout[1] = duplicate
    with pytest.raises(ValueError, match="split tasks"):
        calibrate_fixed_policy_divergence(
            spec, calibration=calibration, holdout=holdout
        )


def test_scope_projection_integrity_and_split_changes_fail_closed():
    spec = protocol(2, 2)
    calibration, holdout = rows(spec)
    with pytest.raises(ValueError, match="scope"):
        calibrate_fixed_policy_divergence(
            replace(spec, scope_id="another-scope"),
            calibration=calibration,
            holdout=holdout,
        )
    with pytest.raises(ValueError, match="projection"):
        calibrate_fixed_policy_divergence(
            replace(spec, projection_commitment="another-projection"),
            calibration=calibration,
            holdout=holdout,
        )
    with pytest.raises(ValueError, match="declared task count"):
        calibrate_fixed_policy_divergence(
            spec, calibration=calibration, holdout=holdout[:1]
        )
    holdout[0]["decision_divergence"] = 1.0
    with pytest.raises(ValueError, match="integrity"):
        calibrate_fixed_policy_divergence(
            spec, calibration=calibration, holdout=holdout
        )


@pytest.mark.parametrize("bad", [0, 1, -0.1, float("nan"), True])
def test_protocol_rejects_invalid_probabilities(bad):
    with pytest.raises(ValueError):
        replace(protocol(2, 2), risk_target=bad)
    with pytest.raises(ValueError):
        replace(protocol(2, 2), alpha=bad)


def test_protocol_rejects_reused_holdout_task():
    with pytest.raises(ValueError, match="disjoint"):
        replace(protocol(2, 2), holdout_task_ids=("cal-0", "test-1"))
    with pytest.raises(ValueError, match="precommit_reference"):
        replace(protocol(2, 2), precommit_reference="")


@pytest.mark.parametrize("failures,trials", [(0, 0), (-1, 5), (6, 5), (True, 5)])
def test_bound_rejects_invalid_counts(failures, trials):
    with pytest.raises(ValueError):
        binomial_upper_bound(failures, trials)


def test_offline_command_reads_exact_declared_cohort_and_records_file_digests(
    tmp_path,
):
    spec = protocol(2, 2)
    calibration, holdout = rows(spec)
    protocol_path = tmp_path / "protocol.json"
    observations_path = tmp_path / "observations.json"
    protocol_path.write_text(json.dumps(asdict(spec)), encoding="utf-8")
    observations_path.write_text(
        json.dumps(holdout + calibration), encoding="utf-8"
    )
    bundle = run(protocol_path, observations_path)
    assert bundle["report"]["holdout"]["tasks"] == 2
    assert len(bundle["protocol_file_sha256"]) == 64
    assert len(bundle["observations_file_sha256"]) == 64
    assert "run the task" not in str(bundle)
    observations_path.write_text(json.dumps(holdout + calibration + [holdout[0]]))
    with pytest.raises(ValueError, match="exactly the declared task set"):
        run(protocol_path, observations_path)
