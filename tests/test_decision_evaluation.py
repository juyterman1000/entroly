from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from entroly.decision_evaluation import (
    DecisionScope,
    JsonFieldProjection,
    context_regret,
    continuity_debt,
    observe_decisions,
)

PROJECTION = JsonFieldProjection("command-exit.v1", ("exit_code",))
SCOPE = DecisionScope(
    "local-command",
    "v1",
    '{"temperature":null}',
    "coding-command",
    "expected-exit.v1",
    PROJECTION.protocol_id,
    "o200k_base",
    "static-100",
    "a" * 64,
    "b" * 40,
)


def observe(**kwargs):
    defaults = dict(
        scope=SCOPE,
        trial_id="trial-1",
        task_id="task-1",
        query="run tests",
        full_output={"exit_code": 0},
        selected_output={"exit_code": 0},
        full_repeat_output={"exit_code": 0},
        projection=PROJECTION,
        full_context_tokens=100,
        selected_tokens=50,
        budget=100,
        assurance_verdict="structurally_valid_risk_unmeasured",
    )
    return observe_decisions(**{**defaults, **kwargs})


def test_projection_ignores_prose_and_records_no_raw_outputs():
    observation = observe(
        selected_output={"exit_code": 0, "explanation": "sensitive prose"}
    )
    assert observation["decision_divergence"] == 0
    assert "sensitive" not in str(observation)
    assert observation["provider_observed_usage"] is None
    assert observation["risk_bound"] is None


def test_self_divergence_control_does_not_get_subtracted_from_context_risk():
    obs = observe(selected_output={"exit_code": 1}, full_repeat_output={"exit_code": 1})
    report = context_regret([obs])
    assert report["decision_divergence_regret"] == 1
    assert report["model_self_divergence"] == 1
    assert report["causal_attribution"].startswith("unestablished")


def test_task_regret_is_distinct_and_can_be_negative():
    obs = observe(
        full_output={"exit_code": 1},
        selected_output={"exit_code": 0},
        task_loss=lambda decision, truth: float(decision["exit_code"] != truth),
        truth=0,
    )
    report = context_regret([obs])
    assert report["decision_divergence_regret"] == 1
    assert report["task_context_regret"] == -1
    assert report["internal_selection_regret"] is None
    assert context_regret([observe()])["task_context_regret"] is None


@pytest.mark.parametrize(
    "output", ["free form answer", "{}", [], {"exit_code": float("nan")}]
)
def test_unstructured_missing_and_nonfinite_decisions_are_rejected(output):
    with pytest.raises((ValueError, TypeError)):
        observe(selected_output=output)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 2, True])
def test_task_loss_requires_declared_bounded_finite_values(value):
    with pytest.raises(ValueError):
        observe(task_loss=lambda *_: value)


def test_projection_protocol_is_bound_to_scope():
    with pytest.raises(ValueError, match="projection"):
        observe(projection=JsonFieldProjection("other", ("exit_code",)))


def test_cross_scope_and_repeated_trials_are_not_aggregated():
    obs = observe()
    changed = observe(scope=replace(SCOPE, model_version="v2"), trial_id="trial-2")
    with pytest.raises(ValueError, match="cross-scope"):
        context_regret([obs, changed])
    with pytest.raises(ValueError, match="duplicate"):
        context_regret([obs, obs])
    assert context_regret([obs, observe(trial_id="trial-2")])["trials"] == 2


def test_edited_observation_cannot_preserve_its_evidence_identity():
    obs = observe()
    obs["decision_divergence"] = 1
    with pytest.raises(ValueError, match="integrity"):
        context_regret([obs])


def test_resealed_observation_cannot_reuse_another_scope_identity():
    from entroly.context_receipts.models import stable_hash

    obs = observe()
    obs["scope"]["model_version"] = "different"
    obs["observation_id"] = stable_hash(
        {k: v for k, v in obs.items() if k != "observation_id"}
    )
    with pytest.raises(ValueError, match="scope commitment"):
        context_regret([obs])


@pytest.mark.parametrize("field", list(asdict(SCOPE)))
def test_scope_metadata_cannot_silently_disappear(field):
    with pytest.raises(ValueError):
        replace(SCOPE, **{field: ""})


@pytest.mark.parametrize(
    "field", ["budget", "full_context_tokens", "selected_tokens", "recovered_tokens"]
)
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_token_metadata_is_not_fabricated(field, value):
    with pytest.raises(ValueError):
        observe(**{field: value})


def test_scope_settings_have_a_canonical_identity():
    left = replace(SCOPE, settings_json='{"seed": 7, "temperature": 0}')
    right = replace(SCOPE, settings_json='{"temperature":0,"seed":7}')
    assert left.scope_id == right.scope_id


def test_continuity_requires_verified_recovery_before_decision():
    def debt(visible, verified):
        return continuity_debt(
            previously_omitted=["early", "irrelevant"],
            newly_required=["early", "never_omitted"],
            visible_before_decision=visible,
            verified_recovered=verified,
            weights={"early": 3},
        )

    assert debt([], ["early"])["missed_context_debt"] == 3
    assert debt(["early"], [])["missed_context_debt"] == 3
    assert debt(["early"], ["early"])["missed_context_debt"] == 0
    assert debt(["early"], ["early"])["newly_relevant_count"] == 1
