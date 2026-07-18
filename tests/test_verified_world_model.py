from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from entroly.ravs.world_model import (
    anytime_hoeffding_radius,
    EbbiforgeWorldModelAdapter,
    EmpiricalWorldModel,
    InsufficientWorldModelData,
    TransitionIntegrityError,
    TransitionLedger,
    VerifiedDreamController,
    VerifiedTransition,
    WorldModelPrediction,
    transition_from_ravs,
)
from entroly.ravs.events import OutcomeEvent, TraceEvent
from entroly.autotune import BenchResult, DreamingLoop, FeedbackJournal
from entroly import autotune as autotune_module


def _real_transition(
    index: int,
    *,
    action: float = 0.1,
    reward: float = 0.8,
    environment: str = "unit_env",
    policy_version: str = "candidate",
) -> VerifiedTransition:
    state = (index / 10.0, 0.25)
    return VerifiedTransition(
        transition_id=f"real-{policy_version}-{index}-{action}",
        state=state,
        action=(action, 0.0),
        next_state=(state[0] + action, state[1]),
        reward=reward,
        environment=environment,
        source="pytest",
        verifier="deterministic_test",
        strength="strong",
        policy_version=policy_version,
    )


def _training_set() -> list[VerifiedTransition]:
    samples = [_real_transition(i, action=0.1, reward=0.8) for i in range(6)]
    samples.extend(
        _real_transition(i + 20, action=-0.1, reward=-0.6) for i in range(6)
    )
    return samples


def test_seeded_native_benchmark_is_exactly_replayable():
    native = pytest.importorskip("entroly_core")
    engine = native.EntrolyEngine()
    if not hasattr(engine, "set_benchmark_seed"):
        pytest.skip("installed native engine predates deterministic benchmark identity")

    cases = autotune_module.load_cases()[:10]
    if not cases:
        pytest.skip("packaged autotune benchmark cases are unavailable")
    config = {
        "weight_recency": 0.30,
        "weight_frequency": 0.25,
        "weight_semantic_sim": 0.25,
        "weight_entropy": 0.20,
        "decay_half_life_turns": 15,
        "min_relevance_threshold": 0.05,
        "exploration_rate": 0.30,
    }

    def projection(result):
        return (
            result.context_efficiency,
            result.recall_accuracy,
            result.total_tokens_used,
            result.total_information,
            tuple(
                (
                    row["case_id"],
                    row["recall"],
                    row["tokens_used"],
                    tuple(sorted(row["selected"])),
                )
                for row in result.per_case
            ),
        )

    expected = projection(
        autotune_module.evaluate(config, cases, benchmark_seed=0xE17A0B1)
    )
    for _ in range(19):
        observed = projection(
            autotune_module.evaluate(config, cases, benchmark_seed=0xE17A0B1)
        )
        assert observed == expected


def test_transition_ledger_separates_real_and_dream_evidence(tmp_path):
    ledger = TransitionLedger(tmp_path)
    real = _real_transition(1)
    receipt = ledger.record_real(real)

    dream = VerifiedTransition(
        state=real.state,
        action=real.action,
        next_state=real.next_state,
        reward=real.reward,
        environment=real.environment,
        source="world_model:unit",
        verifier="model_prediction_not_ground_truth",
        strength="synthetic",
        synthetic=True,
        parent_transition_ids=(real.transition_id,),
        parent_receipt_hashes=(receipt.receipt_hash,),
        prediction_confidence=0.8,
        uncertainty=0.2,
        model_version="unit-model-v1",
        model_backend="unit",
        influence_scope="proposal_only",
    )
    ledger.record_dream(dream)

    assert receipt.evidence_class == "real_verified"
    assert ledger.read_real() == [real]
    assert ledger.read_dreams() == [dream]
    with pytest.raises(ValueError, match="strong external verifier"):
        ledger.record_real(dream)
    with pytest.raises(ValueError, match="synthetic transitions only"):
        ledger.record_dream(real)


def test_experiment_proposal_values_information_but_remains_a_dream(tmp_path):
    ledger = TransitionLedger(tmp_path)
    support = _real_transition(1)
    receipt = ledger.record_real(support)

    class AcquisitionModel:
        def fit(self, _transitions):
            return None

        def predict(self, state, action, *, horizon=1):
            del horizon
            exploratory = float(action[0]) > 0.15
            return WorldModelPrediction(
                next_state=tuple(float(x) for x in state),
                reward=0.40 if exploratory else 0.50,
                confidence=0.50 if exploratory else 0.90,
                uncertainty=0.50 if exploratory else 0.10,
                support_count=1,
                supporting_transition_ids=(support.transition_id,),
                model_version="acquisition-test-v1",
                backend="unit",
            )

        def stats(self):
            return {"backend": "unit"}

    controller = VerifiedDreamController(
        ledger,
        AcquisitionModel(),
        min_confidence=0.50,
        experiment_exploration_bonus=0.50,
    )
    rollout = controller.propose_experiment(
        (0.0, 0.0),
        ((0.10, 0.0), (0.20, 0.0)),
        environment="unit_env",
    )

    assert rollout.transitions[0].action == (0.20, 0.0)
    assert rollout.transitions[0].influence_scope == "proposal_only"
    assert rollout.transitions[0].parent_receipt_hashes == (receipt.receipt_hash,)
    assert len(ledger.read_dreams()) == 1


def test_dream_ledger_requires_closed_real_receipt_lineage(tmp_path):
    ledger = TransitionLedger(tmp_path)
    real = _real_transition(1)
    receipt = ledger.record_real(real)

    missing_commitment = VerifiedTransition(
        state=real.state,
        action=real.action,
        next_state=real.next_state,
        reward=real.reward,
        environment=real.environment,
        source="world_model:unit",
        verifier="model_prediction_not_ground_truth",
        strength="synthetic",
        synthetic=True,
        parent_transition_ids=(real.transition_id,),
        model_version="unit-model-v1",
        model_backend="unit",
    )
    with pytest.raises(TransitionIntegrityError, match="parent receipt hashes"):
        ledger.record_dream(missing_commitment)

    false_lineage = replace(
        missing_commitment,
        parent_transition_ids=("not-in-real-ledger",),
        parent_receipt_hashes=(receipt.receipt_hash,),
    )
    with pytest.raises(TransitionIntegrityError, match="absent from the ledger"):
        ledger.record_dream(false_lineage)

    wrong_commitment = replace(
        missing_commitment,
        parent_receipt_hashes=("0" * 64,),
    )
    with pytest.raises(TransitionIntegrityError, match="commitment mismatch"):
        ledger.record_dream(wrong_commitment)


def test_transition_ledger_fails_closed_on_tampering(tmp_path):
    ledger = TransitionLedger(tmp_path)
    ledger.record_real(_real_transition(1, reward=0.8))
    text = ledger.real_path.read_text(encoding="utf-8")
    ledger.real_path.write_text(text.replace('"reward":0.8', '"reward":0.7'), encoding="utf-8")

    with pytest.raises(TransitionIntegrityError, match="receipt mismatch"):
        TransitionLedger(tmp_path)


def test_empirical_world_model_uses_verified_support_and_ranks_actions():
    model = EmpiricalWorldModel(min_samples=4, neighbors=4)
    samples = _training_set()
    model.fit(samples)

    positive = model.predict(samples[2].state, (0.1, 0.0))
    ranked = model.rank_actions(samples[2].state, [(0.1, 0.0), (-0.1, 0.0)])

    assert positive.reward > 0.5
    assert positive.confidence > 0.5
    assert positive.support_count == 4
    assert set(positive.supporting_transition_ids).issubset(
        {sample.transition_id for sample in samples}
    )
    assert ranked[0][0] == (0.1, 0.0)
    assert model.stats()["synthetic_training_transitions"] == 0


def test_empirical_world_model_refuses_synthetic_training_data():
    real = _real_transition(1)
    dream = VerifiedTransition(
        state=real.state,
        action=real.action,
        next_state=real.next_state,
        reward=real.reward,
        environment=real.environment,
        source="world_model:unit",
        verifier="model_prediction_not_ground_truth",
        strength="synthetic",
        synthetic=True,
        parent_transition_ids=(real.transition_id,),
    )
    model = EmpiricalWorldModel(min_samples=1)
    with pytest.raises(ValueError, match="real verified transitions only"):
        model.fit([real, dream])


def test_ravs_bridge_accepts_external_verification_and_rejects_self_report():
    trace = TraceEvent(request_id="request-1", policy_decision="candidate-policy")
    verified = OutcomeEvent(
        request_id="request-1",
        event_type="test_result",
        value="passed",
        strength="strong",
        source="pytest_collector",
        include_in_default_training=True,
    )
    transition = transition_from_ravs(
        trace,
        verified,
        state=(0.0, 0.0),
        action=(0.1, 0.0),
        next_state=(0.1, 0.0),
        environment="repo_tests",
    )
    assert transition.is_real_verified
    assert transition.reward == 1.0
    assert transition.policy_version == "candidate-policy"

    self_report = OutcomeEvent(
        request_id="request-1",
        event_type="agent_self_report",
        value="success",
        strength="weak",
        source="agent",
        include_in_default_training=False,
    )
    with pytest.raises(ValueError, match="strong default-training evidence"):
        transition_from_ravs(
            trace,
            self_report,
            state=(0.0, 0.0),
            action=(0.1, 0.0),
            next_state=(0.1, 0.0),
            environment="repo_tests",
        )


def test_verified_dreams_never_expand_the_real_training_set(tmp_path):
    ledger = TransitionLedger(tmp_path)
    model = EmpiricalWorldModel(min_samples=4, neighbors=4)
    controller = VerifiedDreamController(
        ledger,
        model,
        min_confidence=0.0,
        max_horizon=3,
    )
    for transition in _training_set():
        controller.observe_real(transition)

    rollout = controller.dream(
        _training_set()[2].state,
        lambda _state: [(0.1, 0.0), (-0.1, 0.0)],
        horizon=3,
        environment="unit_env",
        policy_version="dream-candidate",
    )
    stats = controller.stats()

    assert rollout.transitions
    assert all(transition.synthetic for transition in rollout.transitions)
    assert all(
        transition.parent_receipt_hashes for transition in rollout.transitions
    )
    assert all(
        transition.influence_scope == "proposal_only"
        for transition in rollout.transitions
    )
    assert all(transition.model_version for transition in rollout.transitions)
    assert rollout.promotion_status == "proposal_only"
    assert stats["real_training_transitions"] == len(_training_set())
    assert stats["real_transitions"] == len(_training_set())
    assert stats["dream_transitions"] == len(rollout.transitions)
    assert stats["synthetic_training_transitions"] == 0


def test_promotion_gate_requires_real_holdout_evidence(tmp_path):
    ledger = TransitionLedger(tmp_path)
    controller = VerifiedDreamController(
        ledger, EmpiricalWorldModel(min_samples=1)
    )
    candidate = [
        _real_transition(i, reward=1.0, policy_version="candidate")
        for i in range(64)
    ]
    incumbent = [
        _real_transition(i, reward=-1.0, policy_version="incumbent")
        for i in range(64)
    ]
    for transition in [*candidate, *incumbent]:
        ledger.record_real(transition)

    decision = controller.assess_promotion(candidate, incumbent, min_real_samples=10)
    assert decision.promote is True
    assert "real holdout" in decision.reason
    assert decision.candidate_lower_bound > decision.incumbent_upper_bound
    assert decision.anytime_valid is True
    assert decision.boundary_type == "stitched_hoeffding_cs"

    first = candidate[0]
    synthetic = VerifiedTransition(
        state=first.state,
        action=first.action,
        next_state=first.next_state,
        reward=1.0,
        environment=first.environment,
        source="world_model:test",
        verifier="model_prediction_not_ground_truth",
        strength="synthetic",
        synthetic=True,
        parent_transition_ids=(first.transition_id,),
    )
    rejected = controller.assess_promotion(
        [*candidate[:-1], synthetic], incumbent, min_real_samples=10
    )
    assert rejected.promote is False
    assert "synthetic or unverified" in rejected.reason

    duplicated = controller.assess_promotion(
        [candidate[0]] * 10, incumbent, min_real_samples=10
    )
    assert duplicated.promote is False
    assert "duplicate transition IDs" in duplicated.reason

    forged = replace(candidate[0], reward=0.5)
    forged_decision = controller.assess_promotion(
        [forged, *candidate[1:]], incumbent, min_real_samples=10
    )
    assert forged_decision.promote is False
    assert "exactly committed" in forged_decision.reason


def test_anytime_boundary_pays_for_repeated_peeking():
    early = anytime_hoeffding_radius(10, 0.05)
    later = anytime_hoeffding_radius(1_000, 0.05)
    fixed_time = (2.0 * math.log(4.0 / 0.05) / 1_000) ** 0.5

    assert 0.0 < later < early <= 2.0
    assert later > fixed_time


class _FakeLatentState:
    def __init__(self, vector):
        self.vector = list(vector)


class _FakeEbbiforgePredictor:
    def __init__(self):
        self.steps = 0
        self.validation_loss = float("inf")

    def train(self, _payload, epochs, _learning_rate, _batch_size, _val_split):
        self.steps = epochs
        self.validation_loss = 0.01
        return [SimpleNamespace(epoch=epochs, val_loss=self.validation_loss)]

    def get_total_train_steps(self):
        return self.steps

    def get_validation_loss(self):
        return self.validation_loss

    def predict_sequence(self, initial, actions):
        vector = list(initial.vector)
        states = []
        for action in actions:
            vector = [left + right for left, right in zip(vector, action)]
            states.append(_FakeLatentState(vector))
        return SimpleNamespace(future_states=states, confidence=0.9)


def test_ebbiforge_adapter_combines_dynamics_with_verified_reward_support():
    predictor = _FakeEbbiforgePredictor()
    adapter = EbbiforgeWorldModelAdapter(
        predictor,
        state_factory=_FakeLatentState,
        reward_model=EmpiricalWorldModel(min_samples=4, neighbors=4),
        train_kwargs={"epochs": 5},
    )
    with pytest.raises(InsufficientWorldModelData):
        adapter.predict((0.2, 0.25), (0.1, 0.0))

    adapter.fit(_training_set())
    prediction = adapter.predict((0.2, 0.25), (0.1, 0.0), horizon=2)

    assert prediction.backend == "ebbiforge_autoregressive"
    assert prediction.next_state == pytest.approx((0.4, 0.25))
    assert prediction.reward > 0.5
    assert 0.0 < prediction.confidence <= 0.9
    assert adapter.stats()["training_steps"] == 5
    assert adapter.stats()["synthetic_training_transitions"] == 0


def test_dream_scenario_deterministically_changes_proposed_mutation(tmp_path):
    loop = DreamingLoop(FeedbackJournal(str(tmp_path / "journal")))
    config = {
        "weight_recency": 0.30,
        "weight_frequency": 0.25,
        "weight_semantic_sim": 0.25,
        "weight_entropy": 0.20,
        "decay_half_life_turns": 15,
        "min_relevance_threshold": 0.05,
        "exploration_rate": 0.10,
    }
    auth = {"query": "fix auth retry", "task_type": "Debugging", "budget": 4_000}
    cache = {"query": "optimize cache", "task_type": "Performance", "budget": 12_000}

    first = loop._mutate_dream_config(config, auth)
    repeated = loop._mutate_dream_config(config, auth)
    different = loop._mutate_dream_config(config, cache)

    assert first == repeated
    assert first != different


def test_dreaming_loop_uses_model_only_to_choose_real_benchmark_experiments(
    tmp_path, monkeypatch
):
    journal = FeedbackJournal(str(tmp_path / "journal"))
    journal.log(
        weights={"w_r": 0.3, "w_f": 0.25, "w_s": 0.25, "w_e": 0.2},
        reward=0.8,
        query="fix auth",
    )
    controller = VerifiedDreamController(
        TransitionLedger(tmp_path / "world"),
        EmpiricalWorldModel(min_samples=1, neighbors=1),
        min_confidence=0.0,
    )
    loop = DreamingLoop(
        journal,
        config_path=tmp_path / "learning_config.json",
        max_iterations=2,
        world_model_controller=controller,
    )
    loop._last_activity = time.time() - 120

    def evaluate(config, _cases, time_budget=None, benchmark_seed=None):
        del time_budget, benchmark_seed
        efficiency = float(config.get("weight_recency", 0.3)) / 100.0
        return BenchResult(
            context_efficiency=efficiency,
            recall_accuracy=0.8,
            avg_wall_time_ms=1.0,
            total_tokens_used=100,
            total_information=80.0,
        )

    monkeypatch.setattr("entroly.autotune.load_cases", lambda: [{"id": "real"}])
    monkeypatch.setattr("entroly.autotune.evaluate", evaluate)

    result = loop.run_dream_cycle()
    stats = controller.stats()

    assert result["status"] == "completed"
    assert result["world_model_real_transitions"] == 2
    assert stats["real_training_transitions"] == 2
    assert stats["dream_transitions"] >= 1
    assert stats["synthetic_training_transitions"] == 0


def test_dreaming_loop_accepts_a_positive_pessimistic_model_override(tmp_path):
    fallback = {
        "weight_recency": 0.30,
        "weight_frequency": 0.25,
        "weight_semantic_sim": 0.25,
        "weight_entropy": 0.20,
    }
    proposed = dict(fallback, weight_recency=0.40)
    action = DreamingLoop._config_action(fallback, proposed)

    class PositiveController:
        def dream(self, *_args, **_kwargs):
            return SimpleNamespace(
                transitions=(SimpleNamespace(action=action),),
                pessimistic_return=0.01,
            )

    loop = DreamingLoop(
        FeedbackJournal(str(tmp_path / "journal")),
        config_path=tmp_path / "config.json",
        world_model_controller=PositiveController(),
    )

    selected = loop._select_dream_candidate(fallback, [fallback, proposed])

    assert selected == proposed
    assert loop._world_model_guided_experiments == 1


def test_dreaming_loop_rejects_a_negative_pessimistic_model_override(tmp_path):
    fallback = {
        "weight_recency": 0.30,
        "weight_frequency": 0.25,
        "weight_semantic_sim": 0.25,
        "weight_entropy": 0.20,
    }
    proposed = dict(fallback, weight_recency=0.40)
    action = DreamingLoop._config_action(fallback, proposed)

    class NegativeController:
        def dream(self, *_args, **_kwargs):
            return SimpleNamespace(
                transitions=(SimpleNamespace(action=action),),
                pessimistic_return=-0.01,
            )

    loop = DreamingLoop(
        FeedbackJournal(str(tmp_path / "journal")),
        config_path=tmp_path / "config.json",
        world_model_controller=NegativeController(),
    )

    selected = loop._select_dream_candidate(fallback, [fallback, proposed])

    assert selected == fallback
    assert loop._world_model_guided_experiments == 0


def test_dreaming_reward_is_pareto_aligned_and_bounded(tmp_path):
    ledger = TransitionLedger(tmp_path / "world")
    controller = VerifiedDreamController(
        ledger,
        EmpiricalWorldModel(min_samples=1, neighbors=1),
        min_confidence=0.0,
    )
    loop = DreamingLoop(
        FeedbackJournal(str(tmp_path / "journal")),
        config_path=tmp_path / "config.json",
        world_model_controller=controller,
    )
    baseline = BenchResult(0.03, 0.95, 1.0, 100, 80.0)
    improved = BenchResult(0.04, 0.95, 1.0, 90, 80.0)
    dominated = BenchResult(0.05, 0.90, 1.0, 70, 75.0)
    tied = BenchResult(0.03, 0.95, 1.0, 100, 80.0)
    config = {
        "weight_recency": 0.30,
        "weight_frequency": 0.25,
        "weight_semantic_sim": 0.25,
        "weight_entropy": 0.20,
    }

    loop._record_real_benchmark_transition(
        config, dict(config, weight_recency=0.31), baseline, improved
    )
    loop._record_real_benchmark_transition(
        config, dict(config, weight_recency=0.32), baseline, dominated
    )
    loop._record_real_benchmark_transition(
        config, dict(config, weight_recency=0.33), baseline, tied
    )

    assert [transition.reward for transition in ledger.read_real()] == [1.0, -1.0, 0.0]


def test_dreaming_loop_never_trades_recall_for_token_efficiency(
    tmp_path, monkeypatch
):
    journal = FeedbackJournal(str(tmp_path / "journal"))
    journal.log(
        weights={"w_r": 0.3, "w_f": 0.25, "w_s": 0.25, "w_e": 0.2},
        reward=0.8,
        query="preserve evidence",
    )
    config_path = tmp_path / "learning_config.json"
    original = {
        "weight_recency": 0.3,
        "weight_frequency": 0.25,
        "weight_semantic_sim": 0.25,
        "weight_entropy": 0.2,
    }
    config_path.write_text(json.dumps(original), encoding="utf-8")
    loop = DreamingLoop(
        journal,
        config_path=config_path,
        max_iterations=1,
    )
    loop._last_activity = time.time() - 120
    results = iter(
        [
            BenchResult(
                context_efficiency=0.03,
                recall_accuracy=0.95,
                avg_wall_time_ms=1.0,
                total_tokens_used=100,
                total_information=80.0,
            ),
            BenchResult(
                context_efficiency=0.05,
                recall_accuracy=0.90,
                avg_wall_time_ms=1.0,
                total_tokens_used=60,
                total_information=75.0,
            ),
        ]
    )

    monkeypatch.setattr("entroly.autotune.load_cases", lambda: [{"id": "real"}])
    monkeypatch.setattr(
        "entroly.autotune.evaluate",
        lambda _config, _cases, time_budget=None, benchmark_seed=None: next(results),
    )

    result = loop.run_dream_cycle()

    assert result["status"] == "completed"
    assert result["improvements"] == 0
    assert json.loads(config_path.read_text(encoding="utf-8")) == original
