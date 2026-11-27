"""Durable proposal memory must never replace a fresh benchmark gate."""

import json
import time
from types import SimpleNamespace

import pytest

from entroly import autotune
from entroly.improvement_history import ImprovementHistory, digest


def measured(efficiency=0.02, recall=0.9, *, timeout=False):
    return autotune.BenchResult(
        efficiency, recall, 2.0, 100, 2.0,
        per_case=[{"status": "timeout" if timeout else "ok",
                   "case_id": "case", "tokens_used": 100,
                   "query": "private-benchmark-query", "selected": ["/private/source.py"]}],
    )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    config = dict(autotune._default_runtime_config(), exploration_rate=0.1)
    path = tmp_path / "runtime" / "learning_tuning_config.json"
    autotune.save_config(config, path)
    state = SimpleNamespace(
        config=config, path=path, cases=[{
            "id": "case", "query": "private-benchmark-query", "token_budget": 100,
            "fragments": [{"source": "/private/source.py", "expected_selected": True}],
        }],
        evaluated=[], records=[], mutations=[], saved=[],
        landscape=lambda _config: measured(),
    )
    monkeypatch.setattr(autotune, "runtime_config_path", lambda: path)
    monkeypatch.setattr(autotune, "load_cases", lambda: state.cases)
    monkeypatch.setattr(autotune, "RESULTS_PATH", tmp_path / "results.tsv")
    monkeypatch.setattr(autotune, "_native_history_fingerprint", lambda: "a" * 64)
    real_record = ImprovementHistory.record
    real_save = autotune.save_config

    def record(self, **kwargs):
        receipt = real_record(self, **kwargs)
        state.records.append(kwargs)
        return receipt

    def evaluate(config, cases, time_budget=None, benchmark_seed=None):
        assert cases == state.cases
        assert benchmark_seed == autotune.AUTOTUNE_BENCHMARK_SEED
        state.evaluated.append(dict(config))
        return state.landscape(config)

    def save(config, config_path=None):
        assert config in state.evaluated
        if config != state.config:
            assert any(row["candidate"] == autotune._history_config(config)
                       and row["decision"] == "improved" for row in state.records)
        state.saved.append(dict(config))
        real_save(config, config_path)

    monkeypatch.setattr(ImprovementHistory, "record", record)
    monkeypatch.setattr(autotune, "evaluate", evaluate)
    monkeypatch.setattr(autotune, "save_config", save)

    def proposals(values):
        values = iter(values)

        def mutate(config):
            state.mutations.append(dict(config))
            return dict(config, exploration_rate=next(values), _mutated_param="exploration_rate")

        monkeypatch.setattr(autotune, "mutate_config", mutate)

    state.proposals = proposals
    return state


def make_dream(harness, monkeypatch, *, model=None):
    # Deliberately no journal_dir or load attribute: storage derives from config.
    loop = autotune.DreamingLoop(
        SimpleNamespace(), config_path=harness.path, max_iterations=1,
        cooldown_s=0, world_model_controller=model,
    )
    loop._last_activity = time.time() - 120
    monkeypatch.setattr(loop, "generate_experiment_scenarios", lambda: [
        {"task_type": "Testing", "budget": 8000, "query": "private-proposal-query"},
    ])
    return loop


@pytest.mark.parametrize("improves", [True, False])
def test_autotune_records_actual_gate_before_persisting_and_reopens(harness, improves):
    harness.proposals([0.2])
    harness.landscape = lambda c: measured(
        0.04 if improves else 0.01
    ) if c["exploration_rate"] != 0.1 else measured()
    autotune.run_autotune(iterations=1)

    row = harness.records[0]
    assert row["decision"] == ("improved" if improves else "rejected")
    assert row["prediction"] is None
    assert set(row["scope"]) == {
        "cases_sha256", "evaluator_sha256", "native_sha256", "seed", "objective",
    }
    assert row["scope"]["objective"] == "composite_drift.v1"
    assert set(row["candidate_metrics"]) == {
        "recall_accuracy", "context_efficiency", "avg_wall_time_ms",
        "total_tokens_used", "timed_out",
    }
    reopened = ImprovementHistory(harness.path.parent)
    rejected = reopened.rejected_candidates(scope=row["scope"], incumbent=row["incumbent"])
    assert (digest(row["candidate"]) in rejected) is not improves
    assert harness.path.parent.exists()
    disk = reopened.path.read_text(encoding="utf-8")
    assert disk
    assert "private-benchmark-query" not in disk
    assert "/private/source.py" not in disk
    assert "private-proposal-query" not in disk


@pytest.mark.parametrize("change", ["dataset", "seed", "native", "objective", "baseline", "budget"])
def test_rejections_are_exactly_scoped(harness, monkeypatch, change):
    history = autotune._BenchmarkHistory(harness.path, harness.cases, "composite_drift.v1", 5.0)
    candidate = dict(harness.config, exploration_rate=0.2)
    history.record(harness.config, candidate, measured(), measured(0.01), False)
    assert history.rejected(harness.config) == {digest(candidate)}
    cases = harness.cases
    objective = "composite_drift.v1"
    incumbent = harness.config
    budget = 5.0
    if change == "dataset":
        cases = [dict(cases[0], query="different-private-query")]
    elif change == "seed":
        monkeypatch.setattr(autotune, "AUTOTUNE_BENCHMARK_SEED", 12)
    elif change == "native":
        monkeypatch.setattr(autotune, "_native_history_fingerprint", lambda: "b" * 64)
    elif change == "objective":
        objective = "pareto_recall_efficiency.v1"
    elif change == "baseline":
        incumbent = dict(incumbent, weight_recency=0.4)
    elif change == "budget":
        budget = 2.0
    another = autotune._BenchmarkHistory(harness.path, cases, objective, budget)
    assert another.error is None
    assert another.rejected(incumbent) == set()


def test_next_autotune_run_avoids_rejection_and_actually_evaluates_alternative(harness):
    harness.landscape = lambda c: measured() if c == harness.config else measured(0.01)
    harness.proposals([0.2])
    autotune.run_autotune(iterations=1)
    harness.proposals([0.2, 0.25])
    autotune.run_autotune(iterations=1)
    assert [c["exploration_rate"] for c in harness.evaluated] == [0.1, 0.2, 0.1, 0.25]
    assert len(harness.records) == 2
    assert harness.records[1]["candidate"]["exploration_rate"] == 0.25


def test_autotune_all_known_has_bounded_retries_and_fresh_validation(harness):
    harness.landscape = lambda c: measured() if c == harness.config else measured(0.01)
    harness.proposals([0.2] * 12)
    autotune.run_autotune(iterations=1)
    harness.mutations.clear()
    autotune.run_autotune(iterations=1)
    assert len(harness.mutations) == 9  # original plus at most eight retries
    assert [c["exploration_rate"] for c in harness.evaluated] == [0.1, 0.2, 0.1, 0.2]


@pytest.mark.parametrize("improves", [True, False])
def test_dream_records_keep_and_reject_with_distinct_objective(harness, monkeypatch, improves):
    loop = make_dream(harness, monkeypatch)
    candidate = dict(harness.config, exploration_rate=0.2)
    monkeypatch.setattr(loop, "_mutate_dream_config", lambda *a, **kw: candidate)
    harness.landscape = lambda c: measured() if c == harness.config else measured(
        0.03 if improves else 0.01
    )
    result = loop.run_dream_cycle()
    assert result["status"] == "completed"
    assert result["history_error"] is None
    assert result["improvements"] == int(improves)
    row = harness.records[0]
    assert row["scope"]["objective"] == "pareto_recall_efficiency.v1"
    assert row["decision"] == ("improved" if improves else "rejected")
    rejected = ImprovementHistory(harness.path.parent).rejected_candidates(
        scope=row["scope"], incumbent=row["incumbent"],
    )
    assert (digest(candidate) in rejected) is not improves


def test_next_dream_cycle_uses_fresh_proposal_after_rejection(harness, monkeypatch):
    loop = make_dream(harness, monkeypatch)
    generated = []

    def mutate(config, scenario, *, candidate_index):
        generated.append(candidate_index)
        return dict(config, exploration_rate=0.2 + candidate_index * 0.01)

    monkeypatch.setattr(loop, "_mutate_dream_config", mutate)
    harness.landscape = lambda c: measured() if c == harness.config else measured(0.01)
    assert loop.run_dream_cycle()["status"] == "completed"
    assert generated == [0]  # healthy empty history does not expand normal dreaming
    assert loop.run_dream_cycle()["status"] == "completed"
    assert generated == [0, 0, 1]
    assert [c["exploration_rate"] for c in harness.evaluated] == pytest.approx([0.1, 0.2, 0.1, 0.21])


@pytest.mark.parametrize("model", [None, object()])
def test_dream_all_known_falls_back_to_actual_benchmark(harness, monkeypatch, model):
    loop = make_dream(harness, monkeypatch, model=model)
    candidate = dict(harness.config, exploration_rate=0.2)
    generated = []

    def mutate(*args, **kwargs):
        generated.append(kwargs["candidate_index"])
        return candidate

    monkeypatch.setattr(loop, "_mutate_dream_config", mutate)
    harness.landscape = lambda c: measured() if c == harness.config else measured(0.01)
    loop.run_dream_cycle()
    generated.clear()
    result = loop.run_dream_cycle()
    assert result["status"] == "completed"
    assert len(generated) <= autotune.DREAMING_CANDIDATE_POOL + 1
    assert harness.evaluated[-1] == candidate
    assert len(harness.records) == 2


@pytest.mark.parametrize("operation", ["initialize", "read", "record"])
def test_broken_history_is_visible_and_never_promotes_regression(harness, monkeypatch, caplog, operation):
    def fail(*args, **kwargs):
        raise OSError("private/path should not become a receipt reason")

    monkeypatch.setattr(ImprovementHistory, {
        "initialize": "__init__", "read": "rejected_candidates", "record": "record",
    }[operation], fail)
    loop = make_dream(harness, monkeypatch)
    monkeypatch.setattr(loop, "_mutate_dream_config", lambda c, *a, **kw: dict(c, exploration_rate=0.2))
    harness.landscape = lambda c: measured() if c == harness.config else measured(0.01)
    result = loop.run_dream_cycle()
    assert result["status"] == "completed"
    assert result["improvements"] == 0
    assert result["history_error"] == f"history_{operation}_OSError"
    assert loop.stats()["history_error"] == result["history_error"]
    assert "history unavailable" in caplog.text
    assert len(harness.evaluated) == 2
    assert json.loads(harness.path.read_text()) == harness.config


@pytest.mark.parametrize("dream", [False, True])
def test_evaluation_error_is_recorded_and_never_saves(harness, monkeypatch, caplog, dream):
    def landscape(config):
        if config != harness.config:
            raise RuntimeError("actual evaluator failure")
        return measured()

    harness.landscape = landscape
    if dream:
        loop = make_dream(harness, monkeypatch)
        monkeypatch.setattr(loop, "_mutate_dream_config", lambda c, *a, **kw: dict(c, exploration_rate=0.2))
        result = loop.run_dream_cycle()
        assert result["evaluation_errors"] == 1
        assert "benchmark evaluation failed" in caplog.text
    else:
        harness.proposals([0.2])
        with pytest.raises(RuntimeError, match="actual evaluator failure"):
            autotune.run_autotune(iterations=1)
    assert not harness.saved
    assert harness.records[0]["decision"] == "evaluation_error"
    assert harness.records[0]["candidate_metrics"] is None


def test_history_failure_does_not_hide_benchmark_failure(harness, monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise OSError("history unavailable")

    def landscape(config):
        if config != harness.config:
            raise RuntimeError("benchmark still fails")
        return measured()

    monkeypatch.setattr(ImprovementHistory, "record", fail)
    harness.landscape = landscape
    harness.proposals([0.2])
    with pytest.raises(RuntimeError, match="benchmark still fails"):
        autotune.run_autotune(iterations=1)
    assert "history_record_OSError" in caplog.text
    assert not harness.saved


def test_feedback_prune_calls_history_and_preserves_bool_on_history_failure(harness, monkeypatch, caplog):
    journal = autotune.FeedbackJournal(str(harness.path.parent / "feedback"))
    autotune.DreamingLoop(journal, config_path=harness.path)
    pruned = []

    def prune(self):
        pruned.append(True)
        raise OSError("history prune failed")

    monkeypatch.setattr(ImprovementHistory, "prune", prune)
    assert journal.prune() is True
    assert pruned == [True]
    assert "history prune failed" in caplog.text


def test_timeout_metrics_are_sanitized(harness):
    history = autotune._BenchmarkHistory(harness.path, harness.cases, "composite_drift.v1", 5.0)
    history.record(harness.config, dict(harness.config, exploration_rate=0.2),
                   measured(), measured(0, 0, timeout=True), False)
    assert harness.records[0]["candidate_metrics"] == {
        "recall_accuracy": 0.0, "context_efficiency": 0.0,
        "avg_wall_time_ms": 2.0, "total_tokens_used": 100, "timed_out": True,
    }

@pytest.mark.parametrize("dream", [False, True])
def test_better_score_cannot_replace_a_required_source(harness, monkeypatch, dream):
    harness.cases[0]["fragments"].append(
        {"source": "other.py", "expected_selected": True}
    )

    def landscape(config):
        if config == harness.config:
            return measured(recall=0.5)
        result = measured(efficiency=0.04, recall=0.5)
        result.per_case[0]["selected"] = ["other.py"]
        return result

    harness.landscape = landscape
    if dream:
        loop = make_dream(harness, monkeypatch)
        monkeypatch.setattr(
            loop, "_mutate_dream_config",
            lambda config, *_args, **_kwargs: dict(config, exploration_rate=0.4),
        )
        assert loop.run_dream_cycle()["improvements"] == 0
    else:
        harness.proposals([0.4])
        autotune.run_autotune(iterations=1)

    assert autotune.load_config(harness.path) == harness.config
    assert harness.records[0]["decision"] == "rejected"
    assert harness.records[0]["reason"] == "benchmark_evidence_lost"


def test_evaluator_counts_compact_tokens_without_crediting_partial_evidence(monkeypatch):
    import sys

    class Engine:
        def __init__(self, **_kwargs):
            pass

        def ingest(self, *_args):
            pass

        def optimize(self, *_args):
            return {"selected": [
                {"source": "caller.py", "variant": "full", "token_count": 30},
                {"source": "implementation.py", "variant": "reference", "token_count": 4},
            ]}

    monkeypatch.setitem(sys.modules, "entroly_core", SimpleNamespace(EntrolyEngine=Engine))
    cases = [{
        "id": "rename",
        "query": "rename function",
        "token_budget": 40,
        "fragments": [
            {"source": name, "content": "", "token_count": tokens, "expected_selected": True}
            for name, tokens in [("caller.py", 30), ("implementation.py", 100)]
        ],
    }]
    result = autotune.evaluate({}, cases)

    assert result.total_tokens_used == 34
    assert result.recall_accuracy == 0.5
    assert result.per_case[0]["selected"] == ["caller.py"]
    assert autotune._benchmark_rejection(result, result, cases) is None

@pytest.mark.parametrize("coordinated", [False, True])
@pytest.mark.parametrize("loses_evidence", [False, True])
def test_coordinated_trial_after_plateau_uses_normal_promotion_gate(
    harness, monkeypatch, loses_evidence, coordinated,
):
    harness.config["decay_half_life_turns"] = 15
    # Persist the starting config before installing the save spy.
    harness.path.write_text(json.dumps(harness.config), encoding="utf-8")
    proposals = []

    def mutate(config, *, parameters=1):
        proposals.append(parameters)
        candidate = dict(config, exploration_rate=0.2)
        candidate["_mutated_param"] = "exploration_rate"
        if parameters == 2:
            candidate["decay_half_life_turns"] = 18
            candidate["_mutated_param"] = "exploration_rate,decay_half_life_turns"
        return candidate

    def landscape(config):
        if config == harness.config:
            return measured()
        if config.get("decay_half_life_turns") != 18:
            return measured(0.01)
        result = measured(0.04)
        if loses_evidence:
            result.per_case[0]["selected"] = []
        return result

    monkeypatch.setattr(autotune, "mutate_config", mutate)
    harness.landscape = landscape
    autotune.run_autotune(iterations=4, coordinated=coordinated)

    assert proposals[-1] == (2 if coordinated else 1)
    assert all(count == 1 for count in proposals[:-1])
    assert len(harness.evaluated) == 5  # Baseline plus the four requested trials.
    assert len(harness.records) == 4
    assert harness.records[-1]["kind"] == ("joint_mutation" if coordinated else "mutation")
    accepted = coordinated and not loses_evidence
    assert harness.records[-1]["decision"] == ("improved" if accepted else "rejected")
    saved = autotune.load_config(harness.path)
    if not accepted:
        if coordinated:
            assert harness.records[-1]["reason"] == "benchmark_evidence_lost"
        assert saved == harness.config
    else:
        assert saved["decay_half_life_turns"] == 18
        assert saved["exploration_rate"] == 0.2
    assert "_mutated_param" not in saved


def test_joint_mutations_are_distinct_bounded_and_keep_integer_parameters(monkeypatch):
    monkeypatch.setattr(autotune.random, "sample", lambda population, count: [
        "exploration_rate", "decay_half_life_turns",
    ])
    monkeypatch.setattr(autotune.random, "gauss", lambda *_args: 100)
    monkeypatch.setattr(autotune.random, "randint", lambda *_args: 3)
    config = dict(autotune._default_runtime_config(), exploration_rate=0.29,
                  decay_half_life_turns=49)
    candidate = autotune.mutate_config(config, parameters=2)

    assert candidate["exploration_rate"] == 0.3
    assert candidate["decay_half_life_turns"] == 50
    assert type(candidate["decay_half_life_turns"]) is int
    assert candidate["_mutated_param"] == "exploration_rate,decay_half_life_turns"
    assert config["exploration_rate"] == 0.29
    assert sum(candidate[name] for name in config if name.startswith("weight_")) == pytest.approx(1)

@pytest.mark.parametrize("field,value", [
    ("context_efficiency", float("inf")),
    ("context_efficiency", float("nan")),
    ("recall_accuracy", float("nan")),
    ("recall_accuracy", 1.1),
    ("avg_wall_time_ms", -1.0),
    ("total_information", float("inf")),
    ("total_tokens_used", True),
])
def test_invalid_measurements_never_authorize_promotion(harness, field, value):
    harness.proposals([0.2])
    def landscape(config):
        result = measured()
        if config != harness.config:
            setattr(result, field, value)
        return result
    harness.landscape = landscape

    autotune.run_autotune(iterations=1)

    assert autotune.load_config(harness.path) == harness.config
    assert all(config == harness.config for config in harness.saved)
    assert harness.records[-1]["decision"] == "evaluation_error"
    assert harness.records[-1]["reason"] == "invalid_benchmark_measurement"


def test_invalid_baseline_fails_before_any_promotion(harness):
    harness.landscape = lambda _config: measured(float("inf"))
    with pytest.raises(RuntimeError, match="invalid_benchmark_measurement"):
        autotune.run_autotune(iterations=1)
    assert harness.saved == []
