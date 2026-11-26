import csv
import json
from types import SimpleNamespace

import pytest

from entroly import autotune


@pytest.fixture
def harness(tmp_path, monkeypatch):
    initial = {
        "weight_recency": 0.30,
        "weight_frequency": 0.25,
        "weight_semantic_sim": 0.25,
        "weight_entropy": 0.20,
        "exploration_rate": 0.10,
        "decay_half_life_turns": 15,
    }
    path = tmp_path / "learning_tuning_config.json"
    results_path = tmp_path / "results.tsv"
    autotune.save_config(initial, path)
    cases = [{"id": "mock-case"}]
    state = SimpleNamespace(
        initial=initial, path=path, results_path=results_path,
        evaluated=[], saved=[], logged=[], mutation_inputs=[],
    )
    monkeypatch.setattr(autotune, "runtime_config_path", lambda: path)
    monkeypatch.setattr(autotune, "load_cases", lambda: cases)
    monkeypatch.setattr(autotune, "RESULTS_PATH", results_path)
    real_save = autotune.save_config
    real_log = autotune.log_result

    def save(config, config_path=None):
        assert config_path == path
        assert any(config == tested for tested, _ in state.evaluated)
        real_save(config, config_path)
        state.saved.append(json.loads(path.read_text(encoding="utf-8")))

    def log(iteration, config, result, status, description):
        assert any(config == tested and result is measured
                   for tested, measured in state.evaluated)
        if status == "keep":
            assert json.loads(path.read_text(encoding="utf-8")) == config
        state.logged.append((iteration, dict(config), result, status))
        real_log(iteration, config, result, status, description)

    def install(landscape, mutations, param="exploration_rate"):
        values = iter(mutations)

        def mutate(config):
            state.mutation_inputs.append(dict(config))
            return dict(config, **{param: next(values), "_mutated_param": param})

        def evaluate(config, actual_cases, time_budget, benchmark_seed):
            assert actual_cases is cases
            assert time_budget == 2.5
            assert benchmark_seed == autotune.AUTOTUNE_BENCHMARK_SEED
            score, latency = landscape(config)
            penalty = -autotune.composite_score(
                autotune.BenchResult(0, 0, 0, 0, 0), config, initial,
            )
            result = autotune.BenchResult(
                context_efficiency=(score + penalty - 0.3) / 40,
                recall_accuracy=0.5, avg_wall_time_ms=latency,
                total_tokens_used=100, total_information=1,
                per_case=[{"case_id": "mock-case", "recall": 0.5}],
            )
            state.evaluated.append((dict(config), result))
            return result

        monkeypatch.setattr(autotune, "mutate_config", mutate)
        monkeypatch.setattr(autotune, "evaluate", evaluate)

    def rows():
        with results_path.open(newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    state.install = install
    state.rows = rows
    monkeypatch.setattr(autotune, "save_config", save)
    monkeypatch.setattr(autotune, "log_result", log)
    return state


def test_nonlinear_blend_regression_is_discarded(harness, capsys):
    scores = {0.1: (1.0, 100), 0.14: (1.1, 10), 0.124: (0.9, 5)}
    harness.install(lambda c: scores[c["exploration_rate"]], [0.14])

    autotune.run_autotune(iterations=1, time_budget=2.5)

    assert [c["exploration_rate"] for c, _ in harness.evaluated] == [0.1, 0.14, 0.124]
    assert harness.saved == [harness.initial]
    assert harness.rows()[1]["status"] == "discard"
    assert "Improvements found: 0" in capsys.readouterr().err


def test_kept_metrics_and_next_mutation_use_actual_blend(harness, capsys):
    scores = {
        0.1: (1.0, 100), 0.14: (1.1, 10),
        0.124: (1.05, 20), 0.1311: (0.99, 5),
    }
    harness.install(lambda c: scores[c["exploration_rate"]], [0.14, 0.14])

    autotune.run_autotune(iterations=2, time_budget=2.5)

    kept = dict(harness.initial, exploration_rate=0.124)
    assert harness.mutation_inputs == [harness.initial, kept]
    assert harness.saved == [kept, kept]
    assert [c["exploration_rate"] for c, _ in harness.evaluated] == [
        0.1, 0.14, 0.124, 0.14, 0.1311,
    ]
    row = harness.rows()[1]
    result = harness.evaluated[2][1]
    assert row == {
        "iteration": "1", "score": f"{autotune.composite_score(result):.4f}",
        "recall": f"{result.recall_accuracy:.4f}",
        "efficiency": f"{result.context_efficiency:.6f}", "avg_ms": "20.0",
        "status": "keep", "description": "exploration_rate: 0.1 -> 0.124",
    }
    assert harness.logged[1][1:3] == (kept, result)
    assert "Best score: 1.0500" in capsys.readouterr().err


@pytest.mark.parametrize(
    "baseline_ms,candidate_ms,blend_ms,expected_status,expected_evaluations",
    [(100, 30, 5, "discard", 4),
     (100, 10, 30, "discard", 5),
     (100, 10, 20, "discard", 5),
     (100, 10, 15, "keep", 5),
     (10, 15, 15, "keep", 5)],
)
def test_ties_compare_current_incumbent_and_evaluated_blend(
    harness, baseline_ms, candidate_ms, blend_ms, expected_status,
    expected_evaluations,
):
    scores = {
        0.1: (1.0, baseline_ms), 0.14: (1.1, candidate_ms),
        0.124: (1.1, 20), 0.1264: (1.1, blend_ms),
    }
    harness.install(lambda c: scores[c["exploration_rate"]], [0.14, 0.14])

    autotune.run_autotune(iterations=2, time_budget=2.5)

    assert len(harness.evaluated) == expected_evaluations
    assert harness.rows()[2]["status"] == expected_status
    expected_value = 0.1264 if expected_status == "keep" else 0.124
    assert harness.saved[-1]["exploration_rate"] == expected_value


@pytest.mark.parametrize(
    "param,candidate_value,candidate_score",
    [("exploration_rate", 0.14, 1.5), ("decay_half_life_turns", 16, 1.1)],
)
def test_identical_blend_reuses_candidate_evaluation(
    harness, param, candidate_value, candidate_score,
):
    harness.install(
        lambda c: (candidate_score, 10) if c[param] == candidate_value else (1.0, 100),
        [candidate_value], param=param,
    )

    autotune.run_autotune(iterations=1, time_budget=2.5)

    assert len(harness.evaluated) == 2
    assert harness.saved[-1] == harness.evaluated[-1][0]
    assert harness.saved[-1][param] == candidate_value
    assert harness.rows()[1]["status"] == "keep"


@pytest.mark.parametrize("fail_value", [0.1, 0.14, 0.124])
def test_evaluation_failure_never_persists_untested_config(harness, fail_value):
    def landscape(config):
        value = config["exploration_rate"]
        if value == fail_value:
            raise RuntimeError("evaluation failed")
        return (1.0 if value == 0.1 else 1.1), 10

    harness.install(landscape, [0.14])

    with pytest.raises(RuntimeError, match="evaluation failed"):
        autotune.run_autotune(iterations=1, time_budget=2.5)

    assert harness.saved == []
    assert json.loads(harness.path.read_text(encoding="utf-8")) == harness.initial
    assert all(row[0] == 0 for row in harness.logged)


@pytest.mark.parametrize("polyak_score", [0.9, 1.16, 1.2, None])
def test_final_polyak_uses_its_own_evaluated_config_and_score(
    harness, capsys, polyak_score,
):
    scores = {
        0.1: (1.0, 100), 0.14: (1.1, 10), 0.124: (1.05, 20),
        0.16: (1.2, 10), 0.1502: (1.16, 20),
    }

    def landscape(config):
        value = config["exploration_rate"]
        if value == 0.1247:
            if polyak_score is None:
                raise RuntimeError("Polyak evaluation failed")
            return polyak_score, 50
        return scores[value]

    harness.install(landscape, [0.14, 0.16])
    if polyak_score is None:
        with pytest.raises(RuntimeError, match="Polyak evaluation failed"):
            autotune.run_autotune(iterations=2, time_budget=2.5)
        assert harness.saved[-1]["exploration_rate"] == 0.1502
        assert len(harness.saved) == 2
    else:
        autotune.run_autotune(iterations=2, time_budget=2.5)
        use_polyak = polyak_score >= 1.16
        expected_value = 0.1247 if use_polyak else 0.1502
        expected_score = polyak_score if use_polyak else 1.16
        assert harness.saved[-1]["exploration_rate"] == expected_value
        assert harness.evaluated[-1][0]["exploration_rate"] == 0.1247
        assert f"Best score: {expected_score:.4f}" in capsys.readouterr().err
    assert [row["status"] for row in harness.rows()] == ["keep", "keep", "keep"]


def test_bench_only_preserves_baseline_logging_without_writing(harness, capsys):
    harness.install(lambda c: (1.0, 100), [])

    autotune.run_autotune(iterations=10, time_budget=2.5, bench_only=True)

    assert len(harness.evaluated) == 1
    assert harness.mutation_inputs == []
    assert harness.saved == []
    assert json.loads(harness.path.read_text(encoding="utf-8")) == harness.initial
    assert len(harness.rows()) == 1
    assert harness.rows()[0]["description"] == "baseline"
    assert "Per-case breakdown:" in capsys.readouterr().err
