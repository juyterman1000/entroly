import json
from pathlib import Path
from types import SimpleNamespace

from bench import autotune as benchmark_autotune
from bench import evaluate as benchmark_evaluate
from entroly import autotune as runtime_autotune
from entroly import cli


RUNTIME_CONFIG = {
    "weight_recency": 0.30,
    "weight_frequency": 0.25,
    "weight_semantic_sim": 0.25,
    "weight_entropy": 0.20,
}


def test_runtime_learning_state_is_project_scoped(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    benchmark_path = tmp_path / "bench" / "tuning_config.json"
    monkeypatch.setattr(runtime_autotune, "LEGACY_CONFIG_PATH", benchmark_path)

    runtime_autotune.save_config(RUNTIME_CONFIG)

    target = tmp_path / "state" / "learning_tuning_config.json"
    assert target.exists()
    assert not benchmark_path.exists()
    assert runtime_autotune.load_config() == RUNTIME_CONFIG


def test_runtime_learning_migrates_only_legacy_flat_schema(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    legacy_path = tmp_path / "bench" / "tuning_config.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(json.dumps(RUNTIME_CONFIG), encoding="utf-8")
    monkeypatch.setattr(runtime_autotune, "LEGACY_CONFIG_PATH", legacy_path)

    target = runtime_autotune.runtime_config_path()
    assert runtime_autotune.load_config(target) == RUNTIME_CONFIG
    assert target.exists()


def test_dreaming_loop_honors_explicit_config_path(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "custom-learning.json"
    loaded_paths: list[Path | None] = []
    monkeypatch.setattr(
        runtime_autotune,
        "load_config",
        lambda path=None: loaded_paths.append(path) or dict(RUNTIME_CONFIG),
    )
    monkeypatch.setattr(runtime_autotune, "load_cases", lambda: [])

    loop = runtime_autotune.DreamingLoop(
        journal=SimpleNamespace(load=lambda: []), config_path=config_path
    )
    loop._last_activity = 0.0

    assert loop.run_dream_cycle()["status"] == "no_cases"
    assert loaded_paths == [config_path]


def test_benchmark_loader_rejects_legacy_runtime_schema(tmp_path) -> None:
    path = tmp_path / "tuning_config.json"
    path.write_text(json.dumps(RUNTIME_CONFIG), encoding="utf-8")

    loaded = benchmark_evaluate.load_tuning_config(path)

    assert loaded == benchmark_evaluate.DEFAULT_TUNING_CONFIG


def test_benchmark_loader_completes_older_valid_schema(tmp_path) -> None:
    path = tmp_path / "tuning_config.json"
    older = {
        "weights": dict(benchmark_evaluate.DEFAULT_TUNING_CONFIG["weights"]),
        "decay": dict(benchmark_evaluate.DEFAULT_TUNING_CONFIG["decay"]),
        "knapsack": dict(benchmark_evaluate.DEFAULT_TUNING_CONFIG["knapsack"]),
    }
    path.write_text(json.dumps(older), encoding="utf-8")

    loaded = benchmark_evaluate.load_tuning_config(path)

    assert loaded["weights"] == older["weights"]
    for section in ("sliding_window", "prism", "egtc", "ios", "ecdb"):
        assert section in loaded


def test_benchmark_loader_fails_safe_on_non_object_section(tmp_path) -> None:
    path = tmp_path / "tuning_config.json"
    malformed = dict(benchmark_evaluate.DEFAULT_TUNING_CONFIG)
    malformed["ios"] = 3
    path.write_text(json.dumps(malformed), encoding="utf-8")

    loaded = benchmark_evaluate.load_tuning_config(path)

    assert loaded == benchmark_evaluate.DEFAULT_TUNING_CONFIG


def test_benchmark_loader_fails_safe_on_non_object_root(tmp_path) -> None:
    path = tmp_path / "tuning_config.json"
    path.write_text("[]", encoding="utf-8")

    loaded = benchmark_evaluate.load_tuning_config(path)

    assert loaded == benchmark_evaluate.DEFAULT_TUNING_CONFIG


def test_benchmark_autotune_repairs_invalid_config(tmp_path, monkeypatch) -> None:
    path = tmp_path / "tuning_config.json"
    path.write_text(json.dumps(RUNTIME_CONFIG), encoding="utf-8")

    monkeypatch.setattr(
        benchmark_autotune,
        "evaluate",
        lambda config, cases_path=None: {
            "composite_score": 0.5,
            "avg_recall": 0.5,
            "avg_precision": 0.5,
            "avg_context_efficiency": 0.5,
            "all_latency_ok": True,
        },
    )

    result = benchmark_autotune.autotune(
        iterations=0, config_path=path, verbose=False
    )

    repaired = json.loads(path.read_text(encoding="utf-8"))
    assert result["final_score"] == 0.5
    assert benchmark_evaluate.validate_tuning_config(repaired) == []
    assert list(tmp_path.glob("tuning_config.*.bak.json")) == []


def test_cli_autotune_reports_unexpected_failure(monkeypatch, capsys) -> None:
    def fail(*, iterations):
        raise RuntimeError("broken benchmark config")

    monkeypatch.setattr(benchmark_autotune, "autotune", fail)

    result = cli.cmd_autotune(SimpleNamespace(iterations=1, rollback=False))

    assert result == 1
    output = capsys.readouterr().out
    assert "Autotune failed:" in output
    assert "broken benchmark config" in output
