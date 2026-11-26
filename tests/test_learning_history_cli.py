"""Learning history is reachable through the public CLI and stays read-only."""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    state = tmp_path / "private-state"
    monkeypatch.setenv("ENTROLY_DIR", str(state))
    monkeypatch.setattr(sys, "argv", ["entroly", "learning-history"])
    from entroly import autotune, cli, product_telemetry
    from entroly.docker_launcher_safe import launch

    def forbidden(*args, **kwargs):
        pytest.fail("history inspection attempted tuning, a write probe, or network activity")

    monkeypatch.setattr(cli, "_ENTROLY_DIR", state)
    monkeypatch.setattr(cli, "_FIRST_RUN_MARKER", state / ".welcome_shown")
    monkeypatch.setattr(cli, "_check_first_run", forbidden)
    monkeypatch.setattr(cli, "_check_for_update", forbidden)
    monkeypatch.setattr(product_telemetry, "capture_cli_result", forbidden)
    monkeypatch.setattr(product_telemetry, "flush", forbidden)
    for name in ("load_cases", "load_config", "evaluate", "run_autotune"):
        monkeypatch.setattr(autotune, name, forbidden)
    import socket
    import entroly.private_temp

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(entroly.private_temp, "writable_data_dir", forbidden)
    original_import = builtins.__import__

    def local_import(name, *args, **kwargs):
        if name == "bench" or name.startswith("bench."):
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", local_import)

    def exit_cli(code):
        raise SystemExit(code)

    monkeypatch.setattr(cli.os, "_exit", exit_cli)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["entroly", *args])
        with pytest.raises(SystemExit) as exit_info:
            launch()
        return exit_info.value.code

    return SimpleNamespace(run=run, state=state, cli=cli, root=tmp_path)


def _history_stub(monkeypatch, report=None, error=None):
    module = ModuleType("entroly.improvement_history")
    calls = []

    class ImprovementHistoryError(RuntimeError):
        pass

    class ImprovementHistory:
        def __init__(self, root):
            calls.append(root)

        def summary(self, *, limit):
            calls.append(limit)
            if error:
                raise error
            return report

    module.ImprovementHistory = ImprovementHistory
    module.ImprovementHistoryError = ImprovementHistoryError
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


@pytest.mark.parametrize("args", [("--help",), ("learning-history", "--help")])
def test_public_help_lists_learning_history(runtime, capsys, args):
    before = _snapshot(runtime.root)
    assert runtime.run(*args) == 0
    output = capsys.readouterr()
    assert "learning-history" in output.out
    assert "local learning experiments" in output.out
    assert not output.err
    assert "\x1b" not in output.out
    assert _snapshot(runtime.root) == before


@pytest.mark.parametrize("limit", [None, 1, 128])
def test_public_cli_passes_summary_through_without_writes(runtime, monkeypatch, capsys, limit):
    report = {"total_experiments": 0, "recent_experiments": [], "scopes": {}}
    calls = _history_stub(monkeypatch, report=report)
    before = _snapshot(runtime.root)
    args = ("learning-history",) if limit is None else ("learning-history", "--limit", str(limit))
    assert runtime.run(*args) == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == report
    assert calls == [runtime.state, 20 if limit is None else limit]
    assert not output.err
    assert "\x1b" not in output.out
    assert _snapshot(runtime.root) == before


@pytest.mark.parametrize("limit", ["0", "129", "-1", "1.5", "true", "False", "bad"])
def test_invalid_limits_fail_before_opening_history(runtime, monkeypatch, capsys, limit):
    calls = _history_stub(monkeypatch)
    before = _snapshot(runtime.root)
    assert runtime.run("learning-history", "--limit", limit) == 2
    output = capsys.readouterr()
    assert "integer from 1 to 128" in output.err
    assert not output.out
    assert calls == []
    assert _snapshot(runtime.root) == before


@pytest.mark.parametrize("limit", [True, False, 0, 129, 1.5, "20"])
def test_direct_handler_rejects_non_integer_or_out_of_range_limits(runtime, capsys, limit):
    assert runtime.cli.cmd_learning_history(SimpleNamespace(limit=limit)) == 1
    output = capsys.readouterr()
    assert json.loads(output.out) == {"error": "limit must be an integer from 1 to 128"}
    assert not output.err


@pytest.mark.parametrize("error_type", [OSError, ValueError, TypeError])
def test_unreadable_history_has_static_private_error(runtime, monkeypatch, capsys, error_type):
    private = str(runtime.state / "private-record.json")
    _history_stub(monkeypatch, error=error_type(f"{private}: secret contents"))
    before = _snapshot(runtime.root)
    assert runtime.run("learning-history") == 1
    output = capsys.readouterr()
    assert json.loads(output.out) == {"error": "local learning history is unavailable"}
    assert not output.err
    assert private not in output.out
    assert "secret" not in output.out and "Traceback" not in output.out
    assert "\x1b" not in output.out
    assert _snapshot(runtime.root) == before


def test_unavailable_history_never_reports_healthy_empty(runtime, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "entroly.improvement_history", None)
    assert runtime.run("learning-history") == 1
    output = capsys.readouterr()
    assert json.loads(output.out) == {"error": "local learning history is unavailable"}
    assert not output.err
    assert not runtime.state.exists()


def test_read_only_cli_path_resolution_does_not_probe_storage(runtime):
    assert runtime.cli._resolve_entroly_dir(read_only=True) == runtime.state
    assert not runtime.state.exists()


def test_real_empty_history_via_public_entrypoint(runtime, capsys):
    core = pytest.importorskip("entroly.improvement_history")
    expected = core.ImprovementHistory(runtime.state).summary(limit=20)
    assert not runtime.state.exists()
    assert runtime.run("learning-history") == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == expected
    assert not output.err
    assert not runtime.state.exists()


def test_real_empty_history_in_fresh_process(runtime):
    pytest.importorskip("entroly.improvement_history")
    before = _snapshot(runtime.root)
    result = subprocess.run(
        [sys.executable, "-m", "entroly", "learning-history", "--limit", "20"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["records"] == []
    assert report["retained_records"] == 0
    assert report["returned_records"] == 0
    assert report["promotion_authority"] is False
    assert not result.stderr
    assert "\x1b" not in result.stdout
    assert _snapshot(runtime.root) == before


@pytest.mark.parametrize("limit", [1, 20])
def test_real_records_preserve_scopes_counts_and_types(runtime, capsys, limit):
    core = pytest.importorskip("entroly.improvement_history")
    history = core.ImprovementHistory(runtime.state)
    metrics = {
        "recall_accuracy": 0.8, "context_efficiency": 0.5,
        "avg_wall_time_ms": 2.0, "total_tokens_used": 10, "timed_out": False,
    }
    for index in range(3):
        history.record(
            scope={
                "cases_sha256": core.digest({"workload": index % 2}),
                "evaluator_sha256": core.digest("evaluator"),
                "native_sha256": core.digest("native"),
                "seed": 42, "objective": "context_efficiency",
            },
            incumbent={"weight_recency": 0.3},
            candidate={"weight_recency": 0.4 + index / 10},
            incumbent_metrics=metrics, candidate_metrics=metrics,
            decision="rejected", reason="benchmark_gate_rejected",
        )
    expected = history.summary(limit=limit)
    before = _snapshot(runtime.root)
    assert runtime.run("learning-history", "--limit", str(limit)) == 0
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report == expected
    assert report["retained_records"] == 3
    assert report["returned_records"] == min(3, limit)
    assert report["omitted_records"] == max(0, 3 - limit)
    for key in ("retained_records", "returned_records", "omitted_records", "filtered_records",
                "expired_records", "pruned_records"):
        assert type(report[key]) is int
    assert report["promotion_authority"] is False
    assert report["quality_claim_allowed"] is False
    assert report["evidence_scope"] == "fixed_retrieval_benchmark"
    if limit == 20:
        assert len({record["scope_sha256"] for record in report["records"]}) == 2
    for record in report["records"]:
        assert type(record["scope"]["seed"]) is int
        assert record["scope"]["objective"] == "context_efficiency"
        assert type(record["candidate_metrics"]["timed_out"]) is bool
        assert type(record["candidate_metrics"]["total_tokens_used"]) is int
    assert not output.err
    assert _snapshot(runtime.root) == before


@pytest.mark.parametrize("contents", ["private-secret {", '{"schema":"private-secret"}'])
def test_real_corruption_fails_without_leaking_or_repairing(runtime, capsys, contents):
    core = pytest.importorskip("entroly.improvement_history")
    history = core.ImprovementHistory(runtime.state)
    runtime.state.mkdir()
    history.path.write_text(contents, encoding="utf-8")
    before = _snapshot(runtime.root)
    assert runtime.run("learning-history") == 1
    output = capsys.readouterr()
    assert json.loads(output.out) == {"error": "local learning history is unavailable"}
    assert not output.err
    assert "private-secret" not in output.out
    assert str(runtime.state) not in output.out
    assert "Traceback" not in output.out
    assert _snapshot(runtime.root) == before
