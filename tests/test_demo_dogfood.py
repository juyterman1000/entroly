from __future__ import annotations

import ast
import inspect
import json
from types import SimpleNamespace

from entroly import cli


def test_demo_delegates_to_bounded_shared_simulation(monkeypatch, capsys):
    args = SimpleNamespace(
        max_files=7,
        budget=2048,
        baseline=0,
        query=[],
        json=False,
    )
    report = {
        "files_indexed": 7,
        "repo_tokens_indexed": 20_000,
        "budget": 2048,
        "baseline_tokens_per_query": 20_000,
        "queries": [{"query": "q"}],
        "total_tokens_saved": 12_000,
    }
    observed = {}

    def fake_run(received):
        observed["args"] = received
        return report

    monkeypatch.setattr(cli, "_run_local_simulation", fake_run)
    monkeypatch.setattr(cli, "_print_local_simulation", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_detect_project_type", lambda: "python")
    monkeypatch.setattr(cli, "_recommend_quality", lambda *_: "balanced")

    cli.cmd_demo(args)

    assert observed["args"] is args
    assert "Get started" in capsys.readouterr().out


def test_demo_cannot_reintroduce_direct_auto_indexing():
    tree = ast.parse(inspect.getsource(cli.cmd_demo))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_run_local_simulation" in calls
    assert "auto_index" not in calls


def test_demo_json_output_is_machine_readable(monkeypatch, capsys):
    args = SimpleNamespace(json=True)
    report = {
        "mode": "local_no_llm",
        "files_indexed": 0,
        "queries": [],
        "total_tokens_saved": 0,
    }
    monkeypatch.setattr(cli, "_run_local_simulation", lambda _: report)

    cli.cmd_demo(args)

    assert json.loads(capsys.readouterr().out) == report
