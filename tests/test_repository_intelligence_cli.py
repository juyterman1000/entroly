from __future__ import annotations

import json
from pathlib import Path

from entroly.repository_intelligence.cli import run


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _project(root: Path) -> None:
    _write(root, "pkg/source.py", "def execute():\n    return 1\n")
    _write(
        root,
        "pkg/api.py",
        "from pkg.source import execute\n"
        "def invoke():\n"
        "    return execute()\n",
    )
    _write(
        root,
        "tests/test_api.py",
        "from pkg.api import invoke\n"
        "def test_invoke():\n"
        "    assert invoke() == 1\n",
    )


def test_summary_is_bounded_and_versioned(tmp_path: Path) -> None:
    _project(tmp_path)
    code, payload = run(["--root", str(tmp_path), "summary"])
    assert code == 0
    assert payload["schema_version"] == "entroly.repository-cli.v2"
    assert payload["files"] == 3
    assert payload["tests"] == 1
    assert "symbols" in payload and "call_edges" in payload
    assert "files_payload" not in payload


def test_impact_reports_reverse_dependencies(tmp_path: Path) -> None:
    _project(tmp_path)
    code, payload = run(
        ["--root", str(tmp_path), "impact", "--changed", "pkg/source.py"]
    )
    assert code == 0
    assert payload["report"]["impacted_paths"] == [
        "pkg/api.py",
        "pkg/source.py",
        "tests/test_api.py",
    ]


def test_tests_ranks_directly_related_test(tmp_path: Path) -> None:
    _project(tmp_path)
    code, payload = run(
        ["--root", str(tmp_path), "tests", "--changed", "pkg/source.py"]
    )
    assert code == 0
    assert payload["candidates"][0]["path"] == "tests/test_api.py"


def test_context_returns_verified_partial_graph(tmp_path: Path) -> None:
    _project(tmp_path)
    code, payload = run([
        "--root", str(tmp_path), "context", "--query", "execute",
        "--token-budget", "512",
    ])
    assert code == 0
    assert payload["schema_version"] == "entroly.verified-code-context.v1"
    assert payload["command"] == "context"
    assert payload["fragments"][0]["qualified_name"] == "execute"


def test_graph_returns_verified_static_callers(tmp_path: Path) -> None:
    _project(tmp_path)
    code, payload = run([
        "--root", str(tmp_path), "graph", "--symbol", "execute",
        "--direction", "callers",
    ])
    assert code == 0
    assert payload["schema_version"] == "entroly.verified-symbol-graph.v1"
    assert payload["command"] == "graph"
    assert payload["resolution"] == "resolved"
    assert {node["qualified_name"] for node in payload["nodes"]} >= {
        "execute",
        "invoke",
    }


def test_map_returns_verified_budgeted_repository_priority(tmp_path: Path) -> None:
    _project(tmp_path)
    code, payload = run([
        "--root", str(tmp_path), "map", "--query", "execute",
        "--token-budget", "256",
    ])
    assert code == 0
    assert payload["schema_version"] == "entroly.verified-repository-map.v1"
    assert payload["command"] == "map"
    assert payload["entries"][0]["qualified_name"] == "execute"
    assert payload["budget"]["estimated_tokens"] <= 256


def test_program_returns_verified_control_and_data_flow(tmp_path: Path) -> None:
    _project(tmp_path)
    code, payload = run([
        "--root", str(tmp_path), "program", "--symbol", "invoke",
    ])
    assert code == 0
    assert payload["schema_version"] == "entroly.verified-program-graph.v1"
    assert payload["command"] == "program"
    assert payload["resolution"] == "resolved"
    assert payload["receipt"]["freshness"].startswith("verified")


def test_runtime_binds_json_events_without_values(tmp_path: Path) -> None:
    _project(tmp_path)
    events = tmp_path / "events.json"
    events.write_text(json.dumps([
        {"path": "pkg/source.py", "line": 2, "event": "return", "value": "secret"},
    ]), encoding="utf-8")
    code, payload = run([
        "--root", str(tmp_path), "runtime", "--events-json", str(events),
        "--producer", "pytest",
    ])
    assert code == 0
    assert payload["schema_version"] == "entroly.verified-runtime-overlay.v1"
    assert payload["command"] == "runtime"
    assert payload["observations"][0]["symbol_id"].endswith("::execute::function")
    assert "secret" not in json.dumps(payload)


def test_unknown_changed_path_fails_visibly(tmp_path: Path) -> None:
    _project(tmp_path)
    code, payload = run(
        ["--root", str(tmp_path), "impact", "--changed", "missing.py"]
    )
    assert code == 2
    assert payload["error"] == "unknown_changed_paths"
    assert payload["unknown"] == ["missing.py"]


def test_invalid_root_is_machine_readable(tmp_path: Path) -> None:
    code, payload = run(["--root", str(tmp_path / "absent"), "summary"])
    assert code == 2
    assert payload["error"] == "invalid_repository"
