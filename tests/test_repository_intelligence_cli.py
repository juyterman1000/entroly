from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from entroly.repository_intelligence.cli import run

FAKE_LSP_SERVER = Path(__file__).parent / "fixtures" / "fake_lsp_server.py"


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


def _commit_project(root: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Entroly Test"], cwd=root, check=True
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "baseline"], cwd=root, check=True
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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


def test_query_returns_verified_typed_shortest_path(tmp_path: Path) -> None:
    _project(tmp_path)
    code, payload = run([
        "--root", str(tmp_path), "query", "--query", "pkg/api.py",
        "--operation", "path", "--target", "execute",
        "--direction", "outgoing", "--max-depth", "5",
    ])
    assert code == 0
    assert payload["schema_version"] == "entroly.verified-graph-query.v1"
    assert payload["command"] == "query"
    assert payload["results"][0]["kind"] == "shortest-path"
    assert payload["receipt"]["freshness"].startswith("verified")


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


def test_flow_returns_verified_cross_function_value_summary(tmp_path: Path) -> None:
    _project(tmp_path)
    code, payload = run([
        "--root", str(tmp_path), "flow", "--symbol", "invoke",
    ])
    assert code == 0
    assert payload["schema_version"] == "entroly.verified-interprocedural-flow.v1"
    assert payload["command"] == "flow"
    assert payload["resolution"] == "resolved"
    assert payload["call_relations"]
    assert {edge["kind"] for edge in payload["flow_edges"]} >= {
        "return-to-call-result",
        "call-result-to-consumer",
    }


def test_health_returns_verified_policy_and_commitment(tmp_path: Path) -> None:
    _project(tmp_path)
    code, payload = run([
        "--root", str(tmp_path), "health", "--max-findings", "10",
    ])
    assert code == 0
    assert payload["schema_version"] == "entroly.verified-code-health.v1"
    assert payload["command"] == "health"
    assert payload["policy"]["interpretation"] == "ranking-and-review-aid-not-a-proof-of-defect"
    assert payload["receipt"]["code_health_sha256"]


def test_architecture_returns_verified_layers_routes_and_hotspots(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    code, payload = run([
        "--root", str(tmp_path), "architecture", "--max-routes", "10",
    ])
    assert code == 0
    assert payload["schema_version"] == "entroly.verified-architecture.v1"
    assert payload["command"] == "architecture"
    assert payload["receipt"]["verified_file_count"] == 3
    assert payload["routes"]
    assert payload["hotspots"]


def test_architecture_diff_requires_and_compares_committed_inputs(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    code, before = run(["--root", str(tmp_path), "architecture"])
    assert code == 0
    before_path = tmp_path.parent / f"{tmp_path.name}-before.json"
    before_path.write_text(json.dumps(before), encoding="utf-8")
    _write(tmp_path, "pkg/new.py", "from pkg.source import execute\n")
    code, after = run(["--root", str(tmp_path), "architecture"])
    assert code == 0
    after_path = tmp_path.parent / f"{tmp_path.name}-after.json"
    after_path.write_text(json.dumps(after), encoding="utf-8")
    code, payload = run([
        "--root", str(tmp_path), "architecture-diff",
        "--before-json", str(before_path), "--after-json", str(after_path),
    ])
    assert code == 0
    assert payload["schema_version"] == "entroly.verified-architecture-diff.v1"
    assert payload["command"] == "architecture-diff"
    assert payload["files"]["added"] == ["pkg/new.py"]


def test_git_architecture_diff_uses_commit_objects_without_checkout(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    baseline = _commit_project(tmp_path)
    _write(tmp_path, "pkg/new.py", "from pkg.source import execute\n")
    code, payload = run([
        "--root", str(tmp_path), "git-architecture-diff", "--ref", baseline,
    ])
    assert code == 0
    assert payload["schema_version"] == "entroly.verified-git-architecture-diff.v1"
    assert payload["command"] == "git-architecture-diff"
    assert payload["base_commit"] == baseline
    assert payload["baseline_materialization"]["checkout_mutated"] is False
    assert payload["architecture_diff"]["files"]["added"] == ["pkg/new.py"]


def test_routes_returns_verified_mounted_http_endpoints(tmp_path: Path) -> None:
    _project(tmp_path)
    _write(
        tmp_path,
        "pkg/routes.py",
        "from fastapi import APIRouter\n"
        "router = APIRouter(prefix='/v1')\n"
        "@router.get('/items/{item_id}')\n"
        "def item(item_id: str):\n"
        "    return item_id\n",
    )
    code, payload = run([
        "--root", str(tmp_path), "routes", "--method", "GET",
        "--path-prefix", "/v1",
    ])
    assert code == 0
    assert payload["schema_version"] == "entroly.verified-http-routes.v1"
    assert payload["command"] == "routes"
    assert payload["routes"][0]["path"] == "/v1/items/{item_id}"
    assert payload["routes"][0]["handler"]["status"] == "resolved"


def test_snapshot_round_trip_proves_current_graph_is_importable(tmp_path: Path) -> None:
    _project(tmp_path)
    code, snapshot = run(["--root", str(tmp_path), "snapshot"])
    assert code == 0
    assert snapshot["schema_version"] == "entroly.verified-graph-snapshot.v1"
    snapshot_path = tmp_path.parent / f"{tmp_path.name}-graph.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    code, checked = run([
        "--root", str(tmp_path), "snapshot-check",
        "--snapshot-json", str(snapshot_path),
    ])
    assert code == 0
    assert checked["command"] == "snapshot-check"
    assert checked["snapshot_commitment_valid"] is True
    assert checked["snapshot_importable"] is True
    assert checked["in_sync"] is True


def test_cli_two_phase_rename_requires_ack_and_applies_committed_plan(tmp_path: Path) -> None:
    _project(tmp_path)
    code, plan = run([
        "--root", str(tmp_path), "rename-preview",
        "--symbol", "execute", "--new-name", "perform",
    ])
    assert code == 0
    assert plan["command"] == "rename-preview"
    assert plan["receipt"]["writes_performed"] == 0
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    rejected_code, rejected = run([
        "--root", str(tmp_path), "rename-apply",
        "--plan-json", str(plan_path),
        "--expected-plan-sha", plan["receipt"]["plan_sha256"],
    ])
    assert rejected_code == 2
    assert "acknowledgement" in rejected["detail"]

    applied_code, applied = run([
        "--root", str(tmp_path), "rename-apply",
        "--plan-json", str(plan_path),
        "--expected-plan-sha", plan["receipt"]["plan_sha256"],
        "--acknowledge-incomplete",
    ])
    assert applied_code == 0
    assert applied["command"] == "rename-apply"
    assert applied["apply"]["change_count"] == 3
    assert "def perform" in (tmp_path / "pkg/source.py").read_text(encoding="utf-8")


def test_cli_safe_delete_preview_and_apply_are_committed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "module.py",
        "def keep():\n    return 1\n\ndef unused():\n    return 2\n",
    )
    code, plan = run([
        "--root", str(tmp_path), "safe-delete-preview",
        "--symbol", "unused",
    ])
    assert code == 0
    assert plan["safe_to_apply"] is True
    plan_path = tmp_path / "safe-delete.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    code, applied = run([
        "--root", str(tmp_path), "safe-delete-apply",
        "--plan-json", str(plan_path),
        "--expected-plan-sha", plan["receipt"]["plan_sha256"],
        "--acknowledge-incomplete",
    ])
    assert code == 0
    assert applied["apply"]["operation"] == "safe-delete"
    assert "unused" not in (tmp_path / "module.py").read_text(encoding="utf-8")


def test_cli_file_move_updates_imports_and_refreshes_graph(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/old.py", "def execute():\n    return 1\n")
    _write(tmp_path, "app.py", "from pkg.old import execute\n")
    code, plan = run([
        "--root", str(tmp_path), "file-move-preview",
        "--source", "pkg/old.py", "--target", "pkg/new.py",
    ])
    assert code == 0
    assert plan["safe_to_apply"] is True
    plan_path = tmp_path / "file-move.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    code, applied = run([
        "--root", str(tmp_path), "file-move-apply",
        "--plan-json", str(plan_path),
        "--expected-plan-sha", plan["receipt"]["plan_sha256"],
        "--acknowledge-incomplete",
    ])
    assert code == 0
    assert applied["apply"]["operation"] == "file-move"
    assert applied["refresh"]["status"] == "refreshed"
    assert not (tmp_path / "pkg/old.py").exists()
    assert (tmp_path / "pkg/new.py").exists()


def test_cli_lsp_preview_requires_explicit_command_file(tmp_path: Path) -> None:
    _project(tmp_path)
    command_path = tmp_path / "lsp-command.json"
    command_path.write_text(
        json.dumps([sys.executable, str(FAKE_LSP_SERVER)]),
        encoding="utf-8",
    )
    code, payload = run([
        "--root", str(tmp_path), "lsp-rename-preview",
        "--symbol", "execute", "--new-name", "perform",
        "--language-id", "python", "--command-json", str(command_path),
        "--timeout-seconds", "5",
    ])
    assert code == 0
    assert payload["command"] == "lsp-rename-preview"
    assert payload["plan"]["resolution"] == "resolved"
    assert payload["receipt"]["writes_performed"] == 0


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
