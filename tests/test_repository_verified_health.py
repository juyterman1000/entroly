from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from entroly.repository_intelligence import (
    RepositoryIntelligenceService,
    build_repository_index,
    build_verified_code_health,
    verify_code_health_commitment,
)
from entroly.tree_sitter_support import extract_structural_profiles


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _nested_function(depth: int = 11) -> str:
    lines = ["def navigate(value, a, b, c, d, e, f):"]
    indent = "    "
    for index in range(depth):
        lines.append(f"{indent}if value > {index}:")
        indent += "    "
    lines.append(f"{indent}return value")
    lines.append("    return 0")
    return "\n".join(lines) + "\n"


def test_health_binds_metrics_cycles_and_ambiguity_to_verified_evidence(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "pkg/a.py",
        "from pkg.b import helper\n" + _nested_function() + "\ndef run():\n    return execute()\n",
    )
    _write(
        tmp_path,
        "pkg/b.py",
        "from pkg.a import navigate\ndef helper():\n    return navigate(1, 2, 3, 4, 5, 6, 7)\n",
    )
    _write(tmp_path, "pkg/one.py", "def execute():\n    return 1\n")
    _write(tmp_path, "pkg/two.py", "def execute():\n    return 2\n")

    service = RepositoryIntelligenceService(tmp_path)
    report = service.code_health()

    assert report["schema_version"] == "entroly.verified-code-health.v1"
    assert report["summary"]["architecture_cycle_count"] == 1
    assert report["summary"]["unresolved_calls"] >= 1
    assert report["unresolved_calls_by_reason"]["ambiguous"] >= 1
    navigate = next(
        profile for profile in report["symbol_profiles"]
        if profile["qualified_name"] == "navigate"
    )
    assert navigate["cyclomatic_complexity"] == 12
    assert navigate["max_control_nesting"] == 11
    assert navigate["parameter_count"] == 7
    assert navigate["analysis_backend"] == "python-ast"
    assert any(
        finding["metric"] == "cognitive_complexity"
        and finding["qualified_name"] == "navigate"
        and finding["confidence"] == "parser-exact"
        for finding in report["findings"]
    )
    assert report["receipt"]["remote_calls"] == 0
    assert verify_code_health_commitment(report)
    assert str(tmp_path) not in json.dumps(report)

    tampered = copy.deepcopy(report)
    tampered["summary"]["code_health_score"] = 100
    assert not verify_code_health_commitment(tampered)


def test_health_omits_files_changed_after_the_snapshot(tmp_path: Path) -> None:
    _write(tmp_path, "fresh.py", "def fresh():\n    return 1\n")
    _write(tmp_path, "changed.py", "def changed():\n    return 1\n")
    index = build_repository_index(tmp_path)
    _write(tmp_path, "changed.py", "def changed():\n    return 2\n")

    report = build_verified_code_health(
        tmp_path,
        index,
        index_digest="sha256:test",
    )

    assert report["summary"]["verified_files"] == 1
    assert report["receipt"]["source_omissions_by_reason"] == {"stale-index": 1}
    assert all(profile["path"] != "changed.py" for profile in report["symbol_profiles"])
    assert verify_code_health_commitment(report)


@pytest.mark.parametrize(
    ("path", "source"),
    [
        ("src/lib.rs", "fn decide(x: i32) -> i32 { if x > 0 { x } else { 0 } }\n"),
        ("web/app.ts", "function decide(x: number) { if (x > 0) { return x; } return 0; }\n"),
    ],
)
def test_optional_language_pack_profiles_non_python_control_shape(path: str, source: str) -> None:
    profiles = extract_structural_profiles(source, path)
    if profiles is None:
        pytest.skip("optional local parser grammar unavailable")
    profile = next(item for item in profiles if item.name == "decide")
    assert profile.cyclomatic_complexity >= 2
    assert profile.decision_points >= 1
    assert profile.start_byte == 0
    assert profile.end_byte <= len(source.encode("utf-8"))
