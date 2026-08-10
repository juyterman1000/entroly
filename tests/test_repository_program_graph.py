from __future__ import annotations

import hashlib
from pathlib import Path

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.program_graph import (
    build_verified_program_graph,
    verify_program_graph_commitment,
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _build(root: Path, query: str) -> dict[str, object]:
    index = build_repository_index(root)
    return build_verified_program_graph(
        root,
        index,
        query,
        index_digest="test-index",
    )


def test_program_graph_verifies_control_and_branching_definitions(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "decision.py",
        "def decide(flag, amount):\n"
        "    if flag:\n"
        "        result = amount\n"
        "    else:\n"
        "        result = 0\n"
        "    return result\n",
    )
    payload = _build(tmp_path, "decide")

    assert payload["resolution"] == "resolved"
    assert verify_program_graph_commitment(payload)
    assert {edge["kind"] for edge in payload["control_edges"]} >= {
        "true",
        "false",
        "return",
    }
    result_edges = [edge for edge in payload["data_edges"] if edge["name"] == "result"]
    assert len(result_edges) == 2
    assert {edge["confidence"] for edge in result_edges} == {"may-reach"}
    flag_edges = [edge for edge in payload["data_edges"] if edge["name"] == "flag"]
    assert len(flag_edges) == 1
    assert flag_edges[0]["confidence"] == "must-reach"

    raw = (tmp_path / "decision.py").read_bytes()
    for node in payload["nodes"]:
        if node["trust"] != "verified-source-span":
            continue
        evidence = raw[node["start_byte"]:node["end_byte"]]
        assert hashlib.sha256(evidence).hexdigest() == node["evidence_sha256"]
        for occurrence in node["occurrences"]:
            evidence = raw[occurrence["start_byte"]:occurrence["end_byte"]]
            assert hashlib.sha256(evidence).hexdigest() == occurrence["evidence_sha256"]


def test_program_graph_loop_has_back_edge_and_reaching_definition(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "loop.py",
        "def total(values):\n"
        "    result = 0\n"
        "    for value in values:\n"
        "        result = result + value\n"
        "    return result\n",
    )
    payload = _build(tmp_path, "total")

    assert payload["resolution"] == "resolved"
    kinds = {edge["kind"] for edge in payload["control_edges"]}
    assert {"loop-body", "loop-exit"} <= kinds
    result_uses = [edge for edge in payload["data_edges"] if edge["name"] == "result"]
    assert result_uses
    assert verify_program_graph_commitment(payload)


def test_program_graph_refuses_ambiguous_symbol_and_stale_source(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def duplicate():\n    return 1\n")
    _write(tmp_path, "b.py", "def duplicate():\n    return 2\n")
    index = build_repository_index(tmp_path)
    ambiguous = build_verified_program_graph(
        tmp_path,
        index,
        "duplicate",
        index_digest="test-index",
    )
    assert ambiguous["resolution"] == "ambiguous"
    assert ambiguous["nodes"] == []
    assert verify_program_graph_commitment(ambiguous)

    _write(tmp_path, "a.py", "def duplicate():\n    return 3\n")
    stale = build_verified_program_graph(
        tmp_path,
        index,
        "a.py::duplicate::function",
        index_digest="test-index",
    )
    assert stale["resolution"] == "stale-index"
    assert stale["nodes"] == []
    assert verify_program_graph_commitment(stale)
