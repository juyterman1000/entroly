from __future__ import annotations

import copy
from pathlib import Path

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.graph_query import (
    build_verified_graph_query,
    verify_graph_query_commitment,
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _project(root: Path):
    _write(
        root,
        "service.py",
        "def execute(value):\n    return value + 1\n",
    )
    _write(
        root,
        "api.py",
        "from service import execute\n"
        "def invoke():\n"
        "    return execute(1)\n",
    )
    _write(
        root,
        "app.py",
        "from api import invoke\n"
        "def main():\n"
        "    return invoke()\n",
    )
    return build_repository_index(root)


def test_graph_query_finds_exact_typed_shortest_path_with_witnesses(
    tmp_path: Path,
) -> None:
    index = _project(tmp_path)
    payload = build_verified_graph_query(
        tmp_path,
        index,
        "app.py",
        index_digest="sha256:test",
        operation="path",
        target_query="execute",
        direction="outgoing",
        max_depth=8,
    )
    assert payload["resolution"] == "resolved"
    assert payload["target_resolution"] == "resolved"
    assert payload["results"][0]["kind"] == "shortest-path"
    kinds = [edge["kind"] for edge in payload["results"][0]["edges"]]
    assert kinds == ["contains", "calls", "calls"]
    assert payload["results"][0]["nodes"][0] == "file:app.py"
    assert payload["results"][0]["nodes"][-1].endswith("::execute::function")
    assert verify_graph_query_commitment(payload)


def test_graph_query_impact_and_related_have_explicit_witness_paths(
    tmp_path: Path,
) -> None:
    index = _project(tmp_path)
    impact = build_verified_graph_query(
        tmp_path,
        index,
        "execute",
        index_digest="sha256:test",
        operation="impact",
        max_depth=8,
    )
    result_ids = {result["node_id"] for result in impact["results"]}
    assert any(node.endswith("::invoke::function") for node in result_ids)
    assert any(node.endswith("::main::function") for node in result_ids)
    main = next(
        result
        for result in impact["results"]
        if result["node_id"].endswith("::main::function")
    )
    assert len(main["witness_edges"]) >= 2

    related = build_verified_graph_query(
        tmp_path,
        index,
        "api.py",
        index_digest="sha256:test",
        operation="related",
        max_depth=3,
    )
    assert all("score" in result for result in related["results"])
    assert related["results"][0]["score_policy"] == (
        "inverse-distance-times-bounded-degree"
    )
    assert verify_graph_query_commitment(related)


def test_graph_query_refuses_ambiguous_symbol_identity(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "workers.py",
        "class Alpha:\n"
        "    def execute(self):\n"
        "        return 1\n"
        "class Beta:\n"
        "    def execute(self):\n"
        "        return 2\n",
    )
    index = build_repository_index(tmp_path)
    payload = build_verified_graph_query(
        tmp_path,
        index,
        "execute",
        index_digest="sha256:test",
    )
    assert payload["resolution"] == "ambiguous"
    assert len(payload["candidates"]) == 2
    assert payload["nodes"] == []
    assert verify_graph_query_commitment(payload)


def test_graph_query_removes_stale_intermediate_nodes_and_detects_tampering(
    tmp_path: Path,
) -> None:
    index = _project(tmp_path)
    (tmp_path / "api.py").write_text("VALUE = 2\n", encoding="utf-8")
    payload = build_verified_graph_query(
        tmp_path,
        index,
        "app.py",
        index_digest="sha256:test",
        operation="path",
        target_query="execute",
        direction="outgoing",
        max_depth=8,
    )
    assert payload["results"] == []
    assert payload["receipt"]["omissions_by_reason"] == {"stale-source": 1}
    assert verify_graph_query_commitment(payload)
    tampered = copy.deepcopy(payload)
    tampered["direction"] = "incoming"
    assert not verify_graph_query_commitment(tampered)


def test_graph_query_bounds_large_traversals_visibly(tmp_path: Path) -> None:
    index = _project(tmp_path)
    payload = build_verified_graph_query(
        tmp_path,
        index,
        "execute",
        index_digest="sha256:test",
        operation="impact",
        max_depth=20,
        limit=1,
        max_visited=2,
    )
    assert payload["truncated"] is True
    assert payload["receipt"]["visited_node_count"] == 2
    assert len(payload["results"]) == 1
