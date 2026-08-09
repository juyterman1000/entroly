from __future__ import annotations

import hashlib
import json
from pathlib import Path

from entroly.repository_intelligence import RepositoryIntelligenceService
from entroly.repository_intelligence.program_graph import (
    verify_program_graph_commitment,
)
from entroly.repository_intelligence.repository_map import (
    verify_repository_map_commitment,
)
from entroly.repository_intelligence.graph_query import (
    verify_graph_query_commitment,
)
from entroly.repository_intelligence.verified_architecture import (
    verify_architecture_commitment,
)


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "fixtures"
    / "code_intelligence_conformance"
)


def test_shared_code_intelligence_conformance_fixture() -> None:
    gold = json.loads((FIXTURE / "gold.json").read_text(encoding="utf-8"))
    service = RepositoryIntelligenceService(FIXTURE)
    index, _digest, _generation = service._snapshot()

    typed = gold["typed_call"]
    assert any(
        edge.caller_id == typed["caller"]
        and edge.callee_id == typed["callee"]
        and edge.confidence == "type-inferred"
        for edge in index.call_edges
    )

    unresolved = next(
        call
        for call in index.unresolved_calls
        if call.caller_id == gold["ambiguous_untyped_call"]
    )
    assert unresolved.reason == "untyped-receiver-member"
    assert len(unresolved.candidates) == 2
    assert not any(
        edge.caller_id == gold["ambiguous_untyped_call"]
        for edge in index.call_edges
    )

    ambiguous = service.symbol_graph("execute")
    assert ambiguous["resolution"] == "ambiguous"
    assert len(ambiguous["candidates"]) == 2
    assert not ambiguous["nodes"] and not ambiguous["edges"]

    architecture = service.architecture()
    assert architecture["receipt"]["verified_file_count"] == len(index.files)
    assert architecture["components"]
    assert architecture["routes"]
    assert verify_architecture_commitment(architecture)

    ambiguous_query = service.graph_query("execute")
    assert ambiguous_query["resolution"] == "ambiguous"
    assert not ambiguous_query["nodes"] and not ambiguous_query["edges"]
    typed_path = service.graph_query(
        typed["caller"],
        operation="path",
        target_query=typed["callee"],
        direction="outgoing",
        max_depth=1,
    )
    assert typed_path["results"][0]["distance"] == 1
    assert typed_path["results"][0]["edges"][0]["evidence"]["confidence"] == (
        "type-inferred"
    )
    assert verify_graph_query_commitment(typed_path)

    global_map = service.repository_map(token_budget=2_000)
    query_map = service.repository_map(gold["query"], token_budget=2_000)
    assert global_map["entries"][0]["symbol_id"] == gold["global_hub"]
    assert query_map["entries"][0]["symbol_id"] == gold["query_symbol"]
    assert verify_repository_map_commitment(global_map)
    assert verify_repository_map_commitment(query_map)
    for entry in query_map["entries"]:
        raw = (FIXTURE / entry["path"]).read_bytes()
        evidence = raw[entry["start_byte"]:entry["end_byte"]]
        assert hashlib.sha256(evidence).hexdigest() == entry["evidence_sha256"]

    choose = service.program_graph("choose")
    assert choose["root_symbol_id"] == gold["must_reach_symbol"]
    assert choose["control_edges"]
    assert choose["data_edges"]
    result_edges = [edge for edge in choose["data_edges"] if edge["name"] == "result"]
    flag_edges = [edge for edge in choose["data_edges"] if edge["name"] == "flag"]
    assert len(result_edges) == 2
    assert all(edge["confidence"] == "may-reach" for edge in result_edges)
    assert len(flag_edges) == 1
    assert flag_edges[0]["confidence"] == "must-reach"
    assert verify_program_graph_commitment(choose)
