from __future__ import annotations

import copy
from pathlib import Path

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.verified_slice import (
    build_verified_program_slice,
    verify_program_slice_commitment,
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(text.encode("utf-8"))


def _project(root: Path):
    _write(
        root,
        "normalize.py",
        "def normalize(value):\n    return value.strip()\n",
    )
    _write(
        root,
        "pipeline.py",
        "from normalize import normalize\n"
        "def process(raw):\n"
        "    cleaned = normalize(raw)\n"
        "    return cleaned\n",
    )
    return build_repository_index(root)


def test_program_slice_combines_context_control_and_cross_function_flow(
    tmp_path: Path,
) -> None:
    index = _project(tmp_path)
    payload = build_verified_program_slice(
        tmp_path,
        index,
        "process",
        index_digest="test-index",
        token_budget=1_000,
    )
    assert payload["query_route"] == {
        "kind": "code-entity",
        "identity_status": "unique-exact",
        "exact_candidates": [
            next(symbol.to_dict() for symbol in index.symbols.values() if symbol.name == "process")
        ],
        "exact_candidates_omitted": 0,
    }
    assert payload["entry_points"][0]["origin"] == "exact-query"
    assert payload["intraprocedural_graphs"][0]["resolution"] == "resolved"
    assert payload["interprocedural_flows"][0]["resolution"] == "resolved"
    assert payload["coverage"]["verified_call_relations"] == 1
    assert payload["coverage"]["verified_value_flow_edges"] >= 3
    assert payload["coverage"]["answer_sufficiency"] == "unproven"
    assert verify_program_slice_commitment(payload)


def test_program_slice_accepts_rank_proposal_but_rejects_invented_identity(
    tmp_path: Path,
) -> None:
    index = _project(tmp_path)
    normalize = next(
        symbol for symbol in index.symbols.values() if symbol.name == "normalize"
    )
    payload = build_verified_program_slice(
        tmp_path,
        index,
        "clean user supplied text",
        index_digest="test-index",
        token_budget=256,
        max_fragments=1,
        proposal_scores=[
            {"symbol_id": normalize.symbol_id, "score": 1.0},
            {"symbol_id": "invented::symbol", "score": 1.0},
        ],
        proposal_provider="test-ranker",
    )
    assert payload["query_route"]["kind"] == "natural-language"
    assert payload["entry_points"][0]["symbol_id"] == normalize.symbol_id
    overlay = payload["verified_context"]["proposal_overlay"]
    assert overlay["omissions_by_reason"] == {"unknown-symbol": 1}
    assert payload["neuro_symbolic_contract"]["proposal_may_create_facts"] is False
    assert verify_program_slice_commitment(payload)


def test_program_slice_preserves_ambiguity_and_detects_nested_tampering(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "a.py", "def duplicate():\n    return 1\n")
    _write(tmp_path, "b.py", "def duplicate():\n    return 2\n")
    index = build_repository_index(tmp_path)
    payload = build_verified_program_slice(
        tmp_path,
        index,
        "duplicate",
        index_digest="test-index",
        max_entry_points=2,
    )
    assert payload["query_route"]["identity_status"] == "ambiguous-exact"
    assert len(payload["query_route"]["exact_candidates"]) == 2
    assert {item["origin"] for item in payload["entry_points"]} == {
        "ambiguous-exact-candidate"
    }
    assert verify_program_slice_commitment(payload)

    tampered = copy.deepcopy(payload)
    tampered["verified_context"]["fragments"][0]["content"] += "\n# forged"
    assert not verify_program_slice_commitment(tampered)
