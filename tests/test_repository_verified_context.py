from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from entroly.repository_intelligence import RepositoryIntelligenceService
from entroly.repository_intelligence.verified_context import (
    verify_context_commitment,
    verify_symbol_graph_commitment,
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _project(root: Path) -> None:
    _write(
        root,
        "payments/gateway.py",
        "def authorize(card, amount):\n"
        "    return {'approved': True, 'amount': amount}\n",
    )
    _write(
        root,
        "payments/service.py",
        "from payments.gateway import authorize\n\n"
        "def charge_card(card, amount):\n"
        "    return authorize(card, amount)\n",
    )
    _write(
        root,
        "tests/test_payments.py",
        "from payments.service import charge_card\n\n"
        "def test_charge_card():\n"
        "    assert charge_card('x', 10)['approved']\n",
    )


def test_verified_context_selects_query_seed_and_graph_neighbor(tmp_path: Path) -> None:
    _project(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    payload = service.context(
        "fix charge_card authorization",
        token_budget=1_000,
        max_hops=2,
    )

    assert payload["schema_version"] == "entroly.verified-code-context.v1"
    assert payload["retrieval"]["sufficient"] is True
    names = {fragment["qualified_name"] for fragment in payload["fragments"]}
    assert {"charge_card", "authorize"} <= names
    assert any(relation["kind"] == "calls" for relation in payload["relations"])
    assert payload["receipt"]["remote_calls"] == 0
    assert len(payload["receipt"]["context_sha256"]) == 64
    assert verify_context_commitment(payload)

    for fragment in payload["fragments"]:
        assert fragment["trust"] == "untrusted-source-bytes"
        content = fragment["content"].encode("utf-8")
        assert hashlib.sha256(content).hexdigest() == fragment["fragment_sha256"]
        raw = (tmp_path / fragment["path"]).read_bytes()
        assert raw[fragment["start_byte"]:fragment["end_byte"]] == content
        assert str(tmp_path) not in json.dumps(fragment)

    payload["fragments"][0]["content"] += "\n# tampered"
    assert not verify_context_commitment(payload)


def test_verified_context_is_deterministic_for_same_snapshot(tmp_path: Path) -> None:
    _project(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    first = service.context("charge_card", token_budget=512)
    second = service.context("charge_card", token_budget=512)
    assert first == second


def test_verified_context_constrains_external_proposals_to_indexed_symbols(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    index, _digest, _generation = service._snapshot()
    authorize = next(
        symbol for symbol in index.symbols.values() if symbol.name == "authorize"
    )
    payload = service.context(
        "unrelated natural language request",
        token_budget=256,
        max_hops=0,
        max_fragments=1,
        proposal_scores=[
            {"symbol_id": authorize.symbol_id, "score": 1.0},
            {"symbol_id": "invented::symbol", "score": 1.0},
        ],
        proposal_provider="test-neural-ranker",
    )
    assert payload["fragments"][0]["symbol_id"] == authorize.symbol_id
    assert payload["fragments"][0]["proposal_score"] == 1.0
    assert payload["proposal_overlay"]["accepted"] == [
        {"symbol_id": authorize.symbol_id, "score": 1.0}
    ]
    assert payload["proposal_overlay"]["omissions_by_reason"] == {
        "unknown-symbol": 1
    }
    assert payload["proposal_overlay"]["may_create_symbols_or_edges"] is False
    assert verify_context_commitment(payload)


def test_verified_context_fails_closed_when_source_changed_after_index(tmp_path: Path) -> None:
    _project(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    service.summary()
    _write(
        tmp_path,
        "payments/service.py",
        "def charge_card(card, amount):\n    return False\n",
    )

    payload = service.context("charge_card", token_budget=512, max_hops=0)
    assert all(
        fragment["path"] != "payments/service.py"
        for fragment in payload["fragments"]
    )
    assert payload["receipt"]["omissions_by_reason"]["stale-index"] >= 1


def test_verified_context_uses_signature_resolution_under_tight_budget(tmp_path: Path) -> None:
    body = "\n".join(f"    value_{index} = {index}" for index in range(200))
    _write(
        tmp_path,
        "large.py",
        f"def expensive_operation(customer_id: str) -> int:\n{body}\n    return value_199\n",
    )
    service = RepositoryIntelligenceService(tmp_path)
    payload = service.context(
        "expensive_operation",
        token_budget=128,
        max_hops=0,
        max_fragments=1,
    )
    assert len(payload["fragments"]) == 1
    fragment = payload["fragments"][0]
    assert fragment["resolution"] == "signature"
    assert fragment["content"].startswith("def expensive_operation")
    assert fragment["estimated_tokens"] <= 128


def test_verified_context_can_include_bounded_local_git_history(tmp_path: Path) -> None:
    _project(tmp_path)
    commands = (
        ("init", "-q"),
        ("config", "user.email", "benchmark@example.invalid"),
        ("config", "user.name", "Entroly Benchmark"),
        ("config", "commit.gpgsign", "false"),
        ("add", "."),
        ("commit", "-q", "-m", "add payment authorization"),
    )
    for command in commands:
        subprocess.run(["git", *command], cwd=tmp_path, check=True)

    payload = RepositoryIntelligenceService(tmp_path).context(
        "charge_card authorization",
        token_budget=512,
        include_history=True,
        max_history_commits=3,
    )
    assert payload["history"]["available"] is True
    assert payload["history"]["commits"][0]["subject"] == "add payment authorization"
    assert len(payload["history"]["commits"][0]["commit"]) == 40
    assert payload["history"]["remote_calls"] == 0
    assert verify_context_commitment(payload)


def test_symbol_graph_verifies_call_evidence_and_refuses_tampering(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    payload = RepositoryIntelligenceService(tmp_path).symbol_graph(
        "authorize",
        direction="callers",
        max_depth=2,
    )

    assert payload["schema_version"] == "entroly.verified-symbol-graph.v1"
    assert payload["resolution"] == "resolved"
    assert {node["qualified_name"] for node in payload["nodes"]} >= {
        "authorize",
        "charge_card",
    }
    assert payload["receipt"]["selected_edge_count"] >= 1
    assert verify_symbol_graph_commitment(payload)
    for edge in payload["edges"]:
        raw = (tmp_path / edge["path"]).read_bytes()
        evidence = raw[edge["start_byte"]:edge["end_byte"]]
        assert hashlib.sha256(evidence).hexdigest() == edge["evidence_sha256"]

    payload["edges"][0]["callee_id"] = "invented"
    assert not verify_symbol_graph_commitment(payload)


def test_symbol_graph_preserves_ambiguous_identity(tmp_path: Path) -> None:
    _write(tmp_path, "left.py", "def duplicate():\n    return 1\n")
    _write(tmp_path, "right.py", "def duplicate():\n    return 2\n")
    payload = RepositoryIntelligenceService(tmp_path).symbol_graph("duplicate")

    assert payload["resolution"] == "ambiguous"
    assert len(payload["candidates"]) == 2
    assert payload["nodes"] == []
    assert payload["edges"] == []
    assert verify_symbol_graph_commitment(payload)


def test_symbol_graph_fails_closed_when_indexed_source_is_stale(tmp_path: Path) -> None:
    _project(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)
    service.summary()
    _write(tmp_path, "payments/gateway.py", "def authorize():\n    return False\n")

    payload = service.symbol_graph("authorize")

    assert payload["resolution"] == "stale-index"
    assert payload["nodes"] == []
    assert payload["receipt"]["omissions_by_reason"] == {"stale-index": 1}
    assert verify_symbol_graph_commitment(payload)
