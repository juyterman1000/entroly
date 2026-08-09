from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.interprocedural_flow import (
    build_verified_interprocedural_flow,
    verify_interprocedural_flow_commitment,
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(text.encode("utf-8"))


def _project(root: Path):
    _write(
        root,
        "normalize.py",
        "def normalize(value, limit=10):\n"
        "    if value > limit:\n"
        "        return limit\n"
        "    return value\n",
    )
    _write(
        root,
        "pipeline.py",
        "from normalize import normalize\n"
        "def process(raw):\n"
        "    cleaned = normalize(raw, limit=5)\n"
        "    return cleaned\n",
    )
    return build_repository_index(root)


def test_interprocedural_flow_binds_arguments_returns_and_consumer(
    tmp_path: Path,
) -> None:
    index = _project(tmp_path)
    payload = build_verified_interprocedural_flow(
        tmp_path,
        index,
        "process",
        index_digest="test-index",
    )
    assert payload["resolution"] == "resolved"
    assert verify_interprocedural_flow_commitment(payload)
    kinds = [edge["kind"] for edge in payload["flow_edges"]]
    assert kinds.count("argument-to-parameter") == 2
    assert kinds.count("return-to-call-result") == 2
    assert kinds.count("call-result-to-consumer") == 1
    bindings = {
        (edge["binding"], edge["position"], edge["keyword"])
        for edge in payload["flow_edges"]
        if edge["kind"] == "argument-to-parameter"
    }
    assert bindings == {
        ("positional", 0, None),
        ("keyword", None, "limit"),
    }
    for node in payload["flow_nodes"]:
        raw = (tmp_path / node["path"]).read_bytes()
        evidence = raw[node["start_byte"]:node["end_byte"]]
        assert hashlib.sha256(evidence).hexdigest() == node["evidence_sha256"]
        assert hashlib.sha256(raw).hexdigest() == node["source_sha256"]


def test_interprocedural_flow_traverses_multiple_calls_and_skips_method_self(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "chain.py",
        "class Parser:\n"
        "    def parse(self, text):\n"
        "        return text.strip()\n"
        "def finish(value):\n"
        "    return value.upper()\n"
        "def run(parser: Parser, raw):\n"
        "    return finish(parser.parse(raw))\n",
    )
    index = build_repository_index(tmp_path)
    payload = build_verified_interprocedural_flow(
        tmp_path,
        index,
        "run",
        index_digest="test-index",
        max_depth=2,
    )
    assert payload["resolution"] == "resolved"
    assert {item["qualified_name"] for item in payload["symbols"]} >= {
        "run",
        "Parser.parse",
        "finish",
    }
    parameter_text = {
        node["text"]
        for node in payload["flow_nodes"]
        if node["role"] == "formal-parameter"
    }
    assert "self" not in parameter_text
    assert {"text", "value"} <= parameter_text
    assert any(
        edge["kind"] == "call-result-to-argument"
        for edge in payload["flow_edges"]
    )
    assert verify_interprocedural_flow_commitment(payload)


def test_interprocedural_flow_refuses_ambiguity_staleness_and_tampering(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "a.py", "def duplicate(value):\n    return value\n")
    _write(tmp_path, "b.py", "def duplicate(value):\n    return value\n")
    index = build_repository_index(tmp_path)
    ambiguous = build_verified_interprocedural_flow(
        tmp_path,
        index,
        "duplicate",
        index_digest="test-index",
    )
    assert ambiguous["resolution"] == "ambiguous"
    assert ambiguous["flow_edges"] == []
    assert verify_interprocedural_flow_commitment(ambiguous)

    _write(tmp_path, "caller.py", "from a import duplicate\ndef call(x):\n    return duplicate(x)\n")
    index = build_repository_index(tmp_path)
    _write(tmp_path, "caller.py", "from a import duplicate\ndef call(x):\n    return duplicate(x + 1)\n")
    stale = build_verified_interprocedural_flow(
        tmp_path,
        index,
        "call",
        index_digest="test-index",
    )
    assert stale["call_relations"] == []
    assert any("stale-index" in item for item in stale["diagnostics"])
    assert verify_interprocedural_flow_commitment(stale)

    tampered = copy.deepcopy(ambiguous)
    tampered["resolution"] = "resolved"
    assert not verify_interprocedural_flow_commitment(tampered)
