from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from entroly.repository_intelligence import adaptive_program_graph as adaptive
from entroly.repository_intelligence.models import FileRecord, RepositoryIndex, Symbol
from entroly.repository_intelligence.semantic_ir import (
    EpistemicClass,
    SemanticCapabilities,
    SemanticLevel,
    SemanticNode,
    SourceEvidence,
    UniversalSemanticDocument,
)


def _record(path: str, source: str, language: str) -> FileRecord:
    raw = source.encode("utf-8")
    return FileRecord(
        path=path,
        language=language,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_length=len(raw),
        line_count=max(1, source.count("\n") + 1),
        is_test=False,
    )


def _semantic_document(path: str, source: str, language: str) -> UniversalSemanticDocument:
    raw = source.encode("utf-8")
    evidence = SourceEvidence(
        path=path,
        start_byte=0,
        end_byte=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        start_line=1,
        end_line=max(1, source.count("\n") + 1),
    )
    return UniversalSemanticDocument(
        path=path,
        language=language,
        source_sha256=evidence.sha256,
        byte_length=len(raw),
        capabilities=SemanticCapabilities(
            language=language,
            level=SemanticLevel.STRUCTURE,
            parser_available=True,
            structure=True,
        ),
        nodes=[SemanticNode(
            node_id="node:file",
            kind="file",
            name=Path(path).name,
            language=language,
            evidence=evidence,
            epistemic_class=EpistemicClass.EXACT_SOURCE,
        )],
    )


def test_non_python_symbol_remains_useful_without_invented_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "pub fn launch() void { return; }\n"
    path = "src/main.zig"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    symbol = Symbol(
        symbol_id="src/main.zig::launch::function",
        path=path,
        name="launch",
        qualified_name="launch",
        kind="function",
        line_start=1,
        line_end=1,
        language="zig",
        start_byte=0,
        end_byte=len(source.encode("utf-8")),
        parse_backend="tree-sitter",
    )
    index = RepositoryIndex(
        root=str(tmp_path),
        files={path: _record(path, source, "zig")},
        symbols={symbol.symbol_id: symbol},
    )
    monkeypatch.setattr(
        adaptive,
        "build_verified_context",
        lambda *args, **kwargs: {
            "fragments": [{"path": path, "symbol_id": symbol.symbol_id}],
            "receipt": {"context_sha256": "ctx"},
        },
    )
    monkeypatch.setattr(
        adaptive,
        "build_universal_semantic_document",
        lambda *args, **kwargs: _semantic_document(path, source, "zig"),
    )

    payload = adaptive.build_adaptive_program_graph(
        tmp_path,
        index,
        "launch",
        index_digest="sha256:index",
    )
    assert payload["semantic_files"][0]["language"] == "zig"
    assert payload["deep_semantics"] == []
    assert payload["adapter_boundaries"] == [{
        "symbol_id": symbol.symbol_id,
        "language": "zig",
        "available": "structure",
        "missing": "verified-language-specific-flow-adapter",
    }]
    assert payload["analysis_contract"]["missing_adapters_behavior"] == (
        "report-boundary-never-invent-flow"
    )
    assert adaptive.verify_adaptive_program_graph_commitment(payload)


def test_verified_python_adapter_is_attached_without_changing_universal_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "def run(value):\n    return value\n"
    path = "run.py"
    (tmp_path / path).write_text(source, encoding="utf-8")
    symbol = Symbol(
        symbol_id="run.py::run::function",
        path=path,
        name="run",
        qualified_name="run",
        kind="function",
        line_start=1,
        line_end=2,
        language="python",
        start_byte=0,
        end_byte=len(source.encode("utf-8")),
        parse_backend="python-ast",
    )
    index = RepositoryIndex(
        root=str(tmp_path),
        files={path: _record(path, source, "python")},
        symbols={symbol.symbol_id: symbol},
    )
    monkeypatch.setattr(
        adaptive,
        "build_verified_context",
        lambda *args, **kwargs: {
            "fragments": [{"path": path, "symbol_id": symbol.symbol_id}],
            "receipt": {"context_sha256": "ctx"},
        },
    )
    monkeypatch.setattr(
        adaptive,
        "build_universal_semantic_document",
        lambda *args, **kwargs: _semantic_document(path, source, "python"),
    )
    monkeypatch.setattr(
        adaptive,
        "build_verified_program_graph",
        lambda *args, **kwargs: {
            "schema_version": "test-program",
            "receipt": {"program_graph_sha256": "pg"},
        },
    )
    monkeypatch.setattr(
        adaptive,
        "build_verified_interprocedural_flow",
        lambda *args, **kwargs: {
            "schema_version": "test-flow",
            "receipt": {"interprocedural_flow_sha256": "flow"},
        },
    )

    payload = adaptive.build_adaptive_program_graph(
        tmp_path,
        index,
        "run",
        index_digest="sha256:index",
    )
    assert payload["semantic_files"][0]["language"] == "python"
    assert len(payload["deep_semantics"]) == 1
    assert payload["deep_semantics"][0]["semantic_level"] == "flow"
    assert payload["adapter_boundaries"] == []
    assert adaptive.verify_adaptive_program_graph_commitment(payload)


def test_stale_source_is_reported_and_never_materialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "main.future"
    original = "fn original() { return 1; }\n"
    current = "fn changed() { return 2; }\n"
    (tmp_path / path).write_text(current, encoding="utf-8")
    index = RepositoryIndex(
        root=str(tmp_path),
        files={path: _record(path, original, "unknown")},
    )
    monkeypatch.setattr(
        adaptive,
        "build_verified_context",
        lambda *args, **kwargs: {
            "fragments": [{"path": path}],
            "receipt": {"context_sha256": "ctx"},
        },
    )
    payload = adaptive.build_adaptive_program_graph(
        tmp_path,
        index,
        "changed",
        index_digest="sha256:index",
    )
    assert payload["semantic_files"] == []
    assert payload["diagnostics"] == [{"path": path, "status": "stale-index"}]


def test_top_level_commitment_detects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "main.future"
    source = "thing { nested { value } }\n"
    (tmp_path / path).write_text(source, encoding="utf-8")
    index = RepositoryIndex(
        root=str(tmp_path),
        files={path: _record(path, source, "unknown")},
    )
    monkeypatch.setattr(
        adaptive,
        "build_verified_context",
        lambda *args, **kwargs: {
            "fragments": [{"path": path}],
            "receipt": {"context_sha256": "ctx"},
        },
    )
    monkeypatch.setattr(
        adaptive,
        "build_universal_semantic_document",
        lambda *args, **kwargs: _semantic_document(path, source, "unknown"),
    )
    payload = adaptive.build_adaptive_program_graph(
        tmp_path,
        index,
        "value",
        index_digest="sha256:index",
    )
    assert adaptive.verify_adaptive_program_graph_commitment(payload)
    payload["coverage"]["semantic_nodes"] = 999
    assert not adaptive.verify_adaptive_program_graph_commitment(payload)
