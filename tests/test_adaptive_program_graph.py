from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from entroly.repository_intelligence import adaptive_program_graph as adaptive
from entroly.repository_intelligence.models import FileRecord, RepositoryIndex, Symbol
from entroly.repository_intelligence.registry_frontend import RegistryFacts
from entroly.repository_intelligence.semantic_ir import (
    EpistemicClass,
    SemanticCapabilities,
    SemanticLevel,
    SemanticNode,
    SourceEvidence,
    UniversalSemanticDocument,
)
from entroly.repository_intelligence.syntax_session import SyntaxScan


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


def _context(path: str, symbol_id: str | None = None) -> dict[str, object]:
    fragment: dict[str, object] = {"path": path}
    if symbol_id is not None:
        fragment["symbol_id"] = symbol_id
    return {
        "fragments": [fragment],
        "receipt": {"context_sha256": "ctx"},
    }


def _disable_frontends(
    monkeypatch: pytest.MonkeyPatch,
    language: str,
) -> None:
    monkeypatch.setattr(adaptive, "build_syntax_session", lambda *a, **k: None)
    monkeypatch.setattr(adaptive, "language_for_source", lambda *a, **k: language)
    monkeypatch.setattr(adaptive, "extract_registry_facts", lambda *a, **k: None)


def test_non_python_symbol_remains_useful_without_invented_semantic_flow(
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
        lambda *args, **kwargs: _context(path, symbol.symbol_id),
    )
    _disable_frontends(monkeypatch, "zig")
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
        "available": "syntactic-flow-or-structure",
        "missing": "verified-language-specific-semantic-flow-adapter",
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
        lambda *args, **kwargs: _context(path, symbol.symbol_id),
    )
    _disable_frontends(monkeypatch, "python")
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


def test_adaptive_pipeline_calls_each_frontend_once_per_selected_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "fn launch() { return; }\n"
    path = "main.zig"
    (tmp_path / path).write_text(source, encoding="utf-8")
    index = RepositoryIndex(
        root=str(tmp_path),
        files={path: _record(path, source, "zig")},
    )
    monkeypatch.setattr(
        adaptive,
        "build_verified_context",
        lambda *args, **kwargs: _context(path),
    )
    counts = {"session": 0, "registry": 0, "scan": 0, "semantic": 0}
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    fake_session = SimpleNamespace(
        language="zig",
        source_sha256=source_sha,
        file_path=path,
        syntax_valid=True,
    )
    fake_scan = SyntaxScan(
        language="zig",
        source_sha256=source_sha,
        calls=(),
        flow_shapes=(),
        traversal_complete=True,
        calls_complete=True,
        flow_complete=True,
        nodes_visited=1,
        max_nodes=100,
        max_calls=100,
        max_flow_shapes=100,
    )
    facts = RegistryFacts(
        language="zig",
        imports=(),
        exports=(),
        symbols=(),
        diagnostics=(),
        node_count=1,
        complete=True,
    )

    def build_session(*args, **kwargs):
        counts["session"] += 1
        return fake_session

    def build_facts(*args, **kwargs):
        counts["registry"] += 1
        return facts

    def build_scan(*args, **kwargs):
        counts["scan"] += 1
        return fake_scan

    def build_semantic(*args, **kwargs):
        counts["semantic"] += 1
        assert kwargs["precomputed_registry_facts"] is facts
        assert kwargs["precomputed_syntax_session"] is fake_session
        assert kwargs["precomputed_syntax_scan"] is fake_scan
        return _semantic_document(path, source, "zig")

    monkeypatch.setattr(adaptive, "build_syntax_session", build_session)
    monkeypatch.setattr(adaptive, "extract_registry_facts", build_facts)
    monkeypatch.setattr(adaptive, "scan_syntax_session", build_scan)
    monkeypatch.setattr(adaptive, "build_universal_semantic_document", build_semantic)

    payload = adaptive.build_adaptive_program_graph(
        tmp_path,
        index,
        "launch",
        index_digest="sha256:index",
    )
    assert counts == {"session": 1, "registry": 1, "scan": 1, "semantic": 1}
    assert payload["analysis_contract"]["parser_work_ceiling"] == (
        "one-registry-pass-plus-one-raw-parse-per-selected-file"
    )


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
        lambda *args, **kwargs: _context(path),
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
        lambda *args, **kwargs: _context(path),
    )
    _disable_frontends(monkeypatch, "unknown")
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
