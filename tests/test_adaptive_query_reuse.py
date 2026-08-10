from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from entroly.repository_intelligence import adaptive_program_graph as adaptive
from entroly.repository_intelligence.models import FileRecord, RepositoryIndex
from entroly.repository_intelligence.query_semantics import QuerySemanticGraph
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


def _record(path: str, source: str) -> FileRecord:
    raw = source.encode("utf-8")
    return FileRecord(
        path=path,
        language="zig",
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_length=len(raw),
        line_count=1,
        is_test=False,
    )


def _document(path: str, source: str) -> UniversalSemanticDocument:
    raw = source.encode("utf-8")
    evidence = SourceEvidence(
        path=path,
        start_byte=0,
        end_byte=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        start_line=1,
        end_line=1,
    )
    return UniversalSemanticDocument(
        path=path,
        language="zig",
        source_sha256=evidence.sha256,
        byte_length=len(raw),
        capabilities=SemanticCapabilities(
            language="zig",
            level=SemanticLevel.STRUCTURE,
            parser_available=True,
            structure=True,
        ),
        nodes=[SemanticNode(
            node_id="node:file",
            kind="file",
            name=Path(path).name,
            language="zig",
            evidence=evidence,
            epistemic_class=EpistemicClass.EXACT_SOURCE,
        )],
    )


def test_query_semantics_reuses_existing_tree_without_extra_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "fn launch() { return; }\n"
    path = "main.zig"
    (tmp_path / path).write_text(source, encoding="utf-8")
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    index = RepositoryIndex(
        root=str(tmp_path),
        files={path: _record(path, source)},
    )
    monkeypatch.setattr(
        adaptive,
        "build_verified_context",
        lambda *args, **kwargs: {
            "fragments": [{"path": path}],
            "receipt": {"context_sha256": "ctx"},
        },
    )

    counts = {
        "session": 0,
        "registry": 0,
        "scan": 0,
        "semantic": 0,
        "query": 0,
    }
    sentinel_tree = object()
    session = SimpleNamespace(
        language="zig",
        source_sha256=source_sha,
        file_path=path,
        syntax_valid=True,
        tree=sentinel_tree,
    )
    scan = SyntaxScan(
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
    query_graph = QuerySemanticGraph(
        language="zig",
        source_sha256=source_sha,
        scopes=(),
        definitions=(),
        references=(),
        tag_facts=(),
        bindings=(),
        unresolved=(),
        locals_available=True,
        tags_available=True,
        complete=True,
    )

    def build_session(*args, **kwargs):
        counts["session"] += 1
        return session

    def build_registry(*args, **kwargs):
        counts["registry"] += 1
        return facts

    def build_scan(*args, **kwargs):
        counts["scan"] += 1
        assert args[0] is session
        return scan

    def build_semantic(*args, **kwargs):
        counts["semantic"] += 1
        assert kwargs["precomputed_syntax_session"] is session
        assert kwargs["precomputed_syntax_scan"] is scan
        assert kwargs["precomputed_registry_facts"] is facts
        return _document(path, source)

    def build_query(existing_session, **kwargs):
        counts["query"] += 1
        assert existing_session is session
        assert existing_session.tree is sentinel_tree
        return query_graph

    monkeypatch.setattr(adaptive, "build_syntax_session", build_session)
    monkeypatch.setattr(adaptive, "extract_registry_facts", build_registry)
    monkeypatch.setattr(adaptive, "scan_syntax_session", build_scan)
    monkeypatch.setattr(adaptive, "build_universal_semantic_document", build_semantic)
    monkeypatch.setattr(adaptive, "build_query_semantic_graph", build_query)

    payload = adaptive.build_adaptive_program_graph(
        tmp_path,
        index,
        "launch",
        index_digest="sha256:index",
    )

    assert counts == {
        "session": 1,
        "registry": 1,
        "scan": 1,
        "semantic": 1,
        "query": 1,
    }
    assert payload["coverage"]["query_semantic_files"] == 1
    assert payload["coverage"]["locals_query_files"] == 1
    assert payload["coverage"]["tags_query_files"] == 1
    assert payload["analysis_contract"]["parser_work_ceiling"] == (
        "one-registry-pass-plus-one-raw-parse-per-selected-file"
    )
    assert payload["analysis_contract"]["query_semantics"] == (
        "bundled-locals-tags-on-existing-tree-no-extra-parse"
    )
