from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence import semantic_ir
from entroly.repository_intelligence.registry_frontend import (
    RegistryExport,
    RegistryFacts,
    RegistryImport,
    RegistrySpan,
    RegistryStructure,
    RegistrySymbol,
    extract_registry_facts,
)
from entroly.repository_intelligence.semantic_ir import (
    EpistemicClass,
    build_universal_semantic_document,
)
from entroly.tree_sitter_support import (
    _downloads_allowed,
    available_parser_languages,
    extract_structural_spans,
    language_for_path,
)


def test_registry_detection_is_broad_without_remote_manifest_access() -> None:
    # This accessor is intentionally local-only: it must not fetch the remote
    # language manifest merely to report parser availability. Path detection,
    # however, remains broad through the installed registry plus Entroly's
    # deterministic suffix fallback.
    languages = set(available_parser_languages())
    assert languages
    assert "python" in languages
    assert language_for_path("src/main.c") == "c"
    assert language_for_path("src/main.zig") == "zig"


def test_parser_downloads_require_explicit_opt_in_and_air_gap_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.delenv("ENTROLY_AIR_GAP", raising=False)
    assert _downloads_allowed() is False

    monkeypatch.setenv("ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD", "1")
    assert _downloads_allowed() is True

    monkeypatch.setenv("ENTROLY_AIR_GAP", "1")
    assert _downloads_allowed() is False

    monkeypatch.delenv("ENTROLY_AIR_GAP", raising=False)
    monkeypatch.setenv("ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD", "0")
    assert _downloads_allowed() is False


def test_unregistered_future_language_is_indexed_as_exact_source(
    tmp_path: Path,
) -> None:
    source = (
        "module nebula {\n"
        "  function launch(target) {\n"
        "    return target;\n"
        "  }\n"
        "}\n"
    )
    target = tmp_path / "engine.futurelang"
    target.write_text(source, encoding="utf-8")

    index = build_repository_index(tmp_path)
    assert "engine.futurelang" in index.files
    assert index.files["engine.futurelang"].language == "unknown"


def test_universal_ir_unknown_language_keeps_exact_source_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(semantic_ir, "build_syntax_session", lambda *a, **k: None)
    monkeypatch.setattr(semantic_ir, "extract_registry_facts", lambda *a, **k: None)

    source = "thing {\n  nested { value }\n}\n"
    document = build_universal_semantic_document(source, "sample.future")
    payload = document.to_dict()

    assert payload["capabilities"]["level"] == "source"
    assert payload["nodes"][0]["kind"] == "file"
    assert payload["nodes"][0]["epistemic_class"] == "exact-source"
    assert payload["nodes"][0]["evidence"]["sha256"] == hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()
    assert any(
        node["epistemic_class"] == EpistemicClass.HEURISTIC.value
        for node in payload["nodes"][1:]
    )


def _span(source: str, start: int, end: int, line: int) -> RegistrySpan:
    raw = source.encode("utf-8")
    return RegistrySpan(
        start_byte=start,
        end_byte=end,
        start_line=line,
        end_line=line,
        evidence_sha256=hashlib.sha256(raw[start:end]).hexdigest(),
    )


def test_incomplete_registry_analysis_is_rejected_not_promoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "fn run() { return 1; }\n"
    facts = RegistryFacts(
        language="zig",
        imports=(),
        exports=(),
        symbols=(),
        diagnostics=(),
        node_count=10,
        complete=False,
        structures=(RegistryStructure(
            name="run",
            kind="function",
            signature="fn run()",
            span=_span(source, 0, len(source.encode()) - 1, 1),
        ),),
    )
    monkeypatch.setattr(semantic_ir, "build_syntax_session", lambda *a, **k: None)

    document = build_universal_semantic_document(
        source,
        "main.zig",
        max_nodes=10,
        precomputed_registry_facts=facts,
    )
    assert document.capabilities.structure is False
    assert all(
        not (
            node.name == "run"
            and node.epistemic_class is EpistemicClass.PARSER_VERIFIED
        )
        for node in document.nodes
    )
    assert any("partial structure/facts were rejected" in item for item in document.diagnostics)


def test_registry_normalized_structure_can_serve_new_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import entroly.tree_sitter_support as support

    source = "fn launch() void { return; }\n"
    raw = source.encode("utf-8")

    class Node:
        type = "source_file"
        children: list[object] = []
        has_error = False

    class Parser:
        @staticmethod
        def parse(_raw: bytes) -> object:
            return SimpleNamespace(root_node=Node())

    class Config:
        def __init__(self, **kwargs: object) -> None:
            self.language = kwargs.get("language")

    item = SimpleNamespace(
        kind=SimpleNamespace(name="Function"),
        name="launch",
        signature="fn launch() void",
        span=SimpleNamespace(
            start_byte=0,
            end_byte=len(raw),
            start_line=0,
            end_line=0,
        ),
        children=[],
    )
    fake_pack = SimpleNamespace(
        detect_language_from_path=lambda path: "future_systems",
        get_parser=lambda language: Parser(),
        ProcessConfig=Config,
        process=lambda text, config: SimpleNamespace(structure=[item]),
        available_languages=lambda: ["future_systems"],
    )
    monkeypatch.setattr(support, "_pack", lambda: fake_pack)
    monkeypatch.delenv("ENTROLY_AIR_GAP", raising=False)
    monkeypatch.setenv("ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD", "1")

    spans = extract_structural_spans(source, "main.fsys")
    assert spans is not None
    assert [(span.name, span.kind) for span in spans] == [("launch", "function")]
    assert spans[0].source == source


def test_registry_fact_adapter_hashes_exact_import_export_symbol_and_structure_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = 'import core\nexport fn run() void {}\n'
    raw = source.encode("utf-8")
    import_end = len(b"import core")
    export_start = len(b"import core\n")
    export_end = len(raw) - 1

    class Config:
        def __init__(self, **kwargs: object) -> None:
            self.language = kwargs["language"]

    result = {
        "metrics": {"node_count": 12},
        "structure": [{
            "name": "run",
            "kind": "Function",
            "signature": "export fn run() void",
            "span": {
                "start_byte": export_start,
                "end_byte": export_end,
                "start_line": 1,
                "end_line": 1,
            },
            "children": [],
        }],
        "imports": [{
            "source": "core",
            "items": ["Thing"],
            "alias": "c",
            "is_wildcard": False,
            "span": {
                "start_byte": 0,
                "end_byte": import_end,
                "start_line": 0,
                "end_line": 0,
            },
        }],
        "exports": [{
            "name": "run",
            "kind": "Named",
            "span": {
                "start_byte": export_start,
                "end_byte": export_end,
                "start_line": 1,
                "end_line": 1,
            },
        }],
        "symbols": [{
            "name": "run",
            "kind": "Function",
            "type_annotation": "void",
            "span": {
                "start_byte": export_start,
                "end_byte": export_end,
                "start_line": 1,
                "end_line": 1,
            },
        }],
        "diagnostics": [],
    }
    fake_pack = SimpleNamespace(ProcessConfig=Config, process=lambda text, config: result)
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", fake_pack)

    facts = extract_registry_facts(source, "zig")
    assert facts is not None and facts.complete
    assert facts.structures[0].name == "run"
    assert facts.imports[0].source == "core"
    assert facts.imports[0].items == ("Thing",)
    assert facts.imports[0].span.evidence_sha256 == hashlib.sha256(
        raw[:import_end]
    ).hexdigest()
    assert facts.exports[0].name == "run"
    assert facts.symbols[0].type_annotation == "void"


def test_registry_fact_node_bound_never_promotes_partial_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config:
        def __init__(self, **kwargs: object) -> None:
            self.language = kwargs["language"]

    fake_pack = SimpleNamespace(
        ProcessConfig=Config,
        process=lambda text, config: {
            "metrics": {"node_count": 1001},
            "structure": [],
            "imports": [],
            "exports": [],
            "symbols": [],
            "diagnostics": [],
        },
    )
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", fake_pack)
    facts = extract_registry_facts("fn run() {}", "zig", max_nodes=1000)
    assert facts is not None
    assert facts.complete is False


def test_universal_ir_materializes_structure_and_registry_facts_without_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = 'const x = @import("core.zig");\npub fn run() void {}\n'
    raw = source.encode("utf-8")
    import_end = source.index(";") + 1
    export_start = source.index("pub fn")
    export_end = len(raw) - 1
    facts = RegistryFacts(
        language="zig",
        imports=(RegistryImport(
            source="core.zig",
            items=(),
            alias="x",
            is_wildcard=False,
            span=_span(source, 0, import_end, 1),
        ),),
        exports=(RegistryExport(
            name="run",
            kind="named",
            span=_span(source, export_start, export_end, 2),
        ),),
        symbols=(RegistrySymbol(
            name="run",
            kind="function",
            type_annotation="void",
            span=_span(source, export_start, export_end, 2),
        ),),
        diagnostics=(),
        node_count=20,
        complete=True,
        structures=(RegistryStructure(
            name="run",
            kind="function",
            signature="pub fn run() void",
            span=_span(source, export_start, export_end, 2),
        ),),
    )
    monkeypatch.setattr(semantic_ir, "build_syntax_session", lambda *a, **k: None)

    document = build_universal_semantic_document(
        source,
        "main.zig",
        precomputed_registry_facts=facts,
    )
    payload = document.to_dict()
    assert any(node["kind"] == "function" for node in payload["nodes"])
    assert any(node["kind"] == "import" for node in payload["nodes"])
    assert any(node["kind"] == "export" for node in payload["nodes"])
    assert any(node["kind"] == "symbol:function" for node in payload["nodes"])
    import_edge = next(
        edge for edge in payload["edges"] if edge["relation"] == "imports-source"
    )
    assert import_edge["target_name"] == "core.zig"
    assert import_edge["epistemic_class"] == "parser-verified"
    assert document.capabilities.structure is True
    assert document.capabilities.semantic_binding is False
