from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence import semantic_ir
from entroly.repository_intelligence.semantic_ir import (
    EpistemicClass,
    build_universal_semantic_document,
)
from entroly.tree_sitter_support import (
    StructuralExtraction,
    StructuralSpan,
    _downloads_allowed,
    available_parser_languages,
    extract_structural_spans,
    language_for_path,
)


def test_registry_exposes_broad_language_set_including_c_and_zig() -> None:
    languages = set(available_parser_languages())
    # The base dependency is intentionally the universal registry rather than
    # an Entroly-maintained language allowlist.
    assert len(languages) >= 300
    assert "c" in languages
    assert "zig" in languages
    assert language_for_path("src/main.c") == "c"
    assert language_for_path("src/main.zig") == "zig"


def test_parser_downloads_are_default_but_air_gap_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.delenv("ENTROLY_AIR_GAP", raising=False)
    assert _downloads_allowed() is True

    monkeypatch.setenv("ENTROLY_AIR_GAP", "1")
    monkeypatch.setenv("ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD", "1")
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
    monkeypatch.setattr(semantic_ir, "validate_structural_syntax", lambda *a, **k: None)
    monkeypatch.setattr(semantic_ir, "extract_structural_spans_report", lambda *a, **k: None)
    monkeypatch.setattr(semantic_ir, "extract_structural_calls_report", lambda *a, **k: None)

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


def test_incomplete_parser_traversal_is_rejected_not_promoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "fn run() { return 1; }\n"
    span = StructuralSpan(
        name="run",
        kind="function",
        start_line=1,
        end_line=1,
        source=source.strip(),
        signature="fn run()",
        indent=0,
        start_byte=0,
        end_byte=len(source.encode("utf-8")) - 1,
    )
    incomplete = StructuralExtraction(
        items=(span,),
        language="zig",
        complete=False,
        nodes_visited=10,
        max_nodes=10,
        backend="test",
    )
    monkeypatch.setattr(semantic_ir, "validate_structural_syntax", lambda *a, **k: True)
    monkeypatch.setattr(
        semantic_ir, "extract_structural_spans_report", lambda *a, **k: incomplete
    )
    monkeypatch.setattr(semantic_ir, "extract_structural_calls_report", lambda *a, **k: None)

    document = build_universal_semantic_document(source, "main.zig", max_nodes=10)
    assert document.capabilities.structure is False
    assert all(
        node.epistemic_class is not EpistemicClass.PARSER_VERIFIED
        for node in document.nodes
    )
    assert any("partial parser results were rejected" in item for item in document.diagnostics)


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
