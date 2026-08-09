from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from entroly.semantic_resolution import extract_blocks
from entroly.tree_sitter_support import (
    LANGUAGE_BY_SUFFIX,
    _get_local_parser,
    extract_structural_spans,
)


def test_language_map_covers_at_least_twenty_seven_languages() -> None:
    assert len(set(LANGUAGE_BY_SUFFIX.values())) >= 27


@pytest.mark.parametrize(
    ("path", "source", "names"),
    [
        ("sample.py", "class Cart:\n    def total(self):\n        return 1\n", {"Cart", "total"}),
        ("sample.rs", "pub struct Cart { n: u32 }\nimpl Cart { pub fn total(&self) -> u32 { self.n } }\n", {"Cart", "total"}),
        ("sample.ts", "export class Cart { total(): number { return 1; } }\n", {"Cart", "total"}),
        ("sample.c", "int total(int price) { return price; }\n", {"total"}),
    ],
)
def test_parser_backed_spans_are_exact(path: str, source: str, names: set[str]) -> None:
    pytest.importorskip("tree_sitter_language_pack")
    spans = extract_structural_spans(source, path)
    assert spans is not None
    assert names <= {span.name for span in spans}
    for span in spans:
        assert span.source in source
        assert source.splitlines()[span.start_line - 1].strip()
        assert span.start_line <= span.end_line


def test_semantic_resolution_uses_parser_spans_when_available() -> None:
    pytest.importorskip("tree_sitter_language_pack")
    source = "export class Cart {\n  total(): number { return 1; }\n}\n"
    blocks = extract_blocks(source, "cart.ts")
    assert {block.name for block in blocks} >= {"Cart", "total"}
    assert all(block.start_line <= block.end_line for block in blocks)


def test_pack_v1_never_downloads_without_explicit_opt_in(monkeypatch) -> None:
    calls: list[str] = []
    fake = SimpleNamespace(
        downloaded_languages=lambda: ["python"],
        get_parser=lambda language: calls.append(language),
    )
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", fake)
    monkeypatch.delenv("ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD", raising=False)
    assert _get_local_parser("rust") is None
    assert calls == []


def test_unknown_or_unavailable_parser_fails_open() -> None:
    assert extract_structural_spans("hello", "notes.txt") is None
