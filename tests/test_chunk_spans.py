"""Tests for the AST-aware span→token-budget chunker."""

from __future__ import annotations

from entroly.tree_sitter_support import CodeChunk, StructuralSpan, chunk_spans


def _span(name: str, source: str, start_line: int = 1, end_line: int = 1) -> StructuralSpan:
    return StructuralSpan(
        name=name,
        kind="function",
        start_line=start_line,
        end_line=end_line,
        source=source,
        signature=f"def {name}():",
        indent=0,
        start_byte=0,
        end_byte=len(source.encode("utf-8")),
    )


def test_chunk_spans_empty():
    assert chunk_spans([]) == []


def test_chunk_spans_single():
    spans = [_span("f", "def f(): pass")]
    result = chunk_spans(spans, max_tokens=1000)
    assert len(result) == 1
    assert len(result[0].spans) == 1
    assert result[0].spans[0].name == "f"


def test_chunk_spans_respects_budget():
    big = "x = 1\n" * 500
    spans = [
        _span("a", big, 1, 500),
        _span("b", big, 501, 1000),
    ]
    result = chunk_spans(spans, max_tokens=200)
    assert len(result) == 2
    for chunk in result:
        assert len(chunk.spans) == 1


def test_chunk_spans_packs_small():
    spans = [
        _span("a", "x = 1", 1, 1),
        _span("b", "y = 2", 2, 2),
        _span("c", "z = 3", 3, 3),
    ]
    result = chunk_spans(spans, max_tokens=1000)
    assert len(result) == 1
    assert len(result[0].spans) == 3


def test_chunk_spans_oversized_span_not_dropped():
    big = "x = 1\n" * 1000
    spans = [_span("huge", big, 1, 1000)]
    result = chunk_spans(spans, max_tokens=10)
    assert len(result) == 1
    assert result[0].spans[0].name == "huge"


def test_chunk_spans_text_property():
    spans = [
        _span("a", "def a(): pass", 1, 1),
        _span("b", "def b(): pass", 2, 2),
    ]
    result = chunk_spans(spans, max_tokens=1000)
    assert "def a(): pass" in result[0].text
    assert "def b(): pass" in result[0].text


def test_chunk_spans_line_range():
    spans = [
        _span("a", "def a(): pass", 10, 15),
        _span("b", "def b(): pass", 20, 25),
    ]
    result = chunk_spans(spans, max_tokens=1000)
    assert result[0].start_line == 10
    assert result[0].end_line == 25
