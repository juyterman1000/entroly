"""Byte-canonical SourceSpan coordinate system: exact, fail-closed, verifiable.

Covers the coordinate-system contract: byte-exact location, derived line ranges,
independent verification, and every fail-closed case (CRLF vs LF, Unicode,
duplicate blocks, empty files, missing final newline, overlap/adjacency merging,
whole-file, post-receipt source mutation, out-of-bounds, serialization round-trip).
"""

from __future__ import annotations

from entroly.source_span import (
    SCHEMA_VERSION,
    Representation,
    SourceSpan,
    compute_span,
    derive_lines,
    digest,
    merge_spans,
)

SRC = b"def alpha():\n    return 1\n\ndef beta():\n    x = compute()\n    return x\n"
#      line: 1            2          3   4            5                6


def test_whole_file_span_is_zero_to_len_and_verifies():
    span = compute_span(SRC, SRC, "m.py", representation=Representation.WHOLE_FILE.value)
    assert isinstance(span, SourceSpan)
    assert (span.byte_start, span.byte_end) == (0, len(SRC))
    assert span.representation == "whole_file"
    assert span.verify(SRC) is True


def test_unique_block_byte_and_line_exact():
    block = b"def beta():\n    x = compute()\n    return x"
    span = compute_span(SRC, block, "m.py", representation=Representation.SYNTAX_BLOCK.value)
    assert isinstance(span, SourceSpan)
    assert SRC[span.byte_start:span.byte_end] == block
    assert (span.line_start, span.line_end) == (4, 6)
    assert span.verify(SRC) is True


def test_derive_lines_handles_no_final_newline_and_empty_span():
    no_nl = b"a\nb\nc"           # 3 lines, no trailing newline
    assert derive_lines(no_nl, 0, len(no_nl)) == (1, 3)
    assert derive_lines(no_nl, 4, 5) == (3, 3)   # 'c'
    assert derive_lines(no_nl, 2, 2) == (2, 2)   # empty span sits on one line


def test_duplicate_block_is_ambiguous_fail_closed():
    dup = b"x = 1\ny = 2\nx = 1\n"   # "x = 1" occurs twice
    assert compute_span(dup, b"x = 1", "d.py") == "ambiguous_duplicate"


def test_absent_block_not_found():
    assert compute_span(SRC, b"def omega():\n    pass", "m.py") == "not_found"


def test_empty_block_and_empty_file():
    assert compute_span(SRC, b"", "m.py") == "empty_block"
    # An empty file: only an empty (whole-file) block, which is still empty_block.
    assert compute_span(b"", b"", "e.py") == "empty_block"


def test_crlf_bytes_are_distinct_from_lf_fail_closed():
    crlf = b"def f():\r\n    return 1\r\n"
    # A CRLF block matches the CRLF source exactly at the byte level.
    span = compute_span(crlf, b"    return 1\r\n", "f.py")
    assert isinstance(span, SourceSpan) and span.verify(crlf)
    # The SAME text with LF bytes is genuinely different bytes -> fail closed,
    # never silently normalized into a match.
    assert compute_span(crlf, b"    return 1\n", "f.py") == "not_found"


def test_unicode_bytes_before_and_inside_span():
    src = "# café\nval = 'naïve – 漢字'\nx = 1\n".encode("utf-8")
    block = "val = 'naïve – 漢字'".encode("utf-8")
    span = compute_span(src, block, "u.py")
    assert isinstance(span, SourceSpan)
    assert src[span.byte_start:span.byte_end] == block
    assert (span.line_start, span.line_end) == (2, 2)   # unicode before it doesn't shift the line
    assert span.verify(src)


def test_verify_fails_on_source_mutation_after_receipt():
    span = compute_span(SRC, b"    return 1", "m.py")
    assert isinstance(span, SourceSpan) and span.verify(SRC)
    mutated = SRC.replace(b"return 1", b"return 2")
    assert span.verify(mutated) is False        # source_digest no longer matches


def test_verify_fails_on_out_of_bounds():
    good = compute_span(SRC, b"    return 1", "m.py")
    assert isinstance(good, SourceSpan)
    bad = SourceSpan(good.source_path, good.source_digest, good.byte_start,
                     len(SRC) + 50, good.line_start, good.line_end, good.fragment_digest)
    assert bad.verify(SRC) is False


def test_merge_overlapping_and_adjacent_spans_stay_verifiable():
    s1 = compute_span(SRC, b"def alpha():\n    return 1", "m.py")            # bytes 0..25 (lines 1-2)
    s2 = compute_span(SRC, b"    return 1\n\ndef beta():", "m.py")           # overlaps s1 tail
    assert isinstance(s1, SourceSpan) and isinstance(s2, SourceSpan)
    merged = merge_spans([s2, s1], {"m.py": SRC})
    assert len(merged) == 1
    m = merged[0]
    assert m.representation == "merged"
    assert m.byte_start == 0 and m.byte_end == s2.byte_end
    assert m.verify(SRC) is True                # recomputed digest re-derives exactly


def test_merge_leaves_gapped_spans_separate():
    a = compute_span(SRC, b"def alpha():", "m.py")
    b = compute_span(SRC, b"def beta():", "m.py")
    assert isinstance(a, SourceSpan) and isinstance(b, SourceSpan)
    merged = merge_spans([a, b], {"m.py": SRC})
    assert len(merged) == 2                      # a genuine gap between them

def test_merge_refuses_inconsistent_snapshots():
    a = compute_span(SRC, b"def alpha():", "m.py")
    other = SRC + b"# changed\n"
    b = compute_span(other, b"def beta():", "m.py")   # different source_digest, same path
    assert isinstance(a, SourceSpan) and isinstance(b, SourceSpan)
    merged = merge_spans([a, b], {"m.py": SRC})
    assert len(merged) == 2                      # never merged across snapshots


def test_serialization_round_trip_and_schema_version():
    span = compute_span(SRC, b"    return 1", "m.py", source_commit="abc123")
    assert isinstance(span, SourceSpan)
    d = span.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert SourceSpan.from_dict(d) == span
    assert digest(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
