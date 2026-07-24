"""Exact, fail-closed line attribution for the ContextBench span adapter.

Covers the adapter's required behaviours: multiple spans per file, overlapping
spans, repeated identical text (ambiguity -> fail closed), CRLF vs LF, Unicode,
missing files, renamed/stale paths, and whole-file selection.
"""

from __future__ import annotations

from benchmarks.contextbench_span_adapter import (
    SelectedSpan,
    canonical_path,
    locate_block,
    map_selection,
    to_spans,
)

# A source file with a UNIQUE marker per region plus a DUPLICATED block, so the
# ambiguity path is exercised deterministically.
SRC = (
    "def alpha():\n"          # 1
    "    return 1\n"          # 2
    "\n"                      # 3
    "def beta():\n"           # 4
    "    x = compute()\n"     # 5
    "    return x\n"          # 6
    "\n"                      # 7
    "def gamma():\n"          # 8
    "    x = compute()\n"     # 9  (duplicate of line 5)
    "    return x\n"          # 10 (duplicate of line 6)
)


def _origin(fid: str, source: str, content: str) -> dict:
    return {fid: {"source": source, "content": content}}


def _sel(source: str, ids: list[str], score=1.0, tok=10) -> dict:
    return {"source": source, "source_fragment_ids": ids, "relevance": score, "token_count": tok}


def _reader(files: dict[str, str]):
    def read(repo_dir: str, path: str):
        return files.get(path)
    return read


def test_canonical_path_normalizes_prefix_and_separators():
    assert canonical_path("file:pkg\\mod.py") == "pkg/mod.py"
    assert canonical_path("./a/b.py") == "a/b.py"
    assert canonical_path("/x/y.py") == "x/y.py"


def test_whole_file_selection_maps_all_lines():
    res = locate_block(SRC, SRC)
    assert res == (1, 10)


def test_unique_block_maps_to_exact_interval():
    block = "def beta():\n    x = compute()\n    return x"
    assert locate_block(block, SRC) == (4, 6)


def test_repeated_identical_text_fails_closed():
    # "    x = compute()" appears on lines 5 and 9 -> ambiguous -> no guess.
    assert locate_block("    x = compute()", SRC) == "ambiguous_duplicate"


def test_absent_block_fails_closed():
    assert locate_block("def omega():\n    pass", SRC) == "not_found"


def test_crlf_source_matches_lf_block():
    crlf = SRC.replace("\n", "\r\n")
    block = "def beta():\n    x = compute()\n    return x"
    assert locate_block(block, crlf) == (4, 6)


def test_unicode_block_maps_exactly():
    src = "# ??\ndef f():\n    s = 'café naïve – 漢字'\n    return s\n"
    block = "    s = 'café naïve – 漢字'"
    assert locate_block(block, src) == (3, 3)


def test_multiple_and_overlapping_spans_from_one_file():
    files = {"m.py": SRC}
    origin = {}
    origin.update(_origin("f1", "file:m.py", "def alpha():\n    return 1"))     # 1-2
    origin.update(_origin("f2", "file:m.py", "def beta():\n    x = compute()\n    return x"))  # 4-6
    origin.update(_origin("f3", "file:m.py", "    return x\n\ndef gamma():"))   # overlaps f2 tail (6-8)
    sel = [_sel("file:m.py", ["f1", "f2", "f3"])]
    spans = map_selection(sel, origin, "/repo", read_file=_reader(files))
    assert spans[0].mapped
    assert spans[0].lines == {1, 2, 4, 5, 6, 7, 8}
    assert spans[0].intervals() == [(1, 2), (4, 8)]  # merged, gap at line 3


def test_missing_file_fails_closed():
    sel = [_sel("file:gone.py", ["f1"])]
    origin = _origin("f1", "file:gone.py", "anything")
    spans = map_selection(sel, origin, "/repo", read_file=_reader({}))
    assert spans[0].mapped is False and spans[0].reason == "missing_file"
    assert to_spans(spans) == {}


def test_renamed_or_stale_path_fails_closed():
    # File exists but its content no longer contains the fragment (stale index).
    files = {"m.py": "def totally_different():\n    return None\n"}
    origin = _origin("f1", "file:m.py", "def beta():\n    x = compute()\n    return x")
    spans = map_selection([_sel("file:m.py", ["f1"])], origin, "/repo", read_file=_reader(files))
    assert spans[0].mapped is False and spans[0].reason == "not_found"


def test_carries_score_rank_token_cost_in_order():
    files = {"m.py": SRC}
    origin = _origin("f1", "file:m.py", SRC)
    sel = [_sel("file:m.py", ["f1"], score=2.5, tok=42)]
    spans = map_selection(sel, origin, "/repo", read_file=_reader(files))
    assert spans[0].rank == 0 and spans[0].score == 2.5 and spans[0].token_cost == 42
    assert to_spans(spans) == {"m.py": set(range(1, 11))}


def test_no_origin_metadata_fails_closed_not_guessed():
    # A selected fragment with no source_fragment_ids must not be attributed.
    files = {"m.py": SRC}
    spans = map_selection([_sel("file:m.py", [])], {}, "/repo", read_file=_reader(files))
    assert spans[0].mapped is False and spans[0].reason == "no_origin_metadata"


def test_intervals_helper_merges_runs():
    s = SelectedSpan(path="x", score=0, rank=0, token_cost=0, lines={3, 1, 2, 7, 8})
    assert s.intervals() == [(1, 3), (7, 8)]
