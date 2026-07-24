"""The three selection granularities produce verifiable, deterministic, and
increasingly line-precise spans."""

from __future__ import annotations

from benchmarks.subfile_modes import (
    block_units,
    file_units,
    rank_and_select,
    spans_to_lines,
    verify_rate,
    window_units,
)

# Two files; the query targets `parse_manifest` in a.py specifically.
A = (
    b"import os\n\n"
    b"def unrelated_helper(x):\n    return x + 1\n\n"
    b"def parse_manifest(path):\n"
    b"    data = read_manifest_bytes(path)\n"
    b"    return decode_manifest(data)\n\n"
    b"def another(y):\n    return y * 2\n"
)
B = b"def compute(a, b):\n    return a * b + 1\n\nclass Widget:\n    def render(self):\n        return 'w'\n"
FILES = [("a.py", A), ("b.py", B)]
QUERY = "parse_manifest read_manifest_bytes decode_manifest path"
BUDGET = 4000


def test_all_modes_produce_only_verifiable_spans():
    for units_fn in (file_units, window_units, block_units):
        spans = rank_and_select(FILES, units_fn(FILES), QUERY, BUDGET)
        assert verify_rate(spans, FILES) == 1.0, f"{units_fn.__name__} produced unverifiable spans"


def test_syntax_block_is_line_precise_where_file_level_is_not():
    file_spans = rank_and_select(FILES, file_units(FILES), QUERY, BUDGET)
    block_spans = rank_and_select(FILES, block_units(FILES), QUERY, BUDGET)

    # File mode: the top pick is the whole of a.py (all 12 lines).
    top_file = file_spans[0]
    assert top_file.source_path == "a.py"
    assert (top_file.line_start, top_file.line_end) == (1, A.count(b"\n") + (0 if A.endswith(b"\n") else 1))

    # Block mode: the top pick is exactly the parse_manifest function (lines 6-8),
    # not the whole file — strictly narrower, still inside a.py.
    top_block = block_spans[0]
    assert top_block.source_path == "a.py"
    assert top_block.representation == "syntax_block"
    assert top_block.line_start == 6 and top_block.line_end == 8
    assert top_block.byte_len() < top_file.byte_len()


def test_line_coverage_shrinks_with_finer_granularity():
    def covered(mode_units):
        spans = rank_and_select(FILES, mode_units(FILES), QUERY, BUDGET)
        return sum(len(v) for v in spans_to_lines(spans).values())

    # Finer granularity covers fewer (more targeted) lines for the same query,
    # i.e. higher precision potential: file >= window >= block.
    assert covered(file_units) >= covered(window_units) >= covered(block_units)
    assert covered(block_units) < covered(file_units)


def test_selection_is_deterministic_across_input_order():
    units = block_units(FILES)
    a = rank_and_select(FILES, units, QUERY, BUDGET)
    b = rank_and_select(FILES, list(reversed(units)), QUERY, BUDGET)

    def key(spans):
        return [(s.source_path, s.byte_start, s.byte_end) for s in spans]

    assert key(a) == key(b)  # total-order tie-break => input-order invariant


def test_non_python_and_syntax_error_fall_back_without_crashing():
    files = [("readme.md", b"# Title\n\nsome docs about parse_manifest\n"),
             ("broken.py", b"def x(:\n  pass\n")]
    spans = rank_and_select(files, block_units(files), "parse_manifest docs", BUDGET)
    assert verify_rate(spans, files) == 1.0
