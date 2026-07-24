"""Unit coverage for the ContextBench determinism-tax metric core.

Validates the interval-overlap recall/precision/F1 measurement on known synthetic
spans, so the numbers a real run produces are trustworthy. No dataset or network.
"""

from __future__ import annotations

import math

from benchmarks.contextbench_determinism_tax import (
    decide,
    determinism_tax,
    evidence_drop,
    file_score,
    line_score,
    parse_gold,
)


def test_parse_gold_expands_inclusive_line_ranges():
    gold = parse_gold(
        '[{"file": "a.py", "start_line": 10, "end_line": 12, "content": "x"},'
        ' {"file": "b.py", "start_line": 5, "end_line": 5, "content": "y"}]'
    )
    assert gold == {"a.py": {10, 11, 12}, "b.py": {5}}


def test_parse_gold_is_fail_closed_on_malformed():
    assert parse_gold("not json") == {}
    assert parse_gold('[{"file": "a.py", "start_line": 9, "end_line": 3}]') == {}  # end<start
    assert parse_gold('[{"file": "", "start_line": 1, "end_line": 2}]') == {}  # no path


def test_line_score_perfect_overlap():
    gold = {"a.py": {1, 2, 3, 4}}
    s = line_score(pred=gold, gold=gold)
    assert s.recall == 1.0 and s.precision == 1.0 and s.f1 == 1.0


def test_line_score_partial_overlap():
    gold = {"a.py": {1, 2, 3, 4}}          # 4 gold lines
    pred = {"a.py": {3, 4, 5, 6}}          # 4 pred lines, 2 overlap
    s = line_score(pred, gold)
    assert s.overlap == 2
    assert s.recall == 0.5           # 2/4
    assert s.precision == 0.5        # 2/4
    assert math.isclose(s.f1, 0.5)


def test_line_score_zero_when_disjoint_or_empty():
    assert line_score({"a.py": {1, 2}}, {"a.py": {5, 6}}).f1 == 0.0
    assert line_score({}, {"a.py": {1}}).recall == 0.0
    assert line_score({"a.py": {1}}, {}).precision == 0.0


def test_file_score_counts_files_not_lines():
    gold = {"a.py": {1, 2, 3}, "b.py": {1}}
    pred = {"a.py": {99}, "c.py": {1}}      # a.py file-hit despite no line overlap
    s = file_score(pred, gold)
    assert s.overlap == 1                     # {a.py}
    assert s.recall == 0.5                     # 1 of 2 gold files
    assert s.precision == 0.5                  # 1 of 2 pred files


def test_evidence_drop_is_useless_retrieved_fraction():
    gold = {"a.py": {1, 2}}
    pred = {"a.py": {1, 2, 3, 4}}             # 2 of 4 useful -> 0.5 dropped
    assert evidence_drop(pred, gold) == 0.5
    assert evidence_drop({}, gold) == 0.0     # nothing retrieved, nothing dropped


def test_determinism_tax_and_decision_table():
    # F1 in [0,1]; tax reported in percentage points.
    assert math.isclose(determinism_tax(0.80, 0.78), 2.0, abs_tol=1e-9)
    assert decide(2.0) == "Strong general-purpose thesis"
    assert decide(5.0) == "Strong high-trust product"
    assert decide(12.0) == "Compliance / security niche"
    assert decide(20.0).startswith("Reject")
