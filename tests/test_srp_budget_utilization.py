"""Regression tests for SRP relevance + budget utilization.

Two bugs made smart_read under-resolve:
  1. Jaccard term-overlap collapsed to ~0.01 for every code block (dominated by
     block size), so the query signal was drowned out and nothing was promoted.
  2. There was a demotion pass (over budget) but no promotion pass (under
     budget), so an under-budget read left every block at LOW — e.g. 84 of 700
     tokens used.
"""

from __future__ import annotations

from entroly.semantic_resolution import Resolution, _term_overlap, resolve

_SRC = '''\
def resolve_source_root():
    """Resolve the codebase source root from ENTROLY_SOURCE or cwd."""
    import os
    return os.environ.get("ENTROLY_SOURCE", os.getcwd())


def unrelated_helper():
    """A totally unrelated utility with no query terms."""
    x = 1
    y = 2
    return x + y


def another_unrelated():
    """More filler that should not outrank the matching block."""
    return [i * i for i in range(10)]
'''


def test_coverage_not_diluted_by_block_size():
    # A block containing every query term scores high regardless of its size —
    # Jaccard would have dragged this toward zero.
    query = "resolve source root"
    big_block = "resolve source root " + ("filler token " * 200)
    assert _term_overlap(query, big_block) == 1.0
    # A block with none of the query terms scores zero.
    assert _term_overlap(query, "completely different content here") == 0.0


def test_query_relevant_block_is_rendered_full():
    r = resolve(
        _SRC,
        query="resolve source root from ENTROLY_SOURCE environment variable",
        budget=400,
        file_path="mod.py",
    )
    by_name = {b.block.name: b for b in r.blocks}
    assert by_name["resolve_source_root"].resolution == Resolution.FULL, (
        f"the matching block must be FULL, got {by_name['resolve_source_root'].resolution}"
    )
    # It should be the highest-relevance block.
    assert by_name["resolve_source_root"].relevance == max(b.relevance for b in r.blocks)


def test_spare_budget_is_used_not_left_on_the_table():
    # Ample budget: promote toward FULL instead of stopping at all-LOW (the
    # original bug left every block at LOW even with room to spare). With a
    # budget far exceeding the file, nothing should remain demoted.
    r = resolve(_SRC, query="resolve source root", budget=600, file_path="mod.py")
    assert r.resolution_counts.get("low", 0) == 0, (
        f"blocks stuck at LOW with ample budget: {r.resolution_counts}"
    )
    assert r.resolution_counts.get("full", 0) >= 1, (
        f"expected at least one FULL block with ample budget: {r.resolution_counts}"
    )


def test_tight_budget_still_demotes_within_limit():
    # The demotion path must still hold the line under a tight budget.
    r = resolve(_SRC, query="resolve source root", budget=40, file_path="mod.py")
    assert r.total_tokens <= r.budget
