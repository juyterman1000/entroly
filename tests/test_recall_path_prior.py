"""Locator recall demotes tests/docs/benchmarks so the implementation surfaces.

Dogfooding: for "where the proxy injects context", BM25 term-frequency ranked
tests/_proxy_e2e.py #1 and dropped the implementation (entroly/proxy.py) off
the top-8. A mild path-type prior re-ranks so the impl wins on close calls
without burying a genuinely dominant test/doc.
"""

from __future__ import annotations

from entroly.server import _apply_recall_path_prior, _recall_path_prior


def test_prior_demotes_tests_docs_benchmarks_for_impl_queries() -> None:
    q = "where the proxy injects context"
    assert _recall_path_prior("tests/_proxy_e2e.py", q) == 0.75
    assert _recall_path_prior("entroly/proxy.py", q) == 1.0
    assert _recall_path_prior("docs/proxy.md", q) == 0.85
    assert _recall_path_prior("benchmarks/proxy_scoreboard.py", q) == 0.8


def test_prior_respects_explicit_test_and_doc_queries() -> None:
    assert _recall_path_prior("tests/test_proxy.py", "the proxy test for X") == 1.0
    assert _recall_path_prior("docs/guide.md", "proxy setup guide") == 1.0


def test_rerank_surfaces_impl_over_marginally_higher_test() -> None:
    # Test scores slightly higher on raw BM25, but the impl wins after the mild
    # prior (110 * 0.75 = 82.5 < 100 * 1.0).
    results = [
        {"source": "tests/_proxy_e2e.py", "relevance": 110.0, "fragment_id": "t"},
        {"source": "entroly/proxy.py", "relevance": 100.0, "fragment_id": "i"},
        {"source": "docs/proxy.md", "relevance": 95.0, "fragment_id": "d"},
    ]
    ranked = _apply_recall_path_prior(results, "where the proxy injects context")
    assert ranked[0]["source"] == "entroly/proxy.py"


def test_rerank_keeps_a_dominant_test_when_it_truly_wins() -> None:
    # A test that dominates raw score is not buried by the mild prior
    # (200 * 0.75 = 150 > 100).
    results = [
        {"source": "tests/test_proxy.py", "relevance": 200.0, "fragment_id": "t"},
        {"source": "entroly/proxy.py", "relevance": 100.0, "fragment_id": "i"},
    ]
    ranked = _apply_recall_path_prior(results, "proxy injection")
    assert ranked[0]["source"] == "tests/test_proxy.py"
