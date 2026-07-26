"""A specific query must be searched as asked, not as rewritten.

`optimize_context` expanded every query with terms mined from an initial
`recall(query, 20)` pass -- pseudo-relevance feedback -- and then ranked against
the expansion. When the first pass is mediocre the rewrite inherits its mistakes
and the second pass amplifies them. Classic query drift, applied unconditionally.

Measured on this repository:

    asked:    "what does entroly doctor check and how does it report failures"
    searched: "... [context: tests, verifier, plugin, entry, point]"

None of the injected terms appear in the question. They pulled
test_verifier_plugins.py and plugins.py to the top, while entroly/cli.py -- which
contains cmd_doctor and is the only file that answers it -- fell from rank 4 to
rank 49, far below the ~12 fragments an 8k budget emits. Evidence retention on
the frozen eight-query set was stuck at 0.750 for this reason alone; gating the
rewrite on the analyzer's own `needs_refinement` signal took it to 1.000 at 4k,
8k and 16k.

The analyzer already computed that signal. It gated whether the refinement was
*reported*, not whether it was *used*.
"""

from __future__ import annotations

import pytest


class _Refiner:
    """Stands in for QueryRefiner with a controllable verdict."""

    def __init__(self, needs: bool) -> None:
        self.needs = needs
        self.refined = "REWRITTEN [context: tests, verifier, plugin]"

    def analyze(self, query, summaries):
        return {
            "vagueness_score": 0.9 if self.needs else 0.1,
            "key_terms": [],
            "needs_refinement": self.needs,
            "reason": "stub",
        }

    def refine(self, query, summaries):
        return self.refined


def _searched_query(monkeypatch, tmp_path, needs_refinement: bool) -> str:
    """Return the query the retrieval layer actually searched with."""
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    from entroly.server import EntrolyConfig, EntrolyEngine

    engine = EntrolyEngine(config=EntrolyConfig())
    engine._refiner = _Refiner(needs_refinement)
    engine.ingest_fragment(
        content="def cmd_doctor(args):\n    checks_failed = 0\n    return checks_failed\n",
        source="file:entroly/cli.py",
        token_count=40,
        is_pinned=False,
    )
    engine.ingest_fragment(
        content="def test_verifier_plugin_entry_point():\n    assert True\n",
        source="file:tests/test_verifier_plugins.py",
        token_count=30,
        is_pinned=False,
    )

    seen: list[str] = []
    import entroly.qccr as qccr

    original = qccr._rust_select

    def spy(slim, budget, query, overrides, preferred):
        seen.append(query)
        return original(slim, budget, query, overrides, preferred)

    monkeypatch.setattr(qccr, "_rust_select", spy)
    engine.optimize_context(2000, "what does entroly doctor check and how does it report failures")
    if not seen:
        pytest.skip("selection did not route through qccr in this configuration")
    return seen[0]


def test_specific_query_is_searched_as_asked(monkeypatch, tmp_path):
    """The guard: a query the analyzer deems specific must not be rewritten.

    Reverting to an unconditional `refined_query = self._refiner.refine(...)`
    fails this.
    """
    searched = _searched_query(monkeypatch, tmp_path, needs_refinement=False)
    assert "REWRITTEN" not in searched, (
        f"a specific query was silently rewritten before retrieval: {searched!r}"
    )
    assert "[context:" not in searched, (
        "pseudo-relevance feedback terms leaked into a query that did not need them"
    )
    assert "doctor" in searched


def test_vague_query_still_benefits_from_refinement(monkeypatch, tmp_path):
    """The fix must not disable refinement -- only stop applying it blindly."""
    searched = _searched_query(monkeypatch, tmp_path, needs_refinement=True)
    assert "REWRITTEN" in searched, (
        "refinement must still apply when the analyzer says the query needs it"
    )
