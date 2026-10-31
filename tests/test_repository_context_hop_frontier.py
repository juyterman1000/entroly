"""A higher retrieval score must not spend another path's remaining hops."""

from __future__ import annotations

from pathlib import Path
import random

import pytest

from entroly.repository_intelligence import RepositoryIntelligenceService
from entroly.repository_intelligence import verified_context
from entroly.repository_intelligence.models import RepositoryIndex, Symbol
from entroly.repository_intelligence.verified_context import verify_context_commitment


def _project(root: Path) -> None:
    (root / "task.py").write_text(
        "def outer():\n"
        "    def needle():\n"
        "        return 0\n"
        "    return needle()\n",
        encoding="utf-8",
    )
    (root / "driver.py").write_text(
        "from task import outer\n"
        "from leaf import answer\n\n"
        "def drive():\n"
        "    outer()\n"
        "    return answer()\n",
        encoding="utf-8",
    )
    (root / "leaf.py").write_text(
        "from final import terminal\n\n"
        "def answer():\n"
        "    return terminal()\n",
        encoding="utf-8",
    )
    (root / "final.py").write_text(
        "def terminal():\n"
        "    return 'complete_evidence'\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(("hops", "required"), [(2, "answer"), (3, "terminal")])
def test_shorter_lower_score_path_keeps_its_remaining_hops(tmp_path, hops, required):
    _project(tmp_path)
    service = RepositoryIntelligenceService(tmp_path)

    payload = service.context("needle", token_budget=4_096, max_hops=hops)

    # needle -> outer -> drive scores .88 * .74 = .6512, but spends two
    # hops. The imported-by shortcut needle -> drive scores only .55 and
    # must still be expanded with one hop spent. Otherwise answer (two
    # hops away) and terminal (three hops away) are incorrectly missed.
    names = {fragment["qualified_name"] for fragment in payload["fragments"]}
    assert required in names
    assert payload["receipt"]["omissions_by_reason"] == {}
    assert payload["receipt"]["remote_calls"] == 0
    assert payload["retrieval"]["estimated_tokens"] <= 4_096
    assert verify_context_commitment(payload)


def test_shortcut_does_not_bypass_the_hop_limit(tmp_path):
    _project(tmp_path)
    payload = RepositoryIntelligenceService(tmp_path).context(
        "needle", token_budget=4_096, max_hops=1
    )

    names = {fragment["qualified_name"] for fragment in payload["fragments"]}
    assert "drive" in names
    assert not {"answer", "terminal"} & names


_FACTORS = {
    "calls": 0.82,
    "called-by": 0.74,
    "contained-by": 0.88,
    "contains": 0.68,
    "imports": 0.62,
    "imported-by": 0.55,
}


def _layered_reference(graph, seeds, hops):
    # Independent bounded-walk dynamic program: no heap or production
    # traversal/dominance helper. The production ranking floor is one.
    layer = dict(seeds)
    best = dict(seeds)
    for _ in range(hops):
        following = {}
        for source, score in layer.items():
            for target, relation, _ in graph[source]:
                candidate = score * _FACTORS[relation]
                if candidate >= 1:
                    following[target] = max(candidate, following.get(target, 0))
        layer = following
        for target, score in layer.items():
            best[target] = max(score, best.get(target, 0))
    return best


@pytest.mark.parametrize("graph_seed", range(12))
@pytest.mark.parametrize("hops", range(7))
def test_layered_ranking_matches_bounded_walk_reference(monkeypatch, graph_seed, hops):
    rng = random.Random(graph_seed)
    nodes = [f"node_{i}" for i in range(6)]
    index = RepositoryIndex(
        root="unused",
        symbols={
            node: Symbol(
                symbol_id=node,
                path=f"{node}.py",
                name=node,
                qualified_name=node,
                kind="function",
                line_start=1,
                line_end=2,
                language="python",
            )
            for node in nodes
        },
    )
    # Mixed relation weights, cycles, ties, self-loops, and one/two sources.
    graph = {
        source: [
            (target, rng.choice(tuple(_FACTORS)), None)
            for target in nodes
            if rng.random() < 0.4
        ]
        for source in nodes
    }
    seeds = {nodes[0]: 20.0}
    if graph_seed % 2:
        seeds[nodes[1]] = 13.0
    monkeypatch.setattr(verified_context, "_adjacency", lambda _: (graph, {}))
    monkeypatch.setattr(
        verified_context,
        "_symbol_score",
        lambda symbol, _query, _tokens: seeds.get(symbol.symbol_id, 0.0),
    )

    ranked, seed_ids = verified_context._rank_candidates(
        index, "source", max_hops=hops, max_candidates=100
    )

    assert seed_ids == set(seeds)
    expected = _layered_reference(graph, seeds, hops)
    actual = {symbol_id: score for score, symbol_id, _ in ranked}
    assert actual == pytest.approx(expected)
    assert len(actual) == len(ranked), "return each symbol only once"
    for _, _, reasons in ranked:
        assert len(reasons) - 1 <= hops


def test_candidate_cap_still_bounds_unique_symbols(monkeypatch):
    _project_index = RepositoryIndex(
        root="unused",
        symbols={
            str(i): Symbol(str(i), "unused.py", str(i), str(i), "function", 1, 2, "python")
            for i in range(8)
        },
    )
    graph = {
        node: [(other, "calls", None) for other in _project_index.symbols]
        for node in _project_index.symbols
    }
    monkeypatch.setattr(verified_context, "_adjacency", lambda _: (graph, {}))
    monkeypatch.setattr(
        verified_context,
        "_symbol_score",
        lambda symbol, _query, _tokens: 20.0 if symbol.symbol_id == "0" else 0.0,
    )

    first = verified_context._rank_candidates(
        _project_index, "source", max_hops=6, max_candidates=3
    )
    second = verified_context._rank_candidates(
        _project_index, "source", max_hops=6, max_candidates=3
    )

    assert first == second
    assert len(first[0]) == 3
    assert len({item[1] for item in first[0]}) == 3
