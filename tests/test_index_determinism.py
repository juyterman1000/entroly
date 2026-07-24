"""Regression coverage for deterministic index construction and selection.

Ingest used to assemble its batch in thread-completion order, so a read race
decided which of two SimHash near-duplicate files won — the corpus was not
reproducible. These tests pin the deterministic contract: equal-priority files
ingest in a total order (priority DESC, canonical path ASC), so the dedup winner
is stable and canonical, and query selection is invariant to input order.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from entroly.auto_index import auto_index
from entroly.config import EntrolyConfig
from entroly.server import EntrolyEngine

# A substantial identical body so SimHash reliably judges the two files as
# duplicates (small stubs can fall below the dedup signal threshold).
_IDENTICAL_BODY = (
    '"""A module with enough content for SimHash to lock onto."""\n\n'
    + "\n".join(
        f"def handler_{i}(request):\n"
        f"    value = compute_step_{i}(request.payload, index={i})\n"
        f"    return normalize(value, scale={i * 7 + 3})\n"
        for i in range(40)
    )
    + "\n"
)


def _native_engine(tmp_path: Path) -> EntrolyEngine:
    engine = EntrolyEngine(
        EntrolyConfig(use_persistent_index=False, checkpoint_dir=tmp_path / "cp")
    )
    if not engine._use_rust:
        pytest.skip("deterministic ingest requires the native engine")
    return engine


def _dedup_winner(tmp_path: Path) -> str | None:
    # Two byte-identical files; the alphabetically-first canonical path
    # ("a_original.py" < "b_twin.py") must win under the total-order ingest.
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "a_original.py").write_text(_IDENTICAL_BODY, encoding="utf-8")
    (tmp_path / "b_twin.py").write_text(_IDENTICAL_BODY, encoding="utf-8")
    engine = _native_engine(tmp_path)
    auto_index(engine, project_dir=str(tmp_path), force=True)
    kept = {f.get("source") for f in engine._rust.export_fragments()} & {
        "file:a_original.py",
        "file:b_twin.py",
    }
    if len(kept) != 1:
        pytest.skip("installed entroly-core did not dedup identical files")
    return next(iter(kept))


def test_near_duplicate_dedup_winner_is_deterministic_and_canonical(tmp_path: Path):
    # Repeat cold ingests in fresh temp dirs; the winner must be stable across
    # runs AND equal to the total-order-first source, not a thread-race outcome.
    winners = {_dedup_winner(tmp_path / f"run{i}") for i in range(3)}
    assert winners == {"file:a_original.py"}, (
        f"dedup winner is nondeterministic or non-canonical: {winners}"
    )


def test_selection_is_invariant_to_fragment_input_order(tmp_path: Path):
    # Query-conditioned selection must not depend on the order fragments are
    # presented (e.g. filesystem enumeration order). Same corpus, reversed input
    # order -> byte-identical ordered selection.
    from entroly import qccr

    fragments = [
        {"source": f"file:mod_{i}.py", "content": body, "fragment_id": f"f{i}"}
        for i, body in enumerate(
            [
                "def inject_context(request):\n    return proxy.compress(request)\n",
                "def rank_fragments(query):\n    return bm25_score(query)\n",
                "def knapsack(budget, items):\n    return solve_dp(budget, items)\n",
                "def dedup(fragments):\n    return simhash_filter(fragments)\n",
                "def verify_belief(claim):\n    return witness_check(claim)\n",
            ]
        )
    ]
    query = "how does the proxy inject compressed context into a request"
    forward = qccr.select(fragments, 400, query)
    reverse = qccr.select(list(reversed(fragments)), 400, query)
    if forward == fragments or reverse == list(reversed(fragments)):
        pytest.skip("qccr returned input unchanged (no native compression path)")

    def order(sel: list[dict]) -> list[tuple[str, str]]:
        return [(str(f.get("source") or ""), f.get("content") or "") for f in sel]

    assert order(forward) == order(reverse), (
        "selection depends on fragment input order (tie-break not total-ordered)"
    )
