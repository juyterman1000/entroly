"""Compression — cross-runtime conformance gate (native-engine only).

Companion to test_compression_contract.py. That file proves the Python
fallback honours the universal contract vs brute-force OPT. This one
asserts the *two runtimes* (Python density-greedy+KMN vs the Rust 0/1-DP
hot path) stay within the contract of each other, so a native-engine
user and a pure-Python user never get materially different selections.

Faithful-precision boundary (deliberate): both selections are judged by
ONE canonical objective (`_py_compute_relevance`). We assert

  * C1  feasibility — each runtime's selection ≤ budget
  * C2  symmetric (1−1/e) band — neither runtime scores below
        (1−1/e)·(the other) on the canonical objective
  * C4  determinism — each runtime is stable across repeats

We intentionally do NOT assert strict Rust ≥ Python dominance: the Rust
convenience optimizer uses its own internal weights, so judged by an
external canonical objective it need not dominate. Asserting it would
import an objective-identity assumption that is not proven — exactly the
"confident wrong gate" failure mode. Strict dominance under a unified
objective is the documented deeper follow-up.
"""

from __future__ import annotations

import pytest

from entroly.server import _py_compute_relevance, _py_knapsack_optimize

ec = pytest.importorskip("entroly_core", reason="Rust path not installed")

W = (0.25, 0.25, 0.25, 0.25)
INV_E_COMPLEMENT = 1.0 - 1.0 / 2.718281828459045


class Frag:
    __slots__ = ("recency_score", "frequency_score", "semantic_score",
                 "entropy_score", "token_count", "is_pinned", "fragment_id")

    def __init__(self, fid, rec, freq, sem, ent, tok, pinned=False):
        self.fragment_id = fid
        self.recency_score, self.frequency_score = rec, freq
        self.semantic_score, self.entropy_score = sem, ent
        self.token_count, self.is_pinned = tok, pinned


def _to_rust(f: Frag):
    cf = ec.ContextFragment(f.fragment_id, f.fragment_id)
    cf.recency_score = f.recency_score
    cf.frequency_score = f.frequency_score
    cf.semantic_score = f.semantic_score
    cf.entropy_score = f.entropy_score
    cf.token_count = f.token_count
    cf.is_pinned = f.is_pinned
    return cf


def _canon_value(frags) -> float:
    return sum(_py_compute_relevance(f, *W) for f in frags)


INSTANCES = [
    (
        "adversarial_tiny_blocks_big",
        [Frag("tiny", 0.20, 0.20, 0.20, 0.20, 1),
         Frag("big", 0.85, 0.85, 0.85, 0.85, 10)],
        10,
    ),
    (
        "mixed",
        [Frag(f"f{i}", 0.3 + 0.05 * i, 0.4, 0.5, 0.6 - 0.04 * i, i + 1)
         for i in range(9)],
        18,
    ),
]


@pytest.mark.parametrize("name,frags,budget", INSTANCES,
                         ids=[c[0] for c in INSTANCES])
def test_cross_runtime_contract(name, frags, budget):
    py_sel, _ = _py_knapsack_optimize(list(frags), budget, *W)
    rust_sel, rust_stats = ec.py_knapsack_optimize(
        [_to_rust(f) for f in frags], budget
    )

    py_tok = sum(f.token_count for f in py_sel)
    rust_tok = float(rust_stats["total_tokens"])
    py_v = _canon_value(py_sel)
    rust_v = _canon_value(rust_sel)

    # C1 feasibility
    assert py_tok <= budget, f"[{name}] python infeasible: {py_tok}>{budget}"
    assert rust_tok <= budget + 1e-6, (
        f"[{name}] rust infeasible: {rust_tok}>{budget}"
    )

    # C2 symmetric (1−1/e) band — neither catastrophically worse than the
    # other on the one canonical objective.
    hi = max(py_v, rust_v)
    if hi > 1e-9:
        assert py_v + 1e-9 >= INV_E_COMPLEMENT * hi, (
            f"[{name}] python {py_v:.4f} < (1-1/e)*max={INV_E_COMPLEMENT*hi:.4f}"
        )
        assert rust_v + 1e-9 >= INV_E_COMPLEMENT * hi, (
            f"[{name}] rust {rust_v:.4f} < (1-1/e)*max={INV_E_COMPLEMENT*hi:.4f}"
            f" — the Rust hot path diverges below contract; native-engine "
            f"users would get a materially worse selection."
        )


def test_determinism_each_runtime():
    frags = [Frag(f"f{i}", 0.5, 0.4, 0.6, 0.3, i + 1) for i in range(8)]
    a, _ = _py_knapsack_optimize(list(frags), 15, *W)
    b, _ = _py_knapsack_optimize(list(frags), 15, *W)
    assert [f.fragment_id for f in a] == [f.fragment_id for f in b]

    r1, _ = ec.py_knapsack_optimize([_to_rust(f) for f in frags], 15)
    r2, _ = ec.py_knapsack_optimize([_to_rust(f) for f in frags], 15)
    assert [f.fragment_id for f in r1] == [f.fragment_id for f in r2]
