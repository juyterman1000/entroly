"""Compression selection — universal correctness contract.

The generalized parity gate. WITNESS parity was *decision* equivalence
(classifier). Compression is an *optimizer*: the Python fallback
(`_py_knapsack_optimize`, density-greedy) and the Rust hot path (0/1 DP
knapsack) can legitimately pick different fragment sets — multiple
optima exist — so output identity is the wrong criterion.

The right criterion is the **contract the math guarantees, at the
precision the math allows**:

  C1  Feasibility       selected token count ≤ budget
  C2  Guarantee floor   V(greedy) ≥ (1 − 1/e)·V(OPT)         [KMN 1999]
  C4  Determinism       same input ⇒ same output
  C5  Budget monotone   larger budget ⇒ objective non-decreasing

C2 is checked against **brute-force true OPT** (exact, gold standard) on
small instances — no second runtime needed, so this runs in every CI.
Cross-runtime conformance (Rust vs Python) is the separate, native-gated
layer in test_compression_conformance.py.

This file deliberately includes an *adversarial* instance designed to
expose density-greedy's classic failure (a low-value tiny fragment whose
density outranks a high-value budget-filling fragment, which it then
blocks). If the shipped fallback violates C2 there, this test fails by
design until the Khuller–Moss–Naor singleton-champion fix is applied.
"""

from __future__ import annotations

import itertools
import random

import pytest

from entroly.server import _py_compute_relevance, _py_knapsack_optimize

W = (0.25, 0.25, 0.25, 0.25)  # equal weights; objective is generic
INV_E_COMPLEMENT = 1.0 - 1.0 / 2.718281828459045  # (1 − 1/e) ≈ 0.6321


class Frag:
    """Duck-typed ContextFragment: only the attrs the optimizer reads."""
    __slots__ = ("recency_score", "frequency_score", "semantic_score",
                 "entropy_score", "token_count", "is_pinned", "fragment_id")

    def __init__(self, fid, rec, freq, sem, ent, tok, pinned=False):
        self.fragment_id = fid
        self.recency_score = rec
        self.frequency_score = freq
        self.semantic_score = sem
        self.entropy_score = ent
        self.token_count = tok
        self.is_pinned = pinned


def _val(frags) -> float:
    return sum(_py_compute_relevance(f, *W) for f in frags)


def _true_opt(frags, budget) -> float:
    """Exact optimum by subset enumeration (n ≤ 16). Pinned forced in."""
    pinned = [f for f in frags if f.is_pinned]
    cand = [f for f in frags if not f.is_pinned]
    base_tok = sum(f.token_count for f in pinned)
    best = -1.0
    for r in range(len(cand) + 1):
        for combo in itertools.combinations(cand, r):
            if base_tok + sum(f.token_count for f in combo) <= budget:
                v = _val(pinned) + _val(list(combo))
                if v > best:
                    best = v
    return max(best, _val(pinned))


def _selected(frags, budget):
    sel, stats = _py_knapsack_optimize(list(frags), budget, *W)
    return sel, stats


# ── C1 + C2: feasibility and the (1−1/e) guarantee vs true OPT ────────


ADVERSARIAL = [
    # (name, fragments, budget)
    # Classic density-greedy trap, *within* the [0,0.85] value softcap:
    # a low-value tiny fragment has higher density than a high-value
    # fragment that needs (almost) the whole budget. Pure density-greedy
    # grabs the tiny one and can no longer fit the valuable one.
    (
        "tiny_blocks_big",
        [Frag("tiny", 0.20, 0.20, 0.20, 0.20, 1),
         Frag("big", 0.85, 0.85, 0.85, 0.85, 10)],
        10,
    ),
    (
        "two_tinies_block_big",
        [Frag("t1", 0.15, 0.15, 0.15, 0.15, 1),
         Frag("t2", 0.15, 0.15, 0.15, 0.15, 1),
         Frag("big", 0.80, 0.80, 0.80, 0.80, 12)],
        12,
    ),
]


@pytest.mark.parametrize("name,frags,budget", ADVERSARIAL,
                         ids=[c[0] for c in ADVERSARIAL])
def test_adversarial_guarantee(name, frags, budget):
    sel, stats = _selected(frags, budget)
    tok = sum(f.token_count for f in sel)
    v = _val(sel)
    opt = _true_opt(frags, budget)
    assert tok <= budget, f"[{name}] C1: {tok} > budget {budget}"
    assert v + 1e-9 >= INV_E_COMPLEMENT * opt, (
        f"[{name}] C2 VIOLATED: greedy V={v:.4f} < (1-1/e)*OPT="
        f"{INV_E_COMPLEMENT * opt:.4f} (OPT={opt:.4f}). The shipped "
        f"Python fallback does not honour the (1-1/e) guarantee "
        f"entroly advertises. Fix: Khuller–Moss–Naor singleton champion."
    )


def test_randomized_contract():
    """C1/C2 over many random small instances vs exact OPT."""
    rng = random.Random(20260516)
    worst_ratio = 1.0
    for _ in range(400):
        n = rng.randint(2, 12)
        frags = [
            Frag(f"f{i}",
                 rng.random(), rng.random(), rng.random(), rng.random(),
                 rng.randint(1, 20),
                 pinned=(rng.random() < 0.1))
            for i in range(n)
        ]
        budget = rng.randint(5, 60)
        pinned_tok = sum(f.token_count for f in frags if f.is_pinned)
        if pinned_tok > budget:
            continue  # pinned-overflow is a separate documented invariant
        sel, _ = _selected(frags, budget)
        tok = sum(f.token_count for f in sel)
        assert tok <= budget, f"C1: {tok} > {budget}"
        opt = _true_opt(frags, budget)
        if opt > 1e-9:
            worst_ratio = min(worst_ratio, _val(sel) / opt)
    assert worst_ratio + 1e-9 >= INV_E_COMPLEMENT, (
        f"C2 VIOLATED over random instances: worst V/OPT={worst_ratio:.4f}"
        f" < (1-1/e)={INV_E_COMPLEMENT:.4f}"
    )


# ── C4: determinism ──────────────────────────────────────────────────


def test_determinism():
    frags = [Frag(f"f{i}", 0.5, 0.4, 0.6, 0.3, i + 1) for i in range(8)]
    a, _ = _selected(frags, 15)
    b, _ = _selected(frags, 15)
    assert [f.fragment_id for f in a] == [f.fragment_id for f in b]


# ── C5: budget monotonicity ──────────────────────────────────────────


def test_budget_monotonicity():
    rng = random.Random(7)
    frags = [
        Frag(f"f{i}", rng.random(), rng.random(), rng.random(),
             rng.random(), rng.randint(1, 8))
        for i in range(12)
    ]
    prev = -1.0
    for budget in range(5, 60, 5):
        sel, _ = _selected(frags, budget)
        v = _val(sel)
        assert v + 1e-9 >= prev, (
            f"C5: V dropped from {prev:.4f} to {v:.4f} as budget grew"
        )
        prev = v
