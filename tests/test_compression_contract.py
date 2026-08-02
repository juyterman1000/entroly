"""Compression selection — universal correctness contract.

The generalized parity gate. WITNESS parity was *decision* equivalence
(classifier). Compression is an *optimizer*: the Python fallback
(`_py_knapsack_optimize`, density-greedy) and the Rust hot path (0/1 DP
knapsack) can legitimately pick different fragment sets — multiple
optima exist — so output identity is the wrong criterion.

The right criterion is the **contract the math guarantees, at the
precision the math allows**:

  C1  Feasibility       selected token count ≤ budget
  C2  Regression floor  V(greedy) ≥ 0.632·V(OPT) on these fixtures
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
from dataclasses import dataclass

import pytest

from entroly.server import _py_compute_relevance, _py_knapsack_optimize

W = (0.25, 0.25, 0.25, 0.25)  # equal weights; objective is generic
# The bar C2 holds the selector to on the adversarial fixtures below. It is
# an EMPIRICAL regression floor, not a theorem: the objective here (_val) is a
# plain sum of independently-computed per-fragment scores, i.e. MODULAR, and
# better-of-{density-greedy, best singleton} on a modular knapsack is provably
# ½ (Dantzig/LP rounding). 0.632 is what the shipped selector actually achieves
# on these instances, so it is a stricter bar than the proof supplies -- which
# makes it a useful regression guard and an invalid citation.
# (1 - 1/e) would require a monotone submodular objective (Nemhauser-Wolsey-
# Fisher 1978, tight per Feige 1998) plus, under a knapsack, Sviridenko 2004
# partial enumeration. KMN 1999's better-of-two gives ½(1 - 1/e) for submodular
# coverage. None of those describe this code.
PROVABLE_FLOOR = 0.5
EMPIRICAL_OPT_FLOOR = 0.632


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


@dataclass
class OversizedFrag:
    fragment_id: str
    content: str
    token_count: int
    source: str = "oversized.py"
    recency_score: float = 0.8
    frequency_score: float = 0.5
    semantic_score: float = 0.8
    entropy_score: float = 0.6
    is_pinned: bool = False


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


# ── C1 + C2: feasibility and the value floor vs brute-force OPT ───────


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
    assert v + 1e-9 >= PROVABLE_FLOOR * opt, (
        f"[{name}] C2 VIOLATED below the PROVABLE floor: greedy V={v:.4f} "
        f"< 0.5*OPT={PROVABLE_FLOOR * opt:.4f} (OPT={opt:.4f}). Better-of-"
        f"{{density-greedy, best singleton}} on a modular knapsack cannot do "
        f"this badly -- the singleton champion is probably not being applied."
    )
    assert v + 1e-9 >= EMPIRICAL_OPT_FLOOR * opt, (
        f"[{name}] C2 regression: greedy V={v:.4f} < 0.632*OPT="
        f"{EMPIRICAL_OPT_FLOOR * opt:.4f} (OPT={opt:.4f}). This is the bar the "
        f"selector has been holding on these fixtures, not a theoretical floor "
        f"(see the constant's comment); a drop here means selection quality "
        f"regressed even though the ½ proof is intact."
    )


def test_oversized_python_fallback_returns_budget_excerpt():
    frag = OversizedFrag("huge", "x = 1\n" * 5000, 5000)

    sel, stats = _selected([frag], 128)

    assert len(sel) == 1
    assert sel[0].fragment_id == "huge:excerpt"
    assert sel[0].token_count == 128
    assert sum(f.token_count for f in sel) <= 128
    assert stats["method"] == "greedy_python+oversize_excerpt"


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
    assert worst_ratio + 1e-9 >= PROVABLE_FLOOR, (
        f"C2 VIOLATED below the PROVABLE floor over random instances: worst "
        f"V/OPT={worst_ratio:.4f} < 0.5. Better-of-{{density-greedy, best "
        f"singleton}} on a modular knapsack cannot do this badly."
    )
    assert worst_ratio + 1e-9 >= EMPIRICAL_OPT_FLOOR, (
        f"C2 regression over random instances: worst V/OPT={worst_ratio:.4f}"
        f" < {EMPIRICAL_OPT_FLOOR} (the bar this selector has been holding on "
        f"random instances, not a theoretical floor)"
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
