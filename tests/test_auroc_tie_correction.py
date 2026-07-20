"""Regression tests for tie-corrected AUROC.

The repo historically carried hand-rolled Mann-Whitney AUROC helpers without
midrank tie handling. The worst variant (``sorted(zip(scores, labels))``)
broke score-ties by label and fabricated AUROC near 1.0 under heavy ties —
demonstrated in benchmarks/results/diagnose_ablation_artifact.json, where two
semantically identical inputs "separated" at AUROC 0.9431.

These tests pin the canonical implementation (entroly/metrics.py) against a
brute-force pairwise reference and ban the biased idiom from reappearing
anywhere under entroly/ or benchmarks/.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from entroly.metrics import tie_corrected_auroc

REPO = Path(__file__).resolve().parent.parent


def _reference_auroc(scores, labels):
    """Brute-force pairwise definition: P(s_pos > s_neg) + 0.5 P(s_pos == s_neg)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    total = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                total += 1.0
            elif p == q:
                total += 0.5
    return total / (len(pos) * len(neg))


def test_identical_all_tied_distributions_score_half():
    # ~83% ties at 0.0 plus an identical non-tied tail in each class — the
    # heavy-ties regime that broke the v1 harness. Both classes see the exact
    # same score multiset, so the only correct answer is 0.5.
    scores = [0.0] * 100 + [0.3] * 10 + [0.4] * 10
    labels = [0, 1] * 50 + [0, 1] * 5 + [0, 1] * 5
    assert tie_corrected_auroc(scores, labels) == 0.5


def test_perfect_separation():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert tie_corrected_auroc(scores, labels) == 1.0


def test_matches_pairwise_reference_under_heavy_ties():
    rng = random.Random(7)
    for _ in range(20):
        n = 200
        # Values drawn from a tiny support set to force massive cross-class ties.
        scores = [rng.choice([0.0, 0.0, 0.0, 0.5, 1.0]) for _ in range(n)]
        labels = [rng.randint(0, 1) for _ in range(n)]
        got = tie_corrected_auroc(scores, labels)
        want = _reference_auroc(scores, labels)
        assert abs(got - want) < 1e-12


def test_label_order_invariance():
    # The biased variant depended on label/insertion order at ties.
    scores = [0.0, 0.0, 0.0, 0.0, 0.7]
    a = tie_corrected_auroc(scores, [0, 1, 0, 1, 1])
    b = tie_corrected_auroc(list(reversed(scores)), list(reversed([0, 1, 0, 1, 1])))
    assert a == b


def test_production_helpers_delegate_to_canonical():
    from entroly.rnr import _auroc as rnr_auroc
    from entroly.adversarial_calibration import _auroc as adv_auroc

    scores = [0.0] * 50 + [1.0] * 2
    labels = ([0, 1] * 25) + [1, 0]
    want = _reference_auroc(scores, labels)
    assert abs(rnr_auroc(scores, labels) - want) < 1e-12
    assert abs(adv_auroc(scores, labels) - want) < 1e-12


BIASED_IDIOM = re.compile(r"enumerate\(pairs,\s*1\)")


def test_biased_idiom_banned_from_source():
    """No hand-rolled rank sum over label-tie-broken pairs may reappear."""
    offenders = []
    for root in ("entroly", "benchmarks"):
        for path in (REPO / root).rglob("*.py"):
            if BIASED_IDIOM.search(path.read_text(encoding="utf-8", errors="replace")):
                offenders.append(str(path.relative_to(REPO)))
    assert not offenders, (
        "Tie-biased AUROC idiom found — use entroly.metrics.tie_corrected_auroc: "
        f"{offenders}"
    )
