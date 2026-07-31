"""Shared evaluation metrics — single source of truth.

Every AUROC in this repo must go through `tie_corrected_auroc`. The repo
historically carried two hand-rolled Mann-Whitney variants without midrank
tie handling:

- ``sorted(zip(scores, labels))`` breaks score-ties by *label* (tuple sort),
  deterministically ranking positive items above negative items at every tie;
- ``sorted(..., key=lambda x: x[0])`` leaves tied items in insertion order,
  making the result depend on how the caller interleaved the classes.

With heavily tied score distributions (common for ESG/EICV scores that
saturate at exactly 0.0 or 1.0) the first variant fabricates AUROC near 1.0
from pure noise: two *semantically identical* inputs separated at AUROC 0.94
in benchmarks/results/diagnose_ablation_artifact.json. Midranks fix both;
identical distributions score exactly 0.5.

tests/test_auroc_tie_correction.py bans the biased idioms from reappearing.
"""

from __future__ import annotations

from collections.abc import Sequence


def tie_corrected_auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney AUROC with midrank tie correction.

    Args:
        scores: higher = more likely positive (label 1).
        labels: 0/1 class labels, parallel to ``scores``.

    Returns:
        AUROC in [0, 1]; 0.5 when either class is empty or the score
        distributions are identical.
    """
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        midrank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = midrank
        i = j + 1
    n1 = sum(labels)
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        return 0.5
    rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n0 * n1)
