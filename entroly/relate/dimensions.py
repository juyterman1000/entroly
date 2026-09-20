"""Coverage dimension analysis for compositional omission safety.

Fragments cluster into information dimensions by content overlap.
An omission is safe only if the retained set still covers enough
dimensions.  This is the compositional check that single-fragment
witnesses cannot provide: two individually-safe omissions can be
jointly unsafe when they eliminate the last fragment from distinct
dimensions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_STOPWORDS = frozenset(
    "that this with from have been were being their about would could should "
    "which there these those than some more very just also only other into "
    "does when what where".split()
)


def _content_words(text: str) -> frozenset[str]:
    return frozenset(
        w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", text)
    ) - _STOPWORDS


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class DimensionCoverage:
    total_dimensions: int
    retained_dimensions: int
    lost_dimensions: int
    coverage_ratio: float
    sufficient: bool
    reason: str


def extract_dimensions(
    texts: list[str], *, threshold: float = 0.15,
) -> list[frozenset[int]]:
    """Cluster fragment indices by content-word overlap."""
    n = len(texts)
    if n == 0:
        return []
    words = [_content_words(t) for t in texts]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard(words[i], words[j]) >= threshold:
                union(i, j)

    clusters: dict[int, set[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), set()).add(i)
    return [frozenset(v) for v in clusters.values()]


def check_dimension_coverage(
    retained_indices: set[int],
    dimensions: list[frozenset[int]],
) -> DimensionCoverage:
    """Check whether retained fragments cover enough dimensions."""
    total = len(dimensions)
    if total == 0:
        return DimensionCoverage(0, 0, 0, 1.0, True, "no_dimensions")
    retained_dims = sum(
        1 for d in dimensions if d & retained_indices
    )
    lost = total - retained_dims
    ratio = retained_dims / total if total else 1.0
    min_required = max(2, (total + 1) // 2)
    sufficient = retained_dims >= min(min_required, total)
    reason = f"dimensions:{retained_dims}/{total}"
    if not sufficient:
        reason += f",below_minimum:{min_required}"
    return DimensionCoverage(total, retained_dims, lost, ratio, sufficient, reason)
