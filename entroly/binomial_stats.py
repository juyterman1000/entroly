"""Binomial bounds for small-sample evidence reports."""

from __future__ import annotations

import math
from functools import lru_cache

from .ravs.beta_bounds import beta_cdf


@lru_cache(maxsize=4096)
def binomial_upper(count: int, total: int, confidence: float) -> float:
    """One-sided Clopper–Pearson upper bound for independent Bernoulli trials."""
    if not 0 <= count <= total or not 0 < confidence < 1:
        raise ValueError("invalid binomial count or confidence")
    if count == total:  # Includes the empty sample.
        return 1.0
    if count == 0:
        return math.nextafter(-math.expm1(math.log1p(-confidence) / total), 1.0)
    lo, hi = 0.0, 1.0
    while hi - lo > 1e-12:
        mid = (lo + hi) * 0.5
        if beta_cdf(mid, count + 1, total - count) < confidence:
            lo = mid
        else:
            hi = mid
    return hi
