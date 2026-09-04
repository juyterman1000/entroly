"""Exact lower credible bound for a Beta posterior.

RAVS routes a request to a cheaper model only when it is confident that model
succeeds at the task. That confidence came from a normal approximation to the
Beta posterior::

    ci_lo = max(0.0, mean - 1.96 * std)

A Beta is not symmetric, and this gate lives exactly where the approximation is
worst: few observations and a mean close to 1. Measured against the exact
quantile, the approximation overstates the bound in *every* configuration
tested, and in three of them the difference is what opens the gate:

    cell             mean-1.96s   exact 2.5%
    n=10, 10/0         0.8367       0.7828
    n=20, 19/1         0.8210       0.7892
    n=35, 32/3         0.8073       0.7886

against a ``ci_threshold`` of 0.80. The worst case is the smallest cell that can
ever qualify -- ``min_samples`` observations with a perfect record -- so the
overstatement is largest precisely where the evidence is thinnest.

CLAUDE.md lists RAVS as fail-closed: "always routes to Opus when uncertain;
never sacrifice correctness for cost". A bound that reports more confidence than
the posterior holds fails open. This is the same defect the Jeffreys-prior
comment in ``router.py`` describes one level up: that fixed the prior, this
fixes the bound computed from it.

Pure Python and dependency-free on purpose -- scipy is not a runtime dependency,
and adding one to a routing path for a single quantile is not warranted.
``entroly.engine._WilsonFeedbackTracker`` already uses a Wilson score bound
rather than a naive one for the same reason; this brings RAVS in line.
"""
from __future__ import annotations

import math
from functools import lru_cache

__all__ = ["beta_cdf", "beta_lower_bound"]


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def beta_cdf(x: float, alpha: float, beta: float) -> float:
    """Regularised incomplete beta ``I_x(alpha, beta)`` = ``P(X <= x)``."""
    if not (alpha > 0.0 and beta > 0.0):
        raise ValueError("alpha and beta must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    # Use the reflected series where the continued fraction converges faster.
    if x < (alpha + 1.0) / (alpha + beta + 2.0):
        front = math.exp(alpha * math.log(x) + beta * math.log1p(-x) - log_beta)
        return front * _betacf(alpha, beta, x) / alpha
    front = math.exp(beta * math.log1p(-x) + alpha * math.log(x) - log_beta)
    return 1.0 - front * _betacf(beta, alpha, 1.0 - x) / beta


@lru_cache(maxsize=4096)
def beta_lower_bound(alpha: float, beta: float, tail: float = 0.025) -> float:
    """The ``tail`` quantile of ``Beta(alpha, beta)``.

    With the default this is the lower end of a 95% equal-tailed credible
    interval -- the interval the previous normal approximation was aiming at,
    computed exactly instead of approximated.

    Bisection on a monotone CDF, stopping at 1e-12. Each step evaluates a
    continued fraction, so this is not free: measured at 836 us uncached on a
    routing-sized posterior, against 0.32 us once cached. The arguments are
    discrete (``alpha = 0.5 + passes``) and a cell is reused by every request
    that lands in it, so the cache turns the amortised cost into a dict lookup.

    Stopping at 1e-12 rather than at float precision: the full bisection
    measured 1,464 us and changed no digit that matters, because the result is
    only ever compared against a threshold.
    """
    if not (alpha > 0.0 and beta > 0.0):
        return 0.0
    if not (0.0 < tail < 1.0):
        raise ValueError("tail must be in (0, 1)")

    lo, hi = 0.0, 1.0
    while hi - lo > 1e-12:
        mid = (lo + hi) * 0.5
        if mid <= lo or mid >= hi:  # exhausted float precision
            break
        if beta_cdf(mid, alpha, beta) < tail:
            lo = mid
        else:
            hi = mid
    return max(0.0, min(1.0, (lo + hi) * 0.5))
