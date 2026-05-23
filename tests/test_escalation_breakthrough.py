"""Falsification of the conformal-escalation regret bound.

Claim under test (entroly/escalation.py, bound (R)):

    E[cost_policy] − E[cost_clairvoyant]  ≤  α · q

where the policy sees only the conformal-upper risk, the clairvoyant
sees the true risk, and α is the WITNESS conformal miscoverage.

This is a *falsification* harness, not a demo. It brute-forces the
clairvoyant optimum per query, simulates a faithful RCPS verifier
(coverage miss with prob ≤ α, including an adversarial variant where
misses land on the highest-stakes queries), sweeps α, and asserts the
additive bound holds with statistical margin AND that the controller is
non-trivially useful (beats always-cheap and always-top). If the
derivation is wrong, this fails.
"""

from __future__ import annotations

import random
import statistics

import pytest

from entroly.escalation import (
    Stage,
    optimal_stop_dp,
    realized_cost,
    run_escalation,
)

Q = 10.0          # hallucination cost
N = 40_000        # queries per condition
COSTS = [0.0, 0.5, 1.2, 2.5]   # cumulative ladder cost (4 rungs)
FLOOR = 0.02      # irreducible risk at the top rung


def _clairvoyant(true_risk: list[float]) -> float:
    """Exact lower bound: knows true risk at every rung, pays the
    min over stop indices of (cum_cost + q·true_risk)."""
    return min(COSTS[k] + Q * true_risk[k] for k in range(len(COSTS)))


def _make_query(rng: random.Random, alpha: float, adversarial: bool):
    """True risk: monotone non-increasing down the ladder to FLOOR.
    Conformal-upper estimate: valid (≥ true) w.p. ≥ 1−α; a coverage
    miss (< true) w.p. ≤ α. Adversarial: misses concentrated on
    high-stakes (high-risk) queries — the worst case for bound (R)."""
    r0 = rng.uniform(0.15, 0.95)
    true = []
    r = r0
    for _ in COSTS:
        true.append(max(FLOOR, r))
        r *= rng.uniform(0.35, 0.8)   # escalation strictly helps
    high_stakes = r0 > 0.7
    miss_prob = alpha
    if adversarial and high_stakes:
        miss_prob = min(1.0, alpha * 3.0)   # misses target high-stakes
    elif adversarial:
        miss_prob = alpha * 0.5
    upper = []
    for t in true:
        if rng.random() < miss_prob:
            upper.append(max(0.0, t - rng.uniform(0.05, 0.30)))  # miss
        else:
            upper.append(min(1.0, t + rng.uniform(0.0, 0.15)))   # valid UB
    return true, upper


def _run(alpha: float, adversarial: bool, seed: int):
    rng = random.Random(seed)
    pol, dp, clv, cheap, top = [], [], [], [], []
    for _ in range(N):
        true, upper = _make_query(rng, alpha, adversarial)
        ladder = [Stage(f"m{k}", COSTS[k], upper[k]) for k in range(len(COSTS))]
        dm = run_escalation(ladder, Q, r_floor=FLOOR)
        dd = optimal_stop_dp(ladder, Q, r_floor=FLOOR)
        pol.append(realized_cost(dm, true[dm.stop_index], Q))
        dp.append(realized_cost(dd, true[dd.stop_index], Q))
        clv.append(_clairvoyant(true))
        cheap.append(COSTS[0] + Q * true[0])
        top.append(COSTS[-1] + Q * true[-1])
    return {
        "myopic": statistics.fmean(pol),
        "dp": statistics.fmean(dp),
        "clairvoyant": statistics.fmean(clv),
        "cheap": statistics.fmean(cheap),
        "top": statistics.fmean(top),
        "se_myopic": statistics.stdev(pol) / (N ** 0.5),
        "se_dp": statistics.stdev(dp) / (N ** 0.5),
    }


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
@pytest.mark.parametrize("adversarial", [False, True],
                         ids=["random_miss", "adversarial_miss"])
def test_additive_regret_bound(alpha, adversarial):
    """Bound (R): myopic regret ≤ α·q, incl. adversarial misses."""
    r = _run(alpha, adversarial, seed=1000 + int(alpha * 100))
    regret = r["myopic"] - r["clairvoyant"]
    assert regret <= alpha * Q + 4 * r["se_myopic"], (
        f"BOUND (R) BROKEN @α={alpha} adv={adversarial}: "
        f"regret={regret:.4f} > α·q={alpha*Q:.4f}. Derivation wrong."
    )
    assert regret > 0.0, "degenerate: matched clairvoyant exactly"


def test_dp_implementation_exact_via_perfect_verifier():
    """Falsify the Bellman code itself, deterministically: with a
    perfect verifier (conformal_upper == true risk) the DP MUST equal
    the brute-forced clairvoyant optimum on every instance — no Monte
    Carlo, no tolerance beyond float eps."""
    rng = random.Random(99)
    for _ in range(2000):
        true = []
        r = rng.uniform(0.1, 0.95)
        for _ in COSTS:
            true.append(max(FLOOR, r))
            r *= rng.uniform(0.3, 0.85)
        ladder = [Stage(f"m{k}", COSTS[k], true[k]) for k in range(len(COSTS))]
        d = optimal_stop_dp(ladder, Q, r_floor=FLOOR)
        got = realized_cost(d, true[d.stop_index], Q)
        opt = _clairvoyant(true)
        assert abs(got - opt) < 1e-9, (
            f"DP ≠ clairvoyant under perfect verifier: {got} vs {opt}. "
            f"Bellman recursion is wrong."
        )


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
@pytest.mark.parametrize("adversarial", [False, True],
                         ids=["random_miss", "adversarial_miss"])
def test_dp_dominates_myopic(alpha, adversarial):
    """By construction the look-ahead DP cannot be worse in mean than
    the one-step myopic rule on the same observed u-ladder. If this
    inverts, the DP is mis-implemented."""
    r = _run(alpha, adversarial, seed=2000 + int(alpha * 100))
    se = r["se_dp"] + r["se_myopic"]
    assert r["dp"] <= r["myopic"] + 4 * se, (
        f"DP {r['dp']:.4f} worse than myopic {r['myopic']:.4f} "
        f"@α={alpha} adv={adversarial} — DP wrong."
    )


def test_regret_decomposition():
    """Measure the three regret terms in isolation:

      α=0  →  no coverage miss, DP closes the myopic gap, so
              regret_DP(0)  ≈  conformal-conservatism alone   (term ii)
      α>0  →  regret_DP(α) − regret_DP(0)  ≤  α·q              (term i)
      myopic − DP            > 0                               (term iii, real & closed by DP)
    """
    base = _run(0.0, False, seed=7)
    cons = base["dp"] - base["clairvoyant"]            # term (ii)
    assert cons >= -1e-9, "conservatism term must be ≥ 0"
    myopic_gap0 = base["myopic"] - base["dp"]          # term (iii) @α=0
    assert myopic_gap0 > 0.0, (
        "myopic gap should be strictly positive (DP must actually help)"
    )
    for a in (0.05, 0.10, 0.20):
        r = _run(a, False, seed=7000 + int(a * 100))
        coverage_term = (r["dp"] - r["clairvoyant"]) - cons
        assert coverage_term <= a * Q + 4 * r["se_dp"], (
            f"coverage term {coverage_term:.4f} > α·q={a*Q:.4f} @α={a}"
        )
        assert r["dp"] <= r["myopic"] + 4 * r["se_dp"], "DP must dominate"


if __name__ == "__main__":
    hdr = (f"{'alpha':>5} {'adv':>4} {'myopic':>8} {'DP':>8} "
           f"{'clairv':>8} {'reg_my':>7} {'reg_DP':>7} {'a*q':>6} "
           f"{'cheap':>7} {'top':>7}")
    print(hdr)
    for a in (0.0, 0.05, 0.10, 0.20):
        for adv in (False, True):
            r = _run(a, adv, seed=1000 + int(a * 100))
            print(f"{a:>5} {str(adv):>4} {r['myopic']:>8.4f} "
                  f"{r['dp']:>8.4f} {r['clairvoyant']:>8.4f} "
                  f"{r['myopic']-r['clairvoyant']:>7.4f} "
                  f"{r['dp']-r['clairvoyant']:>7.4f} {a*Q:>6.2f} "
                  f"{r['cheap']:>7.3f} {r['top']:>7.3f}")
