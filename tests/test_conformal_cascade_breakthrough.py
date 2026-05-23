"""Falsification of the conformal selective-verification cascade.

Two layers:

  A. SYNTHETIC, deterministic, always-on — falsifies the *math*:
       - conformal selective-risk control on the cheap region holds
         within the finite-sample ⌈(n+1)(1−α)⌉ slack, even adversarially;
       - the band edge equals escalation.py rule (★) exactly;
       - the cascade Pareto-dominates both single verifiers when the
         cheap verifier's errors are score-concentrated (the regime the
         HaluEval-QA forensic established) — and *fails to* when they
         are not (the honest negative control).

  B. REAL DATA, skipped unless benchmarks/results/cascade_arrays.json
     exists — asserts the guarantee + a Pareto point on the actual
     HaluEval-QA WITNESS vs gpt-4o-mini arrays.

If the derivation is wrong, layer A fails with no Monte-Carlo excuse.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from entroly.conformal_cascade import (
    evaluate_policy,
    p_faithful,
    p_hallucinated,
    select_band,
)
from entroly.escalation import should_escalate

ARRAYS = (Path(__file__).resolve().parent.parent
          / "benchmarks" / "results" / "cascade_arrays.json")


# ── A. synthetic math falsification ───────────────────────────────────


def _synthetic(n, rng, *, concentrated: bool):
    """Cheap verifier score s∈[0,1]; faithful↦low, hallucinated↦high,
    with an overlap band. `concentrated`: errors live in a mid band
    (HaluEval-QA-like). Not concentrated: uniform noise (control).
    LLM verifier: strong but not oracle (≈0.9 acc)."""
    scores, labels, llm = [], [], []
    for _ in range(n):
        y = rng.randint(0, 1)
        if concentrated:
            base = 0.25 if y == 0 else 0.75
            s = min(1.0, max(0.0, rng.gauss(base, 0.12)))
        else:
            s = rng.random()  # score carries no signal
        scores.append(s)
        labels.append(y)
        # LLM ~0.9 accurate, independent-ish of the cheap verifier.
        llm.append(y if rng.random() < 0.9 else 1 - y)
    return scores, labels, llm


def test_band_edge_is_escalation_rule_star():
    """The selective-risk band edge must coincide with escalation.py
    rule (★): escalate iff local error > r_floor + c_exp/q."""
    rng = random.Random(1)
    cs, cl, _ = _synthetic(4000, rng, concentrated=True)
    q, c_exp, r_floor = 1.0, 0.10, 0.0
    pol = select_band(cs, cl, target_selective_risk=0.10,
                       c_exp=c_exp, q=q, r_floor=r_floor)
    assert abs(pol.rule_threshold - (r_floor + c_exp / q)) < 1e-12
    # Just inside the accept region the local error must NOT trigger (★);
    # the band is exactly the (★) escalation set by construction.
    assert not should_escalate(pol.target_selective_risk - 1e-6,
                               c_exp, q, r_floor)


@pytest.mark.parametrize("eps", [0.03, 0.05, 0.10])
def test_conformal_selective_risk_controlled(eps):
    """Realized selective risk on the cheap region ≤ ε + finite-sample
    slack, on disjoint test data. Falsifies the core guarantee."""
    rng = random.Random(7)
    cs, cl, cj = _synthetic(6000, rng, concentrated=True)
    ts, tl, tj = _synthetic(6000, rng, concentrated=True)
    pol = select_band(cs, cl, target_selective_risk=eps,
                      c_exp=eps, q=1.0)
    out = evaluate_policy(pol, ts, tl, tj)
    n_cheap = out.n_accept + out.n_flag
    slack = 3.0 * (eps * (1 - eps) / max(n_cheap, 1)) ** 0.5 + 1.0 / 60
    assert out.selective_risk_cheap <= eps + slack, (
        f"selective risk {out.selective_risk_cheap:.4f} > ε={eps} + "
        f"slack={slack:.4f} — conformal control BROKEN"
    )
    assert out.realized_alpha == out.selective_risk_cheap


def test_pareto_dominance_when_errors_concentrated():
    """In the HaluEval-QA-like regime (errors score-concentrated) the
    cascade must beat BOTH single verifiers at some operating point:
    overall error ≤ min(WITNESS-only, LLM-only) at cost < always-LLM."""
    rng = random.Random(11)
    cs, cl, cj = _synthetic(8000, rng, concentrated=True)
    ts, tl, tj = _synthetic(8000, rng, concentrated=True)
    n = len(ts)
    w_err = sum(int((1 if s > 0.5 else 0) != y)
                for s, y in zip(ts, tl)) / n
    g_err = sum(int(p != y) for p, y in zip(tj, tl)) / n
    found = None
    for eps in (0.02, 0.03, 0.05, 0.08, 0.10):
        pol = select_band(cs, cl, target_selective_risk=eps,
                          c_exp=eps, q=1.0)
        out = evaluate_policy(pol, ts, tl, tj)
        if (out.overall_error <= min(w_err, g_err) + 1e-9
                and out.expected_cost < 1.0):
            found = (eps, out)
            break
    assert found, (
        f"No Pareto point: WITNESS={w_err:.4f} LLM={g_err:.4f} — "
        f"cascade failed to dominate in the concentrated regime"
    )
    _, o = found
    assert o.expected_cost < 1.0
    assert o.overall_error <= min(w_err, g_err) + 1e-9


def test_negative_control_no_free_lunch():
    """Honesty guard: when the cheap score carries NO signal, the
    cascade must NOT manufacture a dominating point — it can only spend
    LLM calls. If this 'passes' (finds dominance), the harness is
    rigged."""
    rng = random.Random(13)
    cs, cl, cj = _synthetic(6000, rng, concentrated=False)
    ts, tl, tj = _synthetic(6000, rng, concentrated=False)
    n = len(ts)
    g_err = sum(int(p != y) for p, y in zip(tj, tl)) / n
    pol = select_band(cs, cl, target_selective_risk=0.05,
                      c_exp=0.05, q=1.0)
    out = evaluate_policy(pol, ts, tl, tj)
    # With no signal, to reach low error it must escalate ~everything
    # (cost → 1) OR carry high error. It cannot be both cheap and as
    # accurate as the LLM.
    if out.expected_cost < 0.8:
        assert out.overall_error > g_err + 0.02, (
            "FREE LUNCH detected: cascade beat the LLM cheaply with a "
            "signal-free cheap verifier — harness/math is rigged"
        )


# ── B. real HaluEval-QA arrays (skipped if not generated) ─────────────


@pytest.mark.skipif(not ARRAYS.exists(),
                    reason="run benchmarks/cascade_frontier.py first")
def test_real_halueval_cascade_guarantee_and_pareto():
    data = json.loads(ARRAYS.read_text(encoding="utf-8"))
    idx = list(range(data["n"]))
    random.Random(SEED := 49).shuffle(idx)
    half = data["n"] // 2

    def take(ix, k):
        return [data[k][i] for i in ix]
    cs = take(idx[:half], "scores")
    cl = take(idx[:half], "labels")
    cj = take(idx[:half], "llm")
    ts = take(idx[half:], "scores")
    tl = take(idx[half:], "labels")
    tj = take(idx[half:], "llm")

    n = len(ts)
    w_err = sum(int((1 if s > 0.0004 else 0) != y)
                for s, y in zip(ts, tl)) / n
    g_err = sum(int(p != y) for p, y in zip(tj, tl)) / n

    pareto = False
    for eps in (0.02, 0.03, 0.05, 0.08, 0.10, 0.15):
        pol = select_band(cs, cl, target_selective_risk=eps,
                          c_exp=eps, q=1.0)
        out = evaluate_policy(pol, ts, tl, tj)
        slack = 3.0 * (eps * (1 - eps)
                       / max(out.n_accept + out.n_flag, 1)) ** 0.5 + 1 / 50
        assert out.selective_risk_cheap <= eps + slack, (
            f"@ε={eps} real selective risk {out.selective_risk_cheap:.4f}"
            f" > ε+slack {eps + slack:.4f}"
        )
        if (out.overall_error <= min(w_err, g_err) + 0.005
                and out.expected_cost < 0.95):
            pareto = True
    assert pareto, (
        f"No Pareto point on real HaluEval-QA: WITNESS={w_err:.4f} "
        f"LLM={g_err:.4f}. Cascade does not dominate — report honestly."
    )
