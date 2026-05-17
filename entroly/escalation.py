"""Verification-gated sequential escalation.

The provable core of the compression↔verification↔routing loop: instead
of a static one-shot (context, model) choice, answer with a cheap model,
let WITNESS verify, and escalate (more context / stronger model) only
when the *calibrated* hallucination risk does not justify accepting.

This is optimal stopping with a costly observation — classic (Wald;
Chow–Robbins). The contribution is **not** that; it is that the
observation is a *conformally-calibrated verifier*. WITNESS's
split-conformal calibration (`witness_calibration.py`, RCPS) certifies a
miscoverage α:

    P[ r_true  >  r̂ + ε_α ]  ≤  α                                  (RCPS)

so the regret of the stopping rule can be written in the very α the
system already controls. That is the part worth shipping.

Derivation (1-D model ladder — proven)
--------------------------------------
Stages k = 1..K, cumulative escalation cost c_k (c_1 ≤ … ≤ c_K), per
stage a candidate answer with conformal-upper risk

    u_k := min(1, r̂_k + ε_α)        (the RCPS-operative risk)

Hallucination costs q if the accepted answer is wrong. Accept-now cost
≈ q·u_k. Escalating one step costs Δc = c_{k+1} − c_k and yields a
better candidate whose conformal-upper risk cannot exceed an
irreducible floor r_floor (the residual even the top model carries).

Escalate iff the avoidable expected hallucination cost beats the price:

    q·(u_k − r_floor)  >  Δc      ⟺      u_k  >  r_floor + Δc/q       (★)

So the optimal rule is a **threshold on the conformal-upper risk**,
τ⁺ = r_floor + Δc/q — independent of r̂'s scale, expressed in the
system's own units. (Equivalently, threshold on the raw estimate is
τ⁺ − ε_α: the conformal slack makes us escalate *more* readily, which is
the fail-closed direction RCPS already commits to.)

Regret (honest, additive — NOT the multiplicative form an earlier draft
hand-waved):

  Condition on the RCPS event E = {u_k is a valid upper bound}, P[E] ≥
  1−α. On E the rule (★) is cost-correct vs. a decision-maker who knew
  the true risk, so it incurs zero excess there. Off E (prob ≤ α) the
  true risk exceeded the certified bound; the worst extra cost on a
  query is bounded by q (a full, unmitigated hallucination). Hence per
  query

      E[cost_policy] − E[cost_clairvoyant]  ≤  α · q .               (R)

  α·q is exactly "the coverage you traded away × the price of being
  wrong." It is tight in the adversarial regime and is what the
  falsification harness (`tests/test_escalation_breakthrough.py`)
  checks against a brute-forced clairvoyant optimum.

What is proven vs. conjectured
------------------------------
Proven & falsified here: the 1-D model-ladder rule (★) and the additive
regret (R). NOT proven: the 2-D case where each escalation also extends
the context budget along the submodular greedy path keeps an interval
stopping region only under a supermodularity condition between marginal
information value and model capability — stated, not established. Do not
present 2-D as a theorem.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    """One rung of the escalation ladder."""
    name: str
    cum_cost: float           # cumulative cost to have produced this stage
    conformal_upper_risk: float  # u_k = min(1, r̂_k + ε_α), in [0, 1]


@dataclass(frozen=True)
class EscalationDecision:
    stop_index: int           # ladder index where we accepted
    spend: float              # cumulative escalation cost paid
    accepted_upper_risk: float
    rule_threshold: float     # τ⁺ at the stopping step (for audit)


def should_escalate(
    upper_risk: float,
    delta_cost: float,
    hallucination_cost: float,
    r_floor: float = 0.0,
) -> bool:
    """Rule (★): escalate iff u_k > r_floor + Δc/q.

    Pure, total, side-effect-free. `hallucination_cost` must be > 0
    (if being wrong is free, never escalate)."""
    if hallucination_cost <= 0.0:
        return False
    return upper_risk > r_floor + delta_cost / hallucination_cost


def run_escalation(
    ladder: list[Stage],
    hallucination_cost: float,
    r_floor: float = 0.0,
) -> EscalationDecision:
    """Walk the ladder under rule (★); stop at the first rung that is
    not worth escalating past (or the top rung).

    The ladder must be ordered by non-decreasing `cum_cost`. Returns the
    stopping rung, the spend, and the operative threshold for audit."""
    if not ladder:
        raise ValueError("empty ladder")
    for k in range(len(ladder) - 1):
        here, nxt = ladder[k], ladder[k + 1]
        delta = nxt.cum_cost - here.cum_cost
        tau = r_floor + (delta / hallucination_cost if hallucination_cost > 0 else float("inf"))
        if not should_escalate(here.conformal_upper_risk, delta,
                               hallucination_cost, r_floor):
            return EscalationDecision(k, here.cum_cost,
                                      here.conformal_upper_risk, tau)
    top = ladder[-1]
    last_delta = top.cum_cost - ladder[-2].cum_cost if len(ladder) > 1 else 0.0
    return EscalationDecision(
        len(ladder) - 1, top.cum_cost, top.conformal_upper_risk,
        r_floor + (last_delta / hallucination_cost
                   if hallucination_cost > 0 else float("inf")),
    )


def realized_cost(
    decision: EscalationDecision,
    true_risk_at_stop: float,
    hallucination_cost: float,
) -> float:
    """Ground-truth cost of a decision: escalation spend + expected
    hallucination cost under the *true* risk of the accepted answer.
    Used by the falsification harness; the controller never sees this."""
    return decision.spend + hallucination_cost * true_risk_at_stop


def optimal_stop_dp(
    ladder: list[Stage],
    hallucination_cost: float,
    r_floor: float = 0.0,
) -> EscalationDecision:
    """Exact backward-induction optimum over the *observed conformal-upper
    risks* — the tightest controller that uses only what WITNESS reveals.

    Backward induction (Bellman):

        V_K       = q · u_K                                 (must stop)
        V_k       = min( q · u_k ,  Δc_k + V_{k+1} )
        stop at k ⟺ q · u_k ≤ Δc_k + V_{k+1}

    where u_k = conformal_upper_risk, Δc_k = c_{k+1} − c_k, q =
    hallucination_cost. The first k satisfying the stop condition is the
    decision (ties → stop, the cheaper action).

    Why this is the right object. The myopic rule (★) compares only the
    *single* next step; this looks ahead over the whole remaining
    ladder. Hence **by construction**

        regret_DP(ω)  ≤  regret_myopic(ω)   for every realization ω

    on the same observed u-ladder — `optimal_stop_dp` is the argmin over
    stop indices of (c_k + q·u_k), which the myopic rule only
    approximates. The falsification harness asserts this inequality
    never inverts (if it does, the DP is wrong).

    Honest decomposition of regret vs. the (unattainable) clairvoyant
    that sees true risk r_k:

      (i)   coverage-miss loss   ≤ α·q   — you see u, not r; on the ≤α
            RCPS-miss mass the bound u ≥ r fails, worst extra cost q.
      (ii)  conformal-conservatism — even when covered, u_k = r_k +
            slack ≥ r_k, so argmin over u can pick a costlier index than
            argmin over r. Irreducible without a tighter verifier;
            shrinks as the conformal interval tightens.
      (iii) myopic look-ahead gap — **fully closed by this DP** (that is
            the contribution here; (i) and (ii) are not closable by
            smarter stopping, only by a better verifier).

    Scope (stated, not overclaimed): this is the *offline* optimum over
    the observed u-ladder — it isolates term (iii) exactly and is a
    clean lower bound on achievable-with-conformal regret. A *realizable
    online* DP (deciding at rung k without having paid to observe
    u_{k+1..K}) additionally needs a transition model for how u evolves
    down the ladder; that is a modelling choice, named here, not proven.
    """
    if not ladder:
        raise ValueError("empty ladder")
    q = hallucination_cost
    n = len(ladder)
    # Value-to-go by backward induction.
    v_next = q * ladder[-1].conformal_upper_risk  # V_K: must stop
    # Walk forward to find the first rung whose stop condition holds,
    # using V_{k+1} computed from the suffix. Precompute suffix values.
    suffix_v = [0.0] * n
    suffix_v[-1] = v_next
    for k in range(n - 2, -1, -1):
        delta = ladder[k + 1].cum_cost - ladder[k].cum_cost
        stop_here = q * ladder[k].conformal_upper_risk
        suffix_v[k] = min(stop_here, delta + suffix_v[k + 1])
    for k in range(n - 1):
        delta = ladder[k + 1].cum_cost - ladder[k].cum_cost
        tau = r_floor + (delta / q if q > 0 else float("inf"))
        stop_here = q * ladder[k].conformal_upper_risk
        if stop_here <= delta + suffix_v[k + 1] + 1e-12:
            return EscalationDecision(k, ladder[k].cum_cost,
                                      ladder[k].conformal_upper_risk, tau)
    top = ladder[-1]
    last_delta = top.cum_cost - ladder[-2].cum_cost if n > 1 else 0.0
    return EscalationDecision(
        n - 1, top.cum_cost, top.conformal_upper_risk,
        r_floor + (last_delta / q if q > 0 else float("inf")),
    )
