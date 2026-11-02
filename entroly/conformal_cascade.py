"""Empirical selective-verification bands with class-rank diagnostics.

select_band greedily fits score cutoffs using calibration error and a cost
threshold. Rank p-values are reported but do not determine actions. Under a
fixed scorer and within-class exchangeability, they are marginally super-uniform
for their class nulls; a small value rejects that null. They do not certify the
population error conditional on a cheap decision.

Empirical prefix errors need not be monotone. The greedy scan stops at the
first failed cutoff and need not find the largest feasible cheap region.
Separate evaluation is descriptive evidence, not a distribution-free selective
risk, local-error, per-user or regret guarantee. cascade_assessment separately
implements an opt-in held-out binomial assessment with explicit assumptions.

evaluate_policy.expected_cost counts resource-call cost only. An error penalty
q additionally requires q times joint error probability:
P(cheap)*R_cheap + P(escalate)*R_escalated. Conditional cheap error alone is not
that penalty. Deterministic scoring also consumes local resources.

No providers are invoked here. Production proxy fusion thresholds are a separate
rule; these fitted bands do not certify that routing policy.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from entroly.escalation import should_escalate

ACCEPT = "accept"
FLAG = "flag"
ESCALATE = "escalate"


@dataclass(frozen=True)
class CascadeCalibration:
    """Sorted class-conditional calibration scores (ascending)."""
    halu_scores: tuple[float, ...]
    faith_scores: tuple[float, ...]

    @property
    def n_h(self) -> int:
        return len(self.halu_scores)

    @property
    def n_g(self) -> int:
        return len(self.faith_scores)


def fit_cascade(scores: list[float], labels: list[int]) -> CascadeCalibration:
    """Split the labelled calibration sample by class and sort. `labels`
    use 1 = hallucinated, 0 = faithful."""
    if len(scores) != len(labels):
        raise ValueError("scores/labels length mismatch")
    halu = sorted(s for s, y in zip(scores, labels) if y == 1)
    faith = sorted(s for s, y in zip(scores, labels) if y == 0)
    if not halu or not faith:
        raise ValueError("calibration needs both classes present")
    return CascadeCalibration(tuple(halu), tuple(faith))


def p_hallucinated(cal: CascadeCalibration, s: float) -> float:
    """p_h(s) = (1 + #{halu cal ≥ s}) / (n_h + 1).  Valid conformal
    p-value under within-class exchangeability; small rejects the hallucinated null."""
    ge = cal.n_h - bisect_left(cal.halu_scores, s)
    return (1 + ge) / (cal.n_h + 1)


def p_faithful(cal: CascadeCalibration, s: float) -> float:
    """p_g(s) = (1 + #{faithful cal ≤ s}) / (n_g + 1).  Small ⇒
    reject the faithful null, under within-class exchangeability."""
    le = bisect_right(cal.faith_scores, s)
    return (1 + le) / (cal.n_g + 1)


@dataclass(frozen=True)
class CascadePolicy:
    """An operating point: accept below `t_lo`, flag at/above `t_hi`,
    escalate in the open band (t_lo, t_hi). Edges are score cutoffs so
    the rule is total and audit-clear; the conformal p-values are
    reported per decision as diagnostics, not selective-risk certification."""
    cal: CascadeCalibration
    t_lo: float                  # accept-as-faithful if s ≤ t_lo
    t_hi: float                  # flag-as-hallucinated if s ≥ t_hi
    target_selective_risk: float
    c_exp: float
    q: float
    r_floor: float
    rule_threshold: float        # r_floor + c_exp/q  (escalation ★)


@dataclass(frozen=True)
class Decision:
    action: str                  # ACCEPT | FLAG | ESCALATE
    predict_hallucinated: bool | None   # None iff escalated
    p_h: float
    p_g: float


def decide(policy: CascadePolicy, s: float) -> Decision:
    """Total, side-effect-free three-way decision for one item."""
    ph = p_hallucinated(policy.cal, s)
    pg = p_faithful(policy.cal, s)
    if s <= policy.t_lo:
        return Decision(ACCEPT, False, ph, pg)
    if s >= policy.t_hi:
        return Decision(FLAG, True, ph, pg)
    return Decision(ESCALATE, None, ph, pg)


def _selective_error_below(sorted_pairs, t_lo):
    """Error rate among calibration points with s ≤ t_lo when we predict
    faithful there (an error is a hallucinated point, y=1)."""
    n = err = 0
    for s, y in sorted_pairs:
        if s > t_lo:
            break
        n += 1
        err += y
    return (err / n if n else 0.0), n


def _selective_error_above(sorted_pairs, t_hi):
    """Error rate among s ≥ t_hi when we predict hallucinated (an error
    is a faithful point, y=0)."""
    n = err = 0
    for s, y in reversed(sorted_pairs):
        if s < t_hi:
            break
        n += 1
        err += (1 - y)
    return (err / n if n else 0.0), n


def select_band(
    scores: list[float],
    labels: list[int],
    *,
    target_selective_risk: float,
    c_exp: float,
    q: float,
    r_floor: float = 0.0,
) -> CascadePolicy:
    """Greedily grow two empirical cheap regions until a cutoff fails.

    The target applies to observed calibration error, not population conditional
    risk. Prefix errors need not be monotone; later feasible cutoffs can be
    missed. Repeated prefix scans can take quadratic time. Independent held-out
    assessment is separate from fitting these cutoffs.
    """
    if len(scores) != len(labels):
        raise ValueError("scores/labels length mismatch")
    cal = fit_cascade(scores, labels)
    pairs = sorted(zip(scores, labels))
    eps = target_selective_risk
    rule_tau = r_floor + (c_exp / q if q > 0 else float("inf"))

    # Grow the accept region upward through candidate cutoffs while both
    # the empirical target and rule (★) hold. Cumulative error is an
    # empirical proxy, not a certified local error probability.
    cand = sorted({s for s, _ in pairs})
    t_lo = float("-inf")
    for t in cand:
        e, _ = _selective_error_below(pairs, t)
        if e <= eps and not should_escalate(e, c_exp, q, r_floor):
            t_lo = t
        else:
            break
    t_hi = float("inf")
    for t in reversed(cand):
        e, _ = _selective_error_above(pairs, t)
        if e <= eps and not should_escalate(e, c_exp, q, r_floor):
            t_hi = t
        else:
            break
    # Degenerate guard: if the two regions cross (no escalation needed),
    # clamp the band to empty at the midpoint.
    if t_lo >= t_hi:
        mid = 0.5 * (min(t_lo, max(cand)) + max(t_hi, min(cand)))
        t_lo = t_hi = mid
    return CascadePolicy(
        cal=cal, t_lo=t_lo, t_hi=t_hi,
        target_selective_risk=eps, c_exp=c_exp, q=q, r_floor=r_floor,
        rule_threshold=rule_tau,
    )


@dataclass(frozen=True)
class CascadeOutcome:
    n: int
    n_accept: int
    n_flag: int
    n_escalate: int
    escalation_rate: float
    selective_risk_cheap: float      # error rate among non-escalated
    overall_error: float             # with LLM resolving escalations
    expected_cost: float             # c_cheap≈0 + esc·c_exp + q·SelRisk
    realized_alpha: float            # = selective_risk_cheap (the α in α·q)


def evaluate_policy(
    policy: CascadePolicy,
    scores: list[float],
    labels: list[int],
    llm_predict: list[int],
    *,
    c_cheap: float = 0.0,
    c_exp_cost: float = 1.0,
) -> CascadeOutcome:
    """Run the policy on a held-out set. `llm_predict[i]` is the
    expensive verifier's label (1/0) used only for escalated items, so
    its measured error enters honestly (no oracle assumption).

    NOTE — two distinct cost roles, kept separate on purpose:
      * `policy.c_exp` is the *band-selection* risk–cost ratio (c_exp/q),
        the lever that ties the escalation band to rule (★);
      * `c_exp_cost` is the *resource* price of one expensive call used
        for accounting. With the defaults (c_cheap=0, c_exp_cost=1) the
        reported `expected_cost` equals the escalation rate — the
        interpretable axis: fraction of LLM calls vs. always-LLM=1.0.
    Conflating these two is a unit error the negative-control test
    `test_negative_control_no_free_lunch` exists to catch."""
    n = len(scores)
    na = nf = ne = 0
    cheap_err = 0
    overall_err = 0
    cost = 0.0
    for s, y, lj in zip(scores, labels, llm_predict):
        d = decide(policy, s)
        cost += c_cheap
        if d.action == ESCALATE:
            ne += 1
            cost += c_exp_cost
            overall_err += int(lj != y)
        else:
            pred = 1 if d.predict_hallucinated else 0
            wrong = int(pred != y)
            cheap_err += wrong
            overall_err += wrong
            if d.action == ACCEPT:
                na += 1
            else:
                nf += 1
    n_cheap = na + nf
    sel_risk = cheap_err / n_cheap if n_cheap else 0.0
    return CascadeOutcome(
        n=n, n_accept=na, n_flag=nf, n_escalate=ne,
        escalation_rate=ne / n if n else 0.0,
        selective_risk_cheap=sel_risk,
        overall_error=overall_err / n if n else 0.0,
        expected_cost=cost / n if n else 0.0,
        realized_alpha=sel_risk,
    )
