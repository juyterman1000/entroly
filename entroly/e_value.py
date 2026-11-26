"""Marginal null evidence from fixed scores, with dependence-safe averaging.

Conservative right-tail ranks include calibration ties. For exchangeable null
calibration/test scores from a fixed scorer, p is super-uniform. The fixed
power calibrator e(p)=0.5/sqrt(p) has integral one, so E[e] <= 1. Clipping it
down preserves this property: an upper bound on the mean survives any
pointwise decrease, which is why the clip is safe to apply unconditionally.
For the validity proof and the admissibility of this calibrator family, see
Vovk and Wang, Sections 2-4: https://arxiv.org/abs/1912.06116

The arithmetic mean of a fixed family of valid same-null e-values is valid
under arbitrary dependence. Their product is NOT valid from marginal
calibration alone; product/max/min diagnostics carry no rejection authority.
There is no optional-stopping, shift, selective-risk or answer-correctness
guarantee here. Failure to reject a null does not prove a claim is supported.
All statistical statements require genuine null data and a scorer/family
fixed independently of that data; this API cannot establish those assumptions.
"""

from __future__ import annotations

import bisect
import math
import sys
from dataclasses import dataclass, field
from typing import Sequence


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric data")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} exceeds numeric range") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _alpha(value: float) -> float:
    value = _finite(value, "alpha")
    if not 0 < value < 1:
        raise ValueError("alpha must be within (0, 1)")
    return value


def _e_values(values: Sequence[float]) -> list[float]:
    checked = [_finite(value, "e-value") for value in values]
    if any(value < 0 for value in checked):
        raise ValueError("e-values must be nonnegative")
    return checked


# ── Empirical CDF helper ──────────────────────────────────────────────


class EmpiricalCDF:
    """Empirical CDF F̂(s) = (1/n) Σ I(x_i ≤ s).

    Returns 0.0 below all calibration points, 1.0 above all.
    """

    def __init__(self, calibration_scores: Sequence[float]):
        self._sorted = sorted(_finite(score, "calibration score") for score in calibration_scores)
        self._n = len(self._sorted)

    def __call__(self, s: float) -> float:
        s = _finite(s, "score")
        if self._n == 0:
            return 0.5
        # Number of calibration points <= s
        n_le = bisect.bisect_right(self._sorted, s)
        return n_le / self._n

    def conformal_p(self, s: float) -> float:
        """Right-tail conformal p-value: P(score >= s | H_0).

        p = (1 + number of calibration scores >= s) / (n+1).
        Ties are included, so a constant null has p=1, not 1/(n+1).
        """
        s = _finite(s, "score")
        if self._n == 0:
            return 1.0
        n_lt = bisect.bisect_left(self._sorted, s)
        # Conservative right-tail p (split-conformal):
        return (self._n - n_lt + 1) / (self._n + 1)

    def quantile(self, q: float) -> float:
        """Inverse CDF at level q ∈ [0,1]. Linear interpolation."""
        q = _finite(q, "quantile")
        if self._n == 0 or q <= 0:
            return 0.0
        if q >= 1:
            return 1.0
        idx = q * (self._n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return self._sorted[lo]
        frac = idx - lo
        return (1 - frac) * self._sorted[lo] + frac * self._sorted[hi]


# ── E-value calibrator ────────────────────────────────────────────────


@dataclass
class EValueCalibrator:
    """Maps a raw hallucination score to a calibrated e-value.

    e(s) = 0.5 / sqrt(p(s)). The integral of this decreasing p-to-e
    calibrator on (0,1) equals one; inverse-p alone is not an e-value.
    Empty calibration returns neutral evidence, e=1.
    """
    cdf: EmpiricalCDF
    max_evalue: float = 1e6
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if not isinstance(self.cdf, EmpiricalCDF):
            raise ValueError("cdf must be an EmpiricalCDF")
        self.max_evalue = _finite(self.max_evalue, "max_evalue")
        self.eps = _finite(self.eps, "eps")
        if self.max_evalue <= 0 or not 0 < self.eps <= 1:
            raise ValueError("max_evalue must be positive and eps within (0,1]")

    @classmethod
    def fit(
        cls,
        calibration_scores: Sequence[float],
        *,
        max_evalue: float = 1e6,
    ) -> "EValueCalibrator":
        return cls(cdf=EmpiricalCDF(calibration_scores), max_evalue=max_evalue)

    def p_value(self, score: float) -> float:
        return self.cdf.conformal_p(score)

    def e_value(self, score: float) -> float:
        """A fixed valid p-to-e transform, clipped only downwards."""
        p = self.p_value(score)
        if not self.cdf._n:
            return min(self.max_evalue, 1.0)
        return min(self.max_evalue, 0.5 / math.sqrt(max(p, self.eps)))


# ── E-Product composition ─────────────────────────────────────────────


def e_product(e_values: Sequence[float]) -> float:
    """Multiplicative composition of e-values.

    Valid under independence or appropriate conditional e-value assumptions,
    which callers here do not establish. A diagnostic only for dependent layers.
    """
    values = _e_values(e_values)
    if any(e == 0 for e in values):
        return 0.0
    # Accumulate binary exponents separately: overflow followed by a small
    # factor must not become inf/NaN or destroy a representable final product.
    mantissa, exponent = 0.5, 1
    for e in values:
        part, shift = math.frexp(e)
        mantissa, adjustment = math.frexp(mantissa * part)
        exponent += shift + adjustment
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return sys.float_info.max  # diagnostic saturated at the finite limit


def e_mean(e_values: Sequence[float]) -> float:
    """Arithmetic mean of e-values — always a valid e-value, without
    requiring independence."""
    checked = _e_values(e_values)
    if not checked:
        return 1.0
    scale = max(checked)
    return scale * (math.fsum(e / scale for e in checked) / len(checked)) if scale else 0.0


def e_to_conservative_p(e: float) -> float:
    """Conservative p-value from an e-value: p = min(1, 1/e).

    Markov's inequality gives P(e >= t) ≤ E[e]/t ≤ 1/t under H_0, so
    p = 1/e is conservatively valid.
    """
    e = _e_values([e])[0]
    if e == 0:
        return 1.0
    return min(1.0, 1.0 / e)


# ── Worst-case bound across manifolds ─────────────────────────────────


@dataclass
class WorstCaseBound:
    """Legacy manifold report; only mean_e is dependence-safe null evidence.

    Max/product/min fields describe observed extremes, not worst-distribution
    guarantees. Valid marginal calibration on every fixed manifold is external.
    """
    worst_e: float
    joint_e: float
    mean_e: float
    worst_p: float
    per_manifold_e: dict[str, float] = field(default_factory=dict)
    per_manifold_p: dict[str, float] = field(default_factory=dict)
    evidence_method: str = "fixed-power-p-to-e-arithmetic-mean"
    e_product_scope: str = "diagnostic-only-independence-not-established"
    statistical_scope: str = "single-null-test-under-exchangeability-not-answer-correctness"
    production_authority: bool = False

    @property
    def reject_at_alpha(self) -> bool:
        """Compatibility property: reject the common null at default alpha=.05."""
        return self.reject()

    def reject(self, alpha: float = 0.05) -> bool:
        """Test the common null using the mean, not a dependent product."""
        return e_to_conservative_p(self.mean_e) <= _alpha(alpha)

    def as_dict(self) -> dict:
        return {
            "worst_e": self.worst_e,
            "joint_e": self.joint_e,
            "mean_e": self.mean_e,
            "worst_p": self.worst_p,
            "per_manifold_e": self.per_manifold_e,
            "per_manifold_p": self.per_manifold_p,
            "evidence_method": "fixed-power-p-to-e-arithmetic-mean",
            "e_product_scope": "diagnostic-only-independence-not-established",
            "statistical_scope": "single-null-test-under-exchangeability-not-answer-correctness",
            "production_authority": False,
        }


def manifold_conditional_crc(
    test_scores: dict[str, float],
    calibrators: dict[str, EValueCalibrator],
) -> WorstCaseBound:
    """Report fixed-manifold rank diagnostics and dependence-safe mean evidence.

    The legacy name does not imply selective-risk control or shift robustness.

    Args:
        test_scores: dict {manifold_name -> score}
        calibrators: dict {manifold_name -> EValueCalibrator}
    """
    per_e: dict[str, float] = {}
    per_p: dict[str, float] = {}
    for name, score in test_scores.items():
        score = _finite(score, "score")
        cal = calibrators.get(name)
        if cal is None:
            # Unknown manifold: be conservative (e=1, p=1)
            per_e[name] = 1.0
            per_p[name] = 1.0
            continue
        per_e[name] = cal.e_value(score)
        per_p[name] = cal.p_value(score)

    if not per_e:
        return WorstCaseBound(worst_e=1.0, joint_e=1.0, mean_e=1.0, worst_p=1.0)

    worst_e = max(per_e.values())
    joint_e = e_product(list(per_e.values()))
    mean_e_ = e_mean(list(per_e.values()))
    worst_p = min(per_p.values())

    return WorstCaseBound(
        worst_e=worst_e,
        joint_e=joint_e,
        mean_e=mean_e_,
        worst_p=worst_p,
        per_manifold_e=per_e,
        per_manifold_p=per_p,
    )


# ── Multi-layer e-value composition ──────────────────────────────────


@dataclass
class LayerEvidence:
    """E-value evidence from one of the EICV 6 layers.

    Each layer produces a hallucination score in [0,1]. The score is
    converted to an e-value using a layer-specific calibrator (fit on
    a held-out grounded set).
    """
    name: str               # e.g. "esg", "rnr", "gamma", "nli_bidir", "sem_entropy"
    score: float            # raw layer score in [0,1] (higher = more hallucinated)
    e_value: float          # calibrated e-value
    p_value: float          # conformal p-value


@dataclass
class CompositeCertificate:
    """Legacy report of null evidence; not a semantic correctness certificate.

    The fixed mean can reject the calibrated common null. Failure to reject
    abstains, because a low e-value cannot establish a claim's correctness.
    """
    layers: list[LayerEvidence]
    e_product: float
    e_mean: float
    decision: str        # "supported" | "hallucinated" | "abstain"
    alpha: float = 0.05
    worst_case: WorstCaseBound | None = None
    evidence_method: str = "fixed-power-p-to-e-arithmetic-mean"
    e_product_scope: str = "diagnostic-only-independence-not-established"
    statistical_scope: str = "single-null-test-under-exchangeability-not-answer-correctness"
    production_authority: bool = False

    def as_dict(self) -> dict:
        return {
            "layers": [layer.__dict__ for layer in self.layers],
            "e_product": self.e_product,
            "e_mean": self.e_mean,
            "decision": self.decision,
            "alpha": self.alpha,
            "worst_case": self.worst_case.as_dict() if self.worst_case else None,
            "evidence_method": "fixed-power-p-to-e-arithmetic-mean",
            "e_product_scope": "diagnostic-only-independence-not-established",
            "statistical_scope": "single-null-test-under-exchangeability-not-answer-correctness",
            "production_authority": False,
        }


def build_certificate(
    layer_scores: dict[str, float],
    calibrators: dict[str, EValueCalibrator],
    *,
    alpha: float = 0.05,
    abstain_band: tuple[float, float] = (0.3, 3.0),
    worst_case: WorstCaseBound | None = None,
) -> CompositeCertificate:
    """Assess the common calibrated null using a dependence-safe mean.

    Args:
        layer_scores: dict {layer_name -> score ∈ [0,1]}.
        calibrators: dict {layer_name -> EValueCalibrator} fit on grounded set.
        alpha: significance level (default 0.05 → e-threshold 20).
        abstain_band: validated legacy argument; low evidence now abstains.
        worst_case: optional manifold-conditional CRC result.

    Returns:
        CompositeCertificate with layer breakdown + decision.
    """
    alpha = _alpha(alpha)
    if not isinstance(abstain_band, (tuple, list)) or len(abstain_band) != 2:
        raise ValueError("abstain_band must contain two ordered nonnegative values")
    low, high = _e_values(abstain_band)
    if low > high:
        raise ValueError("abstain_band must be ordered")
    layers: list[LayerEvidence] = []
    e_vals: list[float] = []
    for name, score in layer_scores.items():
        score = _finite(score, "score")
        cal = calibrators.get(name)
        if cal is None:
            e, p = 1.0, 1.0
        else:
            e = cal.e_value(score)
            p = cal.p_value(score)
        layers.append(LayerEvidence(
            name=name,
            score=score,
            e_value=e,
            p_value=p,
        ))
        e_vals.append(e)

    ep = e_product(e_vals)
    em = e_mean(e_vals)

    # Do not infer support from failure to reject a null hypothesis.
    decision = "hallucinated" if e_to_conservative_p(em) <= alpha else "abstain"

    return CompositeCertificate(
        layers=layers,
        e_product=ep,
        e_mean=em,
        decision=decision,
        alpha=alpha,
        worst_case=worst_case,
    )


# ── Anytime-valid ONS betting e-process (Ville's inequality) ──────────

_ONS_STEP_SCALE = 2.0 / (2.0 - math.log(3.0))


@dataclass(slots=True)
class BettingStepRecord:
    """Audit record for one sequential observation in ``BettingEProcess``."""

    step: int
    loss: float
    bet_lambda: float
    multiplier: float
    wealth: float
    max_wealth: float
    anytime_p: float
    rejected: bool


class BettingEProcess:
    """Online-Newton-Step predictable-betting test supermartingale (Ville's inequality).

    Tests the sequential conditional-mean null hypothesis
        ``H_0: E[loss_t | F_{t-1}] <= null_risk`` for all ``t >= 1``,
    where each observed loss lies in ``[0, 1]``.

    Construction (Waudby-Smith & Ramdas, https://arxiv.org/abs/2010.09686;
    anytime-valid conformal risk control, https://arxiv.org/abs/2602.04364,
    https://arxiv.org/abs/2606.29054):
    - Start with ``M_0 = 1``, ``A_0 = 1``, and predictable bet ``lambda_1 = 0``.
    - At step ``t``, observe ``loss_t in [0, 1]`` and update wealth using the
      pre-observation bet ``lambda_t in [0, 1 / (2 * null_risk)]``:
          ``m_t = 1 + lambda_t * (loss_t - null_risk) >= 0.5 > 0``
          ``M_t = min(1e300, M_{t-1} * m_t)``
    - Update the Online Newton Step (ONS) predictable bet for step ``t + 1``
      using ``z_t = (loss_t - null_risk) / m_t`` and ``A_t = A_{t-1} + z_t^2``:
          ``lambda_{t+1} = clip(lambda_t + (2 / (2 - ln 3)) * z_t / A_t, 0, 1 / (2 * null_risk))``

    Why this holds under optional stopping:
    - Because ``lambda_t >= 0`` is ``F_{t-1}``-measurable and ``m_t >= 0.5``,
      ``E[m_t | F_{t-1}] <= 1`` under ``H_0``; upper-clipping at ``1e300`` is
      concave and nondecreasing, so Jensen's inequality preserves the
      nonnegative supermartingale property ``E[M_t | F_{t-1}] <= M_{t-1}``.
    - By Ville's inequality, for any data-dependent stopping time ``tau``:
          ``P_{H_0}(exists t >= 1 : M_t >= 1 / delta) <= delta``.
    - On ``[-0.5, inf)``, ``x -> ln(1 + x)`` is exp-concave, giving ONS
      ``O(ln T)`` log-wealth regret relative to the best fixed Kelly fraction in
      ``[0, 1 / (2 * null_risk)]``.

    Scope: valid for sequential optional stopping on bounded ``[0, 1]`` losses
    under the stated conditional-mean null. Does not establish distribution
    stationarity or model answer correctness on its own.
    """

    def __init__(
        self,
        *,
        null_risk: float = 0.10,
        delta: float = 0.05,
        max_bet_fraction: float = 0.50,
    ) -> None:
        self.null_risk = _alpha(null_risk)
        self.delta = _alpha(delta)
        frac = _finite(max_bet_fraction, "max_bet_fraction")
        if not 0.0 < frac < 1.0:
            raise ValueError("max_bet_fraction must be within (0, 1)")
        # Default 0.50 keeps lambda_t * (loss_t - null_risk) >= -0.5, inside the
        # exp-concave domain of ln(1 + x) required by ONS.
        self.max_bet = frac / self.null_risk
        self.threshold = 1.0 / self.delta

        self._wealth: float = 1.0
        self._max_wealth: float = 1.0
        self._bet: float = 0.0
        self._grad_sq_sum: float = 1.0
        self._steps: int = 0
        self._loss_sum: float = 0.0
        self._history: list[BettingStepRecord] = []

    def next_bet(self) -> float:
        """Predictable betting fraction ``lambda_{t+1}`` measurable w.r.t. history ``<= t``."""
        return self._bet

    def update(self, loss: float) -> BettingStepRecord:
        """Ingest one bounded loss in ``[0, 1]`` and advance the ONS test supermartingale."""
        val = _finite(loss, "loss")
        if not 0.0 <= val <= 1.0:
            raise ValueError("loss must be within [0, 1]")

        lam = self._bet
        excess = val - self.null_risk
        multiplier = 1.0 + lam * excess
        self._wealth = min(1e300, self._wealth * multiplier)
        if self._wealth > self._max_wealth:
            self._max_wealth = self._wealth

        # ONS update for step t + 1 (strictly after evaluating wealth at step t).
        z_t = excess / multiplier
        self._grad_sq_sum += z_t * z_t
        cand = lam + _ONS_STEP_SCALE * (z_t / self._grad_sq_sum)
        self._bet = max(0.0, min(self.max_bet, cand))

        self._steps += 1
        self._loss_sum += val

        rec = BettingStepRecord(
            step=self._steps,
            loss=val,
            bet_lambda=lam,
            multiplier=multiplier,
            wealth=self._wealth,
            max_wealth=self._max_wealth,
            anytime_p=min(1.0, 1.0 / max(self._max_wealth, 1.0)),
            rejected=self._max_wealth >= self.threshold,
        )
        self._history.append(rec)
        return rec

    @property
    def wealth(self) -> float:
        return self._wealth

    @property
    def max_wealth(self) -> float:
        return self._max_wealth

    @property
    def anytime_p(self) -> float:
        return min(1.0, 1.0 / max(self._max_wealth, 1.0))

    @property
    def rejected(self) -> bool:
        return self._max_wealth >= self.threshold

    @property
    def steps(self) -> int:
        return self._steps

    def summary(self) -> dict[str, object]:
        mean_loss = self._loss_sum / self._steps if self._steps else 0.0
        return {
            "steps": self._steps,
            "null_risk": self.null_risk,
            "delta": self.delta,
            "threshold": self.threshold,
            "wealth": self._wealth,
            "max_wealth": self._max_wealth,
            "anytime_p": self.anytime_p,
            "rejected": self.rejected,
            "empirical_loss_mean": round(mean_loss, 6),
            "next_bet": round(self._bet, 6),
            "method": "ons-predictable-betting-supermartingale-ville",
        }
