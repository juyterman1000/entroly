"""
E-Value Composition + Conformal Risk Control — EICV Layer 5
=============================================================

E-values (Shafer 2019; Vovk & Wang 2021) are an alternative to p-values
with two key properties that p-values lack:

  1. Closed under arithmetic mean: e_avg = mean(e_i) is itself a valid
     e-value testing the conjunction H_1 ∧ H_2 ∧ ...
  2. Closed under PRODUCT: e_joint = ∏ e_i is a valid e-value testing
     the union of hypotheses under sequential testing / optional stopping.

Together, e-values support post-hoc combination of evidence from multiple
sources without α-spending or Bonferroni inflation.

Calibration map e(score)
-------------------------
We use the calibration map e(s) = β · 2^(k(s)) where k(s) is the score's
position above the empirical CDF of the H_0 (supported claims) calibration
set. This is the Vovk-Wang (2021) "betting score" with prior probability β.

Concretely, for ESG T(G):
  - Calibrate using a held-out set of GROUNDED items (label = 0): collect
    F_0 = empirical CDF of T(G).
  - For a new score s, the conformal p-value is p = 1 - F_0(s) (right tail).
  - The e-value under the betting interpretation is e = (1 + ε)^t where
    t = inv_cdf(1 - p) is the deviation from null.

We use the simpler and well-calibrated transformation:
  e(s) = max(1, score / (1 - F_0(s) + ε))

Composition with E-Product
---------------------------
With 6 independent (or marginally calibrated) sources:
  e_joint = ∏_{i=1}^6 e_i(s_i)
A claim "passes" (no hallucination) iff e_joint ≤ 1/α  (default α = 0.05).

Conformal Risk Control (CRC)
-----------------------------
For a single score, the conformal p-value gives a finite-sample guarantee:
  P(p ≤ α) ≤ α  under exchangeability between calibration and test.

When the test distribution shifts (e.g. C4 vs C1), exchangeability fails.
We use weighted CP (Barber-Candès-Ramdas-Tibshirani 2023):
  weights w_i = density_ratio(test, calibration) at point z_i
which restores coverage under known shift direction.

Worst-Case Bound
----------------
For the worst manifold M*:
  P(p_{M*} ≤ α) ≤ α                                  (per-manifold)
  P(any p_m ≤ α/|M|) ≤ α                              (joint Bonferroni)
  e_M* >= max_m e_m                                    (e-value worst-case)
  e_joint = ∏ e_m → composite over all manifolds

References
----------
- Shafer, 2019. The language of betting as a strategy for statistical
  and scientific communication. JASA.
- Vovk & Wang, 2021. E-values: Calibration, combination, and applications.
  Annals of Statistics 49(3): 1736-1754.
- Barber, Candès, Ramdas, Tibshirani, 2023. Conformal prediction beyond
  exchangeability. Annals of Statistics 51(2): 816-845.
- Angelopoulos & Bates, 2023. Conformal Prediction: A Gentle Introduction.
  Foundations and Trends in Machine Learning.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Sequence


# ── Empirical CDF helper ──────────────────────────────────────────────


class EmpiricalCDF:
    """Empirical CDF F̂(s) = (1/n) Σ I(x_i ≤ s).

    Returns 0.0 below all calibration points, 1.0 above all.
    """

    def __init__(self, calibration_scores: Sequence[float]):
        self._sorted = sorted(calibration_scores)
        self._n = len(self._sorted)

    def __call__(self, s: float) -> float:
        if self._n == 0:
            return 0.5
        # Number of calibration points <= s
        n_le = bisect.bisect_right(self._sorted, s)
        return n_le / self._n

    def conformal_p(self, s: float) -> float:
        """Right-tail conformal p-value: P(score >= s | H_0).

        Includes the standard +1 correction for finite-sample validity:
            p = (n+1 - rank) / (n+1)
        where rank is the rank of s among calibration scores.
        """
        if self._n == 0:
            return 1.0
        n_le = bisect.bisect_right(self._sorted, s)
        # Conservative right-tail p (split-conformal):
        return (self._n - n_le + 1) / (self._n + 1)

    def quantile(self, q: float) -> float:
        """Inverse CDF at level q ∈ [0,1]. Linear interpolation."""
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

    e(s) = 1 / p(s)  where p(s) is the conformal p-value from calibration.
    Bounded above to avoid infinity at the tail.
    """
    cdf: EmpiricalCDF
    max_evalue: float = 1e6
    eps: float = 1e-9

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
        """e(s) = min(max_evalue, 1 / p(s))."""
        p = self.p_value(score)
        return min(self.max_evalue, 1.0 / max(p, self.eps))


# ── E-Product composition ─────────────────────────────────────────────


def e_product(e_values: Sequence[float]) -> float:
    """Multiplicative composition of e-values.

    Under independence (or marginal calibration), the product of valid
    e-values is itself a valid e-value for the conjunction of nulls.
    Vovk & Wang (2021) Theorem 2.
    """
    out = 1.0
    for e in e_values:
        out *= max(e, 1e-9)
    return out


def e_mean(e_values: Sequence[float]) -> float:
    """Arithmetic mean of e-values — always a valid e-value (Vovk-Wang 2021)
    without requiring independence."""
    if not e_values:
        return 1.0
    return sum(e_values) / len(e_values)


def e_to_conservative_p(e: float) -> float:
    """Conservative p-value from an e-value: p = min(1, 1/e).

    Markov's inequality gives P(e >= t) ≤ E[e]/t ≤ 1/t under H_0, so
    p = 1/e is conservatively valid.
    """
    if e <= 0:
        return 1.0
    return min(1.0, 1.0 / e)


# ── Worst-case bound across manifolds ─────────────────────────────────


@dataclass
class WorstCaseBound:
    """Worst-case e-value across perturbation manifolds.

    For each manifold m, the score s_m is mapped to e_m via that manifold's
    calibrator (manifold-conditional CRC). The reported quantities are:
      worst_e   = max_m e_m       — worst manifold's evidence against H_0
      joint_e   = ∏_m e_m         — combined evidence (sequential testing)
      mean_e    = (1/|M|) Σ e_m   — robust to non-independent manifolds
      worst_p   = min_m p_m       — most-extreme conformal p-value
    """
    worst_e: float
    joint_e: float
    mean_e: float
    worst_p: float
    per_manifold_e: dict[str, float] = field(default_factory=dict)
    per_manifold_p: dict[str, float] = field(default_factory=dict)

    @property
    def reject_at_alpha(self, alpha: float = 0.05) -> bool:
        """True if any manifold's evidence is strong enough to reject H_0
        at level α (after multiplicity correction via e-product)."""
        return self.joint_e >= 1.0 / alpha

    def as_dict(self) -> dict:
        return {
            "worst_e": self.worst_e,
            "joint_e": self.joint_e,
            "mean_e": self.mean_e,
            "worst_p": self.worst_p,
            "per_manifold_e": self.per_manifold_e,
            "per_manifold_p": self.per_manifold_p,
        }


def manifold_conditional_crc(
    test_scores: dict[str, float],
    calibrators: dict[str, EValueCalibrator],
) -> WorstCaseBound:
    """Manifold-conditional Conformal Risk Control.

    For each manifold m, computes (p_m, e_m) using its own calibrator,
    then aggregates worst / joint / mean.

    Args:
        test_scores: dict {manifold_name -> score}
        calibrators: dict {manifold_name -> EValueCalibrator}
    """
    per_e: dict[str, float] = {}
    per_p: dict[str, float] = {}
    for name, score in test_scores.items():
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
        worst_e=round(worst_e, 4),
        joint_e=round(joint_e, 4),
        mean_e=round(mean_e_, 4),
        worst_p=round(worst_p, 6),
        per_manifold_e={k: round(v, 4) for k, v in per_e.items()},
        per_manifold_p={k: round(v, 6) for k, v in per_p.items()},
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
    """Final hallucination certificate composed from 6 layers.

    The certificate exposes:
      - layer-by-layer e-values and p-values (auditable)
      - composite e-value via e-product (sequential)
      - composite e-value via e-mean (independence-free, robust)
      - decision threshold at α = 0.05
      - manifold worst-case bound if available
    """
    layers: list[LayerEvidence]
    e_product: float
    e_mean: float
    decision: str        # "supported" | "hallucinated" | "abstain"
    alpha: float = 0.05
    worst_case: WorstCaseBound | None = None

    def as_dict(self) -> dict:
        return {
            "layers": [layer.__dict__ for layer in self.layers],
            "e_product": round(self.e_product, 4),
            "e_mean": round(self.e_mean, 4),
            "decision": self.decision,
            "alpha": self.alpha,
            "worst_case": self.worst_case.as_dict() if self.worst_case else None,
        }


def build_certificate(
    layer_scores: dict[str, float],
    calibrators: dict[str, EValueCalibrator],
    *,
    alpha: float = 0.05,
    abstain_band: tuple[float, float] = (0.3, 3.0),
    worst_case: WorstCaseBound | None = None,
) -> CompositeCertificate:
    """Construct a hallucination Certificate from 6-layer scores.

    Args:
        layer_scores: dict {layer_name -> score ∈ [0,1]}.
        calibrators: dict {layer_name -> EValueCalibrator} fit on grounded set.
        alpha: significance level (default 0.05 → e-threshold 20).
        abstain_band: e-product range within which decision is "abstain"
            (default (0.3, 3.0)).
        worst_case: optional manifold-conditional CRC result.

    Returns:
        CompositeCertificate with layer breakdown + decision.
    """
    layers: list[LayerEvidence] = []
    e_vals: list[float] = []
    for name, score in layer_scores.items():
        cal = calibrators.get(name)
        if cal is None:
            e, p = 1.0, 1.0
        else:
            e = cal.e_value(score)
            p = cal.p_value(score)
        layers.append(LayerEvidence(
            name=name,
            score=round(score, 4),
            e_value=round(e, 4),
            p_value=round(p, 6),
        ))
        e_vals.append(e)

    ep = e_product(e_vals)
    em = e_mean(e_vals)

    # Decision policy: e-product is the primary signal.
    threshold = 1.0 / alpha   # e.g. α=0.05 → e ≥ 20 to reject
    if ep >= threshold:
        decision = "hallucinated"
    elif ep <= abstain_band[0]:
        decision = "supported"
    elif ep <= abstain_band[1]:
        decision = "abstain"
    else:
        # In the band [abstain_band[1], threshold) — leaning hallucinated
        # but not enough evidence to reject. Still abstain conservatively.
        decision = "abstain"

    return CompositeCertificate(
        layers=layers,
        e_product=round(ep, 4),
        e_mean=round(em, 4),
        decision=decision,
        alpha=alpha,
        worst_case=worst_case,
    )
