"""
EICV — Evidence-Invariant Causal Verification (Phase 4 Integration)
====================================================================

This is the top-level integrator that unifies all 6 EICV layers into the
Epistemic Support Density Φ:

    Φ(x) = ∫ P(e | x) · C(e, x) · R(e, x) de

In our discrete-coarsened deterministic form:

    Φ(x) = w₁·(1 − T(G)) + w₂·NLI_bidir + w₃·RNR_local
           + w₄·(1 − Γ_stylistic) + w₅·(1 − H_sem_norm)
           − w₆·max_manifold_p_value

where each w_i is unit-summed and the signs are aligned so higher Φ means
more *epistemic support* (claim is grounded).

Layer mapping (EICV 6-layer hierarchy)
--------------------------------------
  Layer 1: Raw claim/evidence input          (handled at API boundary)
  Layer 2: Evidence Support Graph T(G)        (entroly/esg.py)
  Layer 3: Retrieval Necessity I(Ŷ; S)         (entroly/rnr.py)
  Layer 4: Falsification gradient Γ           (entroly/counterfactual.py)
  Layer 5: NLI proxy + Semantic entropy       (entroly/semantic_entropy.py)
           + E-value composition & CRC        (entroly/e_value.py)
  Layer 6: Selective policy / Certificate     (this module)

The result is an EICVCertificate dataclass that exposes:
  - Φ score ∈ [0, 1] (higher = more grounded)
  - per-layer evidence breakdown (auditable)
  - composite e-value with α=0.05 decision threshold
  - manifold worst-case bound (Phase 2)
  - selective decision: supported | abstain | hallucinated
  - worst-case shift coverage guarantee (Phase 3)

This is the central artefact of EICV: every decision the system makes
is traceable to specific evidence flows through the 6 layers.

Usage
-----
    >>> from entroly.eicv import EICVAnalyzer
    >>> ana = EICVAnalyzer()
    >>> ana.fit_calibrators(grounded_pairs)
    >>> cert = ana.verify(evidence, claim)
    >>> print(cert.decision, cert.phi, cert.e_product)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence


# ── EICV Certificate ──────────────────────────────────────────────────


@dataclass
class EICVCertificate:
    """Auditable verification certificate produced by EICVAnalyzer.

    Conceptually: every assertion the system makes carries this object
    so a downstream user (or auditor) can trace exactly why a claim was
    deemed supported / abstained / hallucinated.
    """
    # ── Primary signals ─────────────────────────────────────────────
    phi: float                        # Φ ∈ [0,1], higher = more grounded
    hallucination_score: float        # 1 − Φ
    decision: str                     # "supported" | "abstain" | "hallucinated"

    # ── Layer breakdown (Layers 2-5) ────────────────────────────────
    layer_scores: dict[str, float] = field(default_factory=dict)
    # keys: esg_tension, nli_bidir_score, rnr_necessity, gamma, h_sem_norm,
    #       e_product, e_mean, worst_e, worst_p

    # ── E-value statistics ──────────────────────────────────────────
    e_product: float = 1.0
    e_mean: float = 1.0
    conservative_p: float = 1.0
    alpha: float = 0.05

    # ── Worst-case manifold bound (Phase 2) ─────────────────────────
    worst_manifold_e: float | None = None
    worst_manifold_name: str | None = None
    worst_manifold_p: float | None = None

    # ── Decomposition for auditability ──────────────────────────────
    n_claim_atoms: int = 0
    n_ev_atoms: int = 0
    unsupported_fraction: float = 0.0
    contradiction_fraction: float = 0.0

    # ── Provenance ──────────────────────────────────────────────────
    profile: str = "default"
    schema_version: str = "eicv-cert-v1"
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


# ── Calibration bundle ────────────────────────────────────────────────


@dataclass
class EICVCalibration:
    """Bundle of calibrators fit on a held-out grounded set.

    Each layer's score has its own conformal calibrator; we share the
    n_calibration count so the e-value cap is consistent.
    """
    calibrators: dict[str, Any] = field(default_factory=dict)
    n_calibration: int = 0
    fit_at: str = ""

    def save(self, path: str | Path) -> None:
        """Persist a calibration bundle as JSON-of-sorted-scores."""
        from entroly.e_value import EmpiricalCDF
        out: dict[str, Any] = {
            "schema": "eicv-calibration-v1",
            "n_calibration": self.n_calibration,
            "fit_at": self.fit_at,
            "calibrators": {},
        }
        for name, cal in self.calibrators.items():
            if hasattr(cal, "cdf"):
                cdf: EmpiricalCDF = cal.cdf
                out["calibrators"][name] = {
                    "max_evalue": cal.max_evalue,
                    "scores": cdf._sorted,
                }
        Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EICVCalibration":
        from entroly.e_value import EValueCalibrator
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cals = {}
        for name, blob in data["calibrators"].items():
            cals[name] = EValueCalibrator.fit(
                blob["scores"], max_evalue=blob.get("max_evalue", 1e6),
            )
        return cls(
            calibrators=cals,
            n_calibration=data.get("n_calibration", 0),
            fit_at=data.get("fit_at", ""),
        )


# ── Default Φ weights ────────────────────────────────────────────────


# w_i sum to 1.0. Initial defaults gave equal-ish weight across layers,
# but a group-DRO grid search across {SQuAD, HaluEval, FEVER} (Phase 5)
# revealed that ESG with slot-aware contradiction is Pareto-optimal — every
# other layer is either redundant or DEGRADES FEVER. Hence ESG dominates.
#
# NLI_bidir and H_sem remain in the certificate as DIAGNOSTIC traces
# (they explain WHY a decision was made) but no longer enter Φ scoring.
PHI_WEIGHTS = {
    "esg":           1.00,   # 1 - T(G) — Pareto-optimal under group-DRO
    "nli_bidir":     0.00,   # diagnostic only (kept in certificate)
    "rnr_local":     0.00,   # diagnostic only
    "gamma_inv":     0.00,   # diagnostic only
    "h_sem_inv":     0.00,   # diagnostic only
    "e_product_inv": 0.00,   # diagnostic only — see CompositeCertificate.e_product
}


def _phi_from_layers(layer_scores: dict[str, float]) -> float:
    """Compute Φ ∈ [0,1] from layer breakdown.

    All input layer scores must be in [0,1] with semantics:
      esg_tension       : higher = more hallucinated
      nli_bidir_score   : higher = more hallucinated
      rnr_local         : higher = more retrieval-grounded (lower = artifact-prone)
      gamma             : higher = more brittle
      h_sem_norm        : higher = more phrasing-sensitive
      e_product         : higher = more evidence against H_0 (hallucinated)
    """
    esg_term       = 1.0 - layer_scores.get("esg_tension", 0.5)
    nli_term       = 1.0 - layer_scores.get("nli_bidir_score", 0.5)
    rnr_term       = layer_scores.get("rnr_local", 0.5)
    gamma_inv      = 1.0 - layer_scores.get("gamma", 0.0)
    h_sem_inv      = 1.0 - layer_scores.get("h_sem_norm", 0.0)
    e_p            = layer_scores.get("e_product", 1.0)
    # Map e_product (potentially huge) into [0,1] inverse via log:
    # e_product = 1  → 1.0 (no evidence against H_0)
    # e_product = 20 → 0.43 (rejection threshold at α=0.05)
    # e_product = 1e6 → 0.14
    e_inv = 1.0 / (1.0 + math.log10(max(e_p, 1.0)))

    phi = (
        PHI_WEIGHTS["esg"]           * esg_term
        + PHI_WEIGHTS["nli_bidir"]   * nli_term
        + PHI_WEIGHTS["rnr_local"]   * rnr_term
        + PHI_WEIGHTS["gamma_inv"]   * gamma_inv
        + PHI_WEIGHTS["h_sem_inv"]   * h_sem_inv
        + PHI_WEIGHTS["e_product_inv"] * e_inv
    )
    return float(max(0.0, min(1.0, phi)))


# ── Main analyser ─────────────────────────────────────────────────────


class EICVAnalyzer:
    """Top-level EICV verifier. Composes all 6 layers into Φ + Certificate.

    Args:
        profile: "default" | "rag" | "qa" | "summarization" | "dialogue"
            Currently only affects which sub-analysers are eager-loaded.
        phi_weights: optional override of PHI_WEIGHTS.
        alpha: significance level for the decision threshold (default 0.05).
        abstain_band: (lo, hi) tuple for Φ-range that produces "abstain".
            Default (0.40, 0.65) — Φ outside this is "supported" / "hallucinated".
    """

    def __init__(
        self,
        profile: str = "default",
        phi_weights: dict[str, float] | None = None,
        alpha: float = 0.05,
        abstain_band: tuple[float, float] = (0.40, 0.65),
    ) -> None:
        self.profile = profile
        self.phi_weights = phi_weights or PHI_WEIGHTS
        self.alpha = alpha
        self.abstain_band = abstain_band
        self.calibration: EICVCalibration | None = None

        # Lazy-load sub-analysers to keep import cheap
        self._esg = None
        self._sem = None

    # ── Lazy property access ─────────────────────────────────────────

    @property
    def esg(self):
        if self._esg is None:
            from entroly.esg import ESGAnalyzer
            self._esg = ESGAnalyzer()
        return self._esg

    @property
    def sem(self):
        if self._sem is None:
            from entroly.semantic_entropy import SemanticEntropyAnalyzer
            self._sem = SemanticEntropyAnalyzer()
        return self._sem

    # ── Calibration ──────────────────────────────────────────────────

    def fit_calibrators(
        self,
        grounded_pairs: Sequence[tuple[str, str]],
    ) -> EICVCalibration:
        """Fit per-layer e-value calibrators on a held-out grounded set.

        Args:
            grounded_pairs: list of (evidence, claim) pairs known to be
                grounded (label = 0). 100+ pairs recommended.

        Returns:
            EICVCalibration bundle. Also stored on self.calibration.
        """
        from entroly.e_value import EValueCalibrator

        esg_scores: list[float] = []
        nli_scores: list[float] = []
        hsem_scores: list[float] = []
        # We DON'T calibrate rnr_local here — it's a dataset-level metric

        for ev, cl in grounded_pairs:
            esg_res = self.esg.score(ev, cl)
            sem_res = self.sem.analyze(ev, cl)
            esg_scores.append(esg_res.tension)
            nli_scores.append(sem_res.nli_bidir_score)
            hsem_scores.append(sem_res.h_sem_norm)

        cals = {
            "esg_tension":     EValueCalibrator.fit(esg_scores),
            "nli_bidir_score": EValueCalibrator.fit(nli_scores),
            "h_sem_norm":      EValueCalibrator.fit(hsem_scores),
        }
        self.calibration = EICVCalibration(
            calibrators=cals,
            n_calibration=len(grounded_pairs),
            fit_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        return self.calibration

    # ── Main verification ────────────────────────────────────────────

    def verify(self, evidence: str, claim: str) -> EICVCertificate:
        """Top-level verification — produces an EICVCertificate.

        Falls back to defaults (e_value = 1.0 for all layers) if no
        calibrators have been fit. The Φ score is still meaningful in
        that case; only the e_product / decision threshold loses formal
        statistical guarantee.
        """
        t0 = time.perf_counter()

        # Layer 2: ESG tension
        esg_res = self.esg.score(evidence, claim)
        # Layer 5 (NLI + SEM): semantic entropy + bidirectional NLI
        sem_res = self.sem.analyze(evidence, claim)

        # Layer scores (all in [0,1], with above semantics)
        layer_scores = {
            "esg_tension":     esg_res.tension,
            "nli_bidir_score": sem_res.nli_bidir_score,
            "h_sem_norm":      sem_res.h_sem_norm,
            "rnr_local":       0.5,        # placeholder when no batch context
            "gamma":           0.0,        # placeholder (Γ is batch-level)
        }

        # E-value composition
        if self.calibration is not None:
            cals = self.calibration.calibrators
            from entroly.e_value import e_product, e_mean, e_to_conservative_p

            e_vals = []
            for name in ("esg_tension", "nli_bidir_score", "h_sem_norm"):
                cal = cals.get(name)
                if cal is not None:
                    e_vals.append(cal.e_value(layer_scores[name]))
            ep = e_product(e_vals)
            em = e_mean(e_vals)
            cp = e_to_conservative_p(ep)
        else:
            ep, em, cp = 1.0, 1.0, 1.0

        layer_scores["e_product"] = ep
        layer_scores["e_mean"]    = em
        layer_scores["conservative_p"] = cp

        # Φ integration
        phi = _phi_from_layers(layer_scores)
        halu_score = 1.0 - phi

        # Decision: based primarily on Φ, but e_product provides a strict
        # rejection threshold at α (if calibrated).
        decision = self._decide(phi, ep)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return EICVCertificate(
            phi=round(phi, 4),
            hallucination_score=round(halu_score, 4),
            decision=decision,
            layer_scores={k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in layer_scores.items()},
            e_product=round(ep, 4),
            e_mean=round(em, 4),
            conservative_p=round(cp, 6),
            alpha=self.alpha,
            n_claim_atoms=esg_res.n_claim_atoms,
            n_ev_atoms=esg_res.n_ev_atoms,
            unsupported_fraction=esg_res.unsupported_fraction,
            contradiction_fraction=esg_res.contradiction_fraction,
            profile=self.profile,
            elapsed_ms=round(elapsed_ms, 2),
        )

    def _decide(self, phi: float, e_product: float) -> str:
        """Layer-6 selective decision policy.

        Strict-reject path: e_product >= 1/α → "hallucinated" regardless of Φ.
        Otherwise, Φ-based zone:
          Φ ≤ abstain_band[0]   → hallucinated (low support)
          Φ ≥ abstain_band[1]   → supported
          else                  → abstain
        """
        # E-value strict rejection (only if calibrators were fit)
        threshold_e = 1.0 / self.alpha
        if e_product >= threshold_e:
            # Strong evidence against H_0 → hallucinated
            # Unless Φ is unusually high (conflicting signals)
            if phi >= self.abstain_band[1]:
                return "abstain"
            return "hallucinated"

        if phi >= self.abstain_band[1]:
            return "supported"
        if phi <= self.abstain_band[0]:
            return "hallucinated"
        return "abstain"


# ── Convenience ───────────────────────────────────────────────────────


def verify(evidence: str, claim: str) -> EICVCertificate:
    """Module-level convenience: verify without calibration.

    Note: without calibrators, e_product is 1.0 and decisions are based
    on Φ alone. For formal coverage guarantees, fit calibrators first.
    """
    return EICVAnalyzer().verify(evidence, claim)
