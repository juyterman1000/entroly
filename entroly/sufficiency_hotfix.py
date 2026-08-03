"""Named calibration and fail-closed verdicts for sufficiency diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from . import sufficiency_legacy as _legacy

_EPS = 1e-9


@dataclass(frozen=True)
class CalibrationPolicy:
    """Workload-validated thresholds that may authorize ``sufficient``."""

    calibration_id: str
    residual_risk_limit: float
    minimum_query_coverage: float = 0.5
    maximum_boundary_exposure: float = 0.0
    require_boundary_measurement: bool = True
    allow_empty_query: bool = False

    def __post_init__(self) -> None:
        if not self.calibration_id.strip():
            raise ValueError("calibration_id must be non-empty")
        if not math.isfinite(self.residual_risk_limit) or self.residual_risk_limit < 0:
            raise ValueError("residual_risk_limit must be finite and non-negative")
        if not 0.0 <= self.minimum_query_coverage <= 1.0:
            raise ValueError("minimum_query_coverage must be between 0 and 1")
        if not 0.0 <= self.maximum_boundary_exposure <= 1.0:
            raise ValueError("maximum_boundary_exposure must be between 0 and 1")


@dataclass(frozen=True)
class SufficiencyCertificate:
    captured_mass: float
    shadow_price: float
    cutoff_ambiguity: float
    boundary_exposure: float
    query_coverage: float
    corpus_gap: float
    verdict: str
    reasons: tuple[str, ...] = field(default=())
    calibrated: bool = False
    calibration_id: str | None = None
    boundary_exposure_measured: bool = True

    @property
    def sufficient(self) -> bool:
        return (
            self.calibrated
            and bool(self.calibration_id)
            and self.verdict == "sufficient"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured_mass": round(self.captured_mass, 4),
            "shadow_price": round(self.shadow_price, 6),
            "cutoff_ambiguity": round(self.cutoff_ambiguity, 4),
            "boundary_exposure": round(self.boundary_exposure, 4),
            "boundary_exposure_measured": self.boundary_exposure_measured,
            "query_coverage": round(self.query_coverage, 4),
            "corpus_gap": round(self.corpus_gap, 4),
            "residual_risk": round(
                self.shadow_price / max(self.captured_mass, _EPS), 6
            ),
            "verdict": self.verdict,
            "calibrated": self.calibrated,
            "calibration_id": self.calibration_id,
            "reasons": list(self.reasons),
        }


def certify(
    candidates: Sequence[_legacy.Candidate],
    *,
    query_term_idf: dict[str, float] | None = None,
    retained_terms: Iterable[str] = (),
    unattainable_terms: Iterable[str] = (),
    anchor_weights: dict[str, float] | None = None,
    budget_exhausted: bool = True,
    shadow_price_limit: float = 0.0,
    calibrated: bool = False,
    calibration: CalibrationPolicy | None = None,
    boundary_exposure_measured: bool = True,
) -> SufficiencyCertificate:
    """Compute diagnostics; certify only under an explicit named policy.

    ``calibrated=True`` is accepted for source compatibility but cannot turn
    provisional repository constants into an assurance claim.
    """

    del shadow_price_limit
    term_idf = query_term_idf or {}
    captured = _legacy.captured_mass(candidates)
    shadow = _legacy.shadow_price(candidates)
    ambiguity = _legacy.cutoff_ambiguity(candidates)
    exposure = _legacy.boundary_exposure(candidates, anchor_weights)
    coverage = _legacy.query_coverage(
        retained_terms,
        term_idf,
        unattainable_terms,
    )
    gap = _legacy.corpus_gap(term_idf, unattainable_terms)
    residual_risk = shadow / max(captured, _EPS)

    residual_limit = (
        calibration.residual_risk_limit
        if calibration is not None
        else _legacy.RESIDUAL_RISK_LIMIT
    )
    coverage_limit = (
        calibration.minimum_query_coverage if calibration is not None else 0.5
    )
    exposure_limit = (
        calibration.maximum_boundary_exposure if calibration is not None else 0.0
    )

    reasons: list[str] = []
    if budget_exhausted and residual_risk > residual_limit:
        reasons.append(
            f"residual demand {residual_risk:.4f} per unit captured "
            f"(lambda={shadow:.5f}, captured={captured:.3f})"
        )
    if boundary_exposure_measured:
        if exposure > exposure_limit:
            reasons.append(
                f"{exposure:.0%} of anchor weight lost its required neighbouring context"
            )
    elif calibration is not None and calibration.require_boundary_measurement:
        reasons.append("boundary exposure was not measured for this selection")
    if coverage < coverage_limit and term_idf:
        reasons.append(f"query coverage {coverage:.0%}: discriminative terms dropped")

    if gap > 0.0:
        verdict = "expand_required"
        reasons.insert(
            0,
            f"{gap:.0%} of discriminative query terms appears in no candidate: "
            "the evidence was never retrieved",
        )
    elif reasons:
        verdict = "degraded"
    elif calibration is not None:
        if term_idf or calibration.allow_empty_query:
            verdict = "sufficient"
        else:
            verdict = "uncalibrated"
            reasons.append(
                "the calibration policy does not authorize empty-query sufficiency"
            )
    else:
        verdict = "uncalibrated"
        reasons.append(
            "calibrated=True has no effect without a named CalibrationPolicy"
            if calibrated
            else "no measured gap was detected, but production sufficiency "
            "thresholds are not calibrated on a held-out workload"
        )

    return SufficiencyCertificate(
        captured_mass=captured,
        shadow_price=shadow,
        cutoff_ambiguity=ambiguity,
        boundary_exposure=exposure,
        boundary_exposure_measured=boundary_exposure_measured,
        query_coverage=coverage,
        corpus_gap=gap,
        verdict=verdict,
        reasons=tuple(reasons),
        calibrated=calibration is not None,
        calibration_id=(calibration.calibration_id if calibration else None),
    )


def certify_selection(
    all_fragments,
    selected_fragments,
    query,
    *,
    token_budget,
    shadow_price_limit=None,
) -> SufficiencyCertificate:
    """Refuse to invent optimizer residue absent from these public inputs."""

    del (
        all_fragments,
        selected_fragments,
        query,
        token_budget,
        shadow_price_limit,
    )
    raise RuntimeError(
        "certify_selection cannot reconstruct optimizer residual state; use the "
        "certificate attached by entroly.qccr.select"
    )
