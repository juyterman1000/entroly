"""Calibration and disjoint-holdout validation for selective compression risk.

Candidate-unit evidence may be promoted to semantic scope only after two
separate gates:

1. a calibration split chooses a threshold using atomic residual-risk groups;
2. a disjoint holdout split independently satisfies the declared selective
   failure-rate and coverage contract.

The same residual-risk value is never split by sample ordering. This avoids a
threshold that was fitted on only the favorable members of a tied score group
but later accepts every member of that group in production.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from itertools import groupby
from typing import Any, Iterable, Mapping, Sequence

from .sufficiency_contract import SufficiencyCertificate

_ALLOWED_SPLITS = {"calibration", "holdout"}


@dataclass(frozen=True)
class CalibrationObservation:
    sample_id: str
    dataset: str
    model: str
    split: str
    residual_risk: float
    failed: bool

    def __post_init__(self) -> None:
        if not self.sample_id or not self.dataset or not self.model:
            raise ValueError("sample_id, dataset, and model are required")
        if self.split not in _ALLOWED_SPLITS:
            raise ValueError("split must be 'calibration' or 'holdout'")
        if self.residual_risk < 0 or not math.isfinite(self.residual_risk):
            raise ValueError("residual_risk must be finite and non-negative")

    @property
    def membership_hash(self) -> str:
        body = f"{self.dataset}\0{self.sample_id}".encode("utf-8")
        return hashlib.sha256(body).hexdigest()


@dataclass(frozen=True)
class CalibrationProfile:
    version: str
    threshold: float
    target_failure_rate: float
    accepted_samples: int
    accepted_failures: int
    failure_upper_bound: float
    total_samples: int
    dataset_count: int
    model_count: int
    accepted_dataset_count: int
    accepted_model_count: int
    dataset_fingerprint: str
    calibration_membership: tuple[str, ...]
    calibration_ready: bool
    semantic_validated: bool = False
    holdout_samples: int = 0
    holdout_accepted_samples: int = 0
    holdout_accepted_failures: int = 0
    holdout_failure_upper_bound: float = 1.0
    holdout_coverage: float = 0.0
    holdout_dataset_count: int = 0
    holdout_model_count: int = 0
    holdout_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["calibration_membership"] = list(
            self.calibration_membership
        )
        return payload


@dataclass(frozen=True)
class HoldoutReport:
    calibration_fingerprint: str
    threshold: float
    target_failure_rate: float
    holdout_fingerprint: str
    total_samples: int
    accepted_samples: int
    accepted_failures: int
    failure_upper_bound: float
    coverage: float
    dataset_count: int
    model_count: int
    disjoint: bool
    validated: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


def _wilson_upper(
    failures: int,
    total: int,
    z: float = 1.959963984540054,
) -> float:
    if total <= 0:
        return 1.0
    p = failures / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * (
            p * (1 - p) / total
            + z * z / (4 * total * total)
        )
        ** 0.5
        / denominator
    )
    return min(1.0, centre + radius)


def _fingerprint(
    observations: Iterable[CalibrationObservation],
) -> str:
    body = "\n".join(
        json.dumps(
            asdict(item),
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in sorted(
            observations,
            key=lambda obs: (
                obs.dataset,
                obs.sample_id,
                obs.model,
            ),
        )
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _validate_observations(
    observations: Sequence[CalibrationObservation],
    *,
    required_split: str,
) -> None:
    if required_split not in _ALLOWED_SPLITS:
        raise ValueError("unknown required split")
    wrong = [
        item.sample_id
        for item in observations
        if item.split != required_split
    ]
    if wrong:
        raise ValueError(
            f"{required_split} operation received observations "
            "from another split"
        )
    keys = [
        (item.dataset, item.sample_id)
        for item in observations
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "dataset/sample_id pairs must be unique within a split"
        )


def _accepted_group_stats(
    observations: Sequence[CalibrationObservation],
    threshold: float,
) -> tuple[int, int, int, int]:
    accepted = [
        item
        for item in observations
        if item.residual_risk <= threshold
    ]
    return (
        len(accepted),
        sum(item.failed for item in accepted),
        len({item.dataset for item in accepted}),
        len({item.model for item in accepted}),
    )


def risk_coverage_curve(
    observations: Sequence[CalibrationObservation],
) -> tuple[dict[str, Any], ...]:
    """Return one point per atomic residual-risk group."""
    ordered = sorted(
        observations,
        key=lambda item: (
            item.residual_risk,
            item.dataset,
            item.sample_id,
            item.model,
        ),
    )
    points: list[dict[str, Any]] = []
    accepted = 0
    failures = 0
    datasets: set[str] = set()
    models: set[str] = set()
    for threshold, group_iter in groupby(
        ordered,
        key=lambda item: item.residual_risk,
    ):
        group = list(group_iter)
        accepted += len(group)
        failures += sum(item.failed for item in group)
        datasets.update(item.dataset for item in group)
        models.update(item.model for item in group)
        points.append(
            {
                "threshold": threshold,
                "accepted_samples": accepted,
                "accepted_failures": failures,
                "failure_upper_bound": _wilson_upper(
                    failures,
                    accepted,
                ),
                "coverage": (
                    accepted / len(observations)
                    if observations
                    else 0.0
                ),
                "dataset_count": len(datasets),
                "model_count": len(models),
            }
        )
    return tuple(points)


def fit_profile(
    observations: Sequence[CalibrationObservation],
    *,
    target_failure_rate: float = 0.01,
    min_samples: int = 500,
    min_accepted: int = 100,
    min_datasets: int = 2,
    min_models: int = 2,
    version: str = "semantic-risk-v2",
) -> CalibrationProfile:
    if not 0 < target_failure_rate < 1:
        raise ValueError(
            "target_failure_rate must be in (0, 1)"
        )
    if min(
        min_samples,
        min_accepted,
        min_datasets,
        min_models,
    ) <= 0:
        raise ValueError("minimum gates must be positive")
    _validate_observations(
        observations,
        required_split="calibration",
    )

    best: dict[str, Any] | None = None
    for point in risk_coverage_curve(observations):
        if (
            point["accepted_samples"] >= min_accepted
            and point["dataset_count"] >= min_datasets
            and point["model_count"] >= min_models
            and point["failure_upper_bound"]
            <= target_failure_rate
        ):
            best = point

    if best is None:
        threshold = -1.0
        accepted = 0
        accepted_failures = 0
        accepted_datasets = 0
        accepted_models = 0
        upper = 1.0
    else:
        threshold = float(best["threshold"])
        accepted = int(best["accepted_samples"])
        accepted_failures = int(
            best["accepted_failures"]
        )
        accepted_datasets = int(best["dataset_count"])
        accepted_models = int(best["model_count"])
        upper = float(best["failure_upper_bound"])

    datasets = {item.dataset for item in observations}
    models = {item.model for item in observations}
    calibration_ready = (
        len(observations) >= min_samples
        and accepted >= min_accepted
        and accepted_datasets >= min_datasets
        and accepted_models >= min_models
        and upper <= target_failure_rate
    )
    return CalibrationProfile(
        version=version,
        threshold=threshold,
        target_failure_rate=target_failure_rate,
        accepted_samples=accepted,
        accepted_failures=accepted_failures,
        failure_upper_bound=upper,
        total_samples=len(observations),
        dataset_count=len(datasets),
        model_count=len(models),
        accepted_dataset_count=accepted_datasets,
        accepted_model_count=accepted_models,
        dataset_fingerprint=_fingerprint(observations),
        calibration_membership=tuple(
            sorted(
                item.membership_hash
                for item in observations
            )
        ),
        calibration_ready=calibration_ready,
        semantic_validated=False,
    )


def evaluate_holdout(
    profile: CalibrationProfile,
    observations: Sequence[CalibrationObservation],
    *,
    min_samples: int = 200,
    min_accepted: int = 100,
    min_coverage: float = 0.05,
    min_datasets: int = 2,
    min_models: int = 2,
) -> HoldoutReport:
    if min(
        min_samples,
        min_accepted,
        min_datasets,
        min_models,
    ) <= 0:
        raise ValueError("minimum gates must be positive")
    if not 0 <= min_coverage <= 1:
        raise ValueError(
            "min_coverage must be in [0, 1]"
        )
    _validate_observations(
        observations,
        required_split="holdout",
    )

    calibration_membership = set(
        profile.calibration_membership
    )
    holdout_membership = {
        item.membership_hash
        for item in observations
    }
    disjoint = calibration_membership.isdisjoint(
        holdout_membership
    )
    (
        accepted,
        failures,
        dataset_count,
        model_count,
    ) = _accepted_group_stats(
        observations,
        profile.threshold,
    )
    upper = _wilson_upper(failures, accepted)
    coverage = (
        accepted / len(observations)
        if observations
        else 0.0
    )
    reasons: list[str] = []
    if not profile.calibration_ready:
        reasons.append(
            "calibration profile has not satisfied calibration gates"
        )
    if not disjoint:
        reasons.append(
            "holdout overlaps calibration membership"
        )
    if len(observations) < min_samples:
        reasons.append(
            "holdout sample count is below the declared minimum"
        )
    if accepted < min_accepted:
        reasons.append(
            "accepted holdout sample count is below the declared minimum"
        )
    if coverage < min_coverage:
        reasons.append(
            "accepted holdout coverage is below the declared minimum"
        )
    if dataset_count < min_datasets:
        reasons.append(
            "accepted holdout dataset diversity is below the minimum"
        )
    if model_count < min_models:
        reasons.append(
            "accepted holdout model diversity is below the minimum"
        )
    if upper > profile.target_failure_rate:
        reasons.append(
            "holdout failure upper bound exceeds the declared target"
        )

    return HoldoutReport(
        calibration_fingerprint=(
            profile.dataset_fingerprint
        ),
        threshold=profile.threshold,
        target_failure_rate=(
            profile.target_failure_rate
        ),
        holdout_fingerprint=_fingerprint(observations),
        total_samples=len(observations),
        accepted_samples=accepted,
        accepted_failures=failures,
        failure_upper_bound=upper,
        coverage=coverage,
        dataset_count=dataset_count,
        model_count=model_count,
        disjoint=disjoint,
        validated=not reasons,
        reasons=tuple(reasons),
    )


def validate_profile(
    profile: CalibrationProfile,
    report: HoldoutReport,
) -> CalibrationProfile:
    if (
        report.calibration_fingerprint
        != profile.dataset_fingerprint
    ):
        raise ValueError(
            "holdout report was produced for a different calibration profile"
        )
    if (
        report.threshold != profile.threshold
        or report.target_failure_rate
        != profile.target_failure_rate
    ):
        raise ValueError(
            "holdout report threshold contract does not match the profile"
        )
    return replace(
        profile,
        semantic_validated=report.validated,
        holdout_samples=report.total_samples,
        holdout_accepted_samples=(
            report.accepted_samples
        ),
        holdout_accepted_failures=(
            report.accepted_failures
        ),
        holdout_failure_upper_bound=(
            report.failure_upper_bound
        ),
        holdout_coverage=report.coverage,
        holdout_dataset_count=report.dataset_count,
        holdout_model_count=report.model_count,
        holdout_fingerprint=report.holdout_fingerprint,
    )


def certify_with_profile(
    metrics: Mapping[str, Any],
    profile: CalibrationProfile,
) -> SufficiencyCertificate:
    reasons = [
        str(reason)
        for reason in metrics.get("reasons", [])
    ]
    structurally_safe = (
        bool(metrics.get("source_span_integrity"))
        and float(
            metrics.get("boundary_exposure", 1.0)
        )
        == 0.0
        and float(metrics.get("query_coverage", 0.0))
        >= 0.5
    )
    residual = float(
        metrics.get("residual_risk", float("inf"))
    )
    holdout_validated = (
        profile.semantic_validated
        and bool(profile.holdout_fingerprint)
        and profile.holdout_samples > 0
        and profile.holdout_accepted_samples > 0
        and profile.holdout_failure_upper_bound
        <= profile.target_failure_rate
    )
    if not holdout_validated:
        reasons.append(
            "calibration profile lacks successful disjoint holdout validation"
        )
        verdict, scope = "uncertain", "candidate_units"
    elif not structurally_safe:
        reasons.append(
            "structural evidence-integrity checks failed"
        )
        verdict, scope = "degraded", "semantic"
    elif residual > profile.threshold:
        reasons.append(
            f"residual risk {residual:.6f} exceeds calibrated threshold "
            f"{profile.threshold:.6f}"
        )
        verdict, scope = "uncertain", "semantic"
    else:
        verdict, scope = "sufficient", "semantic"
    payload = dict(metrics)
    payload.update(
        {
            "verdict": verdict,
            "scope": scope,
            "reasons": reasons,
            "calibration_version": profile.version,
            "dataset_fingerprint": (
                profile.dataset_fingerprint
            ),
            "calibration_target_failure_rate": (
                profile.target_failure_rate
            ),
            "calibration_failure_upper_bound": (
                profile.failure_upper_bound
            ),
            "holdout_fingerprint": (
                profile.holdout_fingerprint
            ),
            "holdout_failure_upper_bound": (
                profile.holdout_failure_upper_bound
            ),
            "holdout_coverage": (
                profile.holdout_coverage
            ),
        }
    )
    return SufficiencyCertificate.from_mapping(payload)


__all__ = [
    "CalibrationObservation",
    "CalibrationProfile",
    "HoldoutReport",
    "certify_with_profile",
    "evaluate_holdout",
    "fit_profile",
    "risk_coverage_curve",
    "validate_profile",
]
