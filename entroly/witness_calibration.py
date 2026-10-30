"""Optional WITNESS calibration with class-conditional *marginal* scope.

For a fixed scorer/workload and exchangeable hallucination-labelled scores,
rank floor((n+1)*(1-alpha)) bounds the strict lower-tail escape probability by
1-alpha, averaging over calibration and future data. Alpha is an escalation
target, not confidence or a direct false-pass budget. Ties are conservative.
No positive-class evidence yields unavailable, all-SUPPRESS thresholds.

The upper-coverage helper conformal_quantile has the opposite inequality and
is deliberately not used to choose PASS thresholds. Neither procedure proves
label correctness, exchangeability, optimal retention, per-request safety,
semantic correctness or a high-confidence conditional population-risk bound.
See docs/witness-calibration.md for assumptions, action events and compatibility.
Heuristic defaults for genuinely missing profiles are unchanged and uncalibrated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import time
import zlib
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

from .local_file_lock import exclusive_file_lock

logger = logging.getLogger("entroly.witness_calibration")

CALIBRATION_METHOD = "exchangeable-hallucination-lower-tail-v2"


def _unit_value(value: float, name: str, *, probability: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
        or not math.isfinite(value)
        or (probability and value in (0, 1))
    ):
        interval = "(0,1)" if probability else "[0,1]"
        raise ValueError(f"{name} must be a finite number in {interval}")


def _profile_key(profile: str) -> str:
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("profile must be a nonempty string")
    key = profile.strip().lower().replace("-", "_")
    return "summary" if key == "summarization" else key


def _quantile_rank(n: int, alpha: float, *, upper: bool) -> int:
    # Treat the public probability's decimal representation exactly. Float
    # subtraction can otherwise move a boundary by an entire order statistic.
    rank = (n + 1) * (1 - Fraction(str(alpha)))
    return math.ceil(rank) if upper else math.floor(rank)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    decoded: dict[str, object] = {}
    for key, value in pairs:
        if key in decoded:
            raise ValueError("duplicate calibration JSON key")
        decoded[key] = value
    return decoded


# ── Action enum ──────────────────────────────────────────────────────


class Action:
    """4-action graduated policy. Stringly-typed for JSON-friendly logs."""
    PASS = "pass"           # ρ small — let it through unchanged
    HEDGE = "hedge"         # ρ moderate — surface as "may not be fully verified"
    WARN = "warn"           # ρ high — visible warning, claim stays visible
    SUPPRESS = "suppress"   # ρ very high — claim removed from output

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return (cls.PASS, cls.HEDGE, cls.WARN, cls.SUPPRESS)

    @classmethod
    def is_visible_change(cls, action: str) -> bool:
        return action in (cls.HEDGE, cls.WARN, cls.SUPPRESS)


# ── Threshold representation ─────────────────────────────────────────


@dataclass
class ThresholdSet:
    """Three thresholds defining a 4-action policy on ρ ∈ [0,1].

    Always τ_pass ≤ τ_hedge ≤ τ_warn ≤ 1.

    Action map:
        ρ < τ_pass   →  PASS
        ρ < τ_hedge  →  HEDGE
        ρ < τ_warn   →  WARN
        otherwise    →  SUPPRESS
    """
    tau_pass: float
    tau_hedge: float
    tau_warn: float

    def __post_init__(self) -> None:
        for name in ("tau_pass", "tau_hedge", "tau_warn"):
            _unit_value(getattr(self, name), name)
        # Auto-sort if caller passed them out of order.
        ts = sorted([self.tau_pass, self.tau_hedge, self.tau_warn])
        self.tau_pass, self.tau_hedge, self.tau_warn = ts

    def action(self, rho: float) -> str:
        _unit_value(rho, "rho")
        if rho < self.tau_pass:
            return Action.PASS
        if rho < self.tau_hedge:
            return Action.HEDGE
        if rho < self.tau_warn:
            return Action.WARN
        return Action.SUPPRESS

    def as_dict(self) -> dict:
        return asdict(self)


# Conservative defaults (used when no calibration data is available).
# These values were chosen so that:
#   PASS:     low risk (well-supported claims)
#   HEDGE:    paraphrased but plausible
#   WARN:     unsupported but not contradicted
#   SUPPRESS: high contradiction or unsupported in high-stakes domains
DEFAULT_THRESHOLDS = {
    # Profile           PASS  HEDGE  WARN
    "benchmark_qa":    (0.35, 0.55, 0.80),  # was binary at risk≥0.55
    "qa":              (0.35, 0.55, 0.80),
    "rag":             (0.35, 0.55, 0.80),
    "code":            (0.30, 0.50, 0.75),
    "summary":         (0.40, 0.60, 0.78),  # tighter than chat, looser than qa
    "dialogue":        (0.45, 0.65, 0.82),
    "chat":            (0.45, 0.65, 0.85),
    "auto":            (0.40, 0.60, 0.80),
}


def default_thresholds(profile: str) -> ThresholdSet:
    """Risk-graduated defaults for each workload profile.

    Replace the binary {pass, suppress} logic in apply_witness_policy
    with a 4-action graduated decision tied to ρ.
    """
    key = (profile or "auto").lower().replace("-", "_")
    if key == "summarization":
        key = "summary"
    tup = DEFAULT_THRESHOLDS.get(key, DEFAULT_THRESHOLDS["auto"])
    return ThresholdSet(tau_pass=tup[0], tau_hedge=tup[1], tau_warn=tup[2])


# ── Calibration data ─────────────────────────────────────────────────


@dataclass
class CalibrationSample:
    """One labeled risk observation. Used to compute conformal quantiles."""
    rho: float                 # risk score the analyzer assigned
    y: int                     # ground truth: 1 if claim was hallucinated, 0 if safe
    profile: str = "auto"      # workload profile context (for stratification)

    def __post_init__(self) -> None:
        _unit_value(self.rho, "rho")
        if type(self.y) is not int or self.y not in (0, 1):
            raise ValueError("y must be the integer 0 or 1")
        self.profile = _profile_key(self.profile)


@dataclass
class CalibrationResult:
    """Caller-attested cohort summary, not an authenticated risk certificate."""
    profile: str
    n_samples: int
    n_hallucinated: int
    n_safe: int
    alpha_pass: float
    alpha_hedge: float
    alpha_warn: float
    tau_pass: float
    tau_hedge: float
    tau_warn: float
    empirical_exposure_at_warn: float | None   # hallucination rate below τ_warn
    empirical_retention_at_pass: float | None  # safe-class rate below τ_pass
    dataset_hash: str
    dataset_crc32: str = ""
    timestamp: float = field(default_factory=time.time)
    calibration_method: str = "legacy_unverified"
    calibration_status: str = "unavailable_legacy"

    def __post_init__(self) -> None:
        self.profile = _profile_key(self.profile)
        for name in ("n_samples", "n_hallucinated", "n_safe"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.n_samples != self.n_hallucinated + self.n_safe:
            raise ValueError("sample class counts must sum to n_samples")
        for name in ("alpha_pass", "alpha_hedge", "alpha_warn"):
            _unit_value(getattr(self, name), name, probability=True)
        if not self.alpha_warn < self.alpha_hedge < self.alpha_pass:
            raise ValueError("alpha ordering must be alpha_warn < alpha_hedge < alpha_pass")
        for name in ("tau_pass", "tau_hedge", "tau_warn"):
            _unit_value(getattr(self, name), name)
        if not self.tau_pass <= self.tau_hedge <= self.tau_warn:
            raise ValueError("calibration thresholds must be ordered")
        for name in ("empirical_exposure_at_warn", "empirical_retention_at_pass"):
            value = getattr(self, name)
            if value is not None:
                _unit_value(value, name)
        if not isinstance(self.dataset_hash, str) or not self.dataset_hash:
            raise ValueError("dataset_hash must be a nonempty string")
        if not isinstance(self.dataset_crc32, str):
            raise ValueError("dataset_crc32 must be a string")
        try:
            finite_timestamp = math.isfinite(self.timestamp)
        except (TypeError, OverflowError):
            finite_timestamp = False
        if (isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, float))
                or not finite_timestamp or self.timestamp < 0):
            raise ValueError("timestamp must be finite and nonnegative")
        if not isinstance(self.calibration_method, str) or not isinstance(self.calibration_status, str):
            raise ValueError("calibration method and status must be strings")
        if self.calibration_method == CALIBRATION_METHOD:
            by_rank: dict[int, float] = {}
            for tau, alpha in ((self.tau_pass, self.alpha_pass),
                               (self.tau_hedge, self.alpha_hedge),
                               (self.tau_warn, self.alpha_warn)):
                rank = _quantile_rank(self.n_hallucinated, alpha, upper=False)
                if rank == 0 and tau != 0:
                    raise ValueError("rank-zero calibration threshold must be zero")
                if rank in by_rank and by_rank[rank] != tau:
                    raise ValueError("equal calibration ranks must have equal thresholds")
                by_rank[rank] = tau
            if not self.n_hallucinated:
                if (self.calibration_status != "unavailable_no_hallucinations"
                        or any((self.tau_pass, self.tau_hedge, self.tau_warn))
                        or self.empirical_exposure_at_warn is not None):
                    raise ValueError("no positive-class evidence requires unavailable suppression")
            elif (self.calibration_status != "calibrated_marginal"
                  or self.empirical_exposure_at_warn is None):
                raise ValueError("positive-class calibration requires marginal status and measured rate")
            if not self.n_safe and self.empirical_retention_at_pass is not None:
                raise ValueError("no safe-class evidence requires an unavailable retention rate")

    def as_threshold_set(self) -> ThresholdSet:
        try:
            self.__post_init__()
        except (TypeError, ValueError):
            return ThresholdSet(0.0, 0.0, 0.0)
        if (self.calibration_method != CALIBRATION_METHOD
                or self.calibration_status != "calibrated_marginal"
                or not self.n_hallucinated):
            return ThresholdSet(0.0, 0.0, 0.0)
        return ThresholdSet(
            tau_pass=self.tau_pass,
            tau_hedge=self.tau_hedge,
            tau_warn=self.tau_warn,
        )


def conformal_quantile(values: list[float], alpha: float) -> float:
    """Compute the split-conformal quantile.

    For n exchangeable bounded scores and level α, the conformal quantile is the
        ⌈(n+1)(1−α)⌉ / n
    empirical quantile of the sample. This adjustment ensures the
    finite-sample coverage guarantee

        Pr[future_score ≤ τ] ≥ 1 − α

    holds by exchangeability of (calibration + future) scores.

    Rank n+1 (or empty data) returns the known upper bound 1.0. This is
    trivial coverage, NOT evidence that a policy should PASS a claim.
    Alpha's decimal representation determines the exact rank.
    """
    _unit_value(alpha, "alpha", probability=True)
    for value in values:
        _unit_value(value, "score")
    if not values:
        return 1.0
    n = len(values)
    sorted_v = sorted(values)
    rank = _quantile_rank(n, alpha, upper=True)
    if rank > n:
        return 1.0
    return sorted_v[rank - 1]


def _exposure_threshold(sorted_scores: list[float], alpha: float) -> float:
    """Bound the strict lower-tail event, not upper-coverage miscoverage."""
    rank = _quantile_rank(len(sorted_scores), alpha, upper=False)
    return sorted_scores[rank - 1] if rank else 0.0


def calibrate(
    samples: list[CalibrationSample],
    profile: str = "auto",
    alpha_pass: float = 0.80,
    alpha_hedge: float = 0.50,
    alpha_warn: float = 0.20,
) -> CalibrationResult:
    """Choose conservative lower-tail thresholds for marginal escalation.

    Args:
        samples: Labeled (ρ, y) observations from a held-out set.
        profile: The workload profile these samples represent.
        alpha_pass, alpha_hedge, alpha_warn: Marginal escalation targets,
            in decreasing order. Examples:
                alpha_pass=0.80 → only the bottom 20% of hallucination
                                  scores escape to "pass" — i.e. 80% of
                                  true hallucinations are caught at the
                                  HEDGE or stricter level.
                alpha_hedge=0.50 → at least 50% reach WARN or SUPPRESS.
                alpha_warn=0.20 → at least 20% reach SUPPRESS.

        Constraints:
            alpha_warn < alpha_hedge < alpha_pass

    Returns:
        CalibrationResult with the three τ thresholds and empirical
        validation metrics.

    Raises:
        ValueError for invalid scores, labels, profiles, probabilities or order.

    Assumes a fixed scorer, policy and caller-attested workload, with conditional
    exchangeability within the hallucinated class. This is marginal over data,
    not conditional on the observed calibration set or an individual request.
    No hallucination samples means unavailable, all-SUPPRESS thresholds.
    """
    profile = _profile_key(profile)
    for name, value in (("alpha_pass", alpha_pass), ("alpha_hedge", alpha_hedge), ("alpha_warn", alpha_warn)):
        _unit_value(value, name, probability=True)
    if not (alpha_warn < alpha_hedge < alpha_pass):
        raise ValueError(
            f"α ordering must be α_warn < α_hedge < α_pass, "
            f"got {alpha_warn} < {alpha_hedge} < {alpha_pass}"
        )

    for sample in samples:
        if not isinstance(sample, CalibrationSample):
            raise ValueError("samples must contain CalibrationSample observations")
        sample.__post_init__()
        if sample.profile not in ("auto", profile):
            raise ValueError("sample profile differs from the declared calibration cohort")

    halu_scores = sorted(s.rho for s in samples if s.y == 1)
    safe_scores = [s.rho for s in samples if s.y == 0]

    tau_pass = _exposure_threshold(halu_scores, alpha_pass)
    tau_hedge = _exposure_threshold(halu_scores, alpha_hedge)
    tau_warn = _exposure_threshold(halu_scores, alpha_warn)

    # Empirical validation on calibration set.
    n_below_warn_halu = sum(1 for s in samples if s.y == 1 and s.rho < tau_warn)
    emp_exposure_at_warn = (
        round(n_below_warn_halu / len(halu_scores), 4) if halu_scores else None
    )
    n_below_pass_safe = sum(1 for s in samples if s.y == 0 and s.rho < tau_pass)
    emp_retention_at_pass = (
        round(n_below_pass_safe / len(safe_scores), 4) if safe_scores else None
    )

    # Hash for audit log.
    h = hashlib.sha256()
    crc = 0
    for s in samples:
        line = json.dumps(
            {"rho": s.rho, "y": s.y, "profile": s.profile},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode() + b"\n"
        h.update(line)
        crc = zlib.crc32(line, crc)
    dataset_hash = h.hexdigest()[:16]
    dataset_crc32 = f"{crc & 0xFFFFFFFF:08x}"

    return CalibrationResult(
        profile=profile,
        n_samples=len(samples),
        n_hallucinated=len(halu_scores),
        n_safe=len(safe_scores),
        alpha_pass=alpha_pass,
        alpha_hedge=alpha_hedge,
        alpha_warn=alpha_warn,
        tau_pass=tau_pass,
        tau_hedge=tau_hedge,
        tau_warn=tau_warn,
        empirical_exposure_at_warn=emp_exposure_at_warn,
        empirical_retention_at_pass=emp_retention_at_pass,
        dataset_hash=dataset_hash,
        dataset_crc32=dataset_crc32,
        calibration_method=CALIBRATION_METHOD,
        calibration_status="calibrated_marginal" if halu_scores else "unavailable_no_hallucinations",
    )


# ── Persistence ──────────────────────────────────────────────────────


class CalibrationStore:
    """Per-profile threshold store with JSON persistence and audit log."""

    def __init__(self, path: str | Path):
        self._path = Path(path).resolve()
        self._audit_path = self._path.with_suffix(self._path.suffix + ".audit.jsonl")
        self._results: dict[str, CalibrationResult] = {}
        self._invalid_profiles: set[str] = set()
        self._load_failed = False
        self._load()

    def _load(self) -> None:
        # Refresh from authoritative disk state. Writers replace complete files;
        # a noncooperating partial/corrupt write is unavailable, not missing.
        self._results = {}
        self._invalid_profiles = set()
        self._load_failed = False
        try:
            blob = json.loads(self._path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
        except FileNotFoundError:
            return
        except (OSError, UnicodeError, ValueError) as e:
            logger.warning("witness calibration load failed: %s", e)
            self._load_failed = True
            return
        if not isinstance(blob, dict):
            self._load_failed = True
            return
        for profile, d in blob.items():
            try:
                key = _profile_key(profile)
            except ValueError:
                self._load_failed = True
                continue
            try:
                result = CalibrationResult(**d)
                if (result.profile != key or key in self._results
                        or key in self._invalid_profiles):
                    raise ValueError("stored profile identity is ambiguous")
                self._results[key] = result
            except (TypeError, ValueError):
                self._invalid_profiles.add(key)
                self._results.pop(key, None)
                continue

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._path.parent,
                prefix=f".{self._path.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump({p: asdict(r) for p, r in self._results.items()}, stream,
                          indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self._path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _audit(self, event: dict) -> None:
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def get_thresholds(self, profile: str) -> ThresholdSet:
        """Missing profiles use heuristics; explicit unusable data fails closed."""
        key = _profile_key(profile or "auto")
        self._load()
        if self._load_failed or key in self._invalid_profiles:
            return ThresholdSet(0.0, 0.0, 0.0)
        result = self._results.get(key)
        if result is None:
            return default_thresholds(key)
        return result.as_threshold_set()

    def update(self, result: CalibrationResult) -> None:
        result.__post_init__()
        self._load()
        if self._load_failed or self._invalid_profiles:
            raise ValueError("refusing to overwrite an unreadable or corrupt calibration store")
        lock = self._path.with_suffix(self._path.suffix + ".lock")
        with exclusive_file_lock(lock):
            self._load()
            if self._load_failed or self._invalid_profiles:
                raise ValueError("refusing to overwrite an unreadable or corrupt calibration store")
            self._results[result.profile] = result
            self._save()
            self._audit({
                "ts": result.timestamp,
                "profile": result.profile,
                "n_samples": result.n_samples,
                "tau_pass": result.tau_pass,
                "tau_hedge": result.tau_hedge,
                "tau_warn": result.tau_warn,
                "empirical_exposure": result.empirical_exposure_at_warn,
                "empirical_retention": result.empirical_retention_at_pass,
                "dataset_hash": result.dataset_hash,
                "dataset_crc32": result.dataset_crc32,
                "calibration_method": result.calibration_method,
                "calibration_status": result.calibration_status,
            })

    def all_profiles(self) -> dict[str, CalibrationResult]:
        self._load()
        if self._load_failed:
            return {}
        return {key: result for key, result in self._results.items()
                if key not in self._invalid_profiles}
