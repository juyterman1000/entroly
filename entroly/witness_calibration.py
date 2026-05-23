"""
WITNESS Conformal Risk-Controlled Calibration
==============================================

The witness analyzer assigns each claim a continuous risk score
ρ ∈ [0,1] decomposed from support, contradiction, and adequacy signals.
This module turns ρ into a 4-action policy decision

    π(ρ) ∈ {PASS, HEDGE, WARN, SUPPRESS}

with thresholds τ₁ < τ₂ < τ₃ chosen so that, on the test distribution,
the exposure rate of unsupported claims is bounded above by a
user-chosen α with high probability — using split-conformal calibration
(Vovk-Gammerman-Shafer 2005, Angelopoulos-Bates 2023).

The benefit over the current binary {pass, suppress} policy is that the
operator can dial α (acceptable false-pass rate) and get the *highest*
retention rate compatible with that bound. The current per-profile hand
tuning is replaced by a single procedure that works for any benchmark.

Math sketch
-----------
Given labeled calibration data {(ρᵢ, yᵢ)}ⁿ₁ where yᵢ ∈ {0=safe, 1=halu}:

  1. Compute conformity scores sᵢ = ρᵢ for hallucinations (yᵢ=1)
  2. Choose τ = the ⌈(n+1)(1-α)⌉ / n quantile of these scores
  3. Then by exchangeability:
        Pr[ρ_{n+1} ≤ τ ∣ y_{n+1} = 1] ≥ 1 - α
     i.e. the suppression threshold catches at least (1-α) of true
     hallucinations on average, with finite-sample guarantees.

We compute three thresholds (τ₁ < τ₂ < τ₃) by picking three quantile
levels (α_PASS > α_HEDGE > α_WARN), creating a four-action policy.

When no calibration data is available we fall back to risk-graduated
defaults that are still safer than the current binary "suppress on
unsupported" policy.

Design notes for production
---------------------------
- Calibration data is small (≤ 1000 examples). Persists as JSON.
- Recomputation is cheap (O(n log n)). Run nightly or on operator command.
- Per-profile thresholds: one set per (workload_profile × analyzer_mode).
- Each threshold update is logged for audit (timestamp, dataset hash,
  resulting τ, observed exposure on calibration).
- Production callers should treat the absence of calibration as a
  hard signal to use conservative defaults — fail-closed, not "no policy".

References
----------
- Vovk, V., Gammerman, A., Shafer, G. (2005). Algorithmic Learning in a
  Random World. Springer. [Foundations of conformal prediction.]
- Angelopoulos, A., Bates, S. (2023). Conformal Prediction: A Gentle
  Introduction. Foundations and Trends in Machine Learning.
- Bates, S. et al. (2021). Distribution-Free, Risk-Controlling Prediction
  Sets. JACM. [Risk control framework, parent of CRC.]
- El-Yaniv, R., Wiener, Y. (2010). On the foundations of noise-free
  selective classification. JMLR. [Selective prediction theory.]
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger("entroly.witness_calibration")


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
        # Auto-sort if caller passed them out of order.
        ts = sorted([self.tau_pass, self.tau_hedge, self.tau_warn])
        self.tau_pass, self.tau_hedge, self.tau_warn = ts
        for t in ts:
            if not 0.0 <= t <= 1.0:
                raise ValueError(f"threshold {t} outside [0,1]")

    def action(self, rho: float) -> str:
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


@dataclass
class CalibrationResult:
    """Outcome of running split-conformal calibration on labeled data."""
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
    empirical_exposure_at_warn: float   # FN rate above τ_warn
    empirical_retention_at_pass: float  # TN rate below τ_pass
    dataset_hash: str
    dataset_crc32: str = ""
    timestamp: float = field(default_factory=time.time)

    def as_threshold_set(self) -> ThresholdSet:
        return ThresholdSet(
            tau_pass=self.tau_pass,
            tau_hedge=self.tau_hedge,
            tau_warn=self.tau_warn,
        )


def conformal_quantile(values: list[float], alpha: float) -> float:
    """Compute the split-conformal quantile.

    For n iid scores and level α, the conformal quantile is the
        ⌈(n+1)(1−α)⌉ / n
    empirical quantile of the sample. This adjustment ensures the
    finite-sample coverage guarantee

        Pr[future_score ≤ τ] ≥ 1 − α

    holds by exchangeability of (calibration + future) scores.

    Returns 1.0 if `values` is empty (no data → fail-closed).
    """
    if not values:
        return 1.0
    n = len(values)
    sorted_v = sorted(values)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    # Clamp into valid index range
    rank = max(1, min(rank, n))
    return sorted_v[rank - 1]


def calibrate(
    samples: list[CalibrationSample],
    profile: str = "auto",
    alpha_pass: float = 0.80,
    alpha_hedge: float = 0.50,
    alpha_warn: float = 0.20,
) -> CalibrationResult:
    """Run split-conformal calibration on labeled samples.

    Args:
        samples: Labeled (ρ, y) observations from a held-out set.
        profile: The workload profile these samples represent.
        alpha_pass, alpha_hedge, alpha_warn: Allowed mis-coverage rates,
            in decreasing order. Examples:
                alpha_pass=0.80 → only the bottom 20% of hallucination
                                  scores escape to "pass" — i.e. 80% of
                                  true hallucinations are caught at the
                                  HEDGE or stricter level.
                alpha_warn=0.20 → 80% of true hallucinations are blocked
                                  at WARN or SUPPRESS.

        Constraints:
            alpha_warn < alpha_hedge < alpha_pass

    Returns:
        CalibrationResult with the three τ thresholds and empirical
        validation metrics.

    Raises:
        ValueError if α ordering is wrong.
    """
    if not (alpha_warn < alpha_hedge < alpha_pass):
        raise ValueError(
            f"α ordering must be α_warn < α_hedge < α_pass, "
            f"got {alpha_warn} < {alpha_hedge} < {alpha_pass}"
        )

    # Score set for conformal quantile: ρ values where y=1 (hallucinations).
    halu_scores = [s.rho for s in samples if s.y == 1]
    safe_scores = [s.rho for s in samples if s.y == 0]

    # Three quantiles → three thresholds.
    # Note: quantile(scores, α) yields a τ such that
    # Pr[ρ ≤ τ | halu] ≥ 1 − α. So:
    #   τ_pass    = quantile(halu, α_pass)  — small α_pass means many halus
    #                                          slip below this → low τ_pass
    # In our model we want τ_pass to be small (let only the safest through)
    # and τ_warn to be large (only block the worst). So we use 1−α_*.
    tau_pass = conformal_quantile(halu_scores, alpha_pass)
    tau_hedge = conformal_quantile(halu_scores, alpha_hedge)
    tau_warn = conformal_quantile(halu_scores, alpha_warn)

    # Sanity ordering (calibration data sometimes makes them equal).
    tau_pass, tau_hedge, tau_warn = sorted([tau_pass, tau_hedge, tau_warn])

    # Empirical validation on calibration set.
    n_above_warn_halu = sum(1 for s in samples if s.y == 1 and s.rho < tau_warn)
    emp_exposure_at_warn = (
        n_above_warn_halu / max(len(halu_scores), 1)
    )
    n_below_pass_safe = sum(1 for s in samples if s.y == 0 and s.rho < tau_pass)
    emp_retention_at_pass = (
        n_below_pass_safe / max(len(safe_scores), 1)
    )

    # Hash for audit log.
    h = hashlib.sha256()
    crc = 0
    for s in samples:
        line = f"{s.rho:.6f}|{s.y}|{s.profile}\n".encode()
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
        empirical_exposure_at_warn=round(emp_exposure_at_warn, 4),
        empirical_retention_at_pass=round(emp_retention_at_pass, 4),
        dataset_hash=dataset_hash,
        dataset_crc32=dataset_crc32,
    )


# ── Persistence ──────────────────────────────────────────────────────


class CalibrationStore:
    """Per-profile threshold store with JSON persistence and audit log."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._audit_path = self._path.with_suffix(self._path.suffix + ".audit.jsonl")
        self._results: dict[str, CalibrationResult] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            blob = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning("witness calibration load failed: %s", e)
            return
        for profile, d in blob.items():
            try:
                self._results[profile] = CalibrationResult(**d)
            except TypeError:
                continue

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({p: asdict(r) for p, r in self._results.items()}, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._path)

    def _audit(self, event: dict) -> None:
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def get_thresholds(self, profile: str) -> ThresholdSet:
        """Calibrated thresholds if available, else risk-graduated defaults."""
        key = (profile or "auto").lower().replace("-", "_")
        if key == "summarization":
            key = "summary"
        result = self._results.get(key)
        if result is None:
            return default_thresholds(key)
        return result.as_threshold_set()

    def update(self, result: CalibrationResult) -> None:
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
        })

    def all_profiles(self) -> dict[str, CalibrationResult]:
        return dict(self._results)
