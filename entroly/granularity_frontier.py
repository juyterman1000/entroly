"""Evidence-gated selection of context representation granularity.

The controller in this module is deliberately narrower than a compressor.  It
does not invent summaries, change tokenizers, or infer that fewer tokens are
better.  It answers a later question after an evaluation harness has run the
same task with a full-context baseline and one or more smaller context
representations:

    Has this representation earned promotion for this task/model/tokenizer
    scope, and is the observed token reduction worth the measured risk?

The decision is based on paired outcomes from identical source material.  It
uses two independent gates:

* a one-sided interval for the paired quality difference; and
* a Wilson upper bound for verifier regressions (baseline passes while the
  candidate fails).

Catastrophic failures fail closed.  Sparse evidence and insufficient evaluator
diversity hold the candidate rather than extrapolating.  Every decision emits a
content-addressed receipt that states the statistical assumptions and the exact
evidence digest used.

This is an Entroly-native control algorithm.  It is motivated by research that
shows useful token granularity depends on compute, language, and task, but it
does not claim that language-model training scaling laws directly validate
agent context compression.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "entroly.granularity-frontier.v1"
ALGORITHM_ID = "paired-evidence-resolution-frontier-v1"
_SHA256_HEX_LENGTH = 64


class GranularityFrontierError(ValueError):
    """Base error for malformed or conflicting frontier evidence."""


class ConflictingGranularityEvidence(GranularityFrontierError):
    """Raised when an existing trial key is reused with different evidence."""


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GranularityFrontierError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > 256:
        raise GranularityFrontierError(f"{name} must be at most 256 characters")
    return normalized


def _require_sha256(name: str, value: str) -> str:
    normalized = _require_text(name, value).lower()
    if len(normalized) != _SHA256_HEX_LENGTH or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise GranularityFrontierError(f"{name} must be a lowercase SHA-256 hex digest")
    return normalized


def _require_score(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise GranularityFrontierError(f"{name} must be a finite number in [0, 1]")
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GranularityFrontierError(
            f"{name} must be a finite number in [0, 1]"
        ) from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise GranularityFrontierError(f"{name} must be a finite number in [0, 1]")
    return score


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise GranularityFrontierError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GranularityFrontierError(f"{name} must be a positive integer") from exc
    if integer <= 0 or integer != value:
        raise GranularityFrontierError(f"{name} must be a positive integer")
    return integer


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GranularityScope:
    """A failure-domain boundary for learned context-resolution decisions.

    Evidence never crosses any field in this key.  A result from an English
    coding task on one model/tokenizer cannot silently authorize a different
    language, task family, model, tokenizer, evaluation protocol, or baseline.
    """

    task_class: str
    workload_id: str
    model_id: str
    tokenizer_id: str
    evaluator_protocol: str
    baseline_id: str = "full-context"

    def __post_init__(self) -> None:
        for field_name in (
            "task_class",
            "workload_id",
            "model_id",
            "tokenizer_id",
            "evaluator_protocol",
            "baseline_id",
        ):
            object.__setattr__(self, field_name, _require_text(field_name, getattr(self, field_name)))

    @property
    def scope_id(self) -> str:
        return _stable_digest(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GranularityScope":
        if not isinstance(value, Mapping):
            raise GranularityFrontierError("scope must be an object")
        return cls(
            task_class=str(value.get("task_class", "")),
            workload_id=str(value.get("workload_id", "")),
            model_id=str(value.get("model_id", "")),
            tokenizer_id=str(value.get("tokenizer_id", "")),
            evaluator_protocol=str(value.get("evaluator_protocol", "")),
            baseline_id=str(value.get("baseline_id", "full-context")),
        )


@dataclass(frozen=True)
class EvaluatorVerdict:
    """One versioned evaluator's paired baseline/candidate judgment.

    ``evaluator_family`` names a shared failure domain.  Two model judges with
    different deployment IDs but the same underlying model should normally use
    the same family.  Distinct family labels are operator declarations, not
    proof that evaluators are statistically independent; receipts state that
    limitation explicitly.
    """

    evaluator_id: str
    evaluator_family: str
    baseline_score: float
    candidate_score: float
    baseline_passed: bool
    candidate_passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluator_id", _require_text("evaluator_id", self.evaluator_id))
        object.__setattr__(
            self,
            "evaluator_family",
            _require_text("evaluator_family", self.evaluator_family),
        )
        object.__setattr__(
            self, "baseline_score", _require_score("baseline_score", self.baseline_score)
        )
        object.__setattr__(
            self, "candidate_score", _require_score("candidate_score", self.candidate_score)
        )
        if not isinstance(self.baseline_passed, bool):
            raise GranularityFrontierError("baseline_passed must be boolean")
        if not isinstance(self.candidate_passed, bool):
            raise GranularityFrontierError("candidate_passed must be boolean")

    @property
    def quality_delta(self) -> float:
        return self.candidate_score - self.baseline_score

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluatorVerdict":
        if not isinstance(value, Mapping):
            raise GranularityFrontierError("verdict must be an object")
        return cls(
            evaluator_id=str(value.get("evaluator_id", "")),
            evaluator_family=str(value.get("evaluator_family", "")),
            baseline_score=value.get("baseline_score"),
            candidate_score=value.get("candidate_score"),
            baseline_passed=value.get("baseline_passed"),
            candidate_passed=value.get("candidate_passed"),
        )


@dataclass(frozen=True)
class PairedGranularityObservation:
    """A matched task run comparing full and reduced context.

    The baseline and candidate must be built from the same ``source_sha256``.
    Context digests make the actual delivered representations auditable.  The
    controller cannot prove those digests were produced honestly; the external
    harness remains responsible for execution and evaluator provenance.
    """

    scope: GranularityScope
    trial_id: str
    candidate_id: str
    source_sha256: str
    baseline_context_sha256: str
    candidate_context_sha256: str
    source_bytes: int
    baseline_tokens: int
    candidate_tokens: int
    verdicts: tuple[EvaluatorVerdict, ...]
    catastrophic_failure: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.scope, GranularityScope):
            raise GranularityFrontierError("scope must be a GranularityScope")
        object.__setattr__(self, "trial_id", _require_text("trial_id", self.trial_id))
        object.__setattr__(self, "candidate_id", _require_text("candidate_id", self.candidate_id))
        if self.candidate_id == self.scope.baseline_id:
            raise GranularityFrontierError("candidate_id must differ from baseline_id")
        for name in (
            "source_sha256",
            "baseline_context_sha256",
            "candidate_context_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(name, getattr(self, name)))
        for name in ("source_bytes", "baseline_tokens", "candidate_tokens"):
            object.__setattr__(self, name, _require_positive_int(name, getattr(self, name)))
        verdicts = tuple(self.verdicts)
        if not verdicts or any(not isinstance(v, EvaluatorVerdict) for v in verdicts):
            raise GranularityFrontierError("verdicts must contain at least one EvaluatorVerdict")
        evaluator_ids = [verdict.evaluator_id for verdict in verdicts]
        if len(evaluator_ids) != len(set(evaluator_ids)):
            raise GranularityFrontierError("evaluator_id values must be unique within a trial")
        object.__setattr__(self, "verdicts", verdicts)
        if not isinstance(self.catastrophic_failure, bool):
            raise GranularityFrontierError("catastrophic_failure must be boolean")

    @property
    def trial_key(self) -> str:
        return _stable_digest(
            {
                "scope_id": self.scope.scope_id,
                "trial_id": self.trial_id,
                "candidate_id": self.candidate_id,
            }
        )

    @property
    def observation_digest(self) -> str:
        return _stable_digest(self.to_dict())

    @property
    def conservative_quality_delta(self) -> float:
        """Worst paired score delta across declared evaluator families."""

        return min(verdict.quality_delta for verdict in self.verdicts)

    @property
    def verifier_regressed(self) -> bool:
        return any(
            verdict.baseline_passed and not verdict.candidate_passed
            for verdict in self.verdicts
        )

    @property
    def token_savings_fraction(self) -> float:
        return 1.0 - (self.candidate_tokens / self.baseline_tokens)

    @property
    def evidence_bytes_per_delivered_token(self) -> float:
        return self.source_bytes / self.candidate_tokens

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["verdicts"] = [asdict(verdict) for verdict in self.verdicts]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairedGranularityObservation":
        if not isinstance(value, Mapping):
            raise GranularityFrontierError("observation must be an object")
        raw_verdicts = value.get("verdicts")
        if not isinstance(raw_verdicts, list):
            raise GranularityFrontierError("verdicts must be a list")
        return cls(
            scope=GranularityScope.from_dict(value.get("scope", {})),
            trial_id=str(value.get("trial_id", "")),
            candidate_id=str(value.get("candidate_id", "")),
            source_sha256=str(value.get("source_sha256", "")),
            baseline_context_sha256=str(value.get("baseline_context_sha256", "")),
            candidate_context_sha256=str(value.get("candidate_context_sha256", "")),
            source_bytes=value.get("source_bytes"),
            baseline_tokens=value.get("baseline_tokens"),
            candidate_tokens=value.get("candidate_tokens"),
            verdicts=tuple(EvaluatorVerdict.from_dict(item) for item in raw_verdicts),
            catastrophic_failure=value.get("catastrophic_failure", False),
        )


@dataclass(frozen=True)
class GranularityPromotionPolicy:
    """Evidence requirements for a candidate to earn promotion."""

    min_paired_trials: int = 40
    confidence: float = 0.95
    max_mean_quality_regression: float = 0.02
    max_verifier_regression_rate: float = 0.10
    min_token_savings_fraction: float = 0.05
    min_baseline_pass_rate: float = 0.80
    min_evaluator_families: int = 1
    require_zero_catastrophic_failures: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_paired_trials",
            _require_positive_int("min_paired_trials", self.min_paired_trials),
        )
        object.__setattr__(
            self,
            "min_evaluator_families",
            _require_positive_int("min_evaluator_families", self.min_evaluator_families),
        )
        for name in (
            "confidence",
            "max_mean_quality_regression",
            "max_verifier_regression_rate",
            "min_token_savings_fraction",
            "min_baseline_pass_rate",
        ):
            value = _require_score(name, getattr(self, name))
            object.__setattr__(self, name, value)
        if not 0.5 < self.confidence < 1.0:
            raise GranularityFrontierError("confidence must be strictly between 0.5 and 1")
        if not isinstance(self.require_zero_catastrophic_failures, bool):
            raise GranularityFrontierError(
                "require_zero_catastrophic_failures must be boolean"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GranularityPromotionPolicy":
        if not isinstance(value, Mapping):
            raise GranularityFrontierError("policy must be an object")
        return cls(
            min_paired_trials=value.get("min_paired_trials", 40),
            confidence=value.get("confidence", 0.95),
            max_mean_quality_regression=value.get("max_mean_quality_regression", 0.02),
            max_verifier_regression_rate=value.get("max_verifier_regression_rate", 0.10),
            min_token_savings_fraction=value.get("min_token_savings_fraction", 0.05),
            min_baseline_pass_rate=value.get("min_baseline_pass_rate", 0.80),
            min_evaluator_families=value.get("min_evaluator_families", 1),
            require_zero_catastrophic_failures=value.get(
                "require_zero_catastrophic_failures", True
            ),
        )


def _one_sided_t_critical(confidence: float, degrees_of_freedom: int) -> float:
    """Approximate a Student-t quantile with a Cornish-Fisher expansion.

    The approximation is accurate for the sample sizes allowed to promote by
    the default policy.  The receipt names this assumption instead of
    presenting the interval as distribution-free.
    """

    if degrees_of_freedom <= 0:
        return math.inf
    z = NormalDist().inv_cdf(confidence)
    df = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
    )


def _paired_mean_lower_bound(values: list[float], confidence: float) -> float:
    if not values:
        return -1.0
    mean = statistics.fmean(values)
    if len(values) == 1:
        return max(-1.0, min(1.0, mean))
    sample_sd = statistics.stdev(values)
    critical = _one_sided_t_critical(confidence, len(values) - 1)
    lower = mean - critical * sample_sd / math.sqrt(len(values))
    return max(-1.0, min(1.0, lower))


def _wilson_upper_bound(successes: int, trials: int, confidence: float) -> float:
    """One-sided Wilson upper confidence bound for a Bernoulli rate."""

    if trials <= 0:
        return 1.0
    if not 0 <= successes <= trials:
        raise GranularityFrontierError("successes must be between zero and trials")
    z = NormalDist().inv_cdf(confidence)
    z2 = z * z
    proportion = successes / trials
    denominator = 1.0 + z2 / trials
    centre = (proportion + z2 / (2.0 * trials)) / denominator
    radius = (
        z
        / denominator
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z2 / (4.0 * trials * trials)
        )
    )
    return min(1.0, centre + radius)


class VerifiedGranularityFrontier:
    """Accumulate matched evaluations and select an earned granularity arm."""

    def __init__(self, policy: GranularityPromotionPolicy | None = None):
        self.policy = policy or GranularityPromotionPolicy()
        self._observations: dict[str, PairedGranularityObservation] = {}

    def record(self, observation: PairedGranularityObservation) -> bool:
        """Record one trial idempotently.

        Returns ``True`` for a new observation and ``False`` for an exact
        replay.  Reusing the same scope/candidate/trial key with different
        evidence raises instead of silently rewriting history.
        """

        if not isinstance(observation, PairedGranularityObservation):
            raise GranularityFrontierError(
                "observation must be a PairedGranularityObservation"
            )
        existing = self._observations.get(observation.trial_key)
        if existing is None:
            self._observations[observation.trial_key] = observation
            return True
        if existing == observation:
            return False
        raise ConflictingGranularityEvidence(
            "trial key already exists with different evidence"
        )

    def record_many(self, observations: Iterable[PairedGranularityObservation]) -> int:
        added = 0
        for observation in observations:
            added += int(self.record(observation))
        return added

    def _candidate_observations(
        self, scope: GranularityScope, candidate_id: str
    ) -> list[PairedGranularityObservation]:
        candidate = _require_text("candidate_id", candidate_id)
        return sorted(
            (
                observation
                for observation in self._observations.values()
                if observation.scope == scope and observation.candidate_id == candidate
            ),
            key=lambda observation: observation.trial_id,
        )

    def assess_candidate(
        self, scope: GranularityScope, candidate_id: str
    ) -> dict[str, Any]:
        candidate_id = _require_text("candidate_id", candidate_id)
        observations = self._candidate_observations(scope, candidate_id)
        n = len(observations)
        deltas = [observation.conservative_quality_delta for observation in observations]
        regressions = sum(observation.verifier_regressed for observation in observations)
        catastrophes = sum(observation.catastrophic_failure for observation in observations)
        baseline_passes = sum(
            all(verdict.baseline_passed for verdict in observation.verdicts)
            for observation in observations
        )
        candidate_passes = sum(
            all(verdict.candidate_passed for verdict in observation.verdicts)
            for observation in observations
        )
        evaluator_families = sorted(
            {
                verdict.evaluator_family
                for observation in observations
                for verdict in observation.verdicts
            }
        )
        evaluator_family_trial_counts = {
            family: sum(
                any(verdict.evaluator_family == family for verdict in observation.verdicts)
                for observation in observations
            )
            for family in evaluator_families
        }
        complete_evaluator_families = sorted(
            family
            for family, trial_count in evaluator_family_trial_counts.items()
            if trial_count == n and n > 0
        )

        mean_delta = statistics.fmean(deltas) if deltas else 0.0
        quality_lower = _paired_mean_lower_bound(deltas, self.policy.confidence)
        regression_upper = _wilson_upper_bound(
            regressions, n, self.policy.confidence
        )
        mean_savings = (
            statistics.fmean(
                observation.token_savings_fraction for observation in observations
            )
            if observations
            else 0.0
        )
        mean_candidate_tokens = (
            statistics.fmean(observation.candidate_tokens for observation in observations)
            if observations
            else 0.0
        )
        mean_baseline_tokens = (
            statistics.fmean(observation.baseline_tokens for observation in observations)
            if observations
            else 0.0
        )
        mean_bytes_per_token = (
            statistics.fmean(
                observation.evidence_bytes_per_delivered_token
                for observation in observations
            )
            if observations
            else 0.0
        )

        hold_reasons: list[str] = []
        rejection_reasons: list[str] = []
        if n < self.policy.min_paired_trials:
            hold_reasons.append("insufficient_paired_trials")
        if len(complete_evaluator_families) < self.policy.min_evaluator_families:
            hold_reasons.append("insufficient_evaluator_families")
        baseline_pass_rate = baseline_passes / n if n else 0.0
        candidate_pass_rate = candidate_passes / n if n else 0.0
        if n >= self.policy.min_paired_trials and (
            baseline_pass_rate < self.policy.min_baseline_pass_rate
        ):
            hold_reasons.append("baseline_success_below_policy")
        if self.policy.require_zero_catastrophic_failures and catastrophes:
            rejection_reasons.append("catastrophic_failure_observed")
        # Sparse evidence is not negative evidence. Except for an observed
        # catastrophe, wait until sample-size and evaluator-family gates are
        # satisfied before interpreting statistical/economic thresholds.
        if not hold_reasons:
            if quality_lower < -self.policy.max_mean_quality_regression:
                rejection_reasons.append("quality_noninferiority_not_demonstrated")
            if regression_upper > self.policy.max_verifier_regression_rate:
                rejection_reasons.append("verifier_regression_bound_exceeds_policy")
            if mean_savings < self.policy.min_token_savings_fraction:
                rejection_reasons.append("token_savings_below_policy")

        if rejection_reasons and "catastrophic_failure_observed" in rejection_reasons:
            status = "reject"
        elif hold_reasons:
            status = "hold"
        elif rejection_reasons:
            status = "reject"
        else:
            status = "promote"

        return {
            "candidate_id": candidate_id,
            "status": status,
            "paired_trials": n,
            "evaluator_families": evaluator_families,
            "complete_evaluator_families": complete_evaluator_families,
            "evaluator_family_trial_counts": evaluator_family_trial_counts,
            "quality": {
                "mean_paired_delta": round(mean_delta, 8),
                "one_sided_lower_bound": round(quality_lower, 8),
                "allowed_mean_regression": self.policy.max_mean_quality_regression,
            },
            "verification": {
                "baseline_pass_rate": round(baseline_pass_rate, 8),
                "candidate_pass_rate": round(candidate_pass_rate, 8),
                "required_baseline_pass_rate": self.policy.min_baseline_pass_rate,
                "regression_count": regressions,
                "regression_rate": round(regressions / n, 8) if n else 0.0,
                "wilson_upper_bound": round(regression_upper, 8),
                "allowed_regression_rate": self.policy.max_verifier_regression_rate,
                "catastrophic_failure_count": catastrophes,
            },
            "economics": {
                "mean_baseline_tokens": round(mean_baseline_tokens, 4),
                "mean_candidate_tokens": round(mean_candidate_tokens, 4),
                "mean_token_savings_fraction": round(mean_savings, 8),
                "mean_source_bytes_per_delivered_token": round(
                    mean_bytes_per_token, 8
                ),
                "required_token_savings_fraction": self.policy.min_token_savings_fraction,
            },
            "hold_reasons": hold_reasons,
            "rejection_reasons": rejection_reasons,
            "trial_cohort_digest": _stable_digest(
                sorted(observation.trial_id for observation in observations)
            ),
            "observation_digests": [
                observation.observation_digest for observation in observations
            ],
        }

    def decide(self, scope: GranularityScope) -> dict[str, Any]:
        """Return a content-addressed decision receipt for one exact scope."""

        if not isinstance(scope, GranularityScope):
            raise GranularityFrontierError("scope must be a GranularityScope")
        candidate_ids = sorted(
            {
                observation.candidate_id
                for observation in self._observations.values()
                if observation.scope == scope
            }
        )
        assessments = [
            self.assess_candidate(scope, candidate_id) for candidate_id in candidate_ids
        ]
        promotable = [item for item in assessments if item["status"] == "promote"]
        promotable_cohorts = {
            item["trial_cohort_digest"] for item in promotable
        }
        if len(promotable_cohorts) > 1:
            selection = {
                "status": "retain_baseline",
                "selected_id": scope.baseline_id,
                "baseline_id": scope.baseline_id,
                "reason": "candidate_trial_cohorts_not_aligned",
            }
        elif promotable:
            chosen = min(
                promotable,
                key=lambda item: (
                    -item["economics"]["mean_token_savings_fraction"],
                    -item["quality"]["one_sided_lower_bound"],
                    item["economics"]["mean_candidate_tokens"],
                    item["candidate_id"],
                ),
            )
            selection = {
                "status": "promote_candidate",
                "selected_id": chosen["candidate_id"],
                "baseline_id": scope.baseline_id,
                "reason": "highest_baseline_normalized_token_reduction_among_evidence_eligible_candidates",
            }
        else:
            selection = {
                "status": "retain_baseline",
                "selected_id": scope.baseline_id,
                "baseline_id": scope.baseline_id,
                "reason": (
                    "no_candidate_has_earned_promotion"
                    if assessments
                    else "no_candidate_evidence"
                ),
            }

        evidence_digests = sorted(
            digest
            for assessment in assessments
            for digest in assessment["observation_digests"]
        )
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "algorithm_id": ALGORITHM_ID,
            "scope": asdict(scope),
            "scope_id": scope.scope_id,
            "promotion_policy": asdict(self.policy),
            "selection": selection,
            "candidate_assessments": assessments,
            "evidence_digest": _stable_digest(evidence_digests),
            "assumptions": [
                "paired trials are independent and representative of the deployment scope",
                "quality bounds use a one-sided paired Student-t approximation",
                "verifier regression uses a one-sided Wilson upper bound",
                "evaluator family labels are declared failure domains, not audited independence",
                "source and context digests are harness-provided provenance claims",
                "training-tokenization scaling results are motivation, not validation of context compression",
            ],
        }
        receipt["receipt_sha256"] = _stable_digest(receipt)
        return receipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "promotion_policy": asdict(self.policy),
            "observations": [
                observation.to_dict()
                for observation in sorted(
                    self._observations.values(),
                    key=lambda item: (item.scope.scope_id, item.candidate_id, item.trial_id),
                )
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerifiedGranularityFrontier":
        if not isinstance(value, Mapping):
            raise GranularityFrontierError("frontier state must be an object")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise GranularityFrontierError("unsupported granularity frontier schema")
        frontier = cls(
            GranularityPromotionPolicy.from_dict(value.get("promotion_policy", {}))
        )
        raw_observations = value.get("observations")
        if not isinstance(raw_observations, list):
            raise GranularityFrontierError("observations must be a list")
        frontier.record_many(
            PairedGranularityObservation.from_dict(item) for item in raw_observations
        )
        return frontier

    def save(self, path: str | Path) -> None:
        """Atomically persist validated frontier state."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: str | Path) -> "VerifiedGranularityFrontier":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GranularityFrontierError(
                "unable to load granularity frontier state"
            ) from exc
        return cls.from_dict(value)


def verify_granularity_receipt(receipt: Mapping[str, Any]) -> bool:
    """Verify receipt self-consistency without authenticating evidence claims.

    A successful result detects accidental mutation and malformed derivation.
    Adversarial tamper evidence requires an external signature or transparency
    log anchor; a digest stored beside its own payload is not authentication.
    """

    if not isinstance(receipt, Mapping):
        return False
    snapshot = dict(receipt)
    claimed = snapshot.pop("receipt_sha256", None)
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        return False
    if snapshot.get("algorithm_id") != ALGORITHM_ID:
        return False
    if not isinstance(claimed, str) or claimed != _stable_digest(snapshot):
        return False
    try:
        scope = GranularityScope.from_dict(snapshot.get("scope", {}))
        if snapshot.get("scope_id") != scope.scope_id:
            return False
        assessments = snapshot.get("candidate_assessments")
        if not isinstance(assessments, list):
            return False
        observation_digests: list[str] = []
        candidate_ids: set[str] = set()
        for assessment in assessments:
            if not isinstance(assessment, Mapping):
                return False
            candidate_ids.add(_require_text("candidate_id", assessment.get("candidate_id")))
            digests = assessment.get("observation_digests")
            if not isinstance(digests, list) or any(
                not isinstance(digest, str) or len(digest) != _SHA256_HEX_LENGTH
                for digest in digests
            ):
                return False
            observation_digests.extend(digests)
        if snapshot.get("evidence_digest") != _stable_digest(sorted(observation_digests)):
            return False
        selection = snapshot.get("selection")
        if not isinstance(selection, Mapping):
            return False
        selected_id = selection.get("selected_id")
        if selected_id != scope.baseline_id and selected_id not in candidate_ids:
            return False
    except (GranularityFrontierError, TypeError, ValueError):
        return False
    return True


__all__ = [
    "ALGORITHM_ID",
    "SCHEMA_VERSION",
    "ConflictingGranularityEvidence",
    "EvaluatorVerdict",
    "GranularityFrontierError",
    "GranularityPromotionPolicy",
    "GranularityScope",
    "PairedGranularityObservation",
    "VerifiedGranularityFrontier",
    "verify_granularity_receipt",
]
