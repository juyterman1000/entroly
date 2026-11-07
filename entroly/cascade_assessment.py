"""Held-out statistical assessment of fixed cascade decisions.

Implements the exact-binomial/Bonferroni construction from Learn then Test,
Section 3.2 (https://arxiv.org/abs/2110.01052). Under IID observations within
each profile, correct labels, an independently fixed policy/scorer/profile
family, and matching future distributions, the probability that any supported
pair has population selective error above risk_limit is at most error_level.

These assumptions are not established by this API. Declared sample identities
catch overlap but cannot prove independence or label truth. Query-dependent
switching between supported experts requires its own composite-policy analysis.
This module performs no routing, fallback, provider calls, persistence, or
WITNESS/RAVS authorization. Existing empirical band fitting is unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import re

from .conformal_cascade import CascadeCalibration, CascadePolicy, ESCALATE, decide

MAX_CANDIDATES = 32
MAX_PROFILE_SAMPLES = 4096
MAX_TOTAL_SAMPLES = 16384
MAX_FIT_SAMPLES = 16384
_SCHEMA = "entroly.cascade-risk-assessment.v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class CascadeCandidate:
    candidate_id: str
    profile_id: str
    scorer_sha256: str
    policy: CascadePolicy
    fit_sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    score: float
    label: int


def _hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{name} must be an opaque identifier of 1 to 128 characters")
    return value


def _number(value: object, name: str, *, infinite: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} exceeds numeric range") from None
    if math.isnan(result) or (not infinite and not math.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: object, name: str) -> Fraction:
    number = _number(value, name)
    if not 0 < number < 1:
        raise ValueError(f"{name} must be strictly between zero and one")
    result = Fraction(str(number))
    # Bound exact-arithmetic cost and give decimal targets explicit semantics.
    if result.denominator > 10 ** 12:
        raise ValueError(f"{name} supports at most 12 decimal places")
    return result


def _binomial_tail(errors: int, selected: int, risk: Fraction) -> tuple[int, int]:
    """Exact lower-tail numerator/denominator; no floating-point decisions."""
    if errors == selected:
        return 1, 1
    p, q = risk.numerator, risk.denominator
    denominator = q ** selected
    term = (q - p) ** selected
    numerator = term
    for i in range(errors):
        term = term * (selected - i) * p // ((i + 1) * (q - p))
        numerator += term
    return numerator, denominator


def _policy_binding(candidate: CascadeCandidate) -> tuple[dict, set[str]]:
    if not isinstance(candidate, CascadeCandidate) or not isinstance(candidate.policy, CascadePolicy):
        raise ValueError("a CascadeCandidate with a CascadePolicy is required")
    _identifier(candidate.candidate_id, "candidate_id")
    _identifier(candidate.profile_id, "profile_id")
    if not isinstance(candidate.scorer_sha256, str) or not _SHA.fullmatch(candidate.scorer_sha256):
        raise ValueError("scorer_sha256 must be 64 lowercase hexadecimal characters")
    policy = candidate.policy
    if not isinstance(policy.cal, CascadeCalibration):
        raise ValueError("policy requires a CascadeCalibration")
    low = _number(policy.t_lo, "t_lo", infinite=True)
    high = _number(policy.t_hi, "t_hi", infinite=True)
    if low > high:
        raise ValueError("policy cutoffs must be ordered")
    scores = []
    for values in (policy.cal.halu_scores, policy.cal.faith_scores):
        if not isinstance(values, tuple) or not values or len(values) > MAX_FIT_SAMPLES:
            raise ValueError("class calibration must be a bounded nonempty tuple")
        checked = tuple(_number(value, "fit score") for value in values)
        if any(not 0 <= value <= 1 for value in checked) or checked != tuple(sorted(checked)):
            raise ValueError("class calibration scores must be sorted and between zero and one")
        scores.append(checked)
    count = sum(map(len, scores))
    ids = candidate.fit_sample_ids
    if (not isinstance(ids, tuple) or len(ids) != count or count > MAX_FIT_SAMPLES):
        raise ValueError("fit_sample_ids must match the bounded fitted calibration sample")
    fit_ids = {_identifier(value, "fit sample id") for value in ids}
    if len(fit_ids) != len(ids):
        raise ValueError("fit sample identities must be unique")
    binding = {
        "candidate_id": candidate.candidate_id, "profile_id": candidate.profile_id,
        "scorer_sha256": candidate.scorer_sha256,
        "decision_rule": "conformal_cascade.decide.v1",
        "t_lo": low if math.isfinite(low) else str(low),
        "t_hi": high if math.isfinite(high) else str(high),
        "class_calibration_sha256": _hash(scores),
        "fit_identity_sha256": _hash(sorted(fit_ids)),
    }
    return binding, fit_ids


def assess_cascade_family(
    candidates: Sequence[CascadeCandidate],
    certification_samples: Mapping[str, Sequence[CalibrationSample]],
    *,
    risk_limit: float = 0.05,
    error_level: float = 0.05,
) -> dict:
    """Assess every supplied fixed profile/policy pair on separate data.

    Counts refer to task observations, not duplicated chunks or correlated
    turns. Do not adapt this family/scorer/profile definition to these labels,
    repeatedly peek without error accounting, or transfer results to a dynamic
    composite router. Supported means only the statistical statement above
    under external assumptions; it conveys no execution or content authority.
    """
    risk, level = _probability(risk_limit, "risk_limit"), _probability(error_level, "error_level")
    if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError("candidate family must be a bounded nonempty list or tuple")
    candidates = tuple(candidates)
    bindings, fit_ids, pairs = [], set(), set()
    for candidate in candidates:
        binding, fitted = _policy_binding(candidate)
        key = candidate.profile_id, candidate.candidate_id
        if key in pairs:
            raise ValueError("candidate/profile pairs must be unique")
        pairs.add(key)
        fit_ids.update(fitted)
        bindings.append(binding)
    profiles = {profile for profile, _candidate in pairs}
    if not isinstance(certification_samples, Mapping) or set(certification_samples) != profiles:
        raise ValueError("certification profiles must match the candidate family")
    total, sample_ids, data, data_hashes = 0, set(), {}, {}
    for profile in sorted(profiles):
        samples = certification_samples[profile]
        if not isinstance(samples, (list, tuple)) or len(samples) > MAX_PROFILE_SAMPLES:
            raise ValueError("profile certification sample exceeds its limit")
        total += len(samples)
        if total > MAX_TOTAL_SAMPLES:
            raise ValueError("total certification sample exceeds its limit")
        rows = []
        for sample in samples:
            if not isinstance(sample, CalibrationSample):
                raise ValueError("CalibrationSample values are required")
            identity = _identifier(sample.sample_id, "certification sample id")
            if identity in sample_ids:
                raise ValueError("certification sample identities must be globally unique")
            sample_ids.add(identity)
            score = _number(sample.score, "certification score")
            if not 0 <= score <= 1 or type(sample.label) is not int or sample.label not in (0, 1):
                raise ValueError("certification requires a score in [0,1] and a binary integer label")
            rows.append((identity, score, sample.label))
        data[profile] = sorted(rows)
        data_hashes[profile] = _hash(data[profile])
    if fit_ids & sample_ids:
        raise ValueError("fit and certification sample identities must be disjoint across the family")

    threshold = level / len(candidates)
    entries = []
    for candidate, binding in zip(candidates, bindings):
        selected = errors = 0
        for _identity, score, label in data[candidate.profile_id]:
            decision = decide(candidate.policy, score)
            if decision.action != ESCALATE:
                selected += 1
                errors += int(bool(decision.predict_hallucinated) != bool(label))
        status, p_value, log_p = "insufficient_selected_data", None, None
        if selected:
            numerator, denominator = _binomial_tail(errors, selected, risk)
            supported = numerator * threshold.denominator <= threshold.numerator * denominator
            best = (risk.denominator - risk.numerator) ** selected
            enough = best * threshold.denominator <= threshold.numerator * risk.denominator ** selected
            status = "supported" if supported else "not_supported" if enough else "insufficient_selected_data"
            approximate = numerator / denominator
            p_value = approximate if approximate > 0 else None
            log_p = math.log10(numerator) - math.log10(denominator)
        entries.append({
            "candidate_id": candidate.candidate_id, "profile_id": candidate.profile_id,
            "scorer_sha256": candidate.scorer_sha256, "policy_sha256": _hash(binding),
            "data_sha256": data_hashes[candidate.profile_id], "status": status,
            "sample_count": len(data[candidate.profile_id]), "selected": selected, "errors": errors,
            "empirical_selective_risk": errors / selected if selected else None,
            "p_value_approx": p_value, "p_value_log10_approx": log_p,
            "p_value_underflow": bool(selected and p_value is None),
            "threshold": float(threshold),
        })
    identity = {"schema_version": _SCHEMA,
                "policies": sorted(bindings, key=lambda b: (b["profile_id"], b["candidate_id"])),
                "data": data_hashes, "risk_limit": str(risk), "error_level": str(level)}
    return {
        "schema_version": _SCHEMA,
        "method": "exact-binomial-bonferroni", "production_authority": False,
        "hypothesis_count": len(candidates), "risk_limit": float(risk), "error_level": float(level),
        "family_sha256": _hash(identity), "entries": entries,
        "hash_scope": "declared input configuration and data digests; results are not attested",
        "assumptions_unverified": [
            "fixed scorer, composite decision rules and profiles independent of certification data",
            "IID task observations within profiles and matching future distributions",
            "correct caller-supplied labels and honest sample identity declarations",
            "complete hypothesis, repeated evaluation and deployment-epoch error accounting",
        ],
        "scope": "population selective error of each assessed fixed decision rule under stated assumptions",
        "routing_changes": 0, "provider_calls": 0,
    }
