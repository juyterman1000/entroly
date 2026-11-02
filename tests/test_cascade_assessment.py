"""Independent contract tests for held-out cascade-family assessment.

The oracle uses the established exact binomial lower tail and Bonferroni
correction, with Fraction arithmetic. It imports no assessment internals and
makes no inference calls; these synthetic labels establish no product quality.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from math import comb, log10
import re

import pytest

from entroly.cascade_assessment import (
    CalibrationSample,
    CascadeCandidate,
    assess_cascade_family,
)
from entroly.conformal_cascade import CascadePolicy, fit_cascade


SCORER_SHA = "a" * 64


def _policy():
    return CascadePolicy(
        cal=fit_cascade([0.25, 0.75], [0, 1]),
        t_lo=0.25,
        t_hi=0.75,
        target_selective_risk=0.05,
        c_exp=1.0,
        q=20.0,
        r_floor=0.0,
        rule_threshold=0.05,
    )


def _candidate(candidate_id="candidate", profile_id="profile", **changes):
    candidate = CascadeCandidate(
        candidate_id=candidate_id,
        profile_id=profile_id,
        scorer_sha256=SCORER_SHA,
        policy=_policy(),
        fit_sample_ids=("fit-faithful", "fit-hallucinated"),
    )
    return replace(candidate, **changes)


def _samples(n, errors=0, *, prefix="cert", score=0.0):
    return tuple(
        CalibrationSample(f"{prefix}-{i}", score, int(i < errors))
        for i in range(n)
    )


def _cdf(n, errors, risk):
    """Direct finite sum, independent of the production numerical algorithm."""
    probability = Fraction(str(risk))
    return sum(
        (
            comb(n, k) * probability**k * (1 - probability) ** (n - k)
            for k in range(errors + 1)
        ),
        Fraction(0),
    )


def _entry(result, candidate_id="candidate", profile_id="profile"):
    return next(
        entry
        for entry in result["entries"]
        if (entry["candidate_id"], entry["profile_id"])
        == (candidate_id, profile_id)
    )


def test_records_are_frozen_and_keep_fit_provenance():
    candidate = _candidate()
    sample = CalibrationSample("cert", 0.0, 0)
    with pytest.raises(FrozenInstanceError):
        candidate.profile_id = "changed"
    with pytest.raises(FrozenInstanceError):
        sample.label = 1
    assert candidate.fit_sample_ids == ("fit-faithful", "fit-hallucinated")
    assert len(candidate.fit_sample_ids) == candidate.policy.cal.n_h + candidate.policy.cal.n_g
    assert sample.label == 0


@pytest.mark.parametrize(
    "n,errors,risk,error_level,hypotheses",
    [
        (n, errors, risk, 0.3, hypotheses)
        for n in range(1, 9)
        for errors in range(n + 1)
        for risk in (0.05, 0.7)
        for hypotheses in (1, 3)
    ]
    # Equality must be decided in exact decimal arithmetic. For example,
    # float(0.6) / 3 is below 0.2 although the exact threshold equals 0.2.
    + [
        (1, 0, 0.8, 0.6, 3),
        (1, 0, 0.7, 0.9, 3),
        (1, 0, 0.9, 0.3, 3),
        (1, 0, 0.8, 0.599999999999, 3),
        (1, 0, 0.8, 0.600000000001, 3),
        (8, 1, 0.123456789012, 0.123456789012, 1),
        (8, 0, 0.000000000001, 0.000000000001, 1),
    ],
)
def test_small_samples_match_independent_fraction_oracle(
    n, errors, risk, error_level, hypotheses
):
    candidates = [_candidate(f"candidate-{i}") for i in range(hypotheses)]
    result = assess_cascade_family(
        candidates,
        {"profile": _samples(n, errors)},
        risk_limit=risk,
        error_level=error_level,
    )
    exact_p = _cdf(n, errors, risk)
    threshold = Fraction(str(error_level)) / hypotheses
    expected = (
        "insufficient_selected_data"
        if _cdf(n, 0, risk) > threshold
        else "supported" if exact_p <= threshold else "not_supported"
    )
    assert result["hypothesis_count"] == hypotheses
    assert len(result["entries"]) == hypotheses
    for entry in result["entries"]:
        assert entry["status"] == expected
        assert entry["selected"] == n
        assert entry["errors"] == errors
        assert entry["empirical_selective_risk"] == pytest.approx(errors / n)
        assert float(entry["p_value_approx"]) == pytest.approx(float(exact_p))
        assert entry["p_value_underflow"] is False
        assert float(entry["threshold"]) == pytest.approx(float(threshold))


@pytest.mark.parametrize("hypotheses,minimum", [(1, 59), (2, 72), (4, 86)])
@pytest.mark.parametrize("offset", [-1, 0])
def test_zero_error_minimum_depends_on_entire_family(hypotheses, minimum, offset):
    n = minimum + offset
    result = assess_cascade_family(
        [_candidate(f"candidate-{i}") for i in range(hypotheses)],
        {"profile": _samples(n)},
    )
    threshold = Fraction(1, 20 * hypotheses)
    assert _cdf(minimum - 1, 0, 0.05) > threshold
    assert _cdf(minimum, 0, 0.05) <= threshold
    assert result["hypothesis_count"] == hypotheses
    assert {entry["status"] for entry in result["entries"]} == {
        "supported" if offset == 0 else "insufficient_selected_data"
    }
    assert all(entry["selected"] == n and entry["errors"] == 0 for entry in result["entries"])


@pytest.mark.parametrize("n,errors", [(8, 0), (8, 8), (100, 20), (100, 100)])
def test_small_or_failed_cases_never_receive_support(n, errors):
    result = assess_cascade_family([_candidate()], {"profile": _samples(n, errors)})
    entry = _entry(result)
    assert entry["status"] in {"insufficient_selected_data", "not_supported"}
    assert entry["selected"] == n
    assert entry["errors"] == errors
    assert entry["empirical_selective_risk"] == pytest.approx(errors / n)
    assert result["production_authority"] is False


def test_band_edges_and_both_prediction_errors_use_only_selected_denominator():
    samples = (
        CalibrationSample("accept-edge-right", 0.25, 0),
        CalibrationSample("accept-edge-wrong", 0.25, 1),
        CalibrationSample("accept-right", 0.0, 0),
        CalibrationSample("flag-edge-right", 0.75, 1),
        CalibrationSample("flag-edge-wrong", 0.75, 0),
        CalibrationSample("flag-right", 1.0, 1),
        CalibrationSample("escalate-faithful", 0.5, 0),
        CalibrationSample("escalate-hallucinated", 0.5, 1),
    )
    entry = _entry(assess_cascade_family([_candidate()], {"profile": samples}))
    assert entry["selected"] == 6
    assert entry["errors"] == 2
    assert entry["empirical_selective_risk"] == pytest.approx(1 / 3)
    assert entry["p_value_approx"] == pytest.approx(float(_cdf(6, 2, 0.05)))
    assert entry["status"] != "supported"


@pytest.mark.parametrize("n", [0, 8, 100])
def test_no_selected_samples_is_unknown_risk_even_when_all_escalate(n):
    entry = _entry(
        assess_cascade_family([_candidate()], {"profile": _samples(n, score=0.5)})
    )
    assert entry["selected"] == 0
    assert entry["errors"] == 0
    assert entry["empirical_selective_risk"] is None
    assert entry["status"] == "insufficient_selected_data"


def test_dense_and_sparse_profiles_are_assessed_separately():
    candidates = [_candidate("shared", "dense"), _candidate("shared", "sparse")]
    profiles = {"dense": _samples(100, prefix="dense"), "sparse": _samples(8, prefix="sparse")}
    result = assess_cascade_family(candidates, profiles)
    reordered = assess_cascade_family(
        list(reversed(candidates)), dict(reversed(list(profiles.items())))
    )
    dense = _entry(result, "shared", "dense")
    sparse = _entry(result, "shared", "sparse")
    assert result["hypothesis_count"] == 2
    assert dense["selected"] == 100
    assert dense["status"] == "supported"
    assert sparse["selected"] == 8
    assert sparse["status"] == "insufficient_selected_data"
    assert sparse["empirical_selective_risk"] == 0.0
    assert reordered["family_sha256"] == result["family_sha256"]
    assert _entry(reordered, "shared", "dense") == dense
    assert _entry(reordered, "shared", "sparse") == sparse


def test_empty_and_failing_pairs_still_count_for_bonferroni():
    candidates = [_candidate("good", "good")]
    profiles = {"good": _samples(72, prefix="good")}
    alone = assess_cascade_family(candidates, profiles)
    candidates += [
        _candidate("bad", "bad"),
        _candidate("empty", "empty"),
        _candidate("escalating", "escalating"),
    ]
    profiles.update(
        bad=_samples(100, 100, prefix="bad"),
        empty=(),
        escalating=_samples(100, prefix="escalating", score=0.5),
    )
    family = assess_cascade_family(candidates, profiles)
    assert _entry(alone, "good", "good")["status"] == "supported"
    assert family["hypothesis_count"] == 4
    assert len(family["entries"]) == 4
    assert _entry(family, "good", "good")["status"] == "insufficient_selected_data"
    assert all(float(entry["threshold"]) == pytest.approx(0.0125) for entry in family["entries"])
    assert _entry(family, "bad", "bad")["status"] == "not_supported"


@pytest.mark.parametrize("overlap_profile", ["first", "other"])
def test_fit_ids_cannot_overlap_certification_in_any_profile(overlap_profile):
    candidates = [_candidate("first", "first"), _candidate("other", "other")]
    profiles = {
        "first": _samples(1, prefix="first"),
        "other": _samples(1, prefix="other"),
    }
    profiles[overlap_profile] = (CalibrationSample("fit-faithful", 0.0, 0),)
    with pytest.raises((ValueError, TypeError)) as raised:
        assess_cascade_family(candidates, profiles)
    assert str(raised.value)
    assert profiles[overlap_profile][0].sample_id in candidates[0].fit_sample_ids


@pytest.mark.parametrize(
    "case",
    [
        "duplicate_pair", "duplicate_sample", "cross_profile_sample", "duplicate_fit",
        "fit_count", "missing_profile", "extra_profile", "empty_family",
    ],
)
def test_ambiguous_ids_pairs_and_missing_profiles_fail_closed(case):
    candidates = [_candidate()]
    profiles = {"profile": _samples(2)}
    if case == "duplicate_pair":
        candidates.append(replace(candidates[0], scorer_sha256="b" * 64))
    elif case == "duplicate_sample":
        profiles["profile"] = (CalibrationSample("same", 0.0, 0),) * 2
    elif case == "cross_profile_sample":
        candidates.append(_candidate("candidate", "other"))
        profiles["other"] = profiles["profile"]
    elif case == "duplicate_fit":
        candidates[0] = replace(candidates[0], fit_sample_ids=("fit", "fit"))
    elif case == "fit_count":
        candidates[0] = replace(candidates[0], fit_sample_ids=("only-one",))
    elif case == "missing_profile":
        profiles = {}
    elif case == "extra_profile":
        profiles["extra"] = ()
    else:
        candidates = []
    with pytest.raises((ValueError, TypeError)) as raised:
        assess_cascade_family(candidates, profiles)
    assert str(raised.value)


@pytest.mark.parametrize("field", ["risk_limit", "error_level"])
@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -float("inf"), -0.1, 0, 1, 1.1, True, 0.1234567890123],
)
def test_invalid_risk_levels_and_excess_decimal_precision_are_rejected(field, value):
    with pytest.raises((ValueError, TypeError)) as raised:
        assess_cascade_family([_candidate()], {"profile": _samples(1)}, **{field: value})
    assert str(raised.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("score", float("nan")),
        ("score", float("inf")),
        ("score", -float("inf")),
        ("score", -0.001),
        ("score", 1.001),
        ("label", -1),
        ("label", 2),
        ("label", 0.5),
        ("label", True),
        ("label", 0.0),
        ("label", "0"),
        ("sample_id", ""),
        ("sample_id", "contains spaces"),
        ("candidate_id", ""),
        ("profile_id", ""),
        ("scorer_sha256", "a" * 63),
        ("scorer_sha256", "a" * 65),
        ("scorer_sha256", "A" * 64),
        ("scorer_sha256", "g" * 64),
        ("t_lo", float("nan")),
        ("t_hi", float("nan")),
        ("fit_score", float("nan")),
        ("fit_score", float("inf")),
        ("fit_score", -0.01),
        ("fit_score", 1.01),
        ("class_calibration", "empty"),
        ("class_calibration", "unsorted"),
        ("class_calibration", "list"),
        ("cutoffs", "reversed"),
    ],
)
def test_invalid_records_and_scorer_identity_are_rejected(field, value):
    with pytest.raises((ValueError, TypeError)) as raised:
        candidate = _candidate()
        sample = CalibrationSample("cert", 0.0, 0)
        if field in {"score", "label", "sample_id"}:
            sample = replace(sample, **{field: value})
        elif field in {"t_lo", "t_hi"}:
            candidate = replace(candidate, policy=replace(candidate.policy, **{field: value}))
        elif field in {"fit_score", "class_calibration"}:
            values = (value,)
            if field == "class_calibration":
                values = {"empty": (), "unsorted": (0.8, 0.2), "list": [0.75]}[value]
            cal = replace(candidate.policy.cal, halu_scores=values)
            candidate = replace(
                candidate,
                policy=replace(candidate.policy, cal=cal),
                fit_sample_ids=tuple(f"fit-{i}" for i in range(cal.n_h + cal.n_g)),
            )
        elif field == "cutoffs":
            candidate = replace(candidate, policy=replace(candidate.policy, t_lo=0.8, t_hi=0.2))
        else:
            candidate = replace(candidate, **{field: value})
        assess_cascade_family([candidate], {candidate.profile_id: (sample,)})
    assert str(raised.value)


@pytest.mark.parametrize("resource", ["candidates", "profile", "total", "fit"])
@pytest.mark.parametrize("over_limit", [False, True])
def test_resource_limits_accept_boundary_and_reject_one_more(resource, over_limit):
    candidates = [_candidate()]
    profiles = {"profile": _samples(1)}
    if resource == "candidates":
        candidates = [_candidate(f"candidate-{i}") for i in range(32 + over_limit)]
    elif resource == "profile":
        profiles = {"profile": _samples(4096 + over_limit)}
    elif resource == "total":
        candidates = [_candidate("candidate", f"profile-{i}") for i in range(4 + over_limit)]
        profiles = {
            f"profile-{i}": _samples(4096 if i < 4 else 1, prefix=f"profile-{i}")
            for i in range(4 + over_limit)
        }
    else:
        fit_n = 16384 + over_limit
        policy = replace(
            _policy(), cal=fit_cascade([0.25] * (fit_n - 1) + [0.75], [0] * (fit_n - 1) + [1])
        )
        candidates = [_candidate(policy=policy, fit_sample_ids=tuple(f"fit-{i}" for i in range(fit_n)))]
    if over_limit:
        with pytest.raises((ValueError, TypeError)) as raised:
            assess_cascade_family(candidates, profiles)
        assert str(raised.value)
    else:
        result = assess_cascade_family(candidates, profiles)
        assert result["hypothesis_count"] == len(candidates)
        assert len(result["entries"]) == len(candidates)
        assert result["production_authority"] is False


@pytest.mark.parametrize(
    "mutation", ["score", "label", "sample_id", "policy", "fit_score", "fit_id", "scorer", "risk", "error"]
)
def test_hashes_are_deterministic_and_sensitive_to_evidence(mutation):
    candidate = _candidate()
    samples = _samples(100)
    original = assess_cascade_family([candidate], {"profile": samples})
    repeated = assess_cascade_family([candidate], {"profile": tuple(reversed(samples))})
    changed_candidate = candidate
    changed_samples = samples
    options = {}
    if mutation in {"score", "label", "sample_id"}:
        value = {"score": 0.1, "label": 1, "sample_id": "renamed-cert"}[mutation]
        changed_samples = (replace(samples[0], **{mutation: value}),) + samples[1:]
    elif mutation == "policy":
        changed_candidate = replace(candidate, policy=replace(candidate.policy, t_lo=0.2))
    elif mutation == "fit_id":
        changed_candidate = replace(candidate, fit_sample_ids=("new-fit-faithful", "fit-hallucinated"))
    elif mutation == "fit_score":
        cal = replace(candidate.policy.cal, halu_scores=(0.8,))
        changed_candidate = replace(candidate, policy=replace(candidate.policy, cal=cal))
    elif mutation == "scorer":
        changed_candidate = replace(candidate, scorer_sha256="b" * 64)
    else:
        options = {"risk_limit" if mutation == "risk" else "error_level": 0.04}
    changed = assess_cascade_family([changed_candidate], {"profile": changed_samples}, **options)
    before = _entry(original)
    after = _entry(changed)
    assert original == repeated
    exported = repr(original)
    assert all(sample.sample_id not in exported for sample in samples)
    assert all(identity not in exported for identity in candidate.fit_sample_ids)
    assert changed["family_sha256"] != original["family_sha256"]
    for key in ("policy_sha256", "data_sha256", "scorer_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", before[key])
        assert re.fullmatch(r"[0-9a-f]{64}", after[key])
    assert before["scorer_sha256"] == SCORER_SHA
    if mutation in {"score", "label", "sample_id"}:
        assert after["data_sha256"] != before["data_sha256"]
        assert after["policy_sha256"] == before["policy_sha256"]
    elif mutation in {"policy", "fit_score", "fit_id"}:
        assert after["policy_sha256"] != before["policy_sha256"]
    elif mutation == "scorer":
        assert after["scorer_sha256"] == "b" * 64
        assert after["scorer_sha256"] != before["scorer_sha256"]
    assert changed["production_authority"] is False


@pytest.mark.parametrize("n,risk", [(100, 0.05), (4096, 0.5)])
def test_supported_assessment_declares_scope_without_witness_authority(n, risk):
    result = assess_cascade_family([_candidate()], {"profile": _samples(n)}, risk_limit=risk)
    assert _entry(result)["status"] == "supported"
    assert result["production_authority"] is False
    # Scope metadata must be explicit; it must not turn statistical support
    # on exchangeable held-out labels into per-input WITNESS authorization.
    assumptions = repr(result["assumptions_unverified"]).lower()
    assert "iid" in assumptions
    assert "independent" in assumptions
    assert "label" in assumptions
    assert "future distribution" in assumptions
    assert "population" in result["scope"] and "selective" in result["scope"]
    assert result.get("witness_authority", False) is False
    assert result["provider_calls"] == 0
    assert result["routing_changes"] == 0
    entry = _entry(result)
    assert entry["p_value_log10_approx"] == pytest.approx(n * log10(1 - risk))
    assert entry["p_value_underflow"] is (n == 4096)
    if n == 4096:
        assert entry["p_value_approx"] is None
    else:
        assert entry["p_value_approx"] == pytest.approx(float(_cdf(n, 0, risk)))
