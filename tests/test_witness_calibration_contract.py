"""Exact finite-exchangeability and fail-closed calibration regressions.

Each labelled position in a fixed score pool is equally likely to be the
future observation. Calibration order is irrelevant to these public APIs,
so holding out each position enumerates the full permutation law without
factorial work. Duplicate positions retain their multiplicity for ties.
"""

from __future__ import annotations

import json
import multiprocessing
from dataclasses import asdict
from fractions import Fraction
from itertools import combinations_with_replacement

import pytest

from entroly.witness_calibration import (
    CalibrationResult,
    CalibrationSample,
    CalibrationStore,
    ThresholdSet,
    calibrate,
    conformal_quantile,
)


ALPHAS = (
    (0.80, 0.50, 0.20),
    (0.95, 0.70, 0.40),
    (0.60, 0.30, 0.10),
    (0.75, 0.50, 0.25),
)
POOLS = (
    (0.4,),
    (0.1, 0.9),
    (0.1, 0.4, 0.9),
    (0.0, 0.2, 0.4, 0.7, 1.0),
    (0.0, 0.1, 0.3, 0.5, 0.8, 1.0),
    (0.4, 0.4, 0.4, 0.4, 0.4),
    (0.0, 0.0, 0.4, 0.4, 1.0, 1.0),
)
BAD_SCORES = (float("nan"), float("inf"), -float("inf"), True, False, -0.01, 1.01)
BAD_ALPHAS = (*BAD_SCORES, 0.0, 1.0)
TAUS = ("tau_pass", "tau_hedge", "tau_warn")


def _rank(n: int, alpha: float) -> int:
    """Use exact decimal probabilities, including the 5 * (1 - .8) boundary."""
    return int((n + 1) * (1 - Fraction(str(alpha))))


def _thresholds(result: CalibrationResult | ThresholdSet) -> tuple[float, ...]:
    return tuple(getattr(result, name) for name in TAUS)


def _assert_suppresses_all(thresholds: ThresholdSet) -> None:
    assert _thresholds(thresholds) == (0.0, 0.0, 0.0)
    assert all(thresholds.action(score) == "suppress" for score in (0.0, 0.1, 0.5, 1.0))


@pytest.mark.parametrize("pool", POOLS)
@pytest.mark.parametrize("alpha", (0.01, 0.20, 0.25, 0.40, 0.50, 0.80, 0.95))
def test_upper_quantile_covers_future_exchangeable_score(pool, alpha):
    covered = 0
    for i, future in enumerate(pool):
        calibration = list(pool[:i] + pool[i + 1 :])
        tau = conformal_quantile(calibration, alpha)
        assert 0.0 <= tau <= 1.0
        covered += future <= tau
    assert Fraction(covered, len(pool)) >= 1 - Fraction(str(alpha))


@pytest.mark.parametrize("scores", ([], [0.2], [0.1, 0.4], [0.1, 0.2, 0.3, 0.4]))
def test_upper_quantile_overflow_uses_known_upper_bound(scores):
    assert conformal_quantile(scores, alpha=0.01) == 1.0


def test_upper_quantile_retains_upper_order_statistic_direction():
    scores = [0.1, 0.2, 0.3, 0.4]
    assert conformal_quantile(scores, alpha=0.5) == 0.3
    assert conformal_quantile(scores, alpha=0.8) == 0.1


@pytest.mark.parametrize("pool", POOLS)
@pytest.mark.parametrize("alphas", ALPHAS)
def test_policy_lower_tail_exposure_on_future_exchangeable_score(pool, alphas):
    counts = [0, 0, 0]
    for i, future in enumerate(pool):
        samples = [CalibrationSample(rho=s, y=1) for j, s in enumerate(pool) if j != i]
        result = calibrate(
            samples, alpha_pass=alphas[0], alpha_hedge=alphas[1], alpha_warn=alphas[2]
        )
        policy = result.as_threshold_set()
        action = policy.action(future)
        for k, visible in enumerate((("pass",), ("pass", "hedge"), ("pass", "hedge", "warn"))):
            below = future < getattr(result, TAUS[k])
            assert (action in visible) == below
            counts[k] += below
    for count, alpha in zip(counts, alphas):
        # A marginal escalation target alpha bounds the strict lower tail
        # by 1-alpha, conditional on the future claim being hallucinated.
        assert Fraction(count, len(pool)) <= 1 - Fraction(str(alpha))
        rank = _rank(len(pool) - 1, alpha)
        assert count <= rank
        if len(set(pool)) == len(pool):
            assert count == rank


@pytest.mark.parametrize("n", range(7))
@pytest.mark.parametrize("alphas", ALPHAS)
def test_calibration_uses_conservative_lower_rank_including_decimal_boundary(n, alphas):
    scores = [i / 10 for i in range(1, n + 1)]
    result = calibrate(
        [CalibrationSample(rho=s, y=1) for s in reversed(scores)],
        alpha_pass=alphas[0], alpha_hedge=alphas[1], alpha_warn=alphas[2],
    )
    expected = tuple(scores[r - 1] if (r := _rank(n, a)) else 0.0 for a in alphas)
    assert _thresholds(result) == expected


def test_default_alphas_preserved_and_rank_depends_only_on_hallucinations():
    hallucinations = [CalibrationSample(rho=s, y=1) for s in (0.1, 0.2, 0.3, 0.4)]
    safe = [CalibrationSample(rho=s, y=0) for s in (0.0, 0.1, 0.9, 1.0)]
    result = calibrate(hallucinations + safe)
    assert (result.alpha_pass, result.alpha_hedge, result.alpha_warn) == ALPHAS[0]
    assert _thresholds(result) == (0.1, 0.2, 0.4)
    assert _thresholds(result) == _thresholds(calibrate(hallucinations))


@pytest.mark.parametrize("scores", ([], [0.0], [0.0, 0.5, 1.0]))
def test_no_hallucinations_is_unavailable_and_suppresses_every_score(scores):
    result = calibrate([CalibrationSample(rho=s, y=0) for s in scores])
    assert result.calibration_method == "exchangeable-hallucination-lower-tail-v2"
    assert result.calibration_status == "unavailable_no_hallucinations"
    assert (result.n_samples, result.n_hallucinated, result.n_safe) == (len(scores), 0, len(scores))
    assert result.empirical_exposure_at_warn is None
    assert result.empirical_retention_at_pass == (0.0 if scores else None)
    _assert_suppresses_all(result.as_threshold_set())


@pytest.mark.parametrize("n_safe", (0, 1, 2))
@pytest.mark.parametrize("n_hallucinated", (1, 2, 4))
def test_class_metrics_use_actual_denominators_and_explicit_method(n_safe, n_hallucinated):
    safe = [0.0, 0.5][:n_safe]
    hallucinations = [0.1, 0.3, 0.7, 0.9][:n_hallucinated]
    result = calibrate(
        [CalibrationSample(rho=s, y=0) for s in safe]
        + [CalibrationSample(rho=s, y=1) for s in hallucinations]
    )
    assert result.calibration_method == "exchangeable-hallucination-lower-tail-v2"
    assert result.calibration_status == "calibrated_marginal"
    assert (result.n_samples, result.n_hallucinated, result.n_safe) == (
        n_safe + n_hallucinated, n_hallucinated, n_safe
    )
    exposure = sum(s < result.tau_warn for s in hallucinations) / n_hallucinated
    assert result.empirical_exposure_at_warn == pytest.approx(exposure, abs=0.00005)
    if n_safe:
        retention = sum(s < result.tau_pass for s in safe) / n_safe
        assert result.empirical_retention_at_pass == pytest.approx(retention, abs=0.00005)
    else:
        assert result.empirical_retention_at_pass is None


def test_action_boundaries_are_strict_even_with_tied_thresholds():
    policy = ThresholdSet(0.2, 0.4, 0.6)
    assert [policy.action(s) for s in (0.0, 0.2, 0.4, 0.6, 1.0)] == [
        "pass", "hedge", "warn", "suppress", "suppress"
    ]
    tied = ThresholdSet(0.4, 0.4, 0.4)
    assert tied.action(0.39) == "pass"
    assert tied.action(0.4) == "suppress"


@pytest.mark.parametrize("score", BAD_SCORES)
def test_upper_quantile_rejects_invalid_scores(score):
    with pytest.raises(ValueError):
        conformal_quantile([0.2, score], alpha=0.5)


@pytest.mark.parametrize("alpha", BAD_ALPHAS)
@pytest.mark.parametrize("scores", ([], [0.2, 0.8]))
def test_upper_quantile_rejects_invalid_alpha_even_without_scores(alpha, scores):
    with pytest.raises(ValueError):
        conformal_quantile(scores, alpha)


@pytest.mark.parametrize("score", BAD_SCORES)
@pytest.mark.parametrize("label", (0, 1))
def test_calibrate_rejects_invalid_scores_in_either_class(score, label):
    with pytest.raises(ValueError):
        calibrate([CalibrationSample(rho=score, y=label)])


@pytest.mark.parametrize("alpha", BAD_ALPHAS)
@pytest.mark.parametrize("name", ("alpha_pass", "alpha_hedge", "alpha_warn"))
def test_calibrate_rejects_invalid_alpha_even_without_samples(alpha, name):
    with pytest.raises(ValueError):
        calibrate([], **{name: alpha})


@pytest.mark.parametrize("alphas", ((0.2, 0.5, 0.8), (0.8, 0.8, 0.2), (0.8, 0.2, 0.2)))
def test_calibrate_rejects_invalid_alpha_ordering(alphas):
    with pytest.raises(ValueError):
        calibrate([], alpha_pass=alphas[0], alpha_hedge=alphas[1], alpha_warn=alphas[2])


@pytest.mark.parametrize("label", (-1, 2, 0.5, float("nan"), True, False, "1", None))
def test_calibrate_rejects_invalid_labels(label):
    with pytest.raises(ValueError):
        calibrate([CalibrationSample(rho=0.5, y=label)])


@pytest.mark.parametrize("score", BAD_SCORES)
def test_policy_rejects_invalid_score_instead_of_taking_an_action(score):
    with pytest.raises(ValueError):
        ThresholdSet(0.2, 0.4, 0.6).action(score)


@pytest.mark.parametrize("score", BAD_SCORES)
@pytest.mark.parametrize("name", TAUS)
def test_threshold_set_rejects_invalid_thresholds(score, name):
    values = dict(zip(TAUS, (0.2, 0.4, 0.6)))
    values[name] = score
    with pytest.raises(ValueError):
        ThresholdSet(**values)


def _record() -> dict:
    return asdict(calibrate(
        [CalibrationSample(rho=s, y=1) for s in (0.1, 0.3, 0.7, 0.9)]
        + [CalibrationSample(rho=0.0, y=0)], profile="qa",
    ))


def _legacy_record() -> dict:
    record = _record()
    record.pop("calibration_method", None)
    record.pop("calibration_status", None)
    record.update(tau_pass=0.3, tau_hedge=0.6, tau_warn=0.9)
    return record


def test_direct_legacy_result_cannot_activate_permissive_thresholds():
    result = CalibrationResult(**_legacy_record())
    assert result.calibration_status == "unavailable_legacy"
    _assert_suppresses_all(result.as_threshold_set())


def _load_without_rewrite(tmp_path, payload: str) -> CalibrationStore:
    path = tmp_path / "calibration.json"
    path.write_text(payload, encoding="utf-8")
    before = path.read_bytes()
    store = CalibrationStore(path)
    # Exercise lazy getters as well as construction before checking the bytes.
    store.get_thresholds("qa")
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
    return store


def test_legacy_profile_fails_closed_without_rewrite_but_missing_profile_keeps_defaults(tmp_path):
    store = _load_without_rewrite(tmp_path, json.dumps({"qa": _legacy_record()}))
    _assert_suppresses_all(store.get_thresholds("qa"))
    assert _thresholds(store.get_thresholds("chat")) == (0.45, 0.65, 0.85)


@pytest.mark.parametrize("payload", ("{broken", "null", "[]", '"invalid"', "17"))
def test_corrupt_store_fails_closed_without_rewriting(tmp_path, payload):
    store = _load_without_rewrite(tmp_path, payload)
    _assert_suppresses_all(store.get_thresholds("qa"))
    assert store.get_thresholds("chat").action(0.0) == "suppress"


@pytest.mark.parametrize("record", ({}, None, [], 7, {"unexpected": "field"}))
def test_explicit_corrupt_profile_is_not_treated_as_missing(tmp_path, record):
    store = _load_without_rewrite(tmp_path, json.dumps({"qa": record}))
    _assert_suppresses_all(store.get_thresholds("qa"))
    assert _thresholds(store.get_thresholds("chat")) == (0.45, 0.65, 0.85)


@pytest.mark.parametrize("field,value", (
    ("tau_pass", float("nan")), ("tau_hedge", float("inf")),
    ("tau_warn", 1.01), ("tau_pass", True), ("tau_pass", 0.99),
    ("alpha_pass", 1.1), ("alpha_warn", -0.1), ("alpha_hedge", True),
    ("n_hallucinated", -1), ("n_safe", True), ("n_samples", 999),
    ("empirical_exposure_at_warn", float("nan")),
    ("empirical_retention_at_pass", 1.1),
    ("calibration_method", "unknown-method"),
    ("calibration_status", "unknown-status"),
))
def test_invalid_persisted_fields_cannot_activate_a_profile(tmp_path, field, value):
    record = _record()
    record[field] = value
    store = _load_without_rewrite(tmp_path, json.dumps({"qa": record}))
    _assert_suppresses_all(store.get_thresholds("qa"))
    assert _thresholds(store.get_thresholds("chat")) == (0.45, 0.65, 0.85)


def test_genuinely_missing_store_or_profile_preserves_heuristic_defaults(tmp_path):
    path = tmp_path / "missing.json"
    assert _thresholds(CalibrationStore(path).get_thresholds("qa")) == (0.35, 0.55, 0.80)
    assert not path.exists()
    store = _load_without_rewrite(tmp_path, "{}")
    assert _thresholds(store.get_thresholds("qa")) == (0.35, 0.55, 0.80)
    assert _thresholds(store.get_thresholds("unseen")) == (0.40, 0.60, 0.80)


def test_calibrated_store_round_trip_preserves_status_and_thresholds(tmp_path):
    result = CalibrationResult(**_record())
    path = tmp_path / "roundtrip.json"
    CalibrationStore(path).update(result)
    store = CalibrationStore(path)
    loaded = store.all_profiles()["qa"]
    assert loaded.calibration_method == "exchangeable-hallucination-lower-tail-v2"
    assert loaded.calibration_status == "calibrated_marginal"
    assert _thresholds(store.get_thresholds("qa")) == _thresholds(result)


def test_unavailable_calibration_stays_fail_closed_after_round_trip(tmp_path):
    path = tmp_path / "unavailable.json"
    CalibrationStore(path).update(calibrate([], profile="qa"))
    store = CalibrationStore(path)
    result = store.all_profiles()["qa"]
    assert result.calibration_status == "unavailable_no_hallucinations"
    assert result.empirical_exposure_at_warn is None
    assert result.empirical_retention_at_pass is None
    _assert_suppresses_all(store.get_thresholds("qa"))


def test_exhaustive_small_exchangeable_multisets_and_distinct_pools():
    # Enumerate future indices, not only observations used to fit the threshold.
    pools = []
    for m in range(2, 8):
        pools.extend(combinations_with_replacement((0.0, 0.5, 1.0), m))
        pools.append(tuple(i / (m - 1) for i in range(m)))
    for pool in pools:
        for alphas in ALPHAS:
            below = [0, 0, 0]
            covered = [0, 0, 0]
            for i, future in enumerate(pool):
                calibration = list(pool[:i] + pool[i + 1:])
                result = calibrate(
                    [CalibrationSample(rho=score, y=1) for score in calibration],
                    alpha_pass=alphas[0], alpha_hedge=alphas[1], alpha_warn=alphas[2],
                )
                for j, (alpha, tau) in enumerate(zip(alphas, _thresholds(result))):
                    below[j] += future < tau
                    covered[j] += future <= conformal_quantile(calibration, alpha)
            for j, alpha in enumerate(alphas):
                target = 1 - Fraction(str(alpha))
                assert Fraction(below[j], len(pool)) <= target, (pool, alpha, below[j])
                assert Fraction(covered[j], len(pool)) >= target, (pool, alpha, covered[j])


def test_rank_arithmetic_does_not_promote_across_decimal_boundaries():
    samples = [CalibrationSample(rho=score, y=1) for score in (0.1, 0.2, 0.3, 0.4)]
    assert calibrate(samples, alpha_pass=0.8).tau_pass == 0.1
    assert calibrate(samples, alpha_pass=0.8000000000000001).tau_pass == 0.0
    assert calibrate(samples, alpha_pass=0.7999999999999999).tau_pass == 0.1


def test_differently_scoped_samples_cannot_be_pooled_as_one_calibration():
    with pytest.raises(ValueError, match="profile"):
        calibrate([CalibrationSample(rho=0.4, y=1, profile="code")], profile="qa")


def test_dataset_digest_does_not_round_away_distinct_risk_scores():
    first = calibrate([CalibrationSample(rho=0.12000001, y=1)])
    second = calibrate([CalibrationSample(rho=0.12000002, y=1)])
    assert first.dataset_hash != second.dataset_hash
    assert first.dataset_crc32 != second.dataset_crc32


@pytest.mark.parametrize("payload", ("{broken", '{"qa": {}}'))
def test_updates_cannot_discard_corrupt_user_records(tmp_path, payload):
    path = tmp_path / "calibration.json"
    path.write_text(payload, encoding="utf-8")
    original = path.read_bytes()
    store = CalibrationStore(path)
    with pytest.raises(ValueError, match="corrupt"):
        store.update(calibrate([CalibrationSample(rho=0.4, y=1)], profile="chat"))
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
    _assert_suppresses_all(CalibrationStore(path).get_thresholds("qa"))


def test_alias_collisions_and_profile_identity_mismatch_fail_closed(tmp_path):
    record = _record()
    store = _load_without_rewrite(tmp_path, json.dumps({"qa": record, "QA": record}))
    _assert_suppresses_all(store.get_thresholds("qa"))
    mismatch = _load_without_rewrite(tmp_path, json.dumps({"chat": record}))
    _assert_suppresses_all(mismatch.get_thresholds("chat"))
    assert "chat" not in mismatch.all_profiles()


@pytest.mark.parametrize("entries", (
    (("qa", True), ("QA", True), (" qa ", True)),
    (("QA", False), ("qa", True)),
    (("qa", True), (" QA ", False), ("QA", True)),
))
def test_quarantined_alias_cannot_reappear_in_enumerated_profiles(tmp_path, entries):
    record = _record()
    payload = {name: record if valid else None for name, valid in entries}
    store = _load_without_rewrite(tmp_path, json.dumps(payload))
    _assert_suppresses_all(store.get_thresholds("qa"))
    assert store.all_profiles() == {}


def test_globally_invalid_store_exposes_no_enumerated_profiles(tmp_path):
    store = _load_without_rewrite(tmp_path, json.dumps({"": None, "qa": _record()}))
    _assert_suppresses_all(store.get_thresholds("qa"))
    assert store.all_profiles() == {}


def test_huge_numeric_inputs_do_not_escape_validation():
    with pytest.raises(ValueError):
        conformal_quantile([10 ** 500], 0.1)
    with pytest.raises(ValueError):
        CalibrationResult(**{**_record(), "timestamp": 10 ** 500})


def test_stale_update_preserves_newer_suppression_profile(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{}", encoding="utf-8")
    first, stale = CalibrationStore(path), CalibrationStore(path)
    first.update(calibrate([], profile="qa"))
    stale.update(calibrate([CalibrationSample(rho=0.6, y=1)], profile="chat"))
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {"qa", "chat"}
    _assert_suppresses_all(CalibrationStore(path).get_thresholds("qa"))


def test_existing_reader_observes_newer_suppression_profile(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{}", encoding="utf-8")
    reader, writer = CalibrationStore(path), CalibrationStore(path)
    assert reader.get_thresholds("qa").action(0.0) == "pass"
    writer.update(calibrate([], profile="qa"))
    _assert_suppresses_all(reader.get_thresholds("qa"))


def test_stale_update_refuses_corruption_created_after_construction(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{}", encoding="utf-8")
    stale = CalibrationStore(path)
    path.write_text('{"qa":null}', encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="corrupt"):
        stale.update(calibrate([CalibrationSample(rho=0.6, y=1)], profile="chat"))
    assert path.read_bytes() == original
    _assert_suppresses_all(stale.get_thresholds("qa"))


@pytest.mark.parametrize("duplicate_first", (True, False))
def test_literal_duplicate_json_profile_keys_are_ambiguous_and_not_rewritten(tmp_path, duplicate_first):
    good = json.dumps(_record())
    payload = '{"qa":null,"qa":' + good + '}' if duplicate_first else '{"qa":' + good + ',"qa":null}'
    path = tmp_path / "calibration.json"
    path.write_text(payload, encoding="utf-8")
    original = path.read_bytes()
    store = CalibrationStore(path)
    _assert_suppresses_all(store.get_thresholds("qa"))
    with pytest.raises(ValueError, match="corrupt"):
        store.update(calibrate([], profile="chat"))
    assert path.read_bytes() == original


@pytest.mark.parametrize("alphas", ((0.8, 0.5, 0.2), (0.95, 0.9, 0.8)))
def test_v2_rank_zero_threshold_invariant_is_checked_on_conversion_and_load(tmp_path, alphas):
    result = calibrate([CalibrationSample(rho=0.6, y=1)], profile="qa",
                       alpha_pass=alphas[0], alpha_hedge=alphas[1], alpha_warn=alphas[2])
    result.tau_pass = 0.5
    if alphas[2] == 0.8:
        result.tau_hedge = result.tau_warn = 0.5
    _assert_suppresses_all(result.as_threshold_set())
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"qa": asdict(result)}), encoding="utf-8")
    store = CalibrationStore(path)
    _assert_suppresses_all(store.get_thresholds("qa"))
    assert "qa" not in store.all_profiles()


def test_equal_nonzero_ranks_require_equal_thresholds():
    result = calibrate([CalibrationSample(rho=0.6, y=1)], profile="qa")
    result.tau_warn = 0.9
    with pytest.raises(ValueError, match="equal calibration ranks"):
        result.__post_init__()
    _assert_suppresses_all(result.as_threshold_set())


def _store_writer(path, profile, connection):
    try:
        store = CalibrationStore(path)
        connection.send("loaded")
        if connection.recv() != "update":
            raise AssertionError("unexpected worker command")
        store.update(calibrate([], profile=profile))
        connection.send("updated")
    finally:
        connection.close()


def test_independent_process_updates_preserve_both_profiles(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{}", encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    workers = []
    try:
        for profile in ("qa", "chat"):
            parent, child = context.Pipe()
            process = context.Process(target=_store_writer, args=(path, profile, child))
            process.start()
            child.close()
            workers.append((parent, process))
        for parent, _ in workers:
            assert parent.poll(10), "calibration writer did not load"
            assert parent.recv() == "loaded"
        # Both objects have loaded the original snapshot before either updates.
        for parent, _ in workers:
            parent.send("update")
        for parent, process in workers:
            assert parent.poll(10), "calibration writer did not finish"
            assert parent.recv() == "updated"
            process.join(timeout=5)
            assert process.exitcode == 0
        loaded = CalibrationStore(path)
        assert set(loaded.all_profiles()) == {"qa", "chat"}
        _assert_suppresses_all(loaded.get_thresholds("qa"))
        _assert_suppresses_all(loaded.get_thresholds("chat"))
        audit = path.with_suffix(path.suffix + ".audit.jsonl")
        assert {json.loads(line)["profile"] for line in audit.read_text(encoding="utf-8").splitlines()} == {"qa", "chat"}
    finally:
        for parent, process in workers:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            parent.close()
            assert not process.is_alive()
