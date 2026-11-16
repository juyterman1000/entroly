"""Finite-sample statistics behind `entroly trial` and the accuracy benchmark.

Each guarantee is checked by exhaustive enumeration, not by simulation, so the
tests are deterministic and assert the stated error rate itself.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math

import pytest

from entroly import trial_statistics as ts


def _binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


# ── rank-sum test ─────────────────────────────────────────────────────────


def test_rank_sum_p_value_matches_enumeration_with_ties():
    pooled = [3, 1, 4, 1, 5, 9, 2, 6]
    baseline, optimized = pooled[:4], pooled[4:]

    def statistic(indices):
        return sum(
            sum(1 for v in pooled if v < pooled[i])
            + 0.5 * (sum(1 for v in pooled if v == pooled[i]) + 1)
            for i in indices
        )

    observed = statistic(range(4, 8))
    subsets = list(itertools.combinations(range(8), 4))
    expected = sum(1 for c in subsets if statistic(c) <= observed + 1e-9) / len(subsets)
    assert ts.rank_sum_test(baseline, optimized)["one_sided_p_value"] == pytest.approx(expected)


@pytest.mark.parametrize("runs", [3, 4, 5])
def test_rank_sum_false_positive_rate_is_at_most_alpha(runs):
    """Under exchangeability every labelling is equally likely.

    Enumerate all of them for one fixed set of values: the share with
    p <= alpha is the exact false-positive rate, and it must not exceed alpha.
    """
    values = [float(v) for v in (11, 7, 7, 19, 3, 23, 5, 13, 2, 17)][: 2 * runs]
    rejections = 0
    labelings = list(itertools.combinations(range(2 * runs), runs))
    for chosen in labelings:
        optimized = [values[i] for i in chosen]
        baseline = [values[i] for i in range(2 * runs) if i not in chosen]
        if ts.rank_sum_test(baseline, optimized)["one_sided_p_value"] <= 0.05:
            rejections += 1
    assert rejections / len(labelings) <= 0.05


def test_three_runs_per_arm_can_reach_but_not_beat_one_in_twenty():
    result = ts.rank_sum_test([10_000, 10_100, 9_900], [5_800, 5_900, 6_000])
    assert result["exact"] is True
    assert result["one_sided_p_value"] == pytest.approx(1 / 20)
    assert result["smallest_attainable_p_value"] == pytest.approx(1 / 20)
    # One overlapping run is enough to lose the conclusion at n = 3.
    overlap = ts.rank_sum_test([10_000, 10_100, 5_850], [5_800, 5_900, 6_000])
    assert overlap["one_sided_p_value"] > 0.05


def test_subset_sum_counts_match_enumeration():
    weights = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    for size in range(len(weights) + 1):
        expected = [0] * (sum(sorted(weights, reverse=True)[:size]) + 1)
        for chosen in itertools.combinations(range(len(weights)), size):
            expected[sum(weights[i] for i in chosen)] += 1
        assert ts._subset_sum_counts(weights, size) == expected


def test_results_stay_exact_at_the_run_count_the_report_recommends():
    """36 runs per arm is what the report asks for; it must not fall back to
    an approximation exactly there."""
    baseline = [10_000 + 13 * i for i in range(36)]
    optimized = [6_000 + 11 * i for i in range(36)]
    test = ts.rank_sum_test(baseline, optimized)
    assert test["exact"] is True
    assert test["one_sided_p_value"] == pytest.approx(1 / math.comb(72, 36))
    assert ts.reduction_lower_bound(baseline, optimized)["exact"] is True


def test_large_samples_are_labelled_approximate():
    baseline = [10_000 + 7 * i for i in range(70)]
    optimized = [6_000 + 5 * i for i in range(70)]
    test = ts.rank_sum_test(baseline, optimized)
    assert test["exact"] is False
    assert test["smallest_attainable_p_value"] is None
    assert test["one_sided_p_value"] < 1e-6
    bound = ts.reduction_lower_bound(baseline, optimized)
    assert bound["exact"] is False
    assert 0 < bound["lower_bound"] <= bound["estimate"]


def test_normal_approximation_agrees_with_the_exact_test_near_the_limit(monkeypatch):
    values = [float((i * 37) % 101) for i in range(100)]
    baseline, optimized = values[:50], [v - 9.0 for v in values[50:]]
    exact = ts.rank_sum_test(baseline, optimized)
    monkeypatch.setattr(ts, "_EXACT_POOLED_LIMIT", 10)
    approximate = ts.rank_sum_test(baseline, optimized)
    assert exact["exact"] is True and approximate["exact"] is False
    assert approximate["one_sided_p_value"] == pytest.approx(exact["one_sided_p_value"], abs=0.01)


def test_rank_sum_rejects_empty_and_non_finite_input():
    with pytest.raises(ValueError):
        ts.rank_sum_test([], [1.0])
    with pytest.raises(ValueError):
        ts.rank_sum_test([1.0, float("nan")], [1.0])


# ── reduction bound ───────────────────────────────────────────────────────


def test_reduction_bound_is_the_smallest_difference_at_three_runs():
    result = ts.reduction_lower_bound([10_000, 10_100, 9_900], [5_800, 5_900, 6_000])
    assert result["estimate"] == 4_100
    assert result["lower_bound"] == 3_900  # min baseline - max optimized
    assert result["achieved_confidence"] == pytest.approx(0.95)


def test_reduction_bound_is_withheld_when_no_order_statistic_qualifies():
    result = ts.reduction_lower_bound([10.0, 11.0], [5.0, 6.0])
    assert result["lower_bound"] is None
    assert result["estimate"] == 5.0


def test_reduction_bound_coverage_is_exact_under_a_shift():
    """Enumerate every labelling of shifted data: the bound must sit at or
    below the true shift in at least 95% of them."""
    noise = [0.3, 1.1, 2.6, 4.2, 5.9, 7.5, 8.8, 9.4]
    shift = 40.0
    covered = 0
    labelings = list(itertools.combinations(range(8), 4))
    for chosen in labelings:
        optimized = [noise[i] for i in chosen]
        baseline = [noise[i] + shift for i in range(8) if i not in chosen]
        bound = ts.reduction_lower_bound(baseline, optimized)["lower_bound"]
        assert bound is not None
        covered += bound <= shift + 1e-9
    assert covered / len(labelings) >= 0.95


def test_relative_bound_is_the_worst_ratio_at_three_runs():
    result = ts.relative_reduction_lower_bound([10_000, 10_100, 9_900], [5_800, 5_900, 6_000])
    assert result["estimate"] == pytest.approx(0.41)
    assert result["lower_bound"] == pytest.approx(1 - 6_000 / 9_900)
    assert ts.relative_reduction_lower_bound([10.0, 0.0, 4.0], [5.0, 6.0, 7.0]) is None


def test_relative_bound_coverage_is_exact_under_a_constant_ratio():
    noise = [1.03, 1.11, 1.26, 1.42, 1.59, 1.75, 1.88, 1.94]
    kept = 0.6  # optimized uses 60% of baseline: a 40% reduction
    covered = 0
    labelings = list(itertools.combinations(range(8), 4))
    for chosen in labelings:
        optimized = [1_000 * noise[i] * kept for i in chosen]
        baseline = [1_000 * noise[i] for i in range(8) if i not in chosen]
        bound = ts.relative_reduction_lower_bound(baseline, optimized)["lower_bound"]
        covered += bound <= (1 - kept) + 1e-9
    assert covered / len(labelings) >= 0.95


def test_reduction_bounds_are_withheld_instead_of_sorting_millions_of_pairs(monkeypatch):
    monkeypatch.setattr(ts, "_MAX_PAIRS", 8)
    baseline, optimized = [10.0, 11.0, 12.0], [5.0, 6.0, 7.0]
    assert ts.reduction_lower_bound(baseline, optimized)["lower_bound"] is None
    result = ts.compare_arms(baseline_measurements=baseline, optimized_measurements=optimized)
    assert result["conclusions"]["fewer_input_tokens"] == "supported"  # the test still runs
    assert result["input_tokens"]["relative_reduction_lower_bound"] is None
    assert not any("Typical reduction" in sentence for sentence in result["summary"])


# ── success-rate bounds ───────────────────────────────────────────────────


@pytest.mark.parametrize("trials,rate", [(3, 0.9), (12, 0.8), (40, 0.95), (60, 0.5)])
def test_proportion_bounds_cover_at_the_stated_rate(trials, rate):
    alpha = 0.05
    miss_low = sum(
        _binom_pmf(k, trials, rate)
        for k in range(trials + 1)
        if ts.proportion_lower_bound(k, trials, alpha) > rate
    )
    miss_high = sum(
        _binom_pmf(k, trials, rate)
        for k in range(trials + 1)
        if ts.proportion_upper_bound(k, trials, alpha) < rate
    )
    assert miss_low <= alpha + 1e-9
    assert miss_high <= alpha + 1e-9


def test_three_perfect_runs_do_not_demonstrate_preserved_quality():
    lower = ts.success_difference_lower_bound(3, 3, 3, 3)
    assert lower < -0.5  # a drop of more than half is still compatible
    assert ts.runs_needed_for_noninferiority(0.10) == 36
    assert ts.success_difference_lower_bound(36, 36, 36, 36) >= -0.10
    assert ts.success_difference_lower_bound(35, 35, 35, 35) < -0.10


@pytest.mark.parametrize("rate_base,rate_cand", [(0.9, 0.9), (0.95, 0.7), (0.5, 0.6)])
def test_success_difference_bound_covers_for_independent_arms(rate_base, rate_cand):
    runs = 12
    truth = rate_cand - rate_base
    miss = 0.0
    for b in range(runs + 1):
        for c in range(runs + 1):
            if ts.success_difference_lower_bound(b, runs, c, runs) > truth + 1e-12:
                miss += _binom_pmf(b, runs, rate_base) * _binom_pmf(c, runs, rate_cand)
    assert miss <= 0.05 + 1e-9


def test_paired_bound_is_much_tighter_than_two_separate_intervals():
    """90/100 baseline, 88/100 compressed, arms agree on all but 4 items."""
    paired = ts.paired_success_difference_lower_bound(3, 1, 100)
    unpaired = ts.success_difference_lower_bound(90, 100, 88, 100)
    assert paired > -0.09
    assert unpaired < -0.12
    assert paired - unpaired > 0.05


def test_paired_bound_covers_at_the_stated_rate():
    pairs, p_loss, p_gain = 30, 0.10, 0.05
    truth = p_gain - p_loss
    miss = 0.0
    for loss in range(pairs + 1):
        for gain in range(pairs + 1 - loss):
            probability = (
                math.factorial(pairs)
                / (math.factorial(loss) * math.factorial(gain) * math.factorial(pairs - loss - gain))
                * p_loss**loss
                * p_gain**gain
                * (1 - p_loss - p_gain) ** (pairs - loss - gain)
            )
            if ts.paired_success_difference_lower_bound(loss, gain, pairs) > truth + 1e-12:
                miss += probability
    assert miss <= 0.05 + 1e-9


def test_paired_comparison_counts_and_verdict():
    baseline = [True] * 60 + [False] * 4
    candidate = [True] * 60 + [False] * 4
    same = ts.paired_success_comparison(baseline, candidate, margin=0.10)
    assert same["only_baseline_succeeded"] == same["only_candidate_succeeded"] == 0
    assert same["noninferior"] is True

    worse = ts.paired_success_comparison([True] * 20, [True] * 12 + [False] * 8, margin=0.10)
    assert worse["only_baseline_succeeded"] == 8
    assert worse["observed_difference"] == pytest.approx(-0.4)
    assert worse["noninferior"] is False

    assert ts.paired_success_comparison([], [])["status"] == "no-paired-outcomes"
    with pytest.raises(ValueError):
        ts.paired_success_comparison([True], [True, False])


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        ts.success_difference_lower_bound(4, 3, 3, 3)
    with pytest.raises(ValueError):
        ts.success_difference_lower_bound(3, 3, 3, 3, confidence=1.0)
    with pytest.raises(ValueError):
        ts.paired_success_difference_lower_bound(3, 3, 5)
    with pytest.raises(ValueError):
        ts.runs_needed_for_noninferiority(0.0)


# ── run order ─────────────────────────────────────────────────────────────


def test_arm_order_distinguishes_blocked_from_interleaved():
    blocked = ts.arm_order(["baseline"] * 3 + ["optimized"] * 3)
    assert blocked == {"runs": 6, "switches": 1, "layout": "blocked"}
    assert ts.arm_order(["baseline", "optimized"] * 3)["layout"] == "interleaved"
    assert ts.arm_order(["baseline"] * 3)["layout"] == "single-arm"


# ── the trial report ──────────────────────────────────────────────────────


def _receipt(arm: str, tokens: int, digest: str, *, success=True, retained=True, cost=0.02) -> dict:
    return {
        "schema_version": "entroly.trial-run.v2",
        "experiment": "exp",
        "arm": arm,
        "command_sha256": digest,
        "traffic": {"evidence_gate": "passed"},
        "usage": {"provider_reported_active_input_tokens": tokens},
        "quality": {"task_success": success, "evidence_retained": retained},
        "economics": {"cost_usd": cost},
    }


def _write_runs(directory, runs, *, interleaved=True, evaluated=True, cost=None):
    """``cost`` maps an arm to its base cost per run; None keeps every run equal."""
    digest = "sha256:" + hashlib.sha256(b"pytest -q").hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.json"):
        stale.unlink()
    clock = 1_000
    sequence = (
        [arm for _ in range(runs) for arm in ("baseline", "optimized")]
        if interleaved
        else ["baseline"] * runs + ["optimized"] * runs
    )
    seen = {"baseline": 0, "optimized": 0}
    for arm in sequence:
        index = seen[arm]
        seen[arm] += 1
        tokens = (10_000 if arm == "baseline" else 5_800) + 37 * index
        receipt = _receipt(
            arm, tokens, digest,
            success=True if evaluated else None,
            retained=True if evaluated else None,
            cost=0.02 if cost is None else (
                None if cost[arm] is None else round(cost[arm] + 0.0001 * index, 6)
            ),
        )
        (directory / f"{clock}-{arm}.json").write_text(json.dumps(receipt), encoding="utf-8")
        clock += 1


@pytest.fixture
def workflows(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    from entroly import cli_context_workflows

    return cli_context_workflows


def test_statistics_exist_only_when_the_comparison_is_supported(workflows):
    directory = workflows._experiment_dir("exp")
    for runs in (1, 2):
        _write_runs(directory, runs)
        comparison = workflows._trial_report("exp")["comparison"]
        assert comparison["status"] == "insufficient-evidence"
        assert comparison["statistics"] is None
    _write_runs(directory, 3)
    comparison = workflows._trial_report("exp")["comparison"]
    assert comparison["status"] == "directional"
    assert comparison["statistics"] is not None


def test_three_clean_runs_support_fewer_tokens_but_not_quality(workflows):
    directory = workflows._experiment_dir("exp")
    _write_runs(directory, 3)
    stats = workflows._trial_report("exp")["comparison"]["statistics"]

    assert stats["conclusions"]["fewer_input_tokens"] == "supported"
    assert stats["input_tokens"]["one_sided_p_value"] == pytest.approx(0.05)
    assert stats["input_tokens"]["exact"] is True
    # Smallest baseline run minus largest optimized run.
    assert stats["input_tokens"]["reduction_lower_bound"] == 10_000 - (5_800 + 74)

    assert stats["conclusions"]["task_success"] == "not-demonstrated"
    assert stats["task_success"]["noninferior_within_margin"] is False
    assert stats["task_success"]["largest_drop_not_ruled_out"] > 0.5
    assert stats["task_success"]["runs_per_arm_needed_if_every_run_succeeds"] == 36
    assert stats["order"]["layout"] == "interleaved"


def test_unevaluated_runs_report_quality_as_not_evaluated(workflows):
    directory = workflows._experiment_dir("exp")
    _write_runs(directory, 3, evaluated=False)
    stats = workflows._trial_report("exp")["comparison"]["statistics"]
    assert stats["task_success"]["status"] == "not-evaluated"
    assert stats["conclusions"]["task_success"] == "not-evaluated"


def test_blocked_runs_carry_a_warning(workflows):
    directory = workflows._experiment_dir("exp")
    _write_runs(directory, 3, interleaved=False)
    stats = workflows._trial_report("exp")["comparison"]["statistics"]
    assert stats["order"]["layout"] == "blocked"
    assert any(text.startswith("WARNING") for text in stats["assumptions"])


def test_custom_receipt_names_leave_the_order_unknown(workflows):
    directory = workflows._experiment_dir("exp")
    _write_runs(directory, 3)
    for index, path in enumerate(sorted(directory.glob("*.json"))):
        path.rename(directory / f"run-{index}.json")
    stats = workflows._trial_report("exp")["comparison"]["statistics"]
    assert stats["order"] == {"layout": "unknown"}


def test_enough_perfect_runs_demonstrate_quality_within_margin(workflows):
    directory = workflows._experiment_dir("exp")
    _write_runs(directory, 36)
    stats = workflows._trial_report("exp")["comparison"]["statistics"]
    assert stats["conclusions"] == {
        "fewer_input_tokens": "supported",
        "lower_cost": "not-supported",  # every run cost the same
        "task_success": "noninferior-within-margin",
    }
    assert stats["input_tokens"]["exact"] is True


def test_fewer_tokens_with_a_higher_bill_is_reported_as_a_higher_bill(workflows):
    """Input tokens can fall while cost rises, for example when a changed
    prompt prefix stops the provider's cache from hitting. The report must
    not let the token result stand in for the bill."""
    directory = workflows._experiment_dir("exp")
    _write_runs(directory, 3, cost={"baseline": 0.030, "optimized": 0.040})
    stats = workflows._trial_report("exp")["comparison"]["statistics"]
    assert stats["conclusions"]["fewer_input_tokens"] == "supported"
    assert stats["conclusions"]["lower_cost"] == "contradicted"
    assert stats["cost_usd"]["opposite_direction_p_value"] == pytest.approx(0.05)
    assert any("HIGHER cost" in sentence for sentence in stats["summary"])


def test_lower_cost_is_supported_with_a_bound_when_every_run_has_a_cost(workflows):
    directory = workflows._experiment_dir("exp")
    _write_runs(directory, 4, cost={"baseline": 0.050, "optimized": 0.030})
    stats = workflows._trial_report("exp")["comparison"]["statistics"]
    assert stats["conclusions"]["lower_cost"] == "supported"
    assert stats["cost_usd"]["one_sided_p_value"] == pytest.approx(1 / 70)
    # At four runs per arm the exact bound may pass over one pairing, so it is
    # the second-worst ratio, not the worst (0.0303 against 0.0500).
    assert stats["cost_usd"]["relative_reduction_lower_bound"] == pytest.approx(
        1 - 0.0303 / 0.0501, abs=1e-6
    )
    assert stats["cost_usd"]["reduction_bound_confidence"] == pytest.approx(1 - 2 / 70)


def test_cost_missing_from_any_run_means_no_cost_comparison(workflows):
    directory = workflows._experiment_dir("exp")
    _write_runs(directory, 3, cost={"baseline": 0.05, "optimized": None})
    stats = workflows._trial_report("exp")["comparison"]["statistics"]
    assert stats["cost_usd"] == {"status": "not-recorded-for-every-run"}
    assert stats["conclusions"]["lower_cost"] == "not-evaluated"


def test_summary_never_states_more_than_the_conclusions(workflows):
    directory = workflows._experiment_dir("exp")
    digest = "sha256:" + hashlib.sha256(b"pytest -q").hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    # Overlapping arms: no token conclusion is available.
    for index, (arm, tokens) in enumerate(
        [("baseline", 10_000), ("optimized", 9_990), ("baseline", 10_040),
         ("optimized", 10_060), ("baseline", 10_020), ("optimized", 10_050)]
    ):
        (directory / f"{1000 + index}-{arm}.json").write_text(
            json.dumps(_receipt(arm, tokens, digest)), encoding="utf-8"
        )
    stats = workflows._trial_report("exp")["comparison"]["statistics"]
    assert stats["conclusions"]["fewer_input_tokens"] == "not-supported"
    text = " ".join(stats["summary"])
    assert "do not show lower input tokens" in text
    assert "Typical reduction" not in text
    assert "NOT shown to be preserved" in text
    assert "not a causal" in text


def test_summary_states_the_supported_saving_and_its_floor(workflows):
    directory = workflows._experiment_dir("exp")
    _write_runs(directory, 3)
    stats = workflows._trial_report("exp")["comparison"]["statistics"]
    text = " ".join(stats["summary"])
    assert "lower input tokens" in text
    assert "the smallest value these run counts can produce" in text
    assert "at least 41.3%" in text  # 1 - 5,874 / 10,000
    assert "71 percentage points" in text
    assert "at least 36 runs per arm" in text


def test_optimized_arm_using_more_tokens_is_contradicted(workflows):
    directory = workflows._experiment_dir("exp")
    digest = "sha256:" + hashlib.sha256(b"pytest -q").hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        for offset, (arm, tokens) in enumerate((("baseline", 5_000), ("optimized", 9_000))):
            (directory / f"{1000 + 2 * index + offset}-{arm}.json").write_text(
                json.dumps(_receipt(arm, tokens + index, digest)), encoding="utf-8"
            )
    comparison = workflows._trial_report("exp")["comparison"]
    assert comparison["provider_input_token_difference"] < 0
    assert comparison["statistics"]["conclusions"]["fewer_input_tokens"] == "contradicted"


def test_report_is_json_serialisable_and_deterministic(workflows):
    directory = workflows._experiment_dir("exp")
    _write_runs(directory, 4)
    first = json.dumps(workflows._trial_report("exp"), sort_keys=True)
    second = json.dumps(workflows._trial_report("exp"), sort_keys=True)
    assert first == second


# ── accuracy benchmark ────────────────────────────────────────────────────


def test_accuracy_benchmark_reports_a_paired_comparison():
    from bench.accuracy import BenchmarkResult, _paired_comparison

    def result(mode, outcomes):
        details = [
            {"index": i, "correct": ok} if ok is not None else {"index": i, "error": "boom"}
            for i, ok in enumerate(outcomes)
        ]
        correct = sum(bool(ok) for ok in outcomes)
        return BenchmarkResult(mode, mode, len(outcomes), correct, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               details=details)

    baseline = result("baseline", [True] * 40 + [False] * 8 + [None, True])
    treatment = result("entroly", [True] * 38 + [False] * 10 + [True, None])
    paired = _paired_comparison(baseline, treatment)
    assert paired["pairs"] == 48
    assert paired["items_excluded_for_errors"] == 2
    assert paired["only_baseline_succeeded"] == 2
    assert paired["only_candidate_succeeded"] == 0
    assert paired["difference_lower_bound"] < paired["observed_difference"] < 0
