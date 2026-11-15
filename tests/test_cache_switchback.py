import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import cache_switchback as switchback


def test_stationary_age_residual_and_arm():
    rng = random.Random(125)
    size = 20_000
    samples = [switchback._stationary_run(rng) for _ in range(size)]

    def survival_integral(age):
        early = (math.exp(-0.182 * age) - math.exp(-0.182 * 1.045)) / 0.182
        tail = math.exp(-0.182 * 1.045 - 0.885 * max(0, age - 1.045)) / 0.885
        return max(0, early) + tail

    mean = survival_integral(0)
    per_arm = survival_integral(1) / (2 * mean)
    assert switchback.PURE_PROBABILITY == pytest.approx(per_arm)
    for arm in (0, 1):
        assert sum(a == arm for _, _, a in samples) / size == pytest.approx(0.5, abs=0.012)
        exposed = sum(age >= 1 and a == arm for age, _, a in samples)
        assert exposed / size == pytest.approx(per_arm, abs=0.012)
    # Checking joint tails catches residuals drawn independently of the current age.
    for min_age, min_residual in ((0.0, 1.0), (0.35, 0.55), (0.8, 0.8), (1.2, 0.4)):
        joint = sum(age >= min_age and rem >= min_residual for age, rem, _ in samples)
        assert joint / size == pytest.approx(survival_integral(min_age + min_residual) / mean,
                                            abs=0.012)


def synthetic_rows(plan):
    arm = plan["initial_arm"]
    switches = iter(plan["switch_times"])
    next_switch = next(switches, math.inf)
    last_switch = 0
    rows = []
    counts = {"treatment": 0, "control": 0}
    for time in plan["observation_times"]:
        while next_switch <= time:
            arm = "treatment" if arm == "control" else "control"
            last_switch = next_switch
            next_switch = next(switches, math.inf)
        if last_switch <= time - plan["carryover"]:
            counts[arm] += 1
            outcome = 4 if arm == "treatment" else 1
        else:
            outcome = 10_000
        rows.append({"time": time, "arm": arm, "outcome": outcome})
    return rows, counts


def test_synthetic_recovery_keeps_contaminated_points_in_denominator():
    plan = switchback.plan(2, 3, 10_000, 2026)
    rows, counts = synthetic_rows(plan)
    result = switchback.report(plan, rows)
    assert result["status"] == "ok"
    assert result["production_authority"] is False
    assert result["scheduled_observations"] == 10_000
    assert result["pure_counts"] == counts
    assert result["contaminated_count"] == 10_000 - sum(counts.values()) > 0
    weighted_sum = 4 * counts["treatment"] - counts["control"]
    assert result["effect"] == pytest.approx(weighted_sum / (10_000 * switchback.PURE_PROBABILITY))
    assert result["effect"] == pytest.approx(3, abs=0.15)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "wrong_arm", "nan", "plan_changed"])
def test_invalid_input_fails_closed(defect):
    plan = switchback.plan(1, 1, 24, 12)
    rows, _ = synthetic_rows(plan)
    if defect == "missing":
        del rows[next(index for index, row in enumerate(rows) if row["outcome"] == 10_000)]
    elif defect == "duplicate":
        rows[-1] = rows[0]
    elif defect == "wrong_arm":
        rows[0]["arm"] = "wrong"
    elif defect == "nan":
        rows[0]["outcome"] = math.nan
    elif defect == "plan_changed":
        plan["observation_times"][0] = 0
    with pytest.raises(ValueError):
        switchback.report(plan, rows)


def test_cli_plan_and_report(tmp_path):
    script = Path(switchback.__file__)

    def run(*args):
        return subprocess.run([sys.executable, "-S", str(script), *args],
                              capture_output=True, text=True, check=False)

    for count, expected_exit in ((120, 0), (1, 1)):
        planned = run("plan", "--carryover", "2", "--spacing", "3", "--count", str(count),
                      "--seed", "2026")
        assert planned.returncode == 0, planned.stderr
        plan = json.loads(planned.stdout)
        assert plan["observation_times"] == [2 + 3 * index for index in range(count)]
        assert plan["production_authority"] is False
        plan_file = tmp_path / "plan.json"
        rows_file = tmp_path / "observations.json"
        plan_file.write_text(planned.stdout, encoding="utf-8")
        rows, _ = synthetic_rows(plan)
        rows_file.write_text(json.dumps(rows), encoding="utf-8")

        reported = run("report", "--plan", str(plan_file), "--observations", str(rows_file))
        assert reported.returncode == expected_exit, reported.stderr
        result = json.loads(reported.stdout)
        if expected_exit:
            assert result["status"] == "not_estimable"
            assert result["effect"] is None
        else:
            assert result["status"] == "ok"
            assert result["effect"] is not None

    rows_file.write_text("[]", encoding="utf-8")
    invalid = run("report", "--plan", str(plan_file), "--observations", str(rows_file))
    assert invalid.returncode == 2
    assert invalid.stdout == ""
