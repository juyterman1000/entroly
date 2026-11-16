"""Exact trace tests for loss admission, charging, and paired reporting."""

import hashlib
import json
import subprocess
import sys

import pytest

from entroly.energy_queue import compare_trace


def worker(id="w", *, capacity=10, initial=10, watts=0, charging=()):
    return {
        "id": id,
        "capacity_j": capacity,
        "initial_j": initial,
        "recharge_w": watts,
        "charging": charging,
    }


def job(id, arrival, service, joules):
    return {"id": id, "arrival_s": arrival, "service_s": service, "measured_j": joules}


def trace(workers, jobs):
    return json.dumps({"workers": workers, "jobs": jobs}).encode()


def pairs(result):
    return result["first_fit"], result["energy_aware"]


def test_paired_policies_route_the_same_trace_differently():
    data = trace(
        [worker("high"), worker("low", capacity=4, initial=4)],
        [job("small", 0, 1, 4), job("large", 1, 2, 9), job("last", 3, 1, 1)],
    )
    result = compare_trace(data)
    first, aware = pairs(result)
    assert result["trace_sha256"] == hashlib.sha256(data).hexdigest()
    assert result["offered_jobs"] == 3
    assert result["horizon_s"] == 4
    assert result["paired_completions"] == {
        "only_first_fit": [],
        "only_energy_aware": ["large"],
    }
    assert [(row["id"], row["worker"]) for row in first["completed_jobs"]] == [
        ("small", "high"),
        ("last", "high"),
    ]
    assert first["rejected_jobs"] == [
        {"id": "large", "arrival_s": 1, "service_s": 2, "measured_j": 9},
    ]
    assert first["rejected_job_j"] == 9
    assert [(row["id"], row["worker"]) for row in aware["completed_jobs"]] == [
        ("small", "low"),
        ("large", "high"),
        ("last", "high"),
    ]
    assert aware["rejected"] == 0
    assert first["consumed_j"] == 5
    assert aware["consumed_j"] == 14
    assert aware["mean_latency_s"] == pytest.approx(4 / 3)
    assert aware["max_latency_s"] == 2
    assert aware["completed_jobs"][1]["finish_s"] == 3


def test_energy_aware_can_lose_work_when_it_occupies_the_only_later_option():
    result = compare_trace(
        trace(
            [worker("high", charging=[[2, 4]]), worker("low", capacity=4, initial=4)],
            [job("early", 0, 2, 1), job("later", 1, 2, 2)],
        )
    )
    first, aware = pairs(result)
    assert [row["worker"] for row in first["completed_jobs"]] == ["high", "low"]
    assert [row["worker"] for row in aware["completed_jobs"]] == ["low"]
    assert [row["id"] for row in aware["rejected_jobs"]] == ["later"]
    assert result["paired_completions"] == {
        "only_first_fit": ["later"],
        "only_energy_aware": [],
    }


def test_charging_ties_and_capacity_with_exact_fractional_inputs():
    data = trace(
        [worker(capacity=0.3, initial=0.1, watts=0.6, charging=[[0.3, 0.8], [2, 3]])],
        [
            job("charge_end", 0.8, 0.1, 0.3),
            job("would_cross", 0, 0.4, 0),
            job("ends_at_charge", 0.1, 0.2, 0.1),
            job("charge_start", 0.3, 0.1, 0),
            job("half_charged", 0.5, 0.1, 0),
            job("over_capacity", 3, 0.1, 0.300001),
            job("at_capacity", 3, 0.1, 0.3),
        ],
    )
    result = compare_trace(data)
    first, aware = pairs(result)
    assert first == aware
    assert [row["id"] for row in first["completed_jobs"]] == [
        "ends_at_charge",
        "charge_end",
        "at_capacity",
    ]
    assert [row["id"] for row in first["rejected_jobs"]] == [
        "would_cross",
        "charge_start",
        "half_charged",
        "over_capacity",
    ]
    assert first["completed_jobs"][0]["finish_s"] == 0.3
    balance = first["workers"][0]
    assert balance["recharged_j"] == 0.6
    assert balance["consumed_j"] == 0.7
    assert balance["final_j"] == 0
    assert balance["initial_j"] + balance["recharged_j"] == pytest.approx(
        balance["consumed_j"]
    )


def test_uses_a_feasible_worker_and_preserves_equal_time_file_order():
    result = compare_trace(
        trace(
            [
                worker("charging", charging=[[0, 2]]),
                worker("empty", initial=0),
                worker("ready"),
            ],
            [job("first", 0, 1, 1), job("second", 0, 0.1, 1), job("third", 1, 1, 1)],
        )
    )
    for report in pairs(result):
        assert [(row["id"], row["worker"]) for row in report["completed_jobs"]] == [
            ("first", "ready"),
            ("third", "ready"),
        ]
        assert [row["id"] for row in report["rejected_jobs"]] == ["second"]


def test_adjacent_charging_windows_do_not_open_a_service_gap():
    result = compare_trace(
        trace(
            [worker(initial=0, watts=1, charging=[[0, 1], [1, 2], [3, 6]])],
            [job("closed", 1, 0.1, 1), job("open", 2, 1, 2), job("horizon", 4, 1, 1)],
        )
    )
    for report in pairs(result):
        assert [row["id"] for row in report["completed_jobs"]] == ["open"]
        assert report["workers"][0]["recharged_j"] == 4
        assert report["workers"][0]["final_j"] == 2
    assert result["horizon_s"] == 5


def test_empty_trace_has_no_latency_or_elapsed_charging():
    result = compare_trace(trace([worker(initial=2, watts=2, charging=[[0, 5]])], []))
    assert result["horizon_s"] == 0
    for report in pairs(result):
        assert report["completed"] == report["rejected"] == 0
        assert report["mean_latency_s"] is None
        assert report["max_latency_s"] is None
        assert report["workers"][0]["final_j"] == 2


@pytest.mark.parametrize(
    ("workers", "jobs", "error"),
    [
        ([], [], "at least one"),
        ([worker(initial=11)], [], "initial_j"),
        ([worker(charging=[[0, 2], [1, 3]])], [], "disjoint"),
        ([worker(), worker()], [], "unique"),
        ([worker()], [job("x", 0, 1, 1), job("x", 1, 1, 1)], "unique"),
        ([worker()], [job("x", 0, 0, 1)], "service_s"),
        ([worker()], [job("x", -1, 1, 1)], "arrival_s"),
        ([worker()], [job("x", 0, 1, True)], "measured_j"),
        ([worker()], [job("x", 0, 1, float("nan"))], "measured_j"),
        ([worker()], [job("x", 0, 1, 0.0000001)], "measured_j"),
    ],
)
def test_invalid_measurements_and_configuration_are_rejected(workers, jobs, error):
    with pytest.raises(ValueError, match=error):
        compare_trace(trace(workers, jobs))


def test_runner_help_file_and_machine_readable_error(tmp_path):
    command = [sys.executable, "-m", "entroly.energy_queue"]
    help_result = subprocess.run([*command, "--help"], capture_output=True, check=True)
    assert b"measured_j" in help_result.stdout
    path = tmp_path / "trace.json"
    data = trace([worker()], [job("one", 0, 1, 1)])
    path.write_bytes(data)
    run = subprocess.run([*command, str(path)], capture_output=True, check=True)
    report = json.loads(run.stdout)
    assert report == compare_trace(data)
    assert report["production_authority"] is False
    assumptions = report["assumptions"]
    assert assumptions["oracle_input_replay"] is True
    assert assumptions["runtime_cap_guarantee"] is False
    assert "Retrospectively recorded" in assumptions["job_inputs"]
    assert "idle energy and recharge losses" in assumptions["consumed_j"]
    assert "conditional on admission" in assumptions["mean_latency_s"]
    path.write_text('{"jobs": [], "jobs": [], "workers": []}', encoding="utf-8")
    invalid = subprocess.run([*command, str(path)], capture_output=True, check=False)
    assert invalid.returncode == 1
    assert not invalid.stdout
    assert "duplicate JSON key" in json.loads(invalid.stderr)["error"]
