"""Adversarial persistence and honesty contracts for value accounting."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from entroly.value_tracker import ValueTracker


_CONCURRENT_WRITER = r'''
import sys
import time
from pathlib import Path

from entroly.value_tracker import ValueTracker

root = Path(sys.argv[1])
worker = int(sys.argv[2])
count = int(sys.argv[3])
tracker = ValueTracker(root)
(root / f"ready-{worker}").write_text("ready", encoding="utf-8")
start = root / "start"
deadline = time.monotonic() + 30
while not start.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError("writer did not receive start signal")
    time.sleep(0.005)
for index in range(count):
    tracker.record(
        tokens_saved=10,
        model="",
        optimized=True,
        source="sdk",
        coverage_pct=50.0,
        confidence=0.8,
    )
'''


def test_provider_and_local_evidence_classes_survive_restart(tmp_path: Path) -> None:
    tracker = ValueTracker(tmp_path)
    tracker.record(1000, model="gpt-4o", source="proxy")
    tracker.record(500, model="", source="sdk")
    tracker.record(300, model="private-model", source="proxy")

    restored = ValueTracker(tmp_path)
    lifetime = restored.get_lifetime()
    receipt = restored.get_value_receipt()

    assert lifetime["tokens_saved"] == 1800
    assert lifetime["provider_tokens_saved"] == 1300
    assert lifetime["local_tokens_reduced"] == 500
    assert lifetime["local_operations"] == 1
    assert lifetime["provider_unpriced_tokens"] == 300
    assert lifetime["provider_unpriced_requests"] == 1
    assert lifetime["provider_cost_avoided_usd"] == pytest.approx(0.0025)

    assert receipt["provider_path"]["input_tokens_reduced"] == 1300
    assert receipt["provider_path"]["modeled_input_cost_avoided_usd"] == pytest.approx(0.0025)
    assert receipt["local_operations"]["tokens_reduced"] == 500
    assert receipt["local_operations"]["dollar_claimed_usd"] == 0.0


def test_cross_process_writers_cannot_overwrite_each_others_totals(tmp_path: Path) -> None:
    """All writers load before the barrier, making stale-snapshot loss deterministic."""
    workers = 4
    records_per_worker = 12
    processes: list[subprocess.Popen[str]] = []
    env = {**os.environ, "ENTROLY_DISABLE_UPDATE_CHECK": "1"}

    for worker in range(workers):
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _CONCURRENT_WRITER,
                    str(tmp_path),
                    str(worker),
                    str(records_per_worker),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        )

    deadline = time.monotonic() + 30
    while len(list(tmp_path.glob("ready-*"))) < workers:
        if time.monotonic() >= deadline:
            break
        if any(process.poll() is not None for process in processes):
            break
        time.sleep(0.01)
    assert len(list(tmp_path.glob("ready-*"))) == workers
    (tmp_path / "start").write_text("start", encoding="utf-8")

    for process in processes:
        stdout, stderr = process.communicate(timeout=45)
        assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"

    restored = ValueTracker(tmp_path)
    lifetime = restored.get_lifetime()
    activity = restored.get_activity(last_n=workers * records_per_worker + 10)
    expected_records = workers * records_per_worker

    assert lifetime["requests_total"] == expected_records
    assert lifetime["requests_optimized"] == expected_records
    assert lifetime["local_operations"] == expected_records
    assert lifetime["local_tokens_reduced"] == expected_records * 10
    assert lifetime["tokens_saved"] == expected_records * 10
    assert len([row for row in activity if row.get("kind") == "optimize"]) == expected_records


def test_hostile_numeric_inputs_cannot_decrease_or_poison_public_metrics(
    tmp_path: Path,
) -> None:
    tracker = ValueTracker(tmp_path)
    tracker.record(
        100,
        duplicates=-7,
        confidence=float("nan"),
        coverage_pct=float("inf"),
        source="sdk",
    )
    tracker.record_hallucination_blocked(-5, source="test")
    tracker.record_routing_saving(-12.50, source="test", chosen_model="cheap")

    lifetime = tracker.get_lifetime()
    confidence = tracker.get_confidence()
    receipt = tracker.get_value_receipt()

    assert lifetime["duplicates_caught"] >= 0
    assert lifetime["hallucinations_blocked"] >= 0
    assert lifetime["routing_saved_usd"] >= 0
    assert math.isfinite(float(confidence["confidence"]))
    assert math.isfinite(float(confidence["coverage_pct"]))

    # Public receipts must be strict-JSON serializable; allow_nan=False is the
    # interoperability contract consumed by dashboards and CI.
    json.dumps(receipt, allow_nan=False)
    json.dumps(confidence, allow_nan=False)


def test_activity_feed_rejects_negative_savings_and_nonfinite_metadata(
    tmp_path: Path,
) -> None:
    tracker = ValueTracker(tmp_path)
    tracker.record_event(
        "routing",
        "hostile telemetry",
        source="test",
        tokens_saved=-50,
        cost_saved_usd=float("nan"),
        confidence=float("inf"),
        finite_negative_delta=-12.5,
    )

    restored = ValueTracker(tmp_path)
    activity = restored.get_activity(1)
    assert len(activity) == 1
    row = activity[0]

    assert int(row.get("tokens_saved", 0)) >= 0
    assert float(row.get("cost_saved_usd", 0.0)) >= 0.0
    assert math.isfinite(float(row.get("confidence", 0.0)))
    assert row["finite_negative_delta"] == -12.5
    json.dumps(row, allow_nan=False)


def test_corrupt_tracker_state_fails_safe_without_claiming_old_value(
    tmp_path: Path,
) -> None:
    path = tmp_path / "value_tracker.json"
    path.write_text('{"version":4,"lifetime":', encoding="utf-8")

    tracker = ValueTracker(tmp_path)
    assert tracker.get_lifetime()["tokens_saved"] == 0
    tracker.record(25, source="sdk")

    restored = ValueTracker(tmp_path)
    assert restored.get_lifetime()["tokens_saved"] == 25
    json.loads(path.read_text(encoding="utf-8"))
