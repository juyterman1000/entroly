"""Replay measured jobs through two local loss-system admission policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from operator import attrgetter
from pathlib import Path

ZERO = Fraction(0)

TRACE_HELP = """
JSON input example:
  {
    "workers": [
      {"id": "a", "capacity_j": 12, "initial_j": 8, "recharge_w": 2,
       "charging": [[5, 9]]}
    ],
    "jobs": [
      {"id": "job-1", "arrival_s": 0.5, "service_s": 1, "measured_j": 3.2}
    ]
  }

Times are seconds since trace start. Supply measured arrival times, service
durations, and job energy in joules. This oracle-input replay treats recorded
durations and joules as known at admission and fixed across workers and
policies. It provides no runtime-cap guarantee. Supply battery capacity and
initial energy in joules, recharge rate in watts, and sorted, disjoint charging
windows [start, end) per worker (use [] for none). Use JSON numbers from 0 through 1e9 with
at most six fractional digits. Capacity and service duration must be positive.

A worker runs one job at a time. Admission reserves the full job energy at
arrival. Workers recharge only during the specified windows and cannot run jobs
across them. Completions and charging boundaries take effect before arrivals.
Simultaneous arrivals use file order. An arrival with no feasible worker is
rejected immediately. Both replays end at max(arrival_s + service_s) over the
offered jobs, or zero for an empty trace.

First-fit uses worker order in the file. Energy-aware uses the feasible worker
with the smallest remaining charge after admission, breaking ties by worker
order. Both use only the arriving job, current worker state, and charging
schedule. Results describe this replay under the supplied inputs.
"""


@dataclass(frozen=True)
class Job:
    id: str
    arrival: Fraction
    service: Fraction
    joules: Fraction

    @property
    def finish(self) -> Fraction:
        return self.arrival + self.service


@dataclass(frozen=True)
class Worker:
    id: str
    capacity: Fraction
    initial: Fraction
    recharge: Fraction
    windows: tuple[tuple[Fraction, Fraction], ...]


@dataclass
class WorkerState:
    spec: Worker
    charge: Fraction
    updated_at: Fraction = ZERO
    busy_until: Fraction = ZERO
    window_index: int = 0
    consumed: Fraction = ZERO
    recharged: Fraction = ZERO

    def advance(self, now: Fraction) -> None:
        """Credit only the elapsed part of each charging window, up to capacity."""
        while self.window_index < len(self.spec.windows):
            start, end = self.spec.windows[self.window_index]
            if start > now:
                break
            elapsed = min(end, now) - max(start, self.updated_at)
            added = min(self.spec.capacity - self.charge, elapsed * self.spec.recharge)
            self.charge += added
            self.recharged += added
            if end > now:
                break
            self.window_index += 1
        self.updated_at = now

    def can_serve(self, job: Job) -> bool:
        if self.busy_until > job.arrival or self.charge < job.joules:
            return False
        return (
            self.window_index == len(self.spec.windows)
            or job.finish <= self.spec.windows[self.window_index][0]
        )


def _number(value: object, field: str, *, positive: bool = False) -> Fraction:
    if type(value) is int:
        value = Decimal(value)
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or not 0 <= value <= 10**9
        or value.as_tuple().exponent < -6
        or (positive and value == 0)
    ):
        raise ValueError(
            f"{field} must be a number in range with at most six decimal places"
        )
    return Fraction(value)


def _fields(value: object, fields: str, where: str) -> dict:
    expected = fields.split()
    if not isinstance(value, dict) or value.keys() != set(expected):
        raise ValueError(f"{where} must contain exactly {', '.join(expected)}")
    return value


def _array(value: object, where: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be an array")
    return value


def _distinct_id(value: object, used: set[str], where: str) -> str:
    if not isinstance(value, str) or not value.strip() or value in used:
        raise ValueError(f"{where}.id must be a unique nonempty string")
    used.add(value)
    return value


def _read_trace(data: bytes) -> tuple[list[Worker], list[Job]]:
    def unique_keys(pairs: list[tuple[str, object]]) -> dict:
        obj = {}
        for name, value in pairs:
            if name in obj:
                raise ValueError(f"duplicate JSON key {name!r}")
            obj[name] = value
        return obj

    raw = _fields(
        json.loads(data, parse_float=Decimal, object_pairs_hook=unique_keys),
        "workers jobs",
        "trace",
    )
    workers, jobs = [], []
    worker_ids: set[str] = set()
    job_ids: set[str] = set()
    for index, item in enumerate(_array(raw["workers"], "workers")):
        where = f"workers[{index}]"
        row = _fields(item, "id capacity_j initial_j recharge_w charging", where)
        worker_id = _distinct_id(row["id"], worker_ids, where)
        capacity = _number(row["capacity_j"], f"{where}.capacity_j", positive=True)
        initial = _number(row["initial_j"], f"{where}.initial_j")
        recharge = _number(row["recharge_w"], f"{where}.recharge_w")
        if initial > capacity:
            raise ValueError(f"{where}.initial_j exceeds capacity_j")
        windows = []
        for index_in_worker, pair in enumerate(
            _array(row["charging"], f"{where}.charging")
        ):
            window = f"{where}.charging[{index_in_worker}]"
            if len(_array(pair, window)) != 2:
                raise ValueError(f"{window} must contain start and end times")
            start = _number(pair[0], f"{window} start")
            end = _number(pair[1], f"{window} end")
            if start >= end or (windows and start < windows[-1][1]):
                raise ValueError(
                    f"{window} must be positive-length, sorted, and disjoint"
                )
            windows.append((start, end))
        workers.append(Worker(worker_id, capacity, initial, recharge, tuple(windows)))
    if not workers:
        raise ValueError("workers must contain at least one worker")
    for index, item in enumerate(_array(raw["jobs"], "jobs")):
        where = f"jobs[{index}]"
        row = _fields(item, "id arrival_s service_s measured_j", where)
        jobs.append(
            Job(
                _distinct_id(row["id"], job_ids, where),
                _number(row["arrival_s"], f"{where}.arrival_s"),
                _number(row["service_s"], f"{where}.service_s", positive=True),
                _number(row["measured_j"], f"{where}.measured_j"),
            )
        )
    return workers, sorted(jobs, key=attrgetter("arrival"))


def _replay(
    workers: list[Worker], jobs: list[Job], end: Fraction, *, energy_aware: bool
) -> dict:
    states = [WorkerState(worker, worker.initial) for worker in workers]
    completed, rejected = [], []
    for job in jobs:
        for state in states:
            state.advance(job.arrival)
        feasible = (state for state in states if state.can_serve(job))
        if energy_aware:
            # The job debits the same amount on every worker, so ordering by
            # current charge also orders by charge after admission.
            selected = min(feasible, key=attrgetter("charge"), default=None)
        else:
            selected = next(feasible, None)
        entry = {
            "id": job.id,
            "arrival_s": float(job.arrival),
            "service_s": float(job.service),
            "measured_j": float(job.joules),
        }
        if selected is None:
            rejected.append((job, entry))
            continue
        selected.busy_until = job.finish
        selected.charge -= job.joules
        selected.consumed += job.joules
        completed.append(
            (
                job,
                {
                    **entry,
                    "worker": selected.spec.id,
                    "finish_s": float(job.finish),
                    "latency_s": float(job.service),
                },
            )
        )
    for state in states:
        state.advance(end)
    latencies = [job.service for job, _ in completed]
    return {
        "completed": len(completed),
        "rejected": len(rejected),
        "consumed_j": float(sum(state.consumed for state in states)),
        "rejected_job_j": float(sum(job.joules for job, _ in rejected)),
        "mean_latency_s": float(sum(latencies) / len(latencies)) if latencies else None,
        "max_latency_s": float(max(latencies)) if latencies else None,
        "completed_jobs": [entry for _, entry in completed],
        "rejected_jobs": [entry for _, entry in rejected],
        "workers": [
            {
                "id": state.spec.id,
                "capacity_j": float(state.spec.capacity),
                "initial_j": float(state.spec.initial),
                "consumed_j": float(state.consumed),
                "recharged_j": float(state.recharged),
                "final_j": float(state.charge),
            }
            for state in states
        ],
    }


def compare_trace(data: bytes) -> dict:
    """Replay the same JSON trace bytes independently under both policies."""
    workers, jobs = _read_trace(data)
    end = max((job.finish for job in jobs), default=ZERO)
    first_fit = _replay(workers, jobs, end, energy_aware=False)
    energy_aware = _replay(workers, jobs, end, energy_aware=True)
    first_ids = {entry["id"] for entry in first_fit["completed_jobs"]}
    aware_ids = {entry["id"] for entry in energy_aware["completed_jobs"]}
    return {
        "schema_version": 1,
        "basis": "trace_replay",
        "production_authority": False,
        "assumptions": {
            "oracle_input_replay": True,
            "job_inputs": (
                "Retrospectively recorded joules and service durations are known at "
                "admission and held fixed across workers and policies."
            ),
            "runtime_cap_guarantee": False,
            "consumed_j": "Admitted job energy only. Excludes idle energy and recharge losses.",
            "mean_latency_s": "Arrival to completion, conditional on admission. Null if none.",
        },
        "trace_sha256": hashlib.sha256(data).hexdigest(),
        "offered_jobs": len(jobs),
        "horizon_s": float(end),
        "first_fit": first_fit,
        "energy_aware": energy_aware,
        "paired_completions": {
            "only_first_fit": [
                entry["id"]
                for entry in first_fit["completed_jobs"]
                if entry["id"] not in aware_ids
            ],
            "only_energy_aware": [
                entry["id"]
                for entry in energy_aware["completed_jobs"]
                if entry["id"] not in first_ids
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=TRACE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("trace", type=Path, help="local JSON trace file")
    args = parser.parse_args(argv)
    try:
        result = compare_trace(args.trace.read_bytes())
    except (OSError, ValueError, ArithmeticError, RecursionError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
