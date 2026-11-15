"""Plan and analyze a local two-rate cache switchback at scheduled point times."""

from __future__ import annotations

import argparse
import json
import math
import random
from bisect import bisect_right
from pathlib import Path


EARLY_RATE, LATE_RATE, CHANGE_AGE = 0.182, 0.885, 1.045
ARMS = ("control", "treatment")
ASSUMPTIONS = (
    "The seed is drawn independently of outcomes and the plan is fixed before collection. "
    "Do not redraw based on the schedule or its realized exposure counts.",
    "The schedule is applied continuously from time zero, including between observations. "
    "Recorded point arms must reflect the actual applied arm.",
    "Outcomes depend only on assignments in the preceding carryover window, "
    "including the observation time. Future assignments cannot affect them.",
    "The estimand is treatment minus control averaged over the scheduled point times. "
    "This report supplies a point estimate only.",
)


def _integrated_survival(age: float) -> float:
    """Integral of run survival from age onward, in carryover units."""
    tail_mass = math.exp(-EARLY_RATE * CHANGE_AGE) / LATE_RATE
    if age >= CHANGE_AGE:
        return tail_mass * math.exp(-LATE_RATE * (age - CHANGE_AGE))
    early_mass = -math.expm1(-EARLY_RATE * (CHANGE_AGE - age)) / EARLY_RATE
    return math.exp(-EARLY_RATE * age) * early_mass + tail_mass


MEAN_DURATION = _integrated_survival(0)
PURE_PROBABILITY = _integrated_survival(1) / (2 * MEAN_DURATION)


def _residual(age: float, rng: random.Random) -> float:
    """Sample remaining run length conditional on its age, in carryover units."""
    early_left = CHANGE_AGE - age
    if early_left > 0:
        early_wait = rng.expovariate(EARLY_RATE)
        if early_wait < early_left:
            return early_wait
    return max(0.0, early_left) + rng.expovariate(LATE_RATE)


def _stationary_run(rng: random.Random) -> tuple[float, float, int]:
    # The equilibrium age density is survival divided by mean run length.
    # Invert its early integral, or sample the exponential tail above the changepoint.
    mass = rng.random() * MEAN_DURATION
    early_mass = MEAN_DURATION - _integrated_survival(CHANGE_AGE)
    if mass < early_mass:
        age = -math.log1p(-EARLY_RATE * mass) / EARLY_RATE
    else:
        age = CHANGE_AGE + rng.expovariate(LATE_RATE)
    return age, _residual(age, rng), rng.randrange(2)


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def plan(carryover: float, spacing: float, count: int, seed: int) -> dict:
    """Start assignments at zero and schedule points at carryover + i * spacing."""
    carryover = _number(carryover, "carryover")
    spacing = _number(spacing, "spacing")
    if carryover <= 0 or spacing <= 0:
        raise ValueError("carryover and spacing must be positive and use the same time unit")
    if type(count) is not int or count < 1 or type(seed) is not int:
        raise ValueError("count must be a positive integer and seed must be an integer")
    end = _number(carryover + (count - 1) * spacing, "last observation time")
    times = [carryover + index * spacing for index in range(count)]
    if any(left >= right for left, right in zip(times, times[1:])):
        raise ValueError("spacing is too small to represent distinct observation times")

    rng = random.Random(seed)
    age, residual, arm = _stationary_run(rng)
    initial_age = _number(age * carryover, "initial run age")
    switches: list[float] = []
    start, switch = 0.0, residual * carryover
    while switch <= end:
        if switch <= start:
            raise ValueError("run duration is too small to represent the next switch time")
        switches.append(switch)
        start = switch
        switch = start + _residual(0, rng) * carryover

    return {
        "schema_version": 1,
        "production_authority": False,
        "carryover": carryover,
        "spacing": spacing,
        "count": count,
        "seed": seed,
        "initial_arm": ARMS[arm],
        "initial_run_age": initial_age,
        "switch_times": switches,
        "observation_times": times,
    }


def report(saved_plan: dict, observations: list) -> dict:
    """Verify every scheduled point and estimate the point-average effect."""
    if not isinstance(saved_plan, dict):
        raise ValueError("plan must be the JSON object emitted by plan")
    try:
        expected = plan(*(saved_plan[key] for key in ("carryover", "spacing", "count", "seed")))
    except KeyError as exc:
        raise ValueError(f"plan is missing {exc.args[0]}") from exc
    if saved_plan != expected:
        raise ValueError("plan differs from its seeded schedule")

    times = expected["observation_times"]
    if not isinstance(observations, list) or len(observations) != len(times):
        raise ValueError("observations must contain exactly one row per scheduled point")
    switches = expected["switch_times"]
    initial_arm = ARMS.index(expected["initial_arm"])
    carryover = expected["carryover"]
    pure_counts = dict.fromkeys(ARMS, 0)
    contributions = []
    for time, row in zip(times, observations):
        recorded_time = row.get("time") if isinstance(row, dict) else None
        if type(recorded_time) not in (int, float) or recorded_time != time:
            raise ValueError(f"expected an observation at time {time} in schedule order")
        switched = bisect_right(switches, time)
        arm = ARMS[(initial_arm + switched) % 2]
        if row.get("arm") != arm:
            raise ValueError(f"observed arm does not match the plan at time {time}")
        outcome = _number(row.get("outcome"), "outcome")
        # With the first point at carryover, the observed history starts at zero.
        last_switch = switches[switched - 1] if switched else 0.0
        if time - last_switch >= carryover:
            pure_counts[arm] += 1
            contributions.append(outcome if arm == "treatment" else -outcome)

    result = {
        "production_authority": False,
        "assumptions": ASSUMPTIONS,
        "scheduled_observations": len(times),
        "pure_exposure_probability_per_arm": PURE_PROBABILITY,
        "pure_counts": pure_counts,
        "contaminated_count": len(times) - sum(pure_counts.values()),
    }
    if not all(pure_counts.values()):
        return {
            **result,
            "status": "not_estimable",
            "reason": "both arms require at least one pure observation",
            "effect": None,
        }
    denominator = len(times) * PURE_PROBABILITY
    effect = math.fsum(value / denominator for value in contributions)
    return {**result, "status": "ok", "effect": _number(effect, "estimated effect")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser(
        "plan",
        help="emit a static assignment and observation schedule",
        description="Assignments start at time zero. The first point is at the carryover horizon.",
    )
    planner.add_argument("--carryover", type=float, required=True)
    planner.add_argument("--spacing", type=float, required=True, help="same units as carryover")
    planner.add_argument("--count", type=int, required=True, help="number of scheduled points")
    planner.add_argument("--seed", type=int, required=True, help="independently chosen random seed")
    reporter = commands.add_parser("report", help="verify and analyze locally collected observations")
    reporter.add_argument("--plan", type=Path, required=True, help="original plan JSON file")
    reporter.add_argument(
        "--observations", type=Path, required=True,
        help="JSON array of {time, arm, outcome} objects ordered by scheduled time",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan(args.carryover, args.spacing, args.count, args.seed)
        else:
            saved_plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
            observations = json.loads(args.observations.read_text(encoding="utf-8-sig"))
            result = report(saved_plan, observations)
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(result.get("status") == "not_estimable")


if __name__ == "__main__":
    raise SystemExit(main())
