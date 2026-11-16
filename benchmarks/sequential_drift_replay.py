"""Compare both proxy drift monitors on paired synthetic confusion streams."""

from __future__ import annotations

import json
from random import Random
from statistics import median

from entroly.proxy_feedback import _FrozenBaselineDriftDetector, _SequentialConfusionMonitor

_REPLICATES = 128
_HORIZON = 8192
_CHANGE_AFTER = 4096
_BASE_RATE = 0.20
_SHIFTED_RATE = 0.38
_CLUSTER_REFRESH = 0.10


def _alarms(scenario: str, replicate: int) -> tuple[int | None, int | None]:
    source = Random(65000 + replicate)
    frozen = _FrozenBaselineDriftDetector()
    sequential = _SequentialConfusionMonitor(seed=21000 + replicate)
    frozen_alarm = None
    confused = source.random() < _BASE_RATE
    for t in range(1, _HORIZON + 1):
        if scenario != "clustered_null" or source.random() < _CLUSTER_REFRESH:
            rate = _SHIFTED_RATE if scenario == "abrupt_shift" and t > _CHANGE_AFTER else _BASE_RATE
            confused = source.random() < rate
        frozen.update(-0.5 if confused else 0.3)
        sequential.update(confused)
        if frozen_alarm is None and frozen.trend() == "declining":
            frozen_alarm = t
    return frozen_alarm, sequential.to_dict()["first_alarm_at"]


def _summarize(alarms: tuple[int | None, ...], *, shifted: bool) -> dict:
    if not shifted:
        return {"false_alarms": sum(t is not None for t in alarms)}
    early = sum(t is not None and t <= _CHANGE_AFTER for t in alarms)
    delays = [t - _CHANGE_AFTER for t in alarms if t is not None and t > _CHANGE_AFTER]
    return {
        "pre_change_false_alarms": early,
        "post_change_detections": len(delays),
        "no_alarm": len(alarms) - early - len(delays),
        "median_delay_among_post_change_detections": median(delays) if delays else None,
    }


def run() -> dict:
    scenarios = {}
    for scenario in ("iid_null", "abrupt_shift", "clustered_null"):
        frozen, sequential = zip(*(_alarms(scenario, i) for i in range(_REPLICATES)))
        scenarios[scenario] = {
            "frozen": _summarize(frozen, shifted=scenario == "abrupt_shift"),
            "sequential_shadow": _summarize(sequential, shifted=scenario == "abrupt_shift"),
        }
    return {
        "schema": "entroly.sequential-drift-replay.v1",
        "synthetic": True,
        "production_authority": False,
        "comparison": "default_thresholds",
        "replicates_per_scenario": _REPLICATES,
        "observations_per_replicate": _HORIZON,
        "change_after": _CHANGE_AFTER,
        "base_rate": _BASE_RATE,
        "shifted_rate": _SHIFTED_RATE,
        "cluster_refresh_probability": _CLUSTER_REFRESH,
        "scenarios": scenarios,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
