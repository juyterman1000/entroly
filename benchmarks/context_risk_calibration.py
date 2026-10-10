"""Offline paired-decision divergence calibration for one frozen policy.

This command reads already-collected observation seals. It never contacts a
model/provider, runs an observed decision, or enables production risk gating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from entroly.context_receipts.models import stable_hash
from entroly.decision_evaluation import (
    FixedPolicyCalibrationProtocol,
    calibrate_fixed_policy_divergence,
)


def _read_json(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def run(protocol_path: Path, observations_path: Path) -> dict[str, Any]:
    declared, protocol_file_sha256 = _read_json(protocol_path)
    source, observations_file_sha256 = _read_json(observations_path)
    if not isinstance(declared, dict):
        raise ValueError("protocol must be a JSON object")
    protocol = FixedPolicyCalibrationProtocol(**declared)
    observations = source.get("observations") if isinstance(source, dict) else source
    if not isinstance(observations, list) or any(
        not isinstance(item, dict) for item in observations
    ):
        raise ValueError("observations must be a JSON array of observation objects")
    if any(not isinstance(item.get("task_id"), str) for item in observations):
        raise ValueError("each observation needs a string task_id")
    declared_ids = set(protocol.calibration_task_ids) | set(protocol.holdout_task_ids)
    if len(observations) != len(declared_ids) or {
        item.get("task_id") for item in observations
    } != declared_ids:
        raise ValueError("observation file must contain exactly the declared task set")
    calibration_ids = set(protocol.calibration_task_ids)
    holdout_ids = set(protocol.holdout_task_ids)
    report = calibrate_fixed_policy_divergence(
        protocol,
        calibration=[
            item for item in observations if item["task_id"] in calibration_ids
        ],
        holdout=[item for item in observations if item["task_id"] in holdout_ids],
    )
    payload = {
        "report": report,
        "protocol_file_sha256": protocol_file_sha256,
        "observations_file_sha256": observations_file_sha256,
    }
    return {**payload, "bundle_id": stable_hash(payload)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.protocol, args.observations)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
