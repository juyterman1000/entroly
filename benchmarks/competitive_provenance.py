#!/usr/bin/env python3
"""Create and validate non-publishable-until-pinned benchmark provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
VALID_STATUS = {"valid", "void", "error"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("schema_version") != "entroly.competitive-run.v1":
        failures.append("unsupported schema_version")
    subjects = payload.get("subjects")
    if not isinstance(subjects, list) or len(subjects) < 2:
        failures.append("at least two subjects are required")
        subjects = []
    labels: set[str] = set()
    for index, subject in enumerate(subjects):
        prefix = f"subjects[{index}]"
        if not isinstance(subject, dict):
            failures.append(f"{prefix} must be an object")
            continue
        label = subject.get("label")
        if not isinstance(label, str) or not label.strip() or label in labels:
            failures.append(f"{prefix}.label must be non-empty and unique")
        else:
            labels.add(label)
        if not isinstance(subject.get("version"), str) or not subject["version"].strip():
            failures.append(f"{prefix}.version must be exact")
        if not COMMIT.fullmatch(str(subject.get("commit") or "")):
            failures.append(f"{prefix}.commit must be a full 40-character git SHA")
        if not SHA256.fullmatch(str(subject.get("executable_sha256") or "")):
            failures.append(f"{prefix}.executable_sha256 must be SHA-256")
        if subject.get("status") not in VALID_STATUS:
            failures.append(f"{prefix}.status must be one of {sorted(VALID_STATUS)}")
    for field in ("workload_sha256", "raw_results_sha256"):
        if not SHA256.fullmatch(str(payload.get(field) or "")):
            failures.append(f"{field} must be SHA-256")
    if payload.get("verdict") not in {"supported", "refuted", "inconclusive", "void"}:
        failures.append("verdict is invalid")
    if payload.get("verdict") in {"supported", "refuted"} and any(
        subject.get("status") != "valid" for subject in subjects if isinstance(subject, dict)
    ):
        failures.append("directional verdict requires every subject status=valid")
    return failures


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = validate_manifest(payload)
    if failures:
        print(f"COMPETITIVE PROVENANCE FAILED ({len(failures)})")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"competitive provenance OK - {len(payload['subjects'])} pinned subjects; "
        f"verdict={payload['verdict']}"
    )
    return 0


def new_manifest(
    *, subjects: list[dict[str, Any]], workload: Path, raw_results: Path, verdict: str
) -> dict[str, Any]:
    """Build a manifest after a run; callers still validate before publication."""
    return {
        "schema_version": "entroly.competitive-run.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subjects": subjects,
        "workload_sha256": sha256_file(workload),
        "raw_results_sha256": sha256_file(raw_results),
        "verdict": verdict,
        "claim_policy": {
            "failures_remain_in_sample": True,
            "void_arms_forbid_directional_claims": True,
            "exact_versions_and_commits_required": True,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
