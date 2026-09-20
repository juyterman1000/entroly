"""Frozen NevIR harness stub for RELATE-X.

Requires an explicit dataset file path. This script refuses silent downloads and
counts ties/unresolved cases as failures in the paired metric.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    path = pathlib.Path(args.csv_path)
    if not path.exists():
        raise SystemExit(f"dataset not found: {path}")
    payload = path.read_bytes()
    rows = list(csv.DictReader(payload.decode("utf-8").splitlines()))
    print(json.dumps({
        "dataset_sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(rows),
        "status": "loaded_only_no_model_score",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
