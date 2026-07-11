#!/usr/bin/env python3
"""Synchronize Entroly's package-local model registry snapshots.

The Python package and standalone Rust crate must each contain the snapshot so
that either artifact can be built and installed independently. This tool makes
the Python copy canonical and fails CI when the Rust mirror drifts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "entroly" / "models" / "registry.json"
RUST_TARGET = ROOT / "entroly-core" / "model_registry.json"


def _validated_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError(f"{path} must contain an object with a models list")
    ids = [str(item.get("id", "")) for item in payload["models"]]
    if any(not model_id for model_id in ids):
        raise ValueError(f"{path} contains a model without an id")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate model ids")
    return raw


def synchronize(*, check: bool) -> int:
    source = _validated_bytes(SOURCE)
    target = RUST_TARGET.read_bytes() if RUST_TARGET.exists() else b""
    if target == source:
        print(
            f"model registry synchronized: {len(source):,} bytes, "
            f"sha256={hashlib.sha256(source).hexdigest()}"
        )
        return 0
    if check:
        print(
            "model registry drift: run "
            "`python scripts/sync_model_registry.py` and commit the Rust snapshot"
        )
        return 1
    RUST_TARGET.write_bytes(source)
    print(
        f"updated {RUST_TARGET.relative_to(ROOT)}: {len(source):,} bytes, "
        f"sha256={hashlib.sha256(source).hexdigest()}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the Rust snapshot differs",
    )
    args = parser.parse_args()
    return synchronize(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
