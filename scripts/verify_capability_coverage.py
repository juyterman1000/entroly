#!/usr/bin/env python3
"""Validate Entroly's machine-readable feature-to-proof coverage map."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "capability-coverage.json"
VALID_STATUSES = {"production", "beta", "research"}
VALID_SURFACES = {
    "python-sdk",
    "cli",
    "mcp",
    "proxy",
    "rust",
    "node-wasm",
    "openclaw",
    "release",
}


def _rows(value: object) -> list[dict[str, object]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def validate(manifest: Path = DEFAULT_MANIFEST) -> list[str]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    failures: list[str] = []
    if payload.get("schema_version") != "entroly.capability-coverage.v1":
        failures.append("unsupported schema_version")

    capabilities = _rows(payload.get("capabilities"))
    if not capabilities:
        return failures + ["capabilities must be a non-empty list"]

    seen: set[str] = set()
    for row in capabilities:
        capability_id = row.get("id")
        prefix = str(capability_id or "<missing>")
        if not isinstance(capability_id, str) or not re.fullmatch(
            r"[a-z0-9]+(?:_[a-z0-9]+)*", capability_id
        ):
            failures.append(f"{prefix}: invalid id")
        elif capability_id in seen:
            failures.append(f"{prefix}: duplicate id")
        else:
            seen.add(capability_id)

        status = row.get("status")
        if status not in VALID_STATUSES:
            failures.append(f"{prefix}: invalid status {status!r}")

        for field in ("implementation", "tests", "docs", "package_surfaces", "public_entrypoints"):
            if not isinstance(row.get(field), list):
                failures.append(f"{prefix}: {field} must be a list")

        for relative in row.get("implementation", []):
            if not isinstance(relative, str) or not (REPO_ROOT / relative).is_file():
                failures.append(f"{prefix}: missing implementation {relative!r}")
        for relative in row.get("docs", []):
            if not isinstance(relative, str) or not (REPO_ROOT / relative).is_file():
                failures.append(f"{prefix}: missing documentation {relative!r}")
        for selector in row.get("tests", []):
            if not isinstance(selector, str) or "::" not in selector:
                failures.append(f"{prefix}: invalid test selector {selector!r}")
                continue
            relative, test_name = selector.split("::", 1)
            path = REPO_ROOT / relative
            if not path.is_file():
                failures.append(f"{prefix}: missing test file {relative}")
            elif f"def {test_name}(" not in path.read_text(
                encoding="utf-8", errors="replace"
            ):
                failures.append(f"{prefix}: missing test function {selector}")

        unknown_surfaces = sorted(
            set(row.get("package_surfaces", [])) - VALID_SURFACES
        )
        if unknown_surfaces:
            failures.append(
                f"{prefix}: unknown package surfaces {', '.join(unknown_surfaces)}"
            )

        if status in {"production", "beta"}:
            for required in ("implementation", "tests", "docs", "package_surfaces", "public_entrypoints"):
                if not row.get(required):
                    failures.append(f"{prefix}: shipped capability has no {required}")
        if status == "research" and row.get("public_entrypoints"):
            failures.append(f"{prefix}: research capability declares a public entry point")
        if not isinstance(row.get("limitations"), str) or not row["limitations"].strip():
            failures.append(f"{prefix}: missing limitation")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?", default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    failures = validate(args.manifest)
    if failures:
        print(f"CAPABILITY COVERAGE FAILED ({len(failures)} problems)")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    count = len(_rows(json.loads(args.manifest.read_text(encoding="utf-8")).get("capabilities")))
    print(f"capability coverage OK - {count} major capability families mapped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
