"""Validate the conservative machine-readable platform support matrix."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs" / "platform-readiness.json"


def main() -> int:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("schema_version") != "entroly.platform-readiness.v1":
        errors.append("unexpected platform readiness schema")
    platforms = payload.get("platforms")
    if not isinstance(platforms, list) or not platforms:
        errors.append("platform matrix must contain platforms")
        platforms = []
    ids: set[str] = set()
    allowed = {"ci_verified", "contract_tested", "not_claimed"}
    for platform in platforms:
        platform_id = str(platform.get("id", ""))
        if not platform_id or platform_id in ids:
            errors.append(f"missing or duplicate platform id: {platform_id!r}")
        ids.add(platform_id)
        if not platform.get("limitation"):
            errors.append(f"{platform_id}: missing limitation")
        for key in (
            "python_user_journey",
            "native_wheel",
            "persistent_service_definition",
        ):
            if platform.get(key) not in allowed:
                errors.append(f"{platform_id}: invalid {key}")
        for evidence in platform.get("evidence", []):
            if not (ROOT / evidence).exists():
                errors.append(f"{platform_id}: missing evidence path {evidence}")
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"platform readiness verified: {len(platforms)} platform contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
