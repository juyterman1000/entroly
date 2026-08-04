#!/usr/bin/env python3
"""Validate Entroly's local distribution and discovery control plane.

This check is intentionally offline. It verifies repository-owned metadata and
status honesty; it does not claim that an external listing is currently live.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_STATUSES = {"prepared", "submitted", "published", "blocked", "rejected"}
REQUIRED_TARGET_FIELDS = {
    "id",
    "name",
    "kind",
    "priority",
    "status",
    "target_url",
    "submission_url",
    "fit",
    "required_assets",
    "proof_url",
    "next_action",
}
REQUIRED_DISCOVERY_FILES = (
    Path("README.md"),
    Path("llms.txt"),
    Path("server.json"),
    Path(".claude-plugin/manifest.json"),
    Path("docs/distribution/README.md"),
    Path("docs/distribution/submission-kit.md"),
    Path("docs/distribution/targets.json"),
)
CANONICAL_REPOSITORY = "https://github.com/juyterman1000/entroly"
CANONICAL_DOCUMENTATION = "https://juyterman1000.github.io/entroly/docs/index.html"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a JSON object")
    return data


def _project_version() -> str:
    """Read project.version without adding a Python 3.10 TOML dependency."""

    in_project = False
    for raw_line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[project]":
            in_project = True
            continue
        if in_project and line.startswith("["):
            break
        if in_project and line.startswith("version"):
            key, separator, value = line.partition("=")
            if key.strip() == "version" and separator:
                version = value.strip().strip('"').strip("'")
                if version:
                    return version
    raise ValueError("pyproject.toml is missing project.version")


def validate() -> list[str]:
    errors: list[str] = []

    for relative in REQUIRED_DISCOVERY_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required discovery file: {relative}")

    if errors:
        return errors

    version = _project_version()
    server = _load_json(ROOT / "server.json")
    plugin = _load_json(ROOT / ".claude-plugin/manifest.json")
    registry = _load_json(ROOT / "docs/distribution/targets.json")

    if server.get("version") != version:
        errors.append(
            f"server.json version {server.get('version')!r} does not match {version!r}"
        )
    if plugin.get("version") != version:
        errors.append(
            ".claude-plugin/manifest.json version "
            f"{plugin.get('version')!r} does not match {version!r}"
        )

    if server.get("websiteUrl") not in {CANONICAL_REPOSITORY, CANONICAL_DOCUMENTATION}:
        errors.append("server.json websiteUrl is not an Entroly canonical URL")
    if server.get("repository", {}).get("url") != CANONICAL_REPOSITORY:
        errors.append("server.json repository.url is not canonical")
    if plugin.get("repository") != CANONICAL_REPOSITORY:
        errors.append("Claude plugin repository URL is not canonical")
    if plugin.get("homepage") not in {CANONICAL_REPOSITORY, CANONICAL_DOCUMENTATION}:
        errors.append("Claude plugin homepage is not an Entroly canonical URL")

    npm_command = plugin.get("install", {}).get("alternatives", {}).get("npm")
    if npm_command != "npm install -g entroly":
        errors.append(
            "Claude plugin npm alternative must use the primary `entroly` package"
        )

    documentation = plugin.get("documentation", {})
    if documentation.get("benchmarks") != "docs/BENCHMARKS.md":
        errors.append("Claude plugin benchmark documentation path is stale")

    if registry.get("schema_version") != 1:
        errors.append("distribution registry schema_version must be 1")
    if registry.get("repository") != CANONICAL_REPOSITORY:
        errors.append("distribution registry repository URL is not canonical")

    targets = registry.get("targets")
    if not isinstance(targets, list) or not targets:
        errors.append("distribution registry must contain a non-empty targets list")
        return errors

    seen_ids: set[str] = set()
    for index, target in enumerate(targets):
        prefix = f"targets[{index}]"
        if not isinstance(target, dict):
            errors.append(f"{prefix} must be an object")
            continue

        missing = sorted(REQUIRED_TARGET_FIELDS - target.keys())
        if missing:
            errors.append(f"{prefix} missing fields: {', '.join(missing)}")
            continue

        target_id = target.get("id")
        if not isinstance(target_id, str) or not target_id:
            errors.append(f"{prefix}.id must be a non-empty string")
        elif target_id in seen_ids:
            errors.append(f"duplicate target id: {target_id}")
        else:
            seen_ids.add(target_id)

        status = target.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{prefix}.status {status!r} is not allowed")

        priority = target.get("priority")
        if not isinstance(priority, int) or priority < 0 or priority > 3:
            errors.append(f"{prefix}.priority must be an integer from 0 to 3")

        for field in ("target_url", "submission_url"):
            value = target.get(field)
            if not isinstance(value, str) or not value.startswith("https://"):
                errors.append(f"{prefix}.{field} must be an https URL")

        proof_url = target.get("proof_url")
        if status in {"submitted", "published", "rejected"}:
            if not isinstance(proof_url, str) or not proof_url.startswith("https://"):
                errors.append(
                    f"{prefix} status {status!r} requires a public proof_url"
                )
        elif proof_url is not None:
            errors.append(
                f"{prefix} status {status!r} must not imply external proof"
            )

        required_assets = target.get("required_assets")
        if not isinstance(required_assets, list) or not required_assets:
            errors.append(f"{prefix}.required_assets must be a non-empty list")
        elif status == "prepared":
            for asset in required_assets:
                if not isinstance(asset, str):
                    errors.append(f"{prefix}.required_assets values must be strings")
                    continue
                if asset.startswith(("version-pinned ", "unaltered ", "raw ")):
                    continue
                if not (ROOT / asset).exists():
                    errors.append(
                        f"{prefix} is prepared but required asset is missing: {asset}"
                    )

        next_action = target.get("next_action")
        if not isinstance(next_action, str) or not next_action.strip():
            errors.append(f"{prefix}.next_action must be a non-empty string")

    llms_text = (ROOT / "llms.txt").read_text(encoding="utf-8")
    for required_text in (
        "Context Assurance",
        "No universal guarantee is claimed",
        CANONICAL_REPOSITORY,
    ):
        if required_text not in llms_text:
            errors.append(f"llms.txt is missing required text: {required_text!r}")

    submission_kit = (ROOT / "docs/distribution/submission-kit.md").read_text(
        encoding="utf-8"
    )
    for forbidden_claim in (
        "zero quality loss",
        "better than Headroom",
        "better than LeanCTX",
    ):
        marker = f'- "{forbidden_claim}";'
        if marker not in submission_kit:
            errors.append(
                "submission kit must explicitly prohibit the claim "
                f"{forbidden_claim!r}"
            )

    return errors


def main() -> int:
    try:
        errors = validate()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"distribution surface check failed: {exc}", file=sys.stderr)
        return 1

    if errors:
        print("distribution surface check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("distribution surface check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
