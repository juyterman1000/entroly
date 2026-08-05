#!/usr/bin/env python3
"""Validate Entroly's local distribution and visibility control plane.

This check is intentionally offline. It validates repository-owned metadata,
status honesty, launch assets, citation metadata, and competitive-dimension
coverage. It does not claim that an external listing or review is currently live.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_STATUSES = {"prepared", "submitted", "published", "blocked", "rejected"}
ALLOWED_DIMENSION_STATES = {"strong", "partial", "gap", "blocked", "published"}
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
REQUIRED_DIMENSION_FIELDS = {
    "id",
    "name",
    "state",
    "current_evidence",
    "leadership_target",
    "next_action",
}
REQUIRED_DIMENSION_IDS = {
    "category-positioning",
    "repository-first-impression",
    "python-registry",
    "node-registry",
    "rust-native",
    "homebrew",
    "docker-container",
    "mcp-discovery",
    "claude-plugin-skill",
    "agent-integration-breadth",
    "onboarding-experience",
    "owned-documentation",
    "seo-technical",
    "answer-engine-discovery",
    "multilingual-discovery",
    "comparison-evaluation",
    "independent-reviews",
    "neutral-benchmarks",
    "awesome-lists-directories",
    "os-package-managers",
    "launch-channels",
    "newsletters-media",
    "community-social-proof",
    "release-communications",
    "research-citation",
    "press-media-kit",
    "security-trust",
    "governance-contribution",
    "extension-ecosystem",
    "distribution-observability",
}
REQUIRED_DISCOVERY_FILES = (
    Path("README.md"),
    Path("llms.txt"),
    Path("server.json"),
    Path(".claude-plugin/manifest.json"),
    Path("CITATION.cff"),
    Path("codemeta.json"),
    Path("docs/press-kit.md"),
    Path("docs/independent-review-program.md"),
    Path("docs/distribution/README.md"),
    Path("docs/distribution/submission-kit.md"),
    Path("docs/distribution/targets.json"),
    Path("docs/distribution/visibility-dimensions.json"),
    Path("docs/distribution/competitive-visibility.md"),
    Path("marketing/README.md"),
    Path("marketing/launch/product-hunt.md"),
    Path("marketing/launch/show-hn.md"),
    Path("marketing/launch/community-and-newsletters.md"),
    Path(".github/ISSUE_TEMPLATE/independent-review.yml"),
    Path(".github/ISSUE_TEMPLATE/integration-request.yml"),
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


def _validate_targets(registry: dict[str, Any], errors: list[str]) -> None:
    if registry.get("schema_version") != 1:
        errors.append("distribution registry schema_version must be 1")
    if registry.get("repository") != CANONICAL_REPOSITORY:
        errors.append("distribution registry repository URL is not canonical")

    targets = registry.get("targets")
    if not isinstance(targets, list) or not targets:
        errors.append("distribution registry must contain a non-empty targets list")
        return

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


def _validate_dimensions(matrix: dict[str, Any], errors: list[str]) -> None:
    if matrix.get("schema_version") != 1:
        errors.append("visibility dimension schema_version must be 1")

    leadership_rule = matrix.get("leadership_rule")
    if not isinstance(leadership_rule, str) or "evidence" not in leadership_rule:
        errors.append("visibility matrix must define an evidence-based leadership rule")

    dimensions = matrix.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        errors.append("visibility matrix must contain a non-empty dimensions list")
        return

    seen_ids: set[str] = set()
    for index, dimension in enumerate(dimensions):
        prefix = f"dimensions[{index}]"
        if not isinstance(dimension, dict):
            errors.append(f"{prefix} must be an object")
            continue

        missing = sorted(REQUIRED_DIMENSION_FIELDS - dimension.keys())
        if missing:
            errors.append(f"{prefix} missing fields: {', '.join(missing)}")
            continue

        dimension_id = dimension.get("id")
        if not isinstance(dimension_id, str) or not dimension_id:
            errors.append(f"{prefix}.id must be a non-empty string")
        elif dimension_id in seen_ids:
            errors.append(f"duplicate visibility dimension id: {dimension_id}")
        else:
            seen_ids.add(dimension_id)

        state = dimension.get("state")
        if state not in ALLOWED_DIMENSION_STATES:
            errors.append(f"{prefix}.state {state!r} is not allowed")

        evidence = dimension.get("current_evidence")
        if not isinstance(evidence, list):
            errors.append(f"{prefix}.current_evidence must be a list")
        else:
            for item in evidence:
                if not isinstance(item, str) or not item.strip():
                    errors.append(f"{prefix}.current_evidence values must be strings")
                    continue
                if item.startswith("https://"):
                    continue
                if not (ROOT / item).exists():
                    errors.append(f"{prefix} references missing evidence: {item}")

        for field in ("leadership_target", "next_action"):
            value = dimension.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}.{field} must be a non-empty string")

    missing_dimensions = sorted(REQUIRED_DIMENSION_IDS - seen_ids)
    if missing_dimensions:
        errors.append(
            "visibility matrix is missing required dimensions: "
            + ", ".join(missing_dimensions)
        )


def _validate_citation_metadata(version: str, errors: list[str]) -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    required_citation_lines = (
        "cff-version: 1.2.0",
        f"version: {version}",
        f'repository-code: "{CANONICAL_REPOSITORY}"',
        "license: Apache-2.0",
    )
    for line in required_citation_lines:
        if line not in citation:
            errors.append(f"CITATION.cff is missing required line: {line!r}")

    codemeta = _load_json(ROOT / "codemeta.json")
    if codemeta.get("version") != version:
        errors.append("codemeta.json version does not match pyproject.toml")
    if codemeta.get("codeRepository") != CANONICAL_REPOSITORY:
        errors.append("codemeta.json codeRepository is not canonical")
    if codemeta.get("license") != "https://spdx.org/licenses/Apache-2.0":
        errors.append("codemeta.json license is not the canonical SPDX URL")


def _validate_launch_assets(errors: list[str]) -> None:
    launch_files = (
        ROOT / "marketing/launch/product-hunt.md",
        ROOT / "marketing/launch/show-hn.md",
        ROOT / "marketing/launch/community-and-newsletters.md",
    )
    for path in launch_files:
        content = path.read_text(encoding="utf-8")
        if "Status: prepared, not submitted." not in content:
            errors.append(
                f"{path.relative_to(ROOT)} must state that it is prepared, not submitted"
            )
        if CANONICAL_REPOSITORY not in content:
            errors.append(f"{path.relative_to(ROOT)} is missing the canonical repository URL")

    combined = "\n".join(path.read_text(encoding="utf-8") for path in launch_files)
    prohibited_patterns = (
        r"\bguaranteed savings\b",
        r"\bzero quality loss\b",
        r"\bbest context (?:tool|engine|compressor)\b",
        r"\bbeats? External Baseline\b",
        r"\bbeats? External Context Tool\b",
    )
    for pattern in prohibited_patterns:
        if re.search(pattern, combined, flags=re.IGNORECASE):
            errors.append(f"launch assets contain prohibited claim pattern: {pattern}")

    for phrase in (
        "independent",
        "raw artifacts",
        "limitations",
        "maintain",
    ):
        if phrase not in combined:
            errors.append(f"launch assets are missing trust phrase: {phrase!r}")


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
    dimensions = _load_json(ROOT / "docs/distribution/visibility-dimensions.json")

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

    _validate_targets(registry, errors)
    _validate_dimensions(dimensions, errors)
    _validate_citation_metadata(version, errors)
    _validate_launch_assets(errors)

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
        "any unconditional statement that compression cannot reduce answer quality",
        "better than External Baseline",
        "better than External Context Tool",
    ):
        marker = f'- "{forbidden_claim}";'
        if marker not in submission_kit:
            errors.append(
                "submission kit must explicitly prohibit the claim "
                f"{forbidden_claim!r}"
            )

    press_kit = (ROOT / "docs/press-kit.md").read_text(encoding="utf-8")
    for phrase in (
        "does not guarantee a universal token reduction",
        "Proxy mode still sends the selected request",
        "Claims requiring explicit verification",
    ):
        if phrase not in press_kit:
            errors.append(f"press kit is missing required boundary: {phrase!r}")

    review_program = (ROOT / "docs/independent-review-program.md").read_text(
        encoding="utf-8"
    )
    for phrase in (
        "A ceiling where all arms pass does not establish non-inferiority.",
        "Provider-observed usage is distinguished from tokenizer estimates.",
        "Negative results must not be omitted",
    ):
        if phrase not in review_program:
            errors.append(f"review program is missing required rule: {phrase!r}")

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
