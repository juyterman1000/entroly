from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "distribution" / "targets.json"
DIMENSIONS = ROOT / "docs" / "distribution" / "visibility-dimensions.json"
CHECKER = ROOT / "scripts" / "check_distribution_surface.py"


def _project_version() -> str:
    in_project = False
    for raw_line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[project]":
            in_project = True
            continue
        if in_project and line.startswith("["):
            break
        if in_project and line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise AssertionError("pyproject.toml is missing project.version")


def test_distribution_surface_check_passes_offline() -> None:
    completed = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "distribution surface check passed" in completed.stdout


def test_high_priority_targets_are_actionable_without_implied_submission() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    high_priority = [
        target for target in registry["targets"] if target["priority"] == 1
    ]

    assert high_priority
    assert {target["status"] for target in high_priority} == {"prepared"}
    for target in high_priority:
        assert target["proof_url"] is None
        assert target["next_action"].strip()
        assert target["required_assets"]


def test_published_targets_have_public_proof_urls() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    published = [
        target for target in registry["targets"] if target["status"] == "published"
    ]

    assert published
    for target in published:
        assert target["proof_url"].startswith("https://")


def test_visibility_matrix_covers_every_competitive_dimension() -> None:
    matrix = json.loads(DIMENSIONS.read_text(encoding="utf-8"))
    dimensions = matrix["dimensions"]
    ids = {dimension["id"] for dimension in dimensions}

    assert len(dimensions) >= 30
    assert len(ids) == len(dimensions)
    assert {
        "seo-technical",
        "answer-engine-discovery",
        "mcp-discovery",
        "claude-plugin-skill",
        "independent-reviews",
        "neutral-benchmarks",
        "os-package-managers",
        "launch-channels",
        "newsletters-media",
        "research-citation",
        "press-media-kit",
        "security-trust",
        "distribution-observability",
    } <= ids


def test_gap_dimensions_have_executable_next_actions() -> None:
    matrix = json.loads(DIMENSIONS.read_text(encoding="utf-8"))
    unfinished = [
        dimension
        for dimension in matrix["dimensions"]
        if dimension["state"] in {"partial", "gap", "blocked"}
    ]

    assert unfinished
    for dimension in unfinished:
        assert len(dimension["next_action"].strip()) >= 20
        assert len(dimension["leadership_target"].strip()) >= 20


def test_launch_assets_cannot_imply_publication() -> None:
    launch_dir = ROOT / "marketing" / "launch"
    launch_files = sorted(launch_dir.glob("*.md"))

    assert launch_files
    for path in launch_files:
        content = path.read_text(encoding="utf-8")
        assert "Status: prepared, not submitted." in content
        assert "https://github.com/juyterman1000/entroly" in content


def test_research_metadata_tracks_release_version() -> None:
    project_version = _project_version()
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    codemeta = json.loads((ROOT / "codemeta.json").read_text(encoding="utf-8"))

    assert f"version: {project_version}" in citation
    assert codemeta["version"] == project_version
    assert codemeta["codeRepository"] == "https://github.com/juyterman1000/entroly"


def test_expanded_visibility_assets_are_owned_and_present() -> None:
    required = (
        ".github/CODEOWNERS",
        ".github/workflows/visibility-integrity.yml",
        "docs/choosing-context-assurance.md",
        "docs/localization/README.md",
        "marketing/release-announcement-template.md",
    )
    for relative in required:
        assert (ROOT / relative).is_file(), relative

    codeowners = (ROOT / ".github/CODEOWNERS").read_text(encoding="utf-8")
    for protected_path in (
        "/docs/distribution/",
        "/docs/press-kit.md",
        "/docs/choosing-context-assurance.md",
        "/docs/localization/",
        "/marketing/",
        "/CITATION.cff",
        "/codemeta.json",
        "/scripts/check_distribution_surface.py",
    ):
        assert protected_path in codeowners


def test_release_announcement_template_is_fail_closed() -> None:
    template = (ROOT / "marketing/release-announcement-template.md").read_text(
        encoding="utf-8"
    )

    assert "Status: template, not a published release announcement." in template
    assert "after the coordinated release workflow verifies" in template
    assert "Do not hide a known regression" in template
    assert "Version is synchronized in citation metadata" in template


def test_localization_contract_requires_human_and_technical_review() -> None:
    localization = (ROOT / "docs/localization/README.md").read_text(
        encoding="utf-8"
    )

    assert "one language reviewer and one Entroly technical reviewer" in localization
    assert "Draft pages must not be indexed" in localization
    assert "cannot satisfy the review requirement by itself" in localization
