from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bump_version.py"
SPEC = importlib.util.spec_from_file_location("visibility_bump_version", SCRIPT)
assert SPEC and SPEC.loader
bump_version = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bump_version)


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


def test_citation_metadata_matches_public_release() -> None:
    version = _project_version()
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    codemeta = json.loads((ROOT / "codemeta.json").read_text(encoding="utf-8"))

    assert f"version: {version}" in citation
    assert codemeta["version"] == version


def test_version_bumper_owns_citation_metadata() -> None:
    target_paths = [target[0] for target in bump_version.TARGETS]

    assert "CITATION.cff" in target_paths
    assert "codemeta.json" in target_paths
