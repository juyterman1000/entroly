from __future__ import annotations

import tomllib
from pathlib import Path


def test_root_pyproject_exposes_entrolies_memory_entrypoints() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]

    assert scripts["entroly"] == "entroly._docker_launcher:launch"
    assert scripts["entroly-memory"] == "entroly.memory_cli:main"


def test_nested_pyproject_exposes_memory_entrypoint() -> None:
    data = tomllib.loads(Path("entroly/pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]

    assert scripts["entroly-memory"] == "entroly.memory_cli:main"
