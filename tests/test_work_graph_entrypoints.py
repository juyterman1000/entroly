from __future__ import annotations

import tomllib
from pathlib import Path


def test_work_graph_entry_points_are_published() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]

    assert scripts["entroly-work"] == "entroly.work_graph_cli:main"
    assert scripts["entroly-work-graph-mcp"] == "entroly.work_graph_mcp_server:main"


def test_base_install_contains_mcp_runtime_for_work_graph_server() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = tuple(data["project"]["dependencies"])

    assert any(item.startswith("mcp>=") for item in dependencies)
