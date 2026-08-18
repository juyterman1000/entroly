from __future__ import annotations

import tomllib
from pathlib import Path


def test_work_graph_entrypoints_are_shipped() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = project["project"]["scripts"]
    assert scripts["entroly-work"] == "entroly.work_graph_cli:main"
    assert scripts["entroly-work-graph-mcp"] == "entroly.work_graph_mcp_server:main"
