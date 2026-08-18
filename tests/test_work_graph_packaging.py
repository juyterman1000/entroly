"""Packaging guard: the Work Graph entry points survive a real install.

Reads the manifest through ``tests/pyproject_compat`` -- see that module for why
``tomllib`` cannot be imported at module scope in this repository's test suite.
"""

from __future__ import annotations

from pyproject_compat import read_project_metadata


def test_work_graph_entrypoints_are_shipped() -> None:
    scripts = read_project_metadata("pyproject.toml")["scripts"]

    assert scripts["entroly-work"] == "entroly.work_graph_cli:main"
    assert scripts["entroly-work-graph-mcp"] == "entroly.work_graph_mcp_server:main"
