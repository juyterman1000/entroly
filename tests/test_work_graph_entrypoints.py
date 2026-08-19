"""The Work Graph console scripts and their runtime must actually ship.

Reads the manifest through ``tests/pyproject_compat`` rather than ``tomllib``.
``tomllib`` arrived in Python 3.11 and both manifests declare
``requires-python = ">=3.10"``; a module-level import of it here does not skip
this file on 3.10, it aborts collection for the entire suite, so no test in the
repository runs at all on the oldest supported interpreter.
"""

from __future__ import annotations

from pyproject_compat import read_project_metadata


def test_work_graph_entry_points_are_published() -> None:
    scripts = read_project_metadata("pyproject.toml")["scripts"]

    assert scripts["entroly-work"] == "entroly.work_graph_cli:main"
    assert scripts["entroly-work-graph-mcp"] == "entroly.work_graph_mcp_server:main"


def test_base_install_contains_mcp_runtime_for_work_graph_server() -> None:
    dependencies = tuple(read_project_metadata("pyproject.toml")["dependencies"])

    assert any(item.startswith("mcp>=") for item in dependencies)
