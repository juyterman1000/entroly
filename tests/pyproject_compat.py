"""Minimal pyproject reader for tests that must run on Python 3.10.

``tomllib`` is standard library only from Python 3.11, and both manifests
declare ``requires-python = ">=3.10"``. A module-level ``import tomllib`` in a
test file therefore does not fail that one test on 3.10 -- it aborts pytest
collection for the whole run, so *no* test executes. That is how
``tests/test_work_graph_entrypoints.py`` and ``tests/test_work_graph_packaging.py``
took the entire suite down on the oldest supported interpreter.

The repository had already solved this twice: ``scripts/codebase_graph.py``
guards the import, and ``tests/test_release_surface.py`` carried a local parser
whose docstring said plainly that "Python 3.10 does not ship ``tomllib``".
This module is that parser, promoted to one shared home rather than copied a
third time, and taught to read ``[project.scripts]``.

It is deliberately not a general TOML implementation. It reads the small,
well-formed surface these release guards assert against: version, readme,
dependencies, optional-dependencies, and scripts.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_project_metadata(path: str) -> dict[str, object]:
    """Read the small pyproject surface guarded by the release tests."""
    metadata: dict[str, object] = {
        "optional-dependencies": {},
        "scripts": {},
    }
    current_section = ""
    current_list_key: str | None = None

    for raw_line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line.strip("[]")
            current_list_key = None
            continue
        if current_section == "project" and line.startswith("version"):
            metadata["version"] = line.split("=", 1)[1].strip().strip('"')
            continue
        if current_section == "project" and line.startswith("readme"):
            metadata["readme"] = line.split("=", 1)[1].strip().strip('"')
            continue
        if current_section == "project" and line.startswith("dependencies"):
            current_list_key = "dependencies"
            metadata[current_list_key] = []
            continue
        # `[project.scripts]` is a flat table of console-script name to target,
        # not a list, so it is read here rather than through current_list_key.
        if current_section == "project.scripts" and "=" in line:
            name, _, target = line.partition("=")
            metadata["scripts"][name.strip().strip('"')] = target.strip().strip('"')
            continue
        if (
            current_section == "project.optional-dependencies"
            and "=" in line
            and not line.startswith('"')
        ):
            key = line.split("=", 1)[0].strip()
            current_list_key = key
            metadata["optional-dependencies"][key] = []
            continue
        if current_list_key and line.startswith('"'):
            value = line.rstrip(",").strip().strip('"')
            if current_section == "project":
                metadata[current_list_key].append(value)
            elif current_section == "project.optional-dependencies":
                metadata["optional-dependencies"][current_list_key].append(value)

    return metadata
