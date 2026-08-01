from __future__ import annotations

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_assurance_api_is_importable_from_package_root() -> None:
    entroly = importlib.import_module("entroly")
    assert callable(entroly.compress_assured)
    assert callable(entroly.compress_file_assured)
    assert callable(entroly.compress_messages_assured)
    assert entroly.AssuranceLedger
    assert entroly.RepositoryIntelligence


def test_assurance_mcp_entrypoint_is_consistent_across_pyprojects() -> None:
    expected = 'entroly-assurance-mcp = "entroly.assurance_mcp:main"'
    assert expected in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert expected in (ROOT / "entroly/pyproject.toml").read_text(encoding="utf-8")


def test_assurance_documentation_is_linked_from_public_readmes() -> None:
    assert "docs/ASSURED_CONTEXT.md" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "entroly-assurance-mcp" in (ROOT / "PYPI_README.md").read_text(encoding="utf-8")
