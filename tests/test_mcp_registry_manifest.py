from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_NAME = "io.github.juyterman1000/entroly"
CANONICAL_REPOSITORY = "https://github.com/juyterman1000/entroly"
SCHEMA = "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json"


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _project_value(name: str) -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(rf'(?m)^{re.escape(name)}\s*=\s*"([^"]+)"', text)
    assert match is not None, f"missing {name} in pyproject.toml"
    return match.group(1)


def test_mcp_registry_manifest_respects_schema_bounds() -> None:
    manifest = _json("server.json")

    assert manifest["$schema"] == SCHEMA
    assert 3 <= len(manifest["name"]) <= 200
    assert 1 <= len(manifest["title"]) <= 100
    assert 1 <= len(manifest["description"]) <= 100
    assert 1 <= len(manifest["version"]) <= 255


def test_mcp_registry_identity_and_package_contract() -> None:
    manifest = _json("server.json")
    version = _project_value("version")

    assert manifest["name"] == CANONICAL_NAME
    assert manifest["websiteUrl"] == CANONICAL_REPOSITORY
    assert manifest["repository"] == {
        "url": CANONICAL_REPOSITORY,
        "source": "github",
    }
    assert manifest["version"] == version

    packages = {
        (package["registryType"], package["identifier"]): package
        for package in manifest["packages"]
    }
    assert set(packages) == {
        ("pypi", "entroly"),
        ("npm", "entroly-mcp"),
    }
    expected_runtime = {
        ("pypi", "entroly"): "uvx",
        ("npm", "entroly-mcp"): "npx",
    }
    for key, package in packages.items():
        assert package["version"] == version
        assert package["runtimeHint"] == expected_runtime[key]
        assert package["transport"] == {"type": "stdio"}
        assert package.get("packageArguments", []) == []


def test_registry_package_ownership_proofs_are_canonical() -> None:
    assert _project_value("readme") == "PYPI_README.md"
    pypi_description = (ROOT / "PYPI_README.md").read_text(encoding="utf-8")
    assert f"mcp-name: {CANONICAL_NAME}" in pypi_description

    npm = _json("entroly/npm/package.json")
    assert npm["name"] == "entroly-mcp"
    assert npm["mcpName"] == CANONICAL_NAME
    assert npm["repository"]["url"].removesuffix(".git") == CANONICAL_REPOSITORY
