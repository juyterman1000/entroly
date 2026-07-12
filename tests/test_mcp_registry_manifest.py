from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_mcp_registry_manifest_respects_schema_bounds() -> None:
    manifest = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))

    assert 3 <= len(manifest["name"]) <= 200
    assert 1 <= len(manifest["title"]) <= 100
    assert 1 <= len(manifest["description"]) <= 100
    assert 1 <= len(manifest["version"]) <= 255


def test_mcp_registry_package_ownership_contract() -> None:
    manifest = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    expected_name = "io.github.juyterman1000/entroly"

    assert manifest["name"] == expected_name
    packages = {package["registryType"]: package for package in manifest["packages"]}
    assert packages["pypi"]["identifier"] == "entroly"
    assert packages["npm"]["identifier"] == "entroly-mcp"

    npm_manifest = json.loads(
        (ROOT / "entroly" / "npm" / "package.json").read_text(encoding="utf-8")
    )
    assert npm_manifest["mcpName"] == expected_name

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"mcp-name: {expected_name}" in readme
