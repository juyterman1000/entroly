from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "1.0.51"
CANONICAL_MCP_NAME = "io.github.juyterman1000/entroly"
CANONICAL_REPOSITORY = "https://github.com/juyterman1000/entroly"


def _read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _read_project_metadata(path: str) -> dict[str, object]:
    """Read the small pyproject surface guarded by these tests.

    Python 3.10 does not ship ``tomllib``. Keeping this parser local avoids
    making the release guard depend on an extra test dependency.
    """
    metadata: dict[str, object] = {
        "optional-dependencies": {},
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


def test_public_package_versions_are_1_0_51() -> None:
    assert _read_project_metadata("pyproject.toml")["version"] == RELEASE_VERSION
    assert _read_project_metadata("entroly/pyproject.toml")["version"] == RELEASE_VERSION
    assert _read_json("entroly/npm/package.json")["version"] == RELEASE_VERSION
    assert _read_json("entroly/npm-alias/package.json")["version"] == RELEASE_VERSION
    assert _read_json("entroly-wasm/package.json")["version"] == RELEASE_VERSION
    assert _read_json("integrations/openclaw/package.json")["version"] == RELEASE_VERSION
    assert _read_json(".claude-plugin/manifest.json")["version"] == RELEASE_VERSION
    assert _read_json(".mcpb-build/manifest.json")["version"] == RELEASE_VERSION


def test_mcp_registry_manifest_points_at_release_package() -> None:
    manifest = _read_json("server.json")

    assert manifest["version"] == RELEASE_VERSION
    packages = manifest["packages"]
    assert packages
    assert packages[0]["identifier"] == "entroly"
    assert packages[0]["version"] == RELEASE_VERSION


def test_mcp_registry_identity_is_canonical_and_non_squattable() -> None:
    manifest = _read_json("server.json")
    npm_manifest = _read_json("entroly/npm/package.json")

    assert manifest["name"] == CANONICAL_MCP_NAME
    assert manifest["websiteUrl"] == CANONICAL_REPOSITORY
    assert manifest["repository"] == {
        "url": CANONICAL_REPOSITORY,
        "source": "github",
    }
    assert npm_manifest["mcpName"] == CANONICAL_MCP_NAME
    assert _read_project_metadata("pyproject.toml")["readme"] == "PYPI_README.md"
    assert f"mcp-name: {CANONICAL_MCP_NAME}" in (
        ROOT / "PYPI_README.md"
    ).read_text(encoding="utf-8")

    expected = {
        ("pypi", "entroly", "uvx", ("serve",)),
        ("npm", "entroly-mcp", "npx", ("serve",)),
    }
    actual = {
        (
            package["registryType"],
            package["identifier"],
            package["runtimeHint"],
            tuple(argument["value"] for argument in package["packageArguments"]),
        )
        for package in manifest["packages"]
    }
    assert actual == expected


def test_native_engine_is_optional_for_first_time_install() -> None:
    for path in ("pyproject.toml", "entroly/pyproject.toml"):
        project = _read_project_metadata(path)
        hard_deps = project["dependencies"]
        native_deps = project["optional-dependencies"]["native"]
        full_deps = project["optional-dependencies"]["full"]

        assert not any(dep.startswith("entroly-core") for dep in hard_deps)
        assert f"entroly-core>={RELEASE_VERSION},<2" in native_deps
        assert f"entroly-core>={RELEASE_VERSION},<2" in full_deps


def test_no_stale_package_advertising_versions() -> None:
    stale = []
    for path in (
        "server.json",
        ".mcpb-build/manifest.json",
        ".claude-plugin/manifest.json",
        "entroly/npm/package.json",
        "entroly/npm-alias/package.json",
        "entroly-wasm/package.json",
        "integrations/openclaw/package.json",
        "pyproject.toml",
        "entroly/pyproject.toml",
    ):
        text = (ROOT / path).read_text(encoding="utf-8")
        if re.search(r'"version"\s*:\s*"1\.0\.41"', text):
            stale.append(path)

    assert stale == []


def test_homebrew_formula_targets_release_sdist() -> None:
    text = (ROOT / "packaging/homebrew/entroly.rb").read_text(encoding="utf-8")

    assert f"entroly-{RELEASE_VERSION}.tar.gz" in text
    assert "packages/source/e/entroly/" in text
    assert "94e54692b4e677e9261d2c667a30ecaec3c0f871729c1cc84256854361d9ec1e" in text
