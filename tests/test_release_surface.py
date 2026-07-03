from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "1.0.42"


def _read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _read_toml(path: str) -> dict:
    return tomllib.loads((ROOT / path).read_text(encoding="utf-8"))


def test_public_package_versions_are_1_0_42() -> None:
    assert _read_toml("pyproject.toml")["project"]["version"] == RELEASE_VERSION
    assert _read_toml("entroly/pyproject.toml")["project"]["version"] == RELEASE_VERSION
    assert _read_json("entroly/npm/package.json")["version"] == RELEASE_VERSION
    assert _read_json("entroly/npm-alias/package.json")["version"] == RELEASE_VERSION
    assert _read_json("entroly-wasm/package.json")["version"] == RELEASE_VERSION
    assert _read_json(".claude-plugin/manifest.json")["version"] == RELEASE_VERSION
    assert _read_json(".mcpb-build/manifest.json")["version"] == RELEASE_VERSION


def test_mcp_registry_manifest_points_at_release_package() -> None:
    manifest = _read_json("server.json")

    assert manifest["version"] == RELEASE_VERSION
    packages = manifest["packages"]
    assert packages
    assert packages[0]["identifier"] == "entroly"
    assert packages[0]["version"] == RELEASE_VERSION


def test_native_engine_is_optional_for_first_time_install() -> None:
    for path in ("pyproject.toml", "entroly/pyproject.toml"):
        project = _read_toml(path)["project"]
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
    assert "5a61299142fdddc6a4a761ae0eba6def3caf857d58dc9fe5648858a416b0517c" in text
