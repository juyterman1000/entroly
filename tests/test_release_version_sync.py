from __future__ import annotations

import json
import re
from pathlib import Path

from entroly import __version__


ROOT = Path(__file__).resolve().parent.parent


def _toml_version(text: str) -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', text, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def _json_version(text: str) -> str:
    return str(json.loads(text)["version"])


def _json(text: str) -> dict:
    return json.loads(text)


def _formula_version(text: str) -> str:
    match = re.search(r'url\s+"[^"]+/entroly-([0-9]+\.[0-9]+\.[0-9]+)\.tar\.gz"', text)
    assert match is not None
    return match.group(1)


def _daemon_version(text: str) -> str:
    match = re.search(r'version:\s*str\s*=\s*"([^"]+)"', text)
    assert match is not None
    return match.group(1)


def _native_min_version(text: str) -> str:
    match = re.search(r'MIN_ENTROLY_CORE_VERSION\s*=\s*"([^"]+)"', text)
    assert match is not None
    return match.group(1)


def _fallback_version(text: str, variable: str) -> str:
    match = re.search(rf'{re.escape(variable)}\s*=\s*"([^"]+)"', text)
    assert match is not None
    return match.group(1)


def _native_dependency_min_versions(text: str) -> set[str]:
    return set(re.findall(r'entroly-core>=([0-9]+\.[0-9]+\.[0-9]+)', text))


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_release_version_surfaces_match_package_version() -> None:
    assert _toml_version(_read("pyproject.toml")) == __version__
    assert _toml_version(_read("entroly/pyproject.toml")) == __version__
    assert _toml_version(_read("entroly-core/pyproject.toml")) == __version__
    assert _toml_version(_read("entroly-core/Cargo.toml")) == __version__
    assert _toml_version(_read("entroly-qccr/Cargo.toml")) == __version__
    assert _toml_version(_read("entroly-wasm/Cargo.toml")) == __version__
    assert _json_version(_read("entroly/npm/package.json")) == __version__
    assert _json_version(_read("entroly/npm-alias/package.json")) == __version__
    assert _json_version(_read("entroly-wasm/package.json")) == __version__
    assert _json_version(_read(".claude-plugin/manifest.json")) == __version__
    assert _json_version(_read(".mcpb-build/manifest.json")) == __version__
    assert _formula_version(_read("packaging/homebrew/entroly.rb")) == __version__
    assert _daemon_version(_read("entroly/daemon.py")) == __version__
    assert _native_min_version(_read("entroly/native_status.py")) == __version__
    assert _fallback_version(_read("entroly/cli.py"), "__version__") == __version__
    assert _fallback_version(_read("entroly/server.py"), "_version") == __version__
    assert _native_dependency_min_versions(_read("pyproject.toml")) == {__version__}
    assert _native_dependency_min_versions(_read("entroly/pyproject.toml")) == {__version__}
    assert _native_dependency_min_versions(_read("entroly/cli.py")) == {__version__}
    assert _native_dependency_min_versions(_read("entroly-core/README.md")) == {__version__}

    npm_alias = _json(_read("entroly/npm-alias/package.json"))
    assert npm_alias["dependencies"]["entroly-wasm"] == __version__

    server_manifest = _json(_read("server.json"))
    assert server_manifest["version"] == __version__
    assert {pkg["version"] for pkg in server_manifest["packages"]} == {__version__}
