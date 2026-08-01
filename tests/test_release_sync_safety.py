from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_sync_module():
    path = ROOT / "scripts" / "sync_release_version.py"
    spec = importlib.util.spec_from_file_location("entroly_sync_release_version", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pyproject(version: str) -> str:
    return (
        '[project]\n'
        'name = "entroly"\n'
        f'version = "{version}"\n'
        'dependencies = []\n\n'
        '[project.optional-dependencies]\n'
        f'native = ["entroly-core>={version},<2"]\n'
        f'full = ["entroly-core>={version},<2"]\n'
    )


def test_release_surface_allowlist_excludes_workflows() -> None:
    module = _load_sync_module()

    assert module.RELEASE_SURFACES
    assert all(
        not surface.startswith(".github/workflows/")
        for surface in module.RELEASE_SURFACES
    )


def test_synchronizer_never_rewrites_workflow_definitions(tmp_path: Path) -> None:
    module = _load_sync_module()
    module.RELEASE_SURFACES = ("pyproject.toml", "server.json")

    (tmp_path / "pyproject.toml").write_text(
        _pyproject("1.0.51"),
        encoding="utf-8",
    )
    (tmp_path / "server.json").write_text(
        json.dumps(
            {
                "name": "io.github.juyterman1000/entroly",
                "version": "1.0.51",
                "packages": [
                    {
                        "registryType": "pypi",
                        "identifier": "entroly",
                        "version": "1.0.51",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    workflow = tmp_path / ".github" / "workflows" / "release.yml"
    workflow.parent.mkdir(parents=True)
    original = "name: release\n# example current package 1.0.51\n"
    workflow.write_text(original, encoding="utf-8")

    changed = module.synchronize(tmp_path, "1.0.52")

    assert set(changed) == {
        "docs/releases/v1.0.52.md",
        "pyproject.toml",
        "server.json",
    }
    assert workflow.read_text(encoding="utf-8") == original
    pyproject = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "1.0.52"' in pyproject
    assert "entroly-core>=1.0.52,<2" in pyproject
    registry = json.loads((tmp_path / "server.json").read_text(encoding="utf-8"))
    assert registry["version"] == "1.0.52"
    assert {package["version"] for package in registry["packages"]} == {"1.0.52"}


def test_synchronizer_rebuilds_mcp_bundle_from_updated_manifest(tmp_path: Path) -> None:
    module = _load_sync_module()
    module.RELEASE_SURFACES = (
        "pyproject.toml",
        ".mcpb-build/manifest.json",
    )

    (tmp_path / "pyproject.toml").write_text(
        _pyproject("1.0.51"),
        encoding="utf-8",
    )
    manifest = tmp_path / ".mcpb-build" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '{"name":"entroly","version":"1.0.51"}\n',
        encoding="utf-8",
    )
    (tmp_path / "entroly.mcpb").write_bytes(b"stale bundle")

    changed = module.synchronize(tmp_path, "1.0.52")

    assert "entroly.mcpb" in changed
    with zipfile.ZipFile(tmp_path / "entroly.mcpb") as archive:
        bundled = json.loads(archive.read("manifest.json"))
        assert archive.namelist() == ["manifest.json"]
    assert bundled["version"] == "1.0.52"


def test_synchronizer_updates_npm_alias_dependency_with_package_version(
    tmp_path: Path,
) -> None:
    module = _load_sync_module()
    module.RELEASE_SURFACES = ("entroly/npm-alias/package.json",)

    manifest = tmp_path / "entroly" / "npm-alias" / "package.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "name": "entroly",
                "version": "1.0.51",
                "dependencies": {"entroly-wasm": "1.0.51"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    changed = module.synchronize(tmp_path, "1.0.52")

    assert set(changed) == {
        "docs/releases/v1.0.52.md",
        "entroly/npm-alias/package.json",
    }
    updated = json.loads(manifest.read_text(encoding="utf-8"))
    assert updated["version"] == "1.0.52"
    assert updated["dependencies"]["entroly-wasm"] == "1.0.52"


def test_synchronizer_repairs_partial_bump_without_moving_homebrew_pin(
    tmp_path: Path,
) -> None:
    module = _load_sync_module()
    module.RELEASE_SURFACES = (
        "pyproject.toml",
        "entroly/daemon.py",
        "entroly/cli.py",
        "entroly/server.py",
        "tests/test_release_surface.py",
    )

    (tmp_path / "pyproject.toml").write_text(
        _pyproject("1.0.52"),
        encoding="utf-8",
    )
    package = tmp_path / "entroly"
    package.mkdir()
    (package / "daemon.py").write_text(
        'class State:\n    version: str = "1.0.51"\n',
        encoding="utf-8",
    )
    (package / "cli.py").write_text(
        'try:\n    from entroly import __version__\nexcept ImportError:\n'
        '    __version__ = "1.0.51"\n'
        'NATIVE = "entroly-core>=1.0.51,<2"\n',
        encoding="utf-8",
    )
    (package / "server.py").write_text(
        'try:\n    from entroly import __version__ as _version\nexcept Exception:\n'
        '    _version = "1.0.51"\n',
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    release_test = tests / "test_release_surface.py"
    release_test.write_text(
        'RELEASE_VERSION = "1.0.51"\n'
        'HOMEBREW_FORMULA_VERSION = "1.0.50"\n'
        'HOMEBREW_FORMULA_URL = "entroly-1.0.50.tar.gz"\n'
        'def test_public_package_versions_are_1_0_51():\n'
        '    pass\n',
        encoding="utf-8",
    )

    changed = module.synchronize(tmp_path, "1.0.52")

    assert set(changed) == {
        "docs/releases/v1.0.52.md",
        "entroly/cli.py",
        "entroly/daemon.py",
        "entroly/server.py",
        "tests/test_release_surface.py",
    }
    assert 'version: str = "1.0.52"' in (package / "daemon.py").read_text(
        encoding="utf-8"
    )
    cli = (package / "cli.py").read_text(encoding="utf-8")
    assert '__version__ = "1.0.52"' in cli
    assert "entroly-core>=1.0.52,<2" in cli
    server = (package / "server.py").read_text(encoding="utf-8")
    assert '_version = "1.0.52"' in server
    release_text = release_test.read_text(encoding="utf-8")
    assert 'RELEASE_VERSION = "1.0.52"' in release_text
    assert "def test_public_package_versions_are_1_0_52()" in release_text
    assert 'HOMEBREW_FORMULA_VERSION = "1.0.50"' in release_text
    assert "entroly-1.0.50.tar.gz" in release_text


def test_synchronizer_only_updates_entroly_cargo_lock_packages(tmp_path: Path) -> None:
    module = _load_sync_module()
    module.RELEASE_SURFACES = (
        "pyproject.toml",
        "entroly-wasm/Cargo.lock",
    )

    (tmp_path / "pyproject.toml").write_text(
        _pyproject("1.0.51"),
        encoding="utf-8",
    )
    lock = tmp_path / "entroly-wasm" / "Cargo.lock"
    lock.parent.mkdir()
    lock.write_text(
        'version = 4\n\n'
        '[[package]]\nname = "third-party"\nversion = "1.0.51"\n\n'
        '[[package]]\nname = "entroly-qccr"\nversion = "1.0.51"\n\n'
        '[[package]]\nname = "entroly-qccr-audit"\nversion = "1.0.51"\n\n'
        '[[package]]\nname = "entroly-wasm"\nversion = "1.0.51"\n',
        encoding="utf-8",
    )

    module.synchronize(tmp_path, "1.0.52")

    updated = lock.read_text(encoding="utf-8")
    assert 'name = "third-party"\nversion = "1.0.51"' in updated
    assert 'name = "entroly-qccr"\nversion = "1.0.52"' in updated
    assert 'name = "entroly-qccr-audit"\nversion = "1.0.52"' in updated
    assert 'name = "entroly-wasm"\nversion = "1.0.52"' in updated


def test_synchronizer_updates_openclaw_install_floor_without_rewriting_protocol_history(
    tmp_path: Path,
) -> None:
    module = _load_sync_module()
    module.RELEASE_SURFACES = ("integrations/openclaw/README.md",)

    readme = tmp_path / "integrations" / "openclaw" / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text(
        'pip install "entroly>=1.0.51"\n'
        'pip install "entroly>=1.0.51"\n'
        "The Entroly 1.0.49 bridge v2 protocol remains the compatibility floor.\n",
        encoding="utf-8",
    )

    changed = module.synchronize(tmp_path, "1.0.52")

    assert set(changed) == {
        "docs/releases/v1.0.52.md",
        "integrations/openclaw/README.md",
    }
    updated = readme.read_text(encoding="utf-8")
    assert updated.count('pip install "entroly>=1.0.52"') == 2
    assert "Entroly 1.0.49 bridge v2 protocol" in updated


def test_synchronizer_preflights_all_surfaces_before_writing(tmp_path: Path) -> None:
    module = _load_sync_module()
    module.RELEASE_SURFACES = ("pyproject.toml", "entroly/daemon.py")

    pyproject = tmp_path / "pyproject.toml"
    original = _pyproject("1.0.51")
    pyproject.write_text(original, encoding="utf-8")
    daemon = tmp_path / "entroly" / "daemon.py"
    daemon.parent.mkdir()
    daemon.write_text("class State:\n    version = 'not-semver'\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid release surfaces"):
        module.synchronize(tmp_path, "1.0.52")

    assert pyproject.read_text(encoding="utf-8") == original
    assert not (tmp_path / "docs" / "releases" / "v1.0.52.md").exists()
