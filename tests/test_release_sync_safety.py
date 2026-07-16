from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_sync_module():
    path = ROOT / "scripts" / "sync_release_version.py"
    spec = importlib.util.spec_from_file_location("entroly_sync_release_version", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        '[project]\nname = "entroly"\nversion = "1.0.51"\n',
        encoding="utf-8",
    )
    (tmp_path / "server.json").write_text(
        '{"name":"io.github.juyterman1000/entroly","version":"1.0.51"}\n',
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
    assert 'version = "1.0.52"' in (
        tmp_path / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert '"version":"1.0.52"' in (
        tmp_path / "server.json"
    ).read_text(encoding="utf-8")


def test_synchronizer_rebuilds_mcp_bundle_from_updated_manifest(tmp_path: Path) -> None:
    module = _load_sync_module()
    module.RELEASE_SURFACES = (
        "pyproject.toml",
        ".mcpb-build/manifest.json",
    )

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "entroly"\nversion = "1.0.51"\n',
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
