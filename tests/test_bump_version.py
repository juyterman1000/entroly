"""Regression tests for scripts/bump_version.py."""
from __future__ import annotations

import importlib.util
import json
import re
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bump_version.py"
SPEC = importlib.util.spec_from_file_location("bump_version", SCRIPT)
assert SPEC and SPEC.loader
bump_version = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bump_version)


def test_semver_validation_accepts_release_prerelease_and_build():
    valid = [
        "1.0.46",
        "0.1.0-alpha.1",
        "2.3.4-rc.1+build.7",
    ]

    for version in valid:
        assert bump_version.SEMVER.fullmatch(version), version


def test_semver_validation_rejects_malformed_release_versions():
    invalid = [
        "01.0.0",
        "1.02.0",
        "1.0.03",
        "1.0",
        "1.0.0-",
        "1.0.0+",
        "1.0.0-alpha..1",
        "1.0.0-alpha 1",
        "1.0.0-01",
    ]

    for version in invalid:
        assert not bump_version.SEMVER.fullmatch(version), version


def test_failed_bump_does_not_write_partial_changes(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.toml"
    original = 'version = "1.0.0"\ndep = "stable"\n'
    manifest.write_text(original, encoding="utf-8")

    monkeypatch.setattr(bump_version, "ROOT", tmp_path)
    monkeypatch.setattr(bump_version, "TARGETS", [
        ("manifest.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
        ("manifest.toml", r'^missing\s*=', 'missing = "{v}"'),
    ])

    assert bump_version.main(["bump_version.py", "1.0.1"]) == 1
    assert manifest.read_text(encoding="utf-8") == original


def test_bump_chains_multiple_edits_to_the_same_file(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.toml"
    manifest.write_text('version = "1.0.0"\ndep = "core>=1.0.0,<2"\n', encoding="utf-8")

    monkeypatch.setattr(bump_version, "ROOT", tmp_path)
    monkeypatch.setattr(bump_version, "TARGETS", [
        ("manifest.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
        ("manifest.toml", r'core>=[0-9]+\.[0-9]+\.[0-9]+,<2', 'core>={v},<2'),
    ])

    assert bump_version.main(["bump_version.py", "1.0.1"]) == 0
    assert manifest.read_text(encoding="utf-8") == (
        'version = "1.0.1"\ndep = "core>=1.0.1,<2"\n'
    )


def test_bump_summary_reports_replacements_and_unique_files(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.toml"
    manifest.write_text('version = "1.0.0"\ndep = "core>=1.0.0,<2"\n', encoding="utf-8")

    monkeypatch.setattr(bump_version, "ROOT", tmp_path)
    monkeypatch.setattr(bump_version, "TARGETS", [
        ("manifest.toml", r'^version\s*=\s*"[^"]+"', 'version = "{v}"'),
        ("manifest.toml", r'core>=[0-9]+\.[0-9]+\.[0-9]+,<2', 'core>={v},<2'),
        ("generated/package.json", r'"version"\s*:\s*"[^"]+"', '"version": "{v}"'),
    ])

    assert bump_version.main(["bump_version.py", "1.0.1"]) == 0

    output = capsys.readouterr().out
    assert "generated/package.json missing; skipping generated artifact" in output
    assert output.count("manifest.toml -> 1.0.1") == 1
    assert "bumped 2 target(s) across 1 file(s) to 1.0.1" in output


def test_bump_rebuilds_mcp_bundle_after_manifest_update(
    tmp_path, monkeypatch, capsys
):
    manifest = tmp_path / ".mcpb-build" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '{"name":"entroly","version":"1.0.0"}\n',
        encoding="utf-8",
    )
    (tmp_path / "entroly.mcpb").write_bytes(b"stale bundle")

    monkeypatch.setattr(bump_version, "ROOT", tmp_path)
    monkeypatch.setattr(
        bump_version,
        "TARGETS",
        [
            (
                ".mcpb-build/manifest.json",
                r'"version"\s*:\s*"[^"]+"',
                '"version": "{v}"',
            )
        ],
    )

    assert bump_version.main(["bump_version.py", "1.0.1"]) == 0

    with zipfile.ZipFile(tmp_path / "entroly.mcpb") as archive:
        bundled = json.loads(archive.read("manifest.json"))
        assert archive.namelist() == ["manifest.json"]
    assert bundled["version"] == "1.0.1"
    output = capsys.readouterr().out
    assert "entroly.mcpb -> rebuilt" in output
    assert "across 2 file(s)" in output


def test_homebrew_readme_targets_update_heading_and_command():
    text = (
        "Current release example version: `1.0.39`\n"
        "Set VER=1.0.39\n"
        "Download entroly-1.0.39.tar.gz\n"
    )
    targets = [
        (pattern, template)
        for path, pattern, template in bump_version.TARGETS
        if path == "packaging/homebrew/README.md"
    ]

    for pattern, template in targets:
        text = re.sub(pattern, template.format(v="1.0.40"), text)

    assert "Current release example version: `1.0.40`" in text
    assert "VER=1.0.40" in text
    assert "entroly-1.0.40.tar.gz" in text
