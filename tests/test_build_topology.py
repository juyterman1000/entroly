from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from entroly.repository_intelligence.build_topology import (
    build_verified_build_topology,
    verify_build_topology_commitment,
)
from entroly.repository_intelligence.models import FileRecord, RepositoryIndex


def _record(path: str, source: bytes, language: str, *, is_test: bool = False) -> FileRecord:
    return FileRecord(
        path=path,
        language=language,
        sha256=hashlib.sha256(source).hexdigest(),
        byte_length=len(source),
        line_count=max(1, source.count(b"\n") + 1),
        is_test=is_test,
    )


def test_zig_manifest_establishes_component_without_claiming_build_inclusion(
    tmp_path: Path,
) -> None:
    (tmp_path / "build.zig").write_text("pub fn build(b: *Build) void {}\n", encoding="utf-8")
    source = b"pub fn main() void {}\n"
    test_source = b"test \"works\" {}\n"
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.zig").write_bytes(source)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/main_test.zig").write_bytes(test_source)
    index = RepositoryIndex(
        root=str(tmp_path),
        files={
            "src/main.zig": _record("src/main.zig", source, "zig"),
            "tests/main_test.zig": _record(
                "tests/main_test.zig", test_source, "zig", is_test=True
            ),
        },
    )

    topology = build_verified_build_topology(
        tmp_path,
        index,
        index_digest="sha256:index",
    )
    assert len(topology.components) == 1
    component = topology.components[0]
    assert component.ecosystem == "zig-build"
    assert component.manifest_path == "build.zig"
    by_path = {item.path: item for item in topology.ownership}
    assert by_path["src/main.zig"].component_ids == (component.component_id,)
    assert by_path["tests/main_test.zig"].is_test is True
    payload = topology.to_dict()
    assert payload["analysis_contract"]["ownership_is_build_inclusion"] is False
    assert verify_build_topology_commitment(payload)


def test_nested_manifest_wins_by_nearest_ancestry(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"name":"root"}', encoding="utf-8")
    crate = tmp_path / "native"
    crate.mkdir()
    (crate / "Cargo.toml").write_text('[package]\nname="native"\n', encoding="utf-8")
    source = b"fn run() {}\n"
    (crate / "src").mkdir()
    (crate / "src/lib.rs").write_bytes(source)
    index = RepositoryIndex(
        root=str(tmp_path),
        files={"native/src/lib.rs": _record("native/src/lib.rs", source, "rust")},
    )

    topology = build_verified_build_topology(
        tmp_path,
        index,
        index_digest="sha256:index",
    )
    components = {item.ecosystem: item for item in topology.components}
    ownership = topology.ownership[0]
    assert ownership.component_ids == (components["cargo"].component_id,)
    assert ownership.resolution == "unique-nearest-manifest"


def test_same_root_multiple_build_manifests_preserve_ambiguity(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname="mixed"\n', encoding="utf-8")
    (tmp_path / "package.json").write_text('{"name":"mixed-web"}', encoding="utf-8")
    source = b"print('mixed')\n"
    (tmp_path / "tool.py").write_bytes(source)
    index = RepositoryIndex(
        root=str(tmp_path),
        files={"tool.py": _record("tool.py", source, "python")},
    )

    topology = build_verified_build_topology(
        tmp_path,
        index,
        index_digest="sha256:index",
    )
    assert len(topology.components) == 2
    ownership = topology.ownership[0]
    assert ownership.resolution == "ambiguous-nearest-manifests"
    assert len(ownership.component_ids) == 2


def test_file_without_manifest_ancestor_stays_explicitly_unowned(tmp_path: Path) -> None:
    source = b"fn run() {}\n"
    (tmp_path / "main.future").write_bytes(source)
    index = RepositoryIndex(
        root=str(tmp_path),
        files={"main.future": _record("main.future", source, "unknown")},
    )
    topology = build_verified_build_topology(
        tmp_path,
        index,
        index_digest="sha256:index",
    )
    assert topology.components == ()
    assert topology.ownership[0].resolution == "no-manifest-ancestor"
    assert topology.ownership[0].component_ids == ()


def test_manifest_symlink_is_ignored_without_following_outside_workspace(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-build.zig"
    outside.write_text("SECRET", encoding="utf-8")
    link = tmp_path / "build.zig"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    source = b"pub fn main() void {}\n"
    (tmp_path / "main.zig").write_bytes(source)
    index = RepositoryIndex(
        root=str(tmp_path),
        files={"main.zig": _record("main.zig", source, "zig")},
    )

    topology = build_verified_build_topology(
        tmp_path,
        index,
        index_digest="sha256:index",
    )
    assert topology.components == ()
    assert any(item.startswith("symlink-manifest-ignored:build.zig") for item in topology.diagnostics)
    assert outside.read_text(encoding="utf-8") == "SECRET"


def test_build_topology_commitment_detects_tampering(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/x\n", encoding="utf-8")
    source = b"package main\n"
    (tmp_path / "main.go").write_bytes(source)
    index = RepositoryIndex(
        root=str(tmp_path),
        files={"main.go": _record("main.go", source, "go")},
    )
    payload = build_verified_build_topology(
        tmp_path,
        index,
        index_digest="sha256:index",
    ).to_dict()
    assert verify_build_topology_commitment(payload)
    payload["ownership"][0]["resolution"] = "forged"
    assert not verify_build_topology_commitment(payload)
