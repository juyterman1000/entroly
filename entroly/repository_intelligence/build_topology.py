"""Verified, language-neutral build/test topology from repository manifests.

Source syntax alone does not describe how a repository is packaged, built, or
tested. This module adds a conservative structural build plane: exact manifest
files establish component roots, and files/tests are associated only by nearest
manifest ancestry. It deliberately does *not* claim that an owned file is
compiled into a target. Future build-tool adapters can strengthen the same
component IDs with observed target membership.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

from .models import RepositoryIndex, normalize_relative

BUILD_TOPOLOGY_SCHEMA_VERSION = "entroly.verified-build-topology.v1"
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_MANIFESTS = 2_000
_MAX_FOCUS_PATHS = 2_000

# Build-system identity is based on the manifest filename, not programming
# language. New language/build ecosystems can be added without changing graph
# semantics or agent-facing APIs.
_MANIFESTS: dict[str, str] = {
    "Cargo.toml": "cargo",
    "pyproject.toml": "python-project",
    "package.json": "node-package",
    "go.mod": "go-module",
    "build.zig": "zig-build",
    "build.zig.zon": "zig-package",
    "CMakeLists.txt": "cmake",
    "pom.xml": "maven",
    "build.gradle": "gradle",
    "build.gradle.kts": "gradle-kotlin",
    "settings.gradle": "gradle-settings",
    "settings.gradle.kts": "gradle-kotlin-settings",
    "WORKSPACE": "bazel-workspace",
    "WORKSPACE.bazel": "bazel-workspace",
    "MODULE.bazel": "bazel-module",
    "BUILD": "bazel-package",
    "BUILD.bazel": "bazel-package",
    "Package.swift": "swift-package",
    "mix.exs": "elixir-mix",
    "Gemfile": "ruby-bundler",
    "composer.json": "php-composer",
    "pubspec.yaml": "dart-pub",
    "pubspec.yml": "dart-pub",
    "project.clj": "clojure-leiningen",
    "deps.edn": "clojure-deps",
    "stack.yaml": "haskell-stack",
    "cabal.project": "haskell-cabal",
    "meson.build": "meson",
    "Makefile": "make",
}


@dataclass(frozen=True)
class BuildComponent:
    component_id: str
    ecosystem: str
    root: str
    manifest_path: str
    manifest_sha256: str
    manifest_bytes: int
    indexed_source_match: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "ecosystem": self.ecosystem,
            "root": self.root,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "manifest_bytes": self.manifest_bytes,
            "indexed_source_match": self.indexed_source_match,
            "epistemic_class": "exact-manifest-source",
        }


@dataclass(frozen=True)
class ComponentOwnership:
    path: str
    component_ids: tuple[str, ...]
    resolution: str
    is_test: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "component_ids": list(self.component_ids),
            "resolution": self.resolution,
            "is_test": self.is_test,
            "epistemic_class": "structural-manifest-ancestry",
        }


@dataclass(frozen=True)
class BuildTopology:
    index_digest: str
    components: tuple[BuildComponent, ...]
    ownership: tuple[ComponentOwnership, ...]
    diagnostics: tuple[str, ...]
    topology_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": BUILD_TOPOLOGY_SCHEMA_VERSION,
            "index_digest": self.index_digest,
            "components": [item.to_dict() for item in self.components],
            "ownership": [item.to_dict() for item in self.ownership],
            "diagnostics": list(self.diagnostics),
            "analysis_contract": {
                "component_fact": "exact-manifest-source",
                "ownership_fact": "nearest-manifest-ancestry-only",
                "ownership_is_build_inclusion": False,
                "multiple_nearest_manifests": "preserve-ambiguity",
                "build_tool_execution": False,
                "remote_calls": 0,
            },
            "receipt": {
                "build_topology_sha256": self.topology_sha256,
                "remote_calls": 0,
                "commitment_scope": "payload-excluding-build-topology-sha256",
            },
        }


def _canonical_payload(
    index_digest: str,
    components: tuple[BuildComponent, ...],
    ownership: tuple[ComponentOwnership, ...],
    diagnostics: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": BUILD_TOPOLOGY_SCHEMA_VERSION,
        "index_digest": index_digest,
        "components": [item.to_dict() for item in components],
        "ownership": [item.to_dict() for item in ownership],
        "diagnostics": list(diagnostics),
        "analysis_contract": {
            "component_fact": "exact-manifest-source",
            "ownership_fact": "nearest-manifest-ancestry-only",
            "ownership_is_build_inclusion": False,
            "multiple_nearest_manifests": "preserve-ambiguity",
            "build_tool_execution": False,
            "remote_calls": 0,
        },
        "receipt": {
            "remote_calls": 0,
            "commitment_scope": "payload-excluding-build-topology-sha256",
        },
    }


def _safe_manifest(root: Path, path: Path) -> tuple[bytes | None, str]:
    try:
        if path.is_symlink():
            return None, "symlink-manifest-ignored"
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        if not resolved.is_file():
            return None, "not-regular-file"
        size = resolved.stat().st_size
        if size < 0 or size > _MAX_MANIFEST_BYTES:
            return None, "oversized-manifest"
        return resolved.read_bytes(), "verified"
    except (OSError, RuntimeError, ValueError):
        return None, "unsafe-or-unreadable-manifest"


def _ancestor_roots(path: str) -> tuple[str, ...]:
    parent = PurePosixPath(path).parent
    parts = parent.parts
    roots = [""]
    for depth in range(1, len(parts) + 1):
        roots.append(PurePosixPath(*parts[:depth]).as_posix())
    return tuple(roots)


def _candidate_roots(paths: Iterable[str]) -> tuple[str, ...]:
    roots: set[str] = {""}
    count = 0
    for path in paths:
        count += 1
        if count > _MAX_FOCUS_PATHS:
            break
        roots.update(_ancestor_roots(path))
    return tuple(sorted(roots, key=lambda value: (value.count("/"), value)))


def _component_id(manifest_path: str, digest: str) -> str:
    material = f"{manifest_path}\0{digest}".encode("utf-8")
    return "build-component:" + hashlib.sha256(material).hexdigest()[:24]


def build_verified_build_topology(
    root: Path,
    index: RepositoryIndex,
    *,
    index_digest: str,
    focus_paths: Iterable[str] | None = None,
    max_manifests: int = _MAX_MANIFESTS,
) -> BuildTopology:
    """Build a bounded structural build/test plane for selected repository paths."""
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    selected = tuple(
        sorted(
            dict.fromkeys(
                normalize_relative(path)
                for path in (focus_paths if focus_paths is not None else index.files)
                if normalize_relative(path)
            )
        )
    )
    manifest_limit = max(1, min(int(max_manifests), _MAX_MANIFESTS))
    components: list[BuildComponent] = []
    diagnostics: list[str] = []

    for component_root in _candidate_roots(selected):
        directory = root / component_root if component_root else root
        for manifest_name, ecosystem in sorted(_MANIFESTS.items()):
            if len(components) >= manifest_limit:
                diagnostics.append("manifest-limit-reached")
                break
            manifest = directory / manifest_name
            if not manifest.exists() and not manifest.is_symlink():
                continue
            raw, status = _safe_manifest(root, manifest)
            relative = normalize_relative(manifest.relative_to(root))
            if raw is None:
                diagnostics.append(f"{status}:{relative}")
                continue
            digest = hashlib.sha256(raw).hexdigest()
            indexed = index.files.get(relative)
            components.append(BuildComponent(
                component_id=_component_id(relative, digest),
                ecosystem=ecosystem,
                root=component_root,
                manifest_path=relative,
                manifest_sha256=digest,
                manifest_bytes=len(raw),
                indexed_source_match=bool(indexed and indexed.sha256 == digest),
            ))
        if len(components) >= manifest_limit:
            break

    components = sorted(
        {item.component_id: item for item in components}.values(),
        key=lambda item: (item.root.count("/"), item.root, item.ecosystem, item.manifest_path),
    )
    by_root: dict[str, list[BuildComponent]] = {}
    for component in components:
        by_root.setdefault(component.root, []).append(component)

    ownership: list[ComponentOwnership] = []
    test_paths = set(index.test_paths)
    for path in selected:
        ancestors = [
            component_root
            for component_root in by_root
            if (
                not component_root
                or path == component_root
                or path.startswith(component_root.rstrip("/") + "/")
            )
        ]
        if not ancestors:
            ownership.append(ComponentOwnership(
                path=path,
                component_ids=(),
                resolution="no-manifest-ancestor",
                is_test=path in test_paths,
            ))
            continue
        nearest = max(ancestors, key=lambda value: (value.count("/"), len(value)))
        candidates = tuple(sorted(item.component_id for item in by_root[nearest]))
        ownership.append(ComponentOwnership(
            path=path,
            component_ids=candidates,
            resolution=(
                "unique-nearest-manifest"
                if len(candidates) == 1
                else "ambiguous-nearest-manifests"
            ),
            is_test=path in test_paths,
        ))

    component_tuple = tuple(components)
    ownership_tuple = tuple(sorted(ownership, key=lambda item: item.path))
    diagnostic_tuple = tuple(sorted(dict.fromkeys(diagnostics)))
    canonical = _canonical_payload(
        index_digest,
        component_tuple,
        ownership_tuple,
        diagnostic_tuple,
    )
    digest = hashlib.sha256(json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")).hexdigest()
    return BuildTopology(
        index_digest=index_digest,
        components=component_tuple,
        ownership=ownership_tuple,
        diagnostics=diagnostic_tuple,
        topology_sha256=digest,
    )


def verify_build_topology_commitment(payload: Mapping[str, object]) -> bool:
    try:
        candidate = copy.deepcopy(dict(payload))
        if candidate.get("schema_version") != BUILD_TOPOLOGY_SCHEMA_VERSION:
            return False
        receipt = candidate.get("receipt")
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("build_topology_sha256"))
        return hashlib.sha256(json.dumps(
            candidate,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "BUILD_TOPOLOGY_SCHEMA_VERSION",
    "BuildComponent",
    "BuildTopology",
    "ComponentOwnership",
    "build_verified_build_topology",
    "verify_build_topology_commitment",
]
