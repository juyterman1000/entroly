"""Language-neutral, evidence-conservative workspace dependency resolution.

The parser frontend can observe an import/include/use expression without proving
what workspace artifact it binds to.  This module resolves only relationships
that are unique under deterministic workspace rules and leaves every ambiguous
or external target unresolved.  It intentionally does not invent compiler
search paths, package-manager state, macro expansion, or build-system semantics.
"""
from __future__ import annotations

import posixpath
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Mapping

from .models import normalize_relative
from .parsers import ParsedFile, module_name

WORKSPACE_DEPENDENCY_SCHEMA_VERSION = "entroly.workspace-dependencies.v1"


class DependencyTrust(str, Enum):
    """Strength of the workspace-binding evidence."""

    EXACT_RELATIVE = "exact-relative"
    EXACT_WORKSPACE = "exact-workspace"
    UNIQUE_MODULE = "unique-module"
    UNIQUE_STEM = "unique-stem"
    UNIQUE_SUFFIX = "unique-suffix"


@dataclass(frozen=True)
class WorkspaceDependency:
    importer: str
    imported_text: str
    target: str
    trust: DependencyTrust
    resolution: str

    def to_dict(self) -> dict[str, str]:
        return {
            "importer": self.importer,
            "imported_text": self.imported_text,
            "target": self.target,
            "trust": self.trust.value,
            "resolution": self.resolution,
        }


@dataclass(frozen=True)
class UnresolvedWorkspaceDependency:
    importer: str
    imported_text: str
    reason: str
    candidates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "importer": self.importer,
            "imported_text": self.imported_text,
            "reason": self.reason,
            "candidates": list(self.candidates),
        }


@dataclass(frozen=True)
class WorkspaceDependencyGraph:
    dependencies: tuple[WorkspaceDependency, ...]
    unresolved: tuple[UnresolvedWorkspaceDependency, ...]

    def file_dependencies(self) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, set[str]] = defaultdict(set)
        for edge in self.dependencies:
            grouped[edge.importer].add(edge.target)
        return {
            path: tuple(sorted(values))
            for path, values in sorted(grouped.items())
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": WORKSPACE_DEPENDENCY_SCHEMA_VERSION,
            "dependencies": [item.to_dict() for item in self.dependencies],
            "unresolved": [item.to_dict() for item in self.unresolved],
        }


def _clean_import(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text.replace("\\", "/")


def _module_index(parsed: Mapping[str, ParsedFile]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in sorted(parsed):
        name = module_name(path)
        if name:
            grouped[name].append(path)
        # Rust/Java-style separators are accepted only as alternate spellings
        # of an already-known workspace module; they never create new facts.
        if name:
            grouped[name.replace(".", "::")].append(path)
            grouped[name.replace(".", "/")].append(path)
    return {
        key: tuple(sorted(set(values)))
        for key, values in sorted(grouped.items())
    }


def _stem_index(parsed: Mapping[str, ParsedFile]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in sorted(parsed):
        pure = PurePosixPath(path)
        no_suffix = str(pure.with_suffix(""))
        grouped[no_suffix].append(path)
        grouped[pure.stem].append(path)
    return {
        key: tuple(sorted(set(values)))
        for key, values in sorted(grouped.items())
    }


def _suffix_index(parsed: Mapping[str, ParsedFile]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in sorted(parsed):
        parts = PurePosixPath(path).parts
        for width in range(1, min(len(parts), 4) + 1):
            grouped["/".join(parts[-width:])].append(path)
    return {
        key: tuple(sorted(set(values)))
        for key, values in sorted(grouped.items())
    }


def _unique(values: tuple[str, ...], importer: str) -> str | None:
    matches = tuple(value for value in values if value != importer)
    return matches[0] if len(matches) == 1 else None


def _relative_candidates(importer: str, imported: str) -> tuple[str, ...]:
    """Return deterministic path candidates without inventing search paths."""
    if not imported or imported.startswith(("http://", "https://")):
        return ()
    if imported.startswith("<") and imported.endswith(">"):
        # C/C++ system include syntax deliberately remains external unless a
        # build-system adapter later proves an include search path.
        return ()
    base_dir = posixpath.dirname(importer)
    candidates: list[str] = []
    # Explicit relative imports are exact language-independent path intent.
    if imported.startswith(("./", "../")):
        normalized = normalize_relative(posixpath.normpath(posixpath.join(base_dir, imported)))
        if normalized:
            candidates.append(normalized)
        return tuple(dict.fromkeys(candidates))

    # Path-like imports carrying a file suffix (Zig @import("foo.zig"),
    # C #include "foo.h", generated-language includes) are commonly relative
    # to the importer.  We try only that exact path here. Build adapters may add
    # compiler search paths later.
    suffix = PurePosixPath(imported).suffix
    if suffix:
        normalized = normalize_relative(posixpath.normpath(posixpath.join(base_dir, imported)))
        if normalized:
            candidates.append(normalized)
    return tuple(dict.fromkeys(candidates))


def resolve_workspace_dependencies(
    parsed: Mapping[str, ParsedFile],
) -> WorkspaceDependencyGraph:
    """Resolve parser-observed imports to unique workspace artifacts.

    Resolution order is intentionally strict:
    1. exact explicit-relative/path-like target;
    2. unique existing module spelling;
    3. unique same-directory extensionless stem;
    4. unique short workspace suffix for an explicitly path-like import.

    Package-manager, compiler include-path, build-target, macro, generated-file,
    and runtime module semantics belong to stronger adapters and are never
    guessed here.
    """
    modules = _module_index(parsed)
    stems = _stem_index(parsed)
    suffixes = _suffix_index(parsed)
    edges: set[WorkspaceDependency] = set()
    unresolved: set[UnresolvedWorkspaceDependency] = set()

    for importer, item in sorted(parsed.items()):
        for raw_import in sorted(item.imports):
            imported = _clean_import(str(raw_import))
            if not imported:
                continue

            relative = _relative_candidates(importer, imported)
            exact_matches = tuple(
                candidate
                for candidate in relative
                if candidate in parsed and candidate != importer
            )
            if len(exact_matches) == 1:
                edges.add(WorkspaceDependency(
                    importer,
                    imported,
                    exact_matches[0],
                    DependencyTrust.EXACT_RELATIVE,
                    "relative-path",
                ))
                continue
            if len(exact_matches) > 1:
                unresolved.add(UnresolvedWorkspaceDependency(
                    importer, imported, "ambiguous-relative", tuple(sorted(exact_matches))
                ))
                continue

            module_spellings = (
                imported,
                imported.replace("::", "."),
                imported.replace("/", "."),
            )
            module_candidates = {
                candidate
                for spelling in module_spellings
                for candidate in modules.get(spelling, ())
                if candidate != importer
            }
            if len(module_candidates) == 1:
                edges.add(WorkspaceDependency(
                    importer,
                    imported,
                    next(iter(module_candidates)),
                    DependencyTrust.UNIQUE_MODULE,
                    "module-name",
                ))
                continue
            if len(module_candidates) > 1:
                unresolved.add(UnresolvedWorkspaceDependency(
                    importer,
                    imported,
                    "ambiguous-module",
                    tuple(sorted(module_candidates)),
                ))
                continue

            # Extensionless relative imports such as ./util are resolved by an
            # exact same-directory stem only; unlike older code this is not
            # hard-coded to JS/TS suffixes and therefore works for future
            # languages without adding another suffix table.
            if imported.startswith(("./", "../")) and not PurePosixPath(imported).suffix:
                base = normalize_relative(
                    posixpath.normpath(posixpath.join(posixpath.dirname(importer), imported))
                )
                candidates = tuple(
                    value
                    for value in stems.get(base, ())
                    if value != importer
                ) if base else ()
                target = _unique(candidates, importer)
                if target:
                    edges.add(WorkspaceDependency(
                        importer,
                        imported,
                        target,
                        DependencyTrust.UNIQUE_STEM,
                        "relative-stem",
                    ))
                    continue
                if len(candidates) > 1:
                    unresolved.add(UnresolvedWorkspaceDependency(
                        importer,
                        imported,
                        "ambiguous-relative-stem",
                        tuple(sorted(candidates)),
                    ))
                    continue

            # A path-like import that did not resolve relative to the importer
            # may name a unique workspace suffix (e.g. generated include paths).
            # We accept this only when the textual path contains '/' or a suffix
            # and exactly one workspace artifact matches.
            path_like = "/" in imported or bool(PurePosixPath(imported).suffix)
            if path_like:
                suffix_candidates = tuple(
                    value for value in suffixes.get(imported.lstrip("./"), ())
                    if value != importer
                )
                target = _unique(suffix_candidates, importer)
                if target:
                    edges.add(WorkspaceDependency(
                        importer,
                        imported,
                        target,
                        DependencyTrust.UNIQUE_SUFFIX,
                        "unique-workspace-suffix",
                    ))
                    continue
                if len(suffix_candidates) > 1:
                    unresolved.add(UnresolvedWorkspaceDependency(
                        importer,
                        imported,
                        "ambiguous-workspace-suffix",
                        tuple(sorted(suffix_candidates)),
                    ))
                    continue

            # External/std/package imports are intentionally quiet unless the
            # workspace supplied competing candidates. This keeps the graph
            # useful without pretending package-manager state is known.

    ordered_edges = tuple(sorted(
        edges,
        key=lambda item: (item.importer, item.target, item.imported_text, item.trust.value),
    ))
    ordered_unresolved = tuple(sorted(
        unresolved,
        key=lambda item: (item.importer, item.imported_text, item.reason, item.candidates),
    ))
    return WorkspaceDependencyGraph(ordered_edges, ordered_unresolved)


__all__ = [
    "WORKSPACE_DEPENDENCY_SCHEMA_VERSION",
    "DependencyTrust",
    "UnresolvedWorkspaceDependency",
    "WorkspaceDependency",
    "WorkspaceDependencyGraph",
    "resolve_workspace_dependencies",
]
