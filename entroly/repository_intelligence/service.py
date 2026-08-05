"""Thread-safe shared service for repository-intelligence surfaces."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from .graph import analyze_change_impact, localize_tests
from .models import RepositoryIndex, RepositoryLimits, normalize_relative

SERVICE_SCHEMA_VERSION = "entroly.repository-service.v1"
_MAX_CHANGED_PATHS = 200
_MAX_DIAGNOSTICS = 100


class RepositoryIntelligenceError(ValueError):
    """A bounded, user-correctable repository-intelligence error."""

    code = "repository_intelligence_error"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "error": self.code,
            "detail": str(self),
        }


class UnknownChangedPaths(RepositoryIntelligenceError):
    code = "unknown_changed_paths"

    def __init__(self, paths: Iterable[str]) -> None:
        self.paths = tuple(sorted(dict.fromkeys(paths)))
        super().__init__("unknown changed paths: " + ", ".join(self.paths))

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload["unknown"] = list(self.paths)
        return payload


class InvalidChangedPaths(RepositoryIntelligenceError):
    code = "invalid_changed_paths"

    def __init__(self, count: int) -> None:
        self.count = count
        super().__init__(
            "changed paths must be non-empty workspace-relative paths without "
            "drive prefixes, NUL bytes, or parent traversal"
        )

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload["invalid_count"] = self.count
        return payload


class TooManyChangedPaths(RepositoryIntelligenceError):
    code = "too_many_changed_paths"


Builder = Callable[..., RepositoryIndex]


def _strict_relative(path: str | os.PathLike[str]) -> str | None:
    raw = str(path).replace("\\", "/")
    if not raw or "\x00" in raw:
        return None
    if raw.startswith("/") or raw.startswith("//"):
        return None
    if len(raw) >= 2 and raw[1] == ":":
        return None
    if any(part == ".." for part in raw.split("/")):
        return None
    normalized = normalize_relative(raw)
    return normalized or None


class RepositoryIntelligenceService:
    """Cache one immutable repository snapshot and expose bounded operations.

    Initial construction and explicit refreshes are single-flight. A refresh
    builds outside the state lock and atomically swaps a complete immutable
    snapshot, so readers never observe a partially rebuilt index.
    """

    def __init__(
        self,
        root: str | os.PathLike[str] = ".",
        *,
        limits: RepositoryLimits | None = None,
        builder: Builder | None = None,
    ) -> None:
        resolved = Path(root).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        self.root = resolved
        self.limits = limits or RepositoryLimits()
        self._builder = builder
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._index: RepositoryIndex | None = None
        self._digest = ""
        self._generation = 0

    def _build(self) -> RepositoryIndex:
        builder = self._builder
        if builder is None:
            from . import build_repository_index

            builder = build_repository_index
        return builder(self.root, limits=self.limits)

    @staticmethod
    def _digest_index(index: RepositoryIndex) -> str:
        payload = index.to_dict()
        # The checkout location is operational metadata, not repository
        # content. Canonicalize it so identical trees have identical digests
        # across machines and workspaces.
        payload["root"] = "."
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(canonical).hexdigest()

    def _install(self, index: RepositoryIndex) -> tuple[str, int]:
        digest = self._digest_index(index)
        with self._lock:
            self._index = index
            self._digest = digest
            self._generation += 1
            return digest, self._generation

    def refresh(self) -> dict[str, object]:
        with self._build_lock:
            index = self._build()
            digest, generation = self._install(index)
        return self._summary(index, digest=digest, generation=generation)

    def _snapshot(self) -> tuple[RepositoryIndex, str, int]:
        with self._lock:
            if self._index is not None:
                return self._index, self._digest, self._generation

        # Only one caller may perform the initial build. Recheck after taking
        # the build lock because another caller may have completed it first.
        with self._build_lock:
            with self._lock:
                if self._index is not None:
                    return self._index, self._digest, self._generation
            index = self._build()
            digest, generation = self._install(index)
            return index, digest, generation

    @staticmethod
    def _summary(
        index: RepositoryIndex,
        *,
        digest: str,
        generation: int,
    ) -> dict[str, object]:
        languages = Counter(record.language for record in index.files.values())
        return {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "root": index.root,
            "index_digest": digest,
            "generation": generation,
            "files": len(index.files),
            "symbols": len(index.symbols),
            "call_edges": len(index.call_edges),
            "file_edges": sum(
                len(values) for values in index.file_dependencies.values()
            ),
            "tests": len(index.test_paths),
            "languages": dict(sorted(languages.items())),
            "diagnostics": list(index.diagnostics[:_MAX_DIAGNOSTICS]),
        }

    def summary(self) -> dict[str, object]:
        index, digest, generation = self._snapshot()
        return self._summary(index, digest=digest, generation=generation)

    def _changed(
        self,
        index: RepositoryIndex,
        changed_paths: Iterable[str],
    ) -> tuple[str, ...]:
        normalized_paths: set[str] = set()
        invalid_count = 0
        observed_count = 0
        for path in changed_paths:
            observed_count += 1
            if observed_count > _MAX_CHANGED_PATHS:
                raise TooManyChangedPaths(
                    f"at most {_MAX_CHANGED_PATHS} changed paths are accepted per request"
                )
            normalized = _strict_relative(path)
            if normalized is None:
                invalid_count += 1
            else:
                normalized_paths.add(normalized)
        if invalid_count:
            raise InvalidChangedPaths(invalid_count)
        requested = tuple(sorted(normalized_paths))
        unknown = tuple(path for path in requested if path not in index.files)
        if unknown:
            raise UnknownChangedPaths(unknown)
        return requested

    def impact(
        self,
        changed_paths: Iterable[str],
        *,
        max_depth: int = 4,
        limit: int = 5_000,
    ) -> dict[str, object]:
        index, digest, generation = self._snapshot()
        changed = self._changed(index, changed_paths)
        report = analyze_change_impact(
            index,
            changed,
            max_depth=max_depth,
            max_impacted_paths=limit,
        )
        return {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "index_digest": digest,
            "generation": generation,
            "report": report.to_dict(),
        }

    def tests(
        self,
        changed_paths: Iterable[str],
        *,
        limit: int = 20,
    ) -> dict[str, object]:
        index, digest, generation = self._snapshot()
        changed = self._changed(index, changed_paths)
        candidates = localize_tests(index, changed, limit=limit)
        return {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "index_digest": digest,
            "generation": generation,
            "changed_paths": list(changed),
            "candidates": [candidate.to_dict() for candidate in candidates],
        }
