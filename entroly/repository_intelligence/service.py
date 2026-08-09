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
from .verified_context import build_symbol_graph, build_verified_context

SERVICE_SCHEMA_VERSION = "entroly.repository-service.v2"
_MAX_CHANGED_PATHS = 200
_MAX_DIAGNOSTICS = 100
_MAX_IMPACT_DEPTH = 12
_MAX_IMPACT_PATHS = 5_000
_MAX_TEST_CANDIDATES = 100
_MAX_CONTEXT_TOKENS = 32_768
_MAX_CONTEXT_FRAGMENTS = 100


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


class InvalidContextQuery(RepositoryIntelligenceError):
    code = "invalid_context_query"


class InvalidSymbolQuery(RepositoryIntelligenceError):
    code = "invalid_symbol_query"


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
            "unresolved_calls": len(index.unresolved_calls),
            "file_edges": sum(
                len(values) for values in index.file_dependencies.values()
            ),
            "tests": len(index.test_paths),
            "languages": dict(sorted(languages.items())),
            "parse_backends": dict(sorted(Counter(
                symbol.parse_backend for symbol in index.symbols.values()
            ).items())),
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
        if isinstance(changed_paths, (str, bytes, os.PathLike)):
            raise InvalidChangedPaths(1)
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
        if observed_count == 0:
            raise InvalidChangedPaths(0)
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
            max_depth=max(0, min(int(max_depth), _MAX_IMPACT_DEPTH)),
            max_impacted_paths=max(1, min(int(limit), _MAX_IMPACT_PATHS)),
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
        candidates = localize_tests(
            index,
            changed,
            limit=max(1, min(int(limit), _MAX_TEST_CANDIDATES)),
        )
        return {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "index_digest": digest,
            "generation": generation,
            "changed_paths": list(changed),
            "candidates": [candidate.to_dict() for candidate in candidates],
        }

    def context(
        self,
        query: str,
        *,
        token_budget: int = 2_000,
        max_hops: int = 2,
        max_fragments: int = 24,
        include_history: bool = False,
        max_history_commits: int = 20,
    ) -> dict[str, object]:
        """Return a hash-verified, budgeted partial graph for one task."""
        if not isinstance(query, str) or not query.strip():
            raise InvalidContextQuery("query must not be empty")
        if len(query.strip()) > 4_000:
            raise InvalidContextQuery("query must be at most 4000 characters")
        index, digest, generation = self._snapshot()
        payload = build_verified_context(
            self.root,
            index,
            query,
            index_digest=digest,
            token_budget=max(128, min(int(token_budget), _MAX_CONTEXT_TOKENS)),
            max_hops=max(0, min(int(max_hops), 6)),
            max_fragments=max(1, min(int(max_fragments), _MAX_CONTEXT_FRAGMENTS)),
            include_history=bool(include_history),
            max_history_commits=max(1, min(int(max_history_commits), 100)),
        )
        payload["generation"] = generation
        return payload

    def symbol_graph(
        self,
        symbol_query: str,
        *,
        direction: str = "both",
        max_depth: int = 3,
        limit: int = 200,
    ) -> dict[str, object]:
        """Return a bounded call graph after unambiguous symbol lookup."""
        if not isinstance(symbol_query, str) or not symbol_query.strip():
            raise InvalidSymbolQuery("symbol query must not be empty")
        if len(symbol_query.strip()) > 1_000:
            raise InvalidSymbolQuery("symbol query must be at most 1000 characters")
        if not isinstance(direction, str) or direction.strip().lower() not in {
            "callers", "callees", "both",
        }:
            raise InvalidSymbolQuery("direction must be callers, callees, or both")
        index, digest, generation = self._snapshot()
        payload = build_symbol_graph(
            self.root,
            index,
            symbol_query,
            index_digest=digest,
            direction=direction,
            max_depth=max_depth,
            limit=limit,
        )
        payload["generation"] = generation
        return payload
