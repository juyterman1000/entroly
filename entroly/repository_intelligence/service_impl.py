"""Thread-safe shared service for repository-intelligence surfaces."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .graph import analyze_change_impact, localize_tests
from .graph_query import build_verified_graph_query, prepare_graph_query
from .architecture_diff import build_verified_architecture_diff
from .lsp_orchestrator import (
    build_committed_lsp_rename_preview,
    collect_lsp_references,
)
from .models import RepositoryIndex, RepositoryLimits, normalize_relative
from .program_graph import build_verified_program_graph
from .interprocedural_flow import build_verified_interprocedural_flow
from .repository_map import (
    REPOSITORY_MAP_SCHEMA_VERSION,
    build_verified_repository_map,
    verify_repository_map_commitment,
)
from .runtime_overlay import build_verified_runtime_overlay
from .semantic_overlay import build_verified_semantic_overlay
from .verified_refactor import (
    apply_verified_rename_plan,
    build_verified_rename_plan,
    build_verified_safe_delete_plan,
)
from .verified_context import (
    apply_context_fault,
    build_symbol_graph,
    build_verified_context,
    validate_context_sources,
)
from .verified_slice import build_verified_program_slice
from .verified_health import (
    VERIFIED_HEALTH_SCHEMA_VERSION,
    build_verified_code_health,
    verify_code_health_commitment,
)
from .verified_routes import (
    VERIFIED_ROUTES_SCHEMA_VERSION,
    build_verified_routes,
    verify_routes_commitment,
)
from .verified_snapshot import (
    build_verified_graph_snapshot,
    check_verified_graph_snapshot,
)
from .verified_git_diff import build_verified_git_architecture_diff
from .verified_move import (
    apply_verified_file_move_plan,
    build_verified_file_move_plan,
)
from .verified_architecture import (
    VERIFIED_ARCHITECTURE_SCHEMA_VERSION,
    build_verified_architecture,
    verify_architecture_commitment,
)

SERVICE_SCHEMA_VERSION = "entroly.repository-service.v2"
_MAX_CHANGED_PATHS = 200
_MAX_DIAGNOSTICS = 100
_MAX_IMPACT_DEPTH = 12
_MAX_IMPACT_PATHS = 5_000
_MAX_TEST_CANDIDATES = 100
_MAX_CONTEXT_TOKENS = 32_768
_MAX_CONTEXT_FRAGMENTS = 100
_MAX_MAP_ENTRIES = 1_000
_MAX_HEALTH_FINDINGS = 10_000
_MAX_HEALTH_SYMBOLS = 20_000
_MAX_ARCHITECTURE_COMPONENTS = 20_000
_MAX_ARCHITECTURE_COMMUNITIES = 10_000
_MAX_ARCHITECTURE_CYCLES = 10_000
_MAX_ARCHITECTURE_EDGES = 1_000_000
_MAX_ARCHITECTURE_HOTSPOTS = 1_000
_MAX_ARCHITECTURE_ROUTES = 1_000
_MAX_HTTP_ROUTES = 100_000
_MAX_HTTP_ROUTE_CONFLICTS = 10_000


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


class InvalidContextFault(RepositoryIntelligenceError):
    code = "invalid_context_fault"


class InvalidSymbolQuery(RepositoryIntelligenceError):
    code = "invalid_symbol_query"


class VerifiedRefactorError(RepositoryIntelligenceError):
    code = "verified_refactor_failed"


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
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        resolved = Path(root).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        self.root = resolved
        self.limits = limits or RepositoryLimits()
        if builder is not None and cache_dir is not None:
            raise ValueError("builder and cache_dir are mutually exclusive")
        if cache_dir is not None:
            from .incremental import build_repository_index_incremental
            from .persistent_analysis import PersistentAnalysisCache

            selected_cache = Path(cache_dir).expanduser().resolve()
            self._analysis_cache = PersistentAnalysisCache(selected_cache)

            def incremental_builder(root: Path, *, limits: RepositoryLimits):
                return build_repository_index_incremental(
                    root,
                    cache_dir=selected_cache,
                    limits=limits,
                )

            self._builder = incremental_builder
        else:
            self._builder = builder
            self._analysis_cache = None
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._index: RepositoryIndex | None = None
        self._digest = ""
        self._generation = 0
        self._query_graph = None
        self._query_graph_digest = ""

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
        # Cache and snapshot telemetry describes how this run was served, not
        # what the repository contains, so it must not reach the digest.
        #
        # These prefixes previously carried a trailing space, which matched
        # "incremental-parse-cache hits=..." but not
        # "incremental-parse-cache-retention files=...", because the next
        # character there is a hyphen. The retention counters change between a
        # cold and a warm cache (files=2 bytes=2195 against files=3 bytes=4804
        # on the same unchanged tree), so an identical checkout produced two
        # different digests depending only on whether the cache was warm --
        # which defeats the reuse the persistence layer exists to provide.
        # Matching the family prefix covers every current and future variant.
        diagnostics = payload.get("diagnostics", [])
        if isinstance(diagnostics, list):
            payload["diagnostics"] = [
                item
                for item in diagnostics
                if not str(item).startswith((
                    "incremental-parse-cache",
                    "persistent-index-snapshot",
                ))
            ]
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
            self._query_graph = None
            self._query_graph_digest = ""
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
        proposal_scores: Iterable[Mapping[str, object]] = (),
        proposal_provider: str = "caller-supplied",
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
            proposal_scores=proposal_scores,
            proposal_provider=proposal_provider,
        )
        payload["generation"] = generation
        return payload

    def work_scope_proposals(
        self,
        scope: Mapping[str, object],
    ) -> list[dict[str, object]]:
        """Map a bounded Rust WorkScope back to indexed symbol identities."""
        if not isinstance(scope, Mapping):
            raise InvalidContextQuery("work scope must be an object")
        repo_id = scope.get("repo_id")
        changed_paths = scope.get("changed_paths", [])
        graph_symbol_ids = scope.get("symbol_ids", [])
        if (
            not isinstance(repo_id, str)
            or not repo_id
            or not isinstance(changed_paths, list)
            or len(changed_paths) > 256
            or not all(isinstance(item, str) for item in changed_paths)
            or not isinstance(graph_symbol_ids, list)
            or len(graph_symbol_ids) > 256
            or not all(isinstance(item, str) for item in graph_symbol_ids)
        ):
            raise InvalidContextQuery("work scope identity fields are invalid")
        normalized_paths = {
            normalized
            for item in changed_paths
            if (normalized := _strict_relative(item)) is not None
        }
        selected_graph_ids = set(graph_symbol_ids)
        index, _digest, _generation = self._snapshot()
        scores: dict[str, float] = {}
        for symbol in index.symbols.values():
            if symbol.path in normalized_paths:
                scores[symbol.symbol_id] = 0.85
        if selected_graph_ids:
            from .graph_identity import symbol_node_id

            for symbol in index.symbols.values():
                if symbol_node_id(repo_id, symbol.symbol_id) in selected_graph_ids:
                    scores[symbol.symbol_id] = 1.0
        return [
            {"symbol_id": symbol_id, "score": score}
            for symbol_id, score in sorted(
                scores.items(), key=lambda item: (-item[1], item[0])
            )
        ]

    def context_fault(
        self,
        context: Mapping[str, object],
        context_ref: str,
        *,
        token_budget: int | None = None,
    ) -> dict[str, object]:
        """Recover one committed omission into a new bounded working set."""
        if not isinstance(context, Mapping):
            raise InvalidContextFault("context must be an object")
        if not isinstance(context_ref, str) or not context_ref.strip():
            raise InvalidContextFault("context_ref must not be empty")
        index, digest, generation = self._snapshot()
        try:
            payload = apply_context_fault(
                self.root,
                index,
                context,
                context_ref.strip(),
                index_digest=digest,
                token_budget=token_budget,
            )
        except ValueError as exc:
            raise InvalidContextFault(str(exc)) from None
        payload["generation"] = generation
        return payload

    def validate_context(self, context: Mapping[str, object]) -> dict[str, object]:
        """Verify a receipt and every carried source reference without mutation."""
        if not isinstance(context, Mapping):
            raise InvalidContextFault("context must be an object")
        index, digest, generation = self._snapshot()
        try:
            validate_context_sources(
                self.root,
                index,
                context,
                index_digest=digest,
            )
        except ValueError as exc:
            raise InvalidContextFault(str(exc)) from None
        return {
            "status": "verified-current",
            "index_digest": digest,
            "generation": generation,
            "context_sha256": context["receipt"]["context_sha256"],  # type: ignore[index]
        }

    def program_slice(
        self,
        query: str,
        *,
        token_budget: int = 4_000,
        max_hops: int = 3,
        max_fragments: int = 32,
        max_entry_points: int = 3,
        flow_direction: str = "outgoing",
        flow_depth: int = 3,
        proposal_scores: Iterable[Mapping[str, object]] = (),
        proposal_provider: str = "caller-supplied",
    ) -> dict[str, object]:
        """Return a proof-carrying partial slice with control and value flow."""
        if not isinstance(query, str) or not query.strip():
            raise InvalidContextQuery("query must not be empty")
        if len(query.strip()) > 4_000:
            raise InvalidContextQuery("query must be at most 4000 characters")
        index, digest, generation = self._snapshot()
        payload = build_verified_program_slice(
            self.root,
            index,
            query,
            index_digest=digest,
            token_budget=max(128, min(int(token_budget), _MAX_CONTEXT_TOKENS)),
            max_hops=max(0, min(int(max_hops), 6)),
            max_fragments=max(1, min(int(max_fragments), _MAX_CONTEXT_FRAGMENTS)),
            max_entry_points=max(1, min(int(max_entry_points), 8)),
            flow_direction=flow_direction,
            flow_depth=max(0, min(int(flow_depth), 12)),
            proposal_scores=proposal_scores,
            proposal_provider=proposal_provider,
        )
        payload["generation"] = generation
        return payload

    def repository_map(
        self,
        query: str = "",
        *,
        token_budget: int = 2_000,
        max_entries: int = 100,
    ) -> dict[str, object]:
        """Return a verified whole-repository structural priority map."""
        if not isinstance(query, str):
            raise InvalidContextQuery("query must be a string")
        if len(query.strip()) > 4_000:
            raise InvalidContextQuery("query must be at most 4000 characters")
        index, digest, generation = self._snapshot()
        identity: dict[str, object] = {
            "analysis_schema": REPOSITORY_MAP_SCHEMA_VERSION,
            "index_digest": digest,
            "query": query.strip(),
            "token_budget": max(128, min(int(token_budget), _MAX_CONTEXT_TOKENS)),
            "max_entries": max(1, min(int(max_entries), _MAX_MAP_ENTRIES)),
        }
        cache_status = "disabled"
        if self._analysis_cache is not None:
            cached, cache_status = self._analysis_cache.load(
                "repository-map",
                identity,
                verify=verify_repository_map_commitment,
            )
            if cached is not None:
                cached["generation"] = generation
                return cached
        payload = build_verified_repository_map(
            self.root,
            index,
            query,
            index_digest=digest,
            token_budget=max(128, min(int(token_budget), _MAX_CONTEXT_TOKENS)),
            max_entries=max(1, min(int(max_entries), _MAX_MAP_ENTRIES)),
        )
        if self._analysis_cache is not None:
            self._analysis_cache.store(
                "repository-map", identity, payload, replace=cache_status == "corrupt"
            )
        payload["generation"] = generation
        return payload

    def code_health(
        self,
        *,
        max_findings: int = 500,
        max_symbols: int = 2_000,
    ) -> dict[str, object]:
        """Return freshness-checked structural health and navigability evidence."""
        index, digest, generation = self._snapshot()
        bounded_findings = max(1, min(int(max_findings), _MAX_HEALTH_FINDINGS))
        bounded_symbols = max(1, min(int(max_symbols), _MAX_HEALTH_SYMBOLS))
        identity: dict[str, object] = {
            "analysis_schema": VERIFIED_HEALTH_SCHEMA_VERSION,
            "index_digest": digest,
            "max_findings": bounded_findings,
            "max_symbols": bounded_symbols,
        }
        cache_status = "disabled"
        if self._analysis_cache is not None:
            cached, cache_status = self._analysis_cache.load(
                "code-health", identity, verify=verify_code_health_commitment
            )
            if cached is not None:
                cached["generation"] = generation
                return cached
        payload = build_verified_code_health(
            self.root,
            index,
            index_digest=digest,
            max_findings=bounded_findings,
            max_symbols=bounded_symbols,
        )
        if self._analysis_cache is not None:
            self._analysis_cache.store(
                "code-health", identity, payload, replace=cache_status == "corrupt"
            )
        payload["generation"] = generation
        return payload

    def architecture(
        self,
        *,
        max_components: int = 5_000,
        max_communities: int = 1_000,
        max_cycles: int = 1_000,
        max_dependency_edges: int = 100_000,
        max_hotspots: int = 100,
        max_routes: int = 100,
    ) -> dict[str, object]:
        """Return verified layers, communities, cycles, routes, and hotspots."""
        index, digest, generation = self._snapshot()
        bounds = {
            "max_components": max(
                1, min(int(max_components), _MAX_ARCHITECTURE_COMPONENTS)
            ),
            "max_communities": max(
                1, min(int(max_communities), _MAX_ARCHITECTURE_COMMUNITIES)
            ),
            "max_cycles": max(1, min(int(max_cycles), _MAX_ARCHITECTURE_CYCLES)),
            "max_dependency_edges": max(
                1, min(int(max_dependency_edges), _MAX_ARCHITECTURE_EDGES)
            ),
            "max_hotspots": max(
                1, min(int(max_hotspots), _MAX_ARCHITECTURE_HOTSPOTS)
            ),
            "max_routes": max(1, min(int(max_routes), _MAX_ARCHITECTURE_ROUTES)),
        }
        identity: dict[str, object] = {
            "analysis_schema": VERIFIED_ARCHITECTURE_SCHEMA_VERSION,
            "index_digest": digest,
            **bounds,
        }
        cache_status = "disabled"
        if self._analysis_cache is not None:
            cached, cache_status = self._analysis_cache.load(
                "architecture", identity, verify=verify_architecture_commitment
            )
            if cached is not None:
                cached["generation"] = generation
                return cached
        payload = build_verified_architecture(
            self.root,
            index,
            index_digest=digest,
            **bounds,
        )
        if self._analysis_cache is not None:
            self._analysis_cache.store(
                "architecture", identity, payload, replace=cache_status == "corrupt"
            )
        payload["generation"] = generation
        return payload

    def architecture_diff(
        self,
        before: Mapping[str, object],
        after: Mapping[str, object],
        *,
        limit: int = 5_000,
    ) -> dict[str, object]:
        """Compare two independently committed architecture snapshots."""
        return build_verified_architecture_diff(
            before,
            after,
            limit=max(1, min(int(limit), 50_000)),
        )

    def git_architecture_diff(
        self,
        ref: str = "HEAD",
        *,
        max_changes: int = 10_000,
        max_components: int = 5_000,
        max_communities: int = 1_000,
        max_cycles: int = 1_000,
        max_dependency_edges: int = 100_000,
        max_hotspots: int = 100,
        max_routes: int = 100,
    ) -> dict[str, object]:
        """Compare a local Git commit graph with the verified current worktree."""
        index, digest, generation = self._snapshot()
        builder = self._builder
        if builder is None:
            from . import build_repository_index

            builder = build_repository_index
        payload = build_verified_git_architecture_diff(
            self.root,
            index,
            current_index_digest=digest,
            ref=ref,
            limits=self.limits,
            build_index=builder,
            max_changes=max(1, min(int(max_changes), 50_000)),
            max_components=max(
                1, min(int(max_components), _MAX_ARCHITECTURE_COMPONENTS)
            ),
            max_communities=max(
                1, min(int(max_communities), _MAX_ARCHITECTURE_COMMUNITIES)
            ),
            max_cycles=max(1, min(int(max_cycles), _MAX_ARCHITECTURE_CYCLES)),
            max_dependency_edges=max(
                1, min(int(max_dependency_edges), _MAX_ARCHITECTURE_EDGES)
            ),
            max_hotspots=max(
                1, min(int(max_hotspots), _MAX_ARCHITECTURE_HOTSPOTS)
            ),
            max_routes=max(1, min(int(max_routes), _MAX_ARCHITECTURE_ROUTES)),
        )
        payload["generation"] = generation
        return payload

    def routes(
        self,
        *,
        method: str | None = None,
        path_prefix: str | None = None,
        max_routes: int = 10_000,
        max_conflicts: int = 1_000,
    ) -> dict[str, object]:
        """Return source-verified HTTP routes, mounts, handlers, and collisions."""
        index, digest, generation = self._snapshot()
        selected_method = method.strip().upper() if method else None
        if selected_method is not None and (
            not selected_method or len(selected_method) > 32
        ):
            raise ValueError("method must be at most 32 characters")
        selected_prefix = path_prefix.strip() if path_prefix else None
        if selected_prefix is not None and (
            not selected_prefix.startswith("/") or len(selected_prefix) > 4_096
        ):
            raise ValueError("path prefix must start with / and be at most 4096 characters")
        bounds = {
            "max_routes": max(1, min(int(max_routes), _MAX_HTTP_ROUTES)),
            "max_conflicts": max(
                1, min(int(max_conflicts), _MAX_HTTP_ROUTE_CONFLICTS)
            ),
        }
        identity: dict[str, object] = {
            "analysis_schema": VERIFIED_ROUTES_SCHEMA_VERSION,
            "index_digest": digest,
            "method": selected_method,
            "path_prefix": selected_prefix,
            **bounds,
        }
        cache_status = "disabled"
        if self._analysis_cache is not None:
            cached, cache_status = self._analysis_cache.load(
                "http-routes", identity, verify=verify_routes_commitment
            )
            if cached is not None:
                cached["generation"] = generation
                return cached
        payload = build_verified_routes(
            self.root,
            index,
            index_digest=digest,
            method=selected_method,
            path_prefix=selected_prefix,
            **bounds,
        )
        if self._analysis_cache is not None:
            self._analysis_cache.store(
                "http-routes", identity, payload, replace=cache_status == "corrupt"
            )
        payload["generation"] = generation
        return payload

    def graph_snapshot(self) -> dict[str, object]:
        """Return a deterministic, portable commitment to the complete index."""
        index, digest, generation = self._snapshot()
        payload = build_verified_graph_snapshot(index, index_digest=digest)
        payload["generation"] = generation
        return payload

    def graph_snapshot_check(
        self,
        snapshot: Mapping[str, object],
        *,
        limit: int = 10_000,
    ) -> dict[str, object]:
        """Check whether a portable graph snapshot can be safely imported."""
        index, digest, generation = self._snapshot()
        payload = check_verified_graph_snapshot(
            self.root,
            index,
            snapshot,
            index_digest=digest,
            limit=max(1, min(int(limit), 100_000)),
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

    def graph_query(
        self,
        query: str,
        *,
        operation: str = "neighbors",
        target_query: str | None = None,
        direction: str = "both",
        max_depth: int = 4,
        limit: int = 100,
        max_visited: int = 10_000,
    ) -> dict[str, object]:
        """Query typed file/symbol relationships with freshness on traversal."""
        if not isinstance(query, str) or not query.strip():
            raise InvalidSymbolQuery("graph query must not be empty")
        if len(query.strip()) > 1_000:
            raise InvalidSymbolQuery("graph query must be at most 1000 characters")
        if target_query is not None and (
            not isinstance(target_query, str) or len(target_query.strip()) > 1_000
        ):
            raise InvalidSymbolQuery("target query must be at most 1000 characters")
        index, digest, generation = self._snapshot()
        with self._lock:
            prepared_graph = (
                self._query_graph
                if self._query_graph_digest == digest
                else None
            )
        if prepared_graph is None:
            built_graph = prepare_graph_query(index)
            with self._lock:
                if self._digest == digest:
                    self._query_graph = built_graph
                    self._query_graph_digest = digest
                prepared_graph = built_graph
        payload = build_verified_graph_query(
            self.root,
            index,
            query,
            index_digest=digest,
            operation=operation,
            target_query=target_query,
            direction=direction,
            max_depth=max(0, min(int(max_depth), 20)),
            limit=max(1, min(int(limit), 5_000)),
            max_visited=max(1, min(int(max_visited), 100_000)),
            prepared_graph=prepared_graph,
        )
        payload["generation"] = generation
        return payload

    def program_graph(
        self,
        symbol_query: str,
        *,
        limit: int = 1_000,
    ) -> dict[str, object]:
        """Return verified intraprocedural control and data-flow evidence."""
        if not isinstance(symbol_query, str) or not symbol_query.strip():
            raise InvalidSymbolQuery("symbol query must not be empty")
        if len(symbol_query.strip()) > 1_000:
            raise InvalidSymbolQuery("symbol query must be at most 1000 characters")
        index, digest, generation = self._snapshot()
        payload = build_verified_program_graph(
            self.root,
            index,
            symbol_query,
            index_digest=digest,
            limit=max(16, min(int(limit), 10_000)),
        )
        payload["generation"] = generation
        return payload

    def interprocedural_flow(
        self,
        symbol_query: str,
        *,
        direction: str = "outgoing",
        max_depth: int = 3,
        max_call_edges: int = 1_000,
        max_flow_edges: int = 10_000,
        max_nodes: int = 10_000,
    ) -> dict[str, object]:
        """Return verified cross-function argument, parameter, and return flow."""
        query = symbol_query.strip()
        if not query or len(query) > 1_000:
            raise InvalidSymbolQuery(
                "symbol query must contain between 1 and 1000 characters"
            )
        index, digest, generation = self._snapshot()
        payload = build_verified_interprocedural_flow(
            self.root,
            index,
            query,
            index_digest=digest,
            direction=direction,
            max_depth=max(0, min(int(max_depth), 12)),
            max_call_edges=max(1, min(int(max_call_edges), 100_000)),
            max_flow_edges=max(1, min(int(max_flow_edges), 100_000)),
            max_nodes=max(1, min(int(max_nodes), 100_000)),
        )
        payload["generation"] = generation
        return payload

    def runtime_overlay(
        self,
        events: Iterable[Mapping[str, object]],
        *,
        producer: str = "external-trace",
        max_events: int = 100_000,
    ) -> dict[str, object]:
        """Bind external trace events to verified source without values."""
        index, digest, generation = self._snapshot()
        payload = build_verified_runtime_overlay(
            self.root,
            index,
            events,
            index_digest=digest,
            producer=producer,
            max_events=max(1, min(int(max_events), 1_000_000)),
        )
        payload["generation"] = generation
        return payload

    def semantic_overlay(
        self,
        relationships: Iterable[Mapping[str, object]],
        *,
        provider: str,
        max_relationships: int = 100_000,
    ) -> dict[str, object]:
        """Verify externally reported LSP/compiler semantic locations."""
        index, digest, generation = self._snapshot()
        payload = build_verified_semantic_overlay(
            self.root,
            index,
            relationships,
            index_digest=digest,
            provider=provider,
            max_relationships=max(1, min(int(max_relationships), 1_000_000)),
        )
        payload["generation"] = generation
        return payload

    def rename_preview(
        self,
        symbol_query: str,
        new_name: str,
        *,
        semantic_relationships: Iterable[Mapping[str, object]] = (),
        provider: str = "none",
        max_changes: int = 10_000,
    ) -> dict[str, object]:
        """Build a no-write, source-verified rename transaction plan."""
        if not isinstance(symbol_query, str) or not symbol_query.strip():
            raise InvalidSymbolQuery("symbol query must not be empty")
        index, digest, generation = self._snapshot()
        try:
            payload = build_verified_rename_plan(
                self.root,
                index,
                symbol_query,
                new_name,
                index_digest=digest,
                semantic_relationships=semantic_relationships,
                provider=provider,
                max_changes=max(1, min(int(max_changes), 100_000)),
            )
        except ValueError as exc:
            raise VerifiedRefactorError(str(exc)) from None
        payload["generation"] = generation
        return payload

    def rename_apply(
        self,
        plan: Mapping[str, object],
        *,
        expected_plan_sha256: str,
        acknowledge_incomplete: bool = False,
    ) -> dict[str, object]:
        """Apply an explicitly acknowledged plan and refresh the snapshot."""
        index, digest, _generation = self._snapshot()
        with self._build_lock:
            with self._lock:
                if self._index is not index or self._digest != digest:
                    raise VerifiedRefactorError(
                        "repository index changed after refactor preview"
                    )
            try:
                applied = apply_verified_rename_plan(
                    self.root,
                    index,
                    plan,
                    index_digest=digest,
                    expected_plan_sha256=expected_plan_sha256,
                    acknowledge_incomplete=bool(acknowledge_incomplete),
                )
            except ValueError as exc:
                raise VerifiedRefactorError(str(exc)) from None
            try:
                refreshed_index = self._build()
                refreshed_digest, refreshed_generation = self._install(refreshed_index)
                refresh: dict[str, object] = {
                    "status": "refreshed",
                    "index_digest": refreshed_digest,
                    "generation": refreshed_generation,
                }
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                with self._lock:
                    self._index = None
                    self._digest = ""
                refresh = {
                    "status": "failed-after-apply",
                    "error_type": type(exc).__name__,
                    "detail": "files were changed; the next operation will rebuild the index",
                }
        return {
            "schema_version": "entroly.repository-refactor-service.v1",
            "apply": applied,
            "refresh": refresh,
        }

    def safe_delete_preview(
        self,
        symbol_query: str,
        *,
        max_blockers: int = 10_000,
    ) -> dict[str, object]:
        """Build a no-write delete plan that blocks on every known reference."""
        if not isinstance(symbol_query, str) or not symbol_query.strip():
            raise InvalidSymbolQuery("symbol query must not be empty")
        index, digest, generation = self._snapshot()
        try:
            payload = build_verified_safe_delete_plan(
                self.root,
                index,
                symbol_query,
                index_digest=digest,
                max_blockers=max(1, min(int(max_blockers), 100_000)),
            )
        except ValueError as exc:
            raise VerifiedRefactorError(str(exc)) from None
        payload["generation"] = generation
        return payload

    def safe_delete_apply(
        self,
        plan: Mapping[str, object],
        *,
        expected_plan_sha256: str,
        acknowledge_incomplete: bool = False,
    ) -> dict[str, object]:
        """Apply a committed blocker-free safe-delete and refresh the index."""
        if plan.get("operation") != "safe-delete":
            raise VerifiedRefactorError("safe-delete apply requires a safe-delete plan")
        return self.rename_apply(
            plan,
            expected_plan_sha256=expected_plan_sha256,
            acknowledge_incomplete=acknowledge_incomplete,
        )

    def file_move_preview(
        self,
        source_path: str,
        target_path: str,
        *,
        max_changes: int = 10_000,
        max_blockers: int = 10_000,
    ) -> dict[str, object]:
        """Build a no-write Python module move with exact import rewrites."""
        index, digest, generation = self._snapshot()
        try:
            payload = build_verified_file_move_plan(
                self.root,
                index,
                source_path,
                target_path,
                index_digest=digest,
                max_changes=max(1, min(int(max_changes), 100_000)),
                max_blockers=max(1, min(int(max_blockers), 100_000)),
            )
        except ValueError as exc:
            raise VerifiedRefactorError(str(exc)) from None
        payload["generation"] = generation
        return payload

    def file_move_apply(
        self,
        plan: Mapping[str, object],
        *,
        expected_plan_sha256: str,
        acknowledge_incomplete: bool = False,
    ) -> dict[str, object]:
        """Apply a committed Python module move and refresh the graph."""
        index, digest, _generation = self._snapshot()
        with self._build_lock:
            with self._lock:
                if self._index is not index or self._digest != digest:
                    raise VerifiedRefactorError(
                        "repository index changed after refactor preview"
                    )
            try:
                applied = apply_verified_file_move_plan(
                    self.root,
                    index,
                    plan,
                    index_digest=digest,
                    expected_plan_sha256=expected_plan_sha256,
                    acknowledge_incomplete=bool(acknowledge_incomplete),
                )
            except ValueError as exc:
                raise VerifiedRefactorError(str(exc)) from None
            try:
                refreshed_index = self._build()
                refreshed_digest, refreshed_generation = self._install(refreshed_index)
                refresh: dict[str, object] = {
                    "status": "refreshed",
                    "index_digest": refreshed_digest,
                    "generation": refreshed_generation,
                }
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                with self._lock:
                    self._index = None
                    self._digest = ""
                refresh = {
                    "status": "failed-after-apply",
                    "error_type": type(exc).__name__,
                    "detail": "files were changed; the next operation will rebuild the index",
                }
        return {
            "schema_version": "entroly.repository-refactor-service.v1",
            "apply": applied,
            "refresh": refresh,
        }

    def lsp_rename_preview(
        self,
        symbol_query: str,
        new_name: str,
        *,
        command: Iterable[str],
        language_id: str,
        timeout_seconds: float = 15.0,
        max_relationships: int = 10_000,
    ) -> dict[str, object]:
        """Run one configured local LSP and return a committed no-write plan."""
        index, digest, generation = self._snapshot()
        try:
            orchestration = collect_lsp_references(
                self.root,
                index,
                symbol_query,
                command=tuple(command),
                language_id=language_id,
                timeout_seconds=timeout_seconds,
                max_relationships=max_relationships,
            )
            plan = build_verified_rename_plan(
                self.root,
                index,
                symbol_query,
                new_name,
                index_digest=digest,
                semantic_relationships=orchestration["relationships"],
                provider=str(orchestration["provider"]),
                max_changes=max_relationships,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise VerifiedRefactorError(str(exc)) from None
        payload = build_committed_lsp_rename_preview(orchestration, plan)
        payload["generation"] = generation
        return payload
