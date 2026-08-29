"""Bounded, deterministic repository intelligence.

Repository intelligence is a base Entroly capability. Source files are mapped
into a language-independent semantic representation; parser/compiler/LSP
frontends strengthen evidence when available, while unknown languages retain a
safe exact-source fallback. All paths remain workspace-relative and escaping
symlinks are rejected.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..parser_compatibility import warn_if_incompatible_language_pack
from .graph import analyze_change_impact, localize_tests, resolve_calls, resolve_imports
from .models import (
    CallEdge,
    FileRecord,
    ImpactReport,
    RepositoryIndex,
    RepositoryLimits,
    Symbol,
    TestCandidate,
    UnresolvedCall,
)
from .parsers import scan_repository
from .semantic_ir import (
    SEMANTIC_IR_SCHEMA_VERSION,
    EpistemicClass,
    SemanticCapabilities,
    SemanticEdge,
    SemanticLevel,
    SemanticNode,
    SourceEvidence,
    UniversalSemanticDocument,
    build_universal_semantic_document,
)
from .workspace_dependencies import (
    WorkspaceDependencyGraph,
    resolve_workspace_dependencies,
)

__all__ = [
    "CallEdge", "FileRecord", "ImpactReport", "RepositoryIndex",
    "RepositoryLimits", "Symbol", "TestCandidate", "UnresolvedCall",
    "SEMANTIC_IR_SCHEMA_VERSION", "EpistemicClass", "SemanticCapabilities",
    "SemanticEdge", "SemanticLevel", "SemanticNode", "SourceEvidence",
    "UniversalSemanticDocument", "build_universal_semantic_document",
    "WorkspaceDependencyGraph", "resolve_workspace_dependencies",
    "analyze_change_impact",
    "build_repository_index", "localize_tests", "InvalidChangedPaths",
    "InvalidContextFault", "InvalidContextQuery", "InvalidSymbolQuery",
    "RepositoryIntelligenceError", "RepositoryIntelligenceService",
    "UnknownChangedPaths", "build_symbol_graph", "build_verified_context",
    "build_repository_index_incremental", "build_verified_program_graph",
    "build_verified_interprocedural_flow",
    "build_verified_program_slice",
    "build_verified_runtime_overlay", "verify_context_commitment",
    "build_verified_semantic_overlay",
    "build_verified_repository_map",
    "verify_program_graph_commitment", "verify_symbol_graph_commitment",
    "verify_interprocedural_flow_commitment",
    "verify_program_slice_commitment",
    "verify_runtime_overlay_commitment",
    "verify_semantic_overlay_commitment",
    "verify_repository_map_commitment",
    "build_verified_code_health", "verify_code_health_commitment",
    "build_verified_rename_plan", "apply_verified_rename_plan",
    "build_verified_safe_delete_plan", "apply_verified_refactor_plan",
    "verify_refactor_plan_commitment", "verify_refactor_apply_commitment",
    "collect_lsp_references", "verify_lsp_rename_preview_commitment",
    "build_verified_architecture", "verify_architecture_commitment",
    "build_verified_graph_query", "verify_graph_query_commitment",
    "build_verified_architecture_diff", "verify_architecture_diff_commitment",
    "build_verified_git_architecture_diff",
    "verify_git_architecture_diff_commitment",
    "build_verified_routes", "verify_routes_commitment",
    "build_verified_graph_snapshot", "check_verified_graph_snapshot",
    "load_verified_graph_snapshot", "verify_graph_snapshot_commitment",
    "verify_graph_snapshot_check_commitment",
    "build_verified_file_move_plan", "apply_verified_file_move_plan",
]


def _merge_file_dependencies(
    legacy: dict[str, tuple[str, ...]],
    universal: WorkspaceDependencyGraph,
    paths: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Add proven universal edges without removing established resolver edges.

    The legacy resolver remains authoritative for behaviors already covered by
    production tests (including framework-specific index-module conventions).
    The universal resolver contributes only deterministic workspace bindings.
    This makes the rollout monotonic: language breadth can grow without losing
    any existing Python/JS/Rust dependency evidence.
    """
    universal_map = universal.file_dependencies()
    return {
        path: tuple(sorted(
            set(legacy.get(path, ())) | set(universal_map.get(path, ()))
        ))
        for path in paths
    }


def build_repository_index(
    root: str | os.PathLike[str],
    *,
    limits: RepositoryLimits | None = None,
) -> RepositoryIndex:
    """Build a deterministic, resource-bounded repository index."""
    policy = limits or RepositoryLimits()
    root_path = Path(root).expanduser().resolve(strict=True)
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)
    parser_status = warn_if_incompatible_language_pack()
    parsed, diagnostics = scan_repository(root_path, policy)
    if parser_status.installed and not parser_status.compatible:
        version = parser_status.version or "unknown"
        diagnostics.append(
            "parser registry degraded: tree-sitter-language-pack "
            f"{version} is below supported >={parser_status.minimum_version}; "
            "language coverage may be materially reduced"
        )
    symbols = {
        symbol.symbol_id: symbol
        for path in sorted(parsed)
        for symbol in sorted(parsed[path].symbols, key=lambda item: item.symbol_id)
    }
    legacy_dependencies = resolve_imports(parsed)
    workspace_dependencies = resolve_workspace_dependencies(parsed)
    dependencies = _merge_file_dependencies(
        legacy_dependencies,
        workspace_dependencies,
        tuple(sorted(parsed)),
    )
    calls, unresolved_calls = resolve_calls(parsed, symbols, policy)
    if len(calls) + len(unresolved_calls) >= policy.max_edges:
        diagnostics.append("relationship limit reached; remaining evidence omitted")
    return RepositoryIndex(
        root=str(root_path),
        files={path: parsed[path].record for path in sorted(parsed)},
        symbols=symbols,
        call_edges=calls,
        unresolved_calls=unresolved_calls,
        file_dependencies=dependencies,
        diagnostics=tuple(sorted(dict.fromkeys(diagnostics))),
    )


from .service import (  # noqa: E402
    InvalidChangedPaths,
    InvalidContextFault,
    InvalidContextQuery,
    InvalidSymbolQuery,
    RepositoryIntelligenceError,
    RepositoryIntelligenceService,
    UnknownChangedPaths,
)
from .program_graph import (  # noqa: E402
    build_verified_program_graph,
    verify_program_graph_commitment,
)
from .interprocedural_flow import (  # noqa: E402
    build_verified_interprocedural_flow,
    verify_interprocedural_flow_commitment,
)
from .verified_slice import (  # noqa: E402
    build_verified_program_slice,
    verify_program_slice_commitment,
)
from .incremental import build_repository_index_incremental  # noqa: E402
from .runtime_overlay import (  # noqa: E402
    build_verified_runtime_overlay,
    verify_runtime_overlay_commitment,
)
from .semantic_overlay import (  # noqa: E402
    build_verified_semantic_overlay,
    verify_semantic_overlay_commitment,
)
from .repository_map import (  # noqa: E402
    build_verified_repository_map,
    verify_repository_map_commitment,
)
from .verified_health import (  # noqa: E402
    build_verified_code_health,
    verify_code_health_commitment,
)
from .verified_refactor import (  # noqa: E402
    apply_verified_refactor_plan,
    apply_verified_rename_plan,
    build_verified_rename_plan,
    build_verified_safe_delete_plan,
    verify_refactor_apply_commitment,
    verify_refactor_plan_commitment,
)
from .lsp_orchestrator import (  # noqa: E402
    collect_lsp_references,
    verify_lsp_rename_preview_commitment,
)
from .verified_architecture import (  # noqa: E402
    build_verified_architecture,
    verify_architecture_commitment,
)
from .graph_query import (  # noqa: E402
    build_verified_graph_query,
    verify_graph_query_commitment,
)
from .architecture_diff import (  # noqa: E402
    build_verified_architecture_diff,
    verify_architecture_diff_commitment,
)
from .verified_git_diff import (  # noqa: E402
    build_verified_git_architecture_diff,
    verify_git_architecture_diff_commitment,
)
from .verified_routes import (  # noqa: E402
    build_verified_routes,
    verify_routes_commitment,
)
from .verified_snapshot import (  # noqa: E402
    build_verified_graph_snapshot,
    check_verified_graph_snapshot,
    load_verified_graph_snapshot,
    verify_graph_snapshot_check_commitment,
    verify_graph_snapshot_commitment,
)
from .verified_move import (  # noqa: E402
    apply_verified_file_move_plan,
    build_verified_file_move_plan,
)
from .verified_context import (  # noqa: E402
    build_symbol_graph,
    build_verified_context,
    verify_context_commitment,
    verify_symbol_graph_commitment,
)
