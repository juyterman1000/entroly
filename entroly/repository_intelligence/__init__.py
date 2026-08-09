"""Bounded, deterministic repository intelligence.

Python uses exact AST symbol ranges. Rust and JavaScript/TypeScript use
conservative syntax recognition; ambiguous relationships are omitted rather
than guessed. All paths remain workspace-relative and escaping symlinks are
rejected.
"""
from __future__ import annotations

import os
from pathlib import Path

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

__all__ = [
    "CallEdge", "FileRecord", "ImpactReport", "RepositoryIndex",
    "RepositoryLimits", "Symbol", "TestCandidate", "UnresolvedCall",
    "analyze_change_impact",
    "build_repository_index", "localize_tests", "InvalidChangedPaths",
    "InvalidContextQuery", "InvalidSymbolQuery",
    "RepositoryIntelligenceError", "RepositoryIntelligenceService",
    "UnknownChangedPaths", "build_symbol_graph", "build_verified_context",
    "verify_context_commitment", "verify_symbol_graph_commitment",
]


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
    parsed, diagnostics = scan_repository(root_path, policy)
    symbols = {
        symbol.symbol_id: symbol
        for path in sorted(parsed)
        for symbol in sorted(parsed[path].symbols, key=lambda item: item.symbol_id)
    }
    dependencies = resolve_imports(parsed)
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
    InvalidContextQuery,
    InvalidSymbolQuery,
    RepositoryIntelligenceError,
    RepositoryIntelligenceService,
    UnknownChangedPaths,
)
from .verified_context import (  # noqa: E402
    build_symbol_graph,
    build_verified_context,
    verify_context_commitment,
    verify_symbol_graph_commitment,
)
