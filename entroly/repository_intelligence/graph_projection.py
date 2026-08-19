"""Project the repository code graph into the Work Graph.

Section 4.1 of the master implementation prompt asks for the repository to be
graph-addressable as::

    Repository --CONTAINS--> File --DEFINES--> Symbol
                                 --IMPORTS--> File

`graph_identity` made the two sides agree on *names*. This module supplies the
*edges*: it turns what repository intelligence already knows -- files, symbols,
import relations -- into Work Graph operations, so a file and the work done on
it are reachable from one another instead of living in two disconnected graphs.

Bounded, per section 4.2
------------------------
The instruction is explicit that this must not eagerly materialize every file,
symbol and dependency on every observation. So this projection:

* takes the caller's *scope* -- typically changed or active files -- rather than
  walking the repository;
* caps files, symbols per file, and total operations, and reports what it
  dropped instead of silently truncating;
* stores compact references only. Paths, language, digests and line counts go
  into attributes; source text never does. The digest is the recovery handle,
  and copying content into graph state would make every observation grow the
  persisted graph without bound.

Trust, per section 3.3
----------------------
Everything emitted here is `observed`: it is durable filesystem state read from
the working tree, not a verified claim and not an agent's assertion. Nothing in
this module may emit `verified` -- that level is reserved for independently
checked evidence such as a passing test or a receipt, and minting it from a
directory listing is exactly the fabricated-completeness failure the handoff
forbids.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..work_graph import stable_edge_id
from .graph_identity import (
    file_node_id,
    repository_node_id,
    symbol_node_id,
)
from .models import FileRecord, Symbol

# Bounds chosen to stay far below the engine's own MAX_OPERATIONS_PER_EVENT
# (4,096) so a projection is never the reason an event is rejected.
DEFAULT_MAX_FILES = 256
DEFAULT_MAX_SYMBOLS_PER_FILE = 64
DEFAULT_MAX_OPERATIONS = 3_000

_OBSERVED = "observed"


def _node(node_id: str, kind: str, label: str, attributes: Mapping[str, Any]) -> dict:
    return {
        "op": "upsert_node",
        "node": {
            "node_id": node_id,
            "kind": kind,
            "label": label,
            "trust": _OBSERVED,
            "attributes": dict(attributes),
        },
    }


def _edge(edge_from: str, kind: str, edge_to: str) -> dict:
    # `edge_id` is required by the engine -- `WorkEdge.edge_id` carries no serde
    # default, so an event without it is rejected outright. It is still not
    # computed here: `stable_edge_id` calls the engine, so the value is the one
    # the graph would derive itself rather than a Python reimplementation.
    return {
        "op": "upsert_edge",
        "edge": {
            "edge_id": stable_edge_id(edge_from, kind, edge_to),
            "from_node": edge_from,
            "to_node": edge_to,
            "kind": kind,
            "trust": _OBSERVED,
        },
    }


def project_repository_scope(
    repo_id: str,
    *,
    files: Sequence[FileRecord],
    symbols: Mapping[str, Sequence[Symbol]] | None = None,
    imports: Iterable[tuple[str, str]] = (),
    observed_at_ms: int,
    max_files: int = DEFAULT_MAX_FILES,
    max_symbols_per_file: int = DEFAULT_MAX_SYMBOLS_PER_FILE,
    max_operations: int = DEFAULT_MAX_OPERATIONS,
) -> dict[str, Any]:
    """Build one Work Graph event describing a bounded slice of the repository.

    Returns an event payload ready for `WorkGraph.apply_event`. The caller
    chooses the slice; this function does not decide what is interesting, and it
    does not read the filesystem.

    The returned payload carries a ``projection`` block reporting how many files
    and symbols were dropped by the caps, so a truncated view is visible to the
    caller rather than passing as a complete one.
    """
    symbols = symbols or {}
    operations: list[dict] = []
    repo_node = repository_node_id(repo_id)

    operations.append(
        _node(repo_node, "repository", repo_id, {"repo_id": repo_id})
    )

    selected = list(files)[:max_files]
    dropped_files = max(0, len(files) - len(selected))
    dropped_symbols = 0
    truncated = False

    for record in selected:
        if len(operations) >= max_operations:
            truncated = True
            break
        node_id = file_node_id(repo_id, record.path)
        operations.append(
            _node(
                node_id,
                "file",
                record.path,
                {
                    "path": record.path,
                    "language": record.language,
                    # A content commitment, not the content. This is what makes
                    # the node a reference rather than a copy of the file.
                    "sha256": record.sha256,
                    "byte_length": record.byte_length,
                    "line_count": record.line_count,
                    "is_test": record.is_test,
                },
            )
        )
        operations.append(_edge(repo_node, "contains", node_id))

        file_symbols = list(symbols.get(record.path, ()))
        kept = file_symbols[:max_symbols_per_file]
        dropped_symbols += len(file_symbols) - len(kept)
        for symbol in kept:
            if len(operations) >= max_operations:
                truncated = True
                break
            symbol_node = symbol_node_id(repo_id, symbol.symbol_id)
            operations.append(
                _node(
                    symbol_node,
                    "symbol",
                    symbol.qualified_name or symbol.name,
                    {
                        "path": symbol.path,
                        "name": symbol.name,
                        "qualified_name": symbol.qualified_name,
                        "symbol_kind": symbol.kind,
                        "language": symbol.language,
                        "line_start": symbol.line_start,
                        "line_end": symbol.line_end,
                        # The local repository-intelligence key, kept so a graph
                        # node can be traced back to the index that produced it.
                        "symbol_id": symbol.symbol_id,
                    },
                )
            )
            operations.append(_edge(node_id, "defines", symbol_node))

    # Import targets outside the scope become boundary nodes.
    #
    # The engine rejects an edge whose endpoint it does not know, so a dangling
    # `imports` edge is not an option. Dropping those edges instead would mean
    # `File IMPORTS File` almost never materializes, since a changed file
    # usually imports unchanged ones. A minimal node -- path only, no digest, no
    # symbols -- is the "bounded traversal outward from the active workstream"
    # section 4.2 allows: one hop, and it is counted and reported rather than
    # quietly widening the caller's scope.
    projected_paths = {record.path for record in selected}
    boundary_files = 0
    for source_path, target_path in imports:
        if len(operations) >= max_operations:
            truncated = True
            break
        if source_path not in projected_paths:
            # The importer itself is out of scope; its edge is not this
            # projection's to assert.
            continue
        if target_path not in projected_paths:
            projected_paths.add(target_path)
            boundary_files += 1
            operations.append(
                _node(
                    file_node_id(repo_id, target_path),
                    "file",
                    target_path,
                    {"path": target_path, "boundary": True},
                )
            )
            operations.append(
                _edge(repo_node, "contains", file_node_id(repo_id, target_path))
            )
        operations.append(
            _edge(
                file_node_id(repo_id, source_path),
                "imports",
                file_node_id(repo_id, target_path),
            )
        )

    return {
        "observed_at_ms": int(observed_at_ms),
        "source_kind": "repository_fact",
        "source_ref": f"repository_intelligence:{repo_id}",
        "operations": operations,
        "projection": {
            "files_projected": len(selected),
            "files_dropped": dropped_files,
            "symbols_dropped": dropped_symbols,
            "boundary_files": boundary_files,
            "truncated": truncated,
            "operation_count": len(operations),
        },
    }


def apply_repository_scope(graph: Any, repo_id: str, **kwargs: Any) -> dict[str, Any]:
    """Project a scope and apply it to ``graph``.

    Convenience for callers that already hold a `WorkGraph`. The projection
    block is returned so a truncated result stays visible.
    """
    payload = project_repository_scope(repo_id, **kwargs)
    projection = payload.pop("projection")
    graph.apply_event(payload)
    return projection


__all__ = [
    "DEFAULT_MAX_FILES",
    "DEFAULT_MAX_OPERATIONS",
    "DEFAULT_MAX_SYMBOLS_PER_FILE",
    "apply_repository_scope",
    "project_repository_scope",
]
