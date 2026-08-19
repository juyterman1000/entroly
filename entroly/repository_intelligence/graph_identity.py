"""Canonical Work Graph identity for repository-intelligence artifacts.

Repository intelligence and the Work Graph describe the same repository. The
graph has `NodeKind::{Repository, File, Symbol}` with identities computed by
`entroly_engine::work_graph::stable_node_id`; repository intelligence has
`FileRecord` and `Symbol` keyed by readable local strings such as
``entroly/server.py::EntrolyEngine.optimize_context::function``. Both are
correct for their own purpose, and until now they had nothing in common, so a
file could not be looked up in one graph from the other.

This module supplies the join. The readable local key is kept exactly as it is
-- it is referenced by dozens of modules, persisted in caches, and quoted in
receipts, and renumbering it would be a large break for no gain -- and is passed
to the engine as the ``key`` it hashes. One artifact therefore has one canonical
node id, derived rather than invented.

The hash itself is deliberately not reimplemented here. `stable_node_id` is
shared product semantics; a Python copy of
``sha256("node|{kind}|{repo}|{key}")[:24]`` would be a second definition that
could drift from the engine silently, which is the precise failure this module
exists to end. The consequence is that these helpers require the native engine
and fail closed without it, exactly as the rest of the Work Graph surface does.
"""

from __future__ import annotations

from .models import FileRecord, Symbol


def repository_node_id(repo_id: str) -> str:
    """Canonical id for the repository itself."""
    from ..work_graph import stable_node_id

    return stable_node_id("repository", repo_id, repo_id)


def file_node_id(repo_id: str, path: str) -> str:
    """Canonical id for a file, keyed by its repository-relative path.

    The path is the key rather than a content digest: a file keeps its identity
    across edits, which is what lets change evidence attach to the same node
    over time. Content identity is carried separately by `FileRecord.sha256`.
    """
    from ..work_graph import stable_node_id

    return stable_node_id("file", repo_id, path)


def symbol_node_id(repo_id: str, symbol_id: str) -> str:
    """Canonical id for a symbol, keyed by its repository-intelligence id.

    ``symbol_id`` is the readable ``path::qualified::kind`` key produced by the
    parsers. Passing it through unchanged means the two graphs agree without
    either side having to adopt the other's naming.
    """
    from ..work_graph import stable_node_id

    return stable_node_id("symbol", repo_id, symbol_id)


def file_record_node_id(repo_id: str, record: FileRecord) -> str:
    """Canonical id for a `FileRecord`."""
    return file_node_id(repo_id, record.path)


def symbol_record_node_id(repo_id: str, symbol: Symbol) -> str:
    """Canonical id for a `Symbol`."""
    return symbol_node_id(repo_id, symbol.symbol_id)


def contains_edge_id(repo_id: str, path: str) -> str:
    """`Repository CONTAINS File`, the edge in section 4.1 of the architecture."""
    from ..work_graph import stable_edge_id

    return stable_edge_id(
        repository_node_id(repo_id), "contains", file_node_id(repo_id, path)
    )


def defines_edge_id(repo_id: str, path: str, symbol_id: str) -> str:
    """`File DEFINES Symbol`."""
    from ..work_graph import stable_edge_id

    return stable_edge_id(
        file_node_id(repo_id, path), "defines", symbol_node_id(repo_id, symbol_id)
    )


def imports_edge_id(repo_id: str, from_path: str, to_path: str) -> str:
    """`File IMPORTS File`."""
    from ..work_graph import stable_edge_id

    return stable_edge_id(
        file_node_id(repo_id, from_path), "imports", file_node_id(repo_id, to_path)
    )


__all__ = [
    "contains_edge_id",
    "defines_edge_id",
    "file_node_id",
    "file_record_node_id",
    "imports_edge_id",
    "repository_node_id",
    "symbol_node_id",
    "symbol_record_node_id",
]
