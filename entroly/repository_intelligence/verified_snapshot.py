"""Portable, content-addressed repository graph snapshots.

Snapshots are root-independent and deterministic.  Import is deliberately
fail-closed: the commitment, graph identities, edge endpoints, index digest,
and every workspace source hash must match before a RepositoryIndex is built.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .models import (
    CallEdge,
    FileRecord,
    RepositoryIndex,
    Symbol,
    UnresolvedCall,
)

VERIFIED_GRAPH_SNAPSHOT_SCHEMA_VERSION = "entroly.verified-graph-snapshot.v1"
VERIFIED_GRAPH_SNAPSHOT_CHECK_SCHEMA_VERSION = (
    "entroly.verified-graph-snapshot-check.v1"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _portable_index(index: RepositoryIndex) -> dict[str, object]:
    payload = index.to_dict()
    payload["root"] = "."
    diagnostics = payload.get("diagnostics", [])
    if isinstance(diagnostics, list):
        payload["diagnostics"] = [
            item
            for item in diagnostics
            if not str(item).startswith((
                "incremental-parse-cache ",
                "persistent-index-snapshot ",
            ))
        ]
    return payload


def _digest(index_payload: Mapping[str, object]) -> str:
    candidate = copy.deepcopy(dict(index_payload))
    candidate["root"] = "."
    return "sha256:" + hashlib.sha256(_canonical(candidate)).hexdigest()


def build_verified_graph_snapshot(
    index: RepositoryIndex,
    *,
    index_digest: str,
) -> dict[str, object]:
    """Serialize the complete bounded graph as a committable JSON object."""
    graph = _portable_index(index)
    calculated = _digest(graph)
    if calculated != index_digest:
        raise ValueError("index digest does not bind the portable graph")
    files = graph["files"]
    symbols = graph["symbols"]
    call_edges = graph["call_edges"]
    unresolved = graph["unresolved_calls"]
    dependencies = graph["file_dependencies"]
    assert isinstance(files, list)
    assert isinstance(symbols, list)
    assert isinstance(call_edges, list)
    assert isinstance(unresolved, list)
    assert isinstance(dependencies, dict)
    sources = {
        str(item["path"]): str(item["sha256"])
        for item in files
        if isinstance(item, dict)
    }
    payload: dict[str, object] = {
        "schema_version": VERIFIED_GRAPH_SNAPSHOT_SCHEMA_VERSION,
        "index_digest": index_digest,
        "graph": graph,
        "counts": {
            "files": len(files),
            "symbols": len(symbols),
            "call_edges": len(call_edges),
            "unresolved_calls": len(unresolved),
            "dependency_edges": sum(
                len(value) for value in dependencies.values()
                if isinstance(value, list)
            ),
        },
        "receipt": {
            "root_independent": True,
            "complete_within_index_bounds": True,
            "merge_policy": "reject-drift-never-union-stale-facts",
            "source_manifest_sha256": _sha(sources),
            "graph_sha256": _sha(graph),
            "commitment_scope": (
                "payload-excluding-command-generation-and-snapshot-sha256"
            ),
        },
    }
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt["snapshot_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def verify_graph_snapshot_commitment(payload: Mapping[str, object]) -> bool:
    """Verify a detached snapshot commitment and structural invariants."""
    try:
        candidate = copy.deepcopy(dict(payload))
        candidate.pop("command", None)
        candidate.pop("generation", None)
        if candidate.get("schema_version") != VERIFIED_GRAPH_SNAPSHOT_SCHEMA_VERSION:
            return False
        receipt = candidate["receipt"]
        graph = candidate["graph"]
        if not isinstance(receipt, dict) or not isinstance(graph, dict):
            return False
        expected = str(receipt.pop("snapshot_sha256"))
        if hashlib.sha256(_canonical(candidate)).hexdigest() != expected:
            return False
        if str(receipt["graph_sha256"]) != _sha(graph):
            return False
        if str(candidate["index_digest"]) != _digest(graph):
            return False
        _validate_graph(graph)
        files = graph["files"]
        assert isinstance(files, list)
        sources = {str(item["path"]): str(item["sha256"]) for item in files}
        return str(receipt["source_manifest_sha256"]) == _sha(sources)
    except (AssertionError, KeyError, TypeError, ValueError):
        return False


def _records(graph: Mapping[str, object], key: str) -> list[dict[str, object]]:
    value = graph.get(key)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"snapshot graph {key} must be an array of objects")
    return value


def _validate_graph(graph: Mapping[str, object]) -> None:
    if graph.get("schema_version") != "entroly.repository-index.v2":
        raise ValueError("snapshot graph schema mismatch")
    if graph.get("root") != ".":
        raise ValueError("snapshot graph root must be portable")
    files = _records(graph, "files")
    symbols = _records(graph, "symbols")
    calls = _records(graph, "call_edges")
    unresolved = _records(graph, "unresolved_calls")
    file_ids = [str(item.get("path", "")) for item in files]
    symbol_ids = [str(item.get("symbol_id", "")) for item in symbols]
    if not all(file_ids) or len(set(file_ids)) != len(file_ids):
        raise ValueError("snapshot file identities are empty or duplicated")
    if not all(symbol_ids) or len(set(symbol_ids)) != len(symbol_ids):
        raise ValueError("snapshot symbol identities are empty or duplicated")
    known_files = set(file_ids)
    known_symbols = set(symbol_ids)
    if any(str(item.get("path", "")) not in known_files for item in symbols):
        raise ValueError("snapshot symbol references an unknown file")
    for edge in calls:
        if (
            str(edge.get("caller_id", "")) not in known_symbols
            or str(edge.get("callee_id", "")) not in known_symbols
            or str(edge.get("path", "")) not in known_files
        ):
            raise ValueError("snapshot call edge has an unknown endpoint")
    for item in unresolved:
        if (
            str(item.get("caller_id", "")) not in known_symbols
            or str(item.get("path", "")) not in known_files
            or any(str(value) not in known_symbols for value in item.get("candidates", []))
        ):
            raise ValueError("snapshot unresolved call has an unknown endpoint")
    dependencies = graph.get("file_dependencies")
    if not isinstance(dependencies, dict):
        raise ValueError("snapshot file dependencies must be an object")
    for source, targets in dependencies.items():
        if str(source) not in known_files or not isinstance(targets, list):
            raise ValueError("snapshot dependency source is invalid")
        if any(str(target) not in known_files for target in targets):
            raise ValueError("snapshot dependency target is unknown")


def load_verified_graph_snapshot(
    root: Path,
    payload: Mapping[str, object],
) -> RepositoryIndex:
    """Reconstruct a graph only when every committed source still matches."""
    if not verify_graph_snapshot_commitment(payload):
        raise ValueError("graph snapshot commitment or structure is invalid")
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    graph = payload["graph"]
    assert isinstance(graph, dict)
    files_payload = _records(graph, "files")
    files: dict[str, FileRecord] = {}
    for item in files_payload:
        record = FileRecord(**item)
        try:
            candidate = (root / record.path).resolve(strict=True)
            candidate.relative_to(root)
            raw = candidate.read_bytes()
        except (OSError, RuntimeError, ValueError):
            raise ValueError(f"snapshot source unavailable: {record.path}") from None
        if hashlib.sha256(raw).hexdigest() != record.sha256:
            raise ValueError(f"snapshot source drift: {record.path}")
        files[record.path] = record
    symbols = {
        item["symbol_id"]: Symbol(**item)
        for item in _records(graph, "symbols")
    }
    dependencies = graph["file_dependencies"]
    assert isinstance(dependencies, dict)
    diagnostics = graph.get("diagnostics", [])
    if not isinstance(diagnostics, list) or any(
        not isinstance(item, str) for item in diagnostics
    ):
        raise ValueError("snapshot diagnostics must be an array of strings")
    return RepositoryIndex(
        root=str(root),
        files=files,
        symbols=symbols,
        call_edges=tuple(CallEdge(**item) for item in _records(graph, "call_edges")),
        unresolved_calls=tuple(
            UnresolvedCall(**{**item, "candidates": tuple(item.get("candidates", []))})
            for item in _records(graph, "unresolved_calls")
        ),
        file_dependencies={
            str(path): tuple(str(target) for target in targets)
            for path, targets in dependencies.items()
        },
        diagnostics=tuple(diagnostics),
    )


def _identities(index: RepositoryIndex) -> dict[str, set[str]]:
    return {
        "files": set(index.files),
        "symbols": set(index.symbols),
        "call_edges": {
            _sha(edge.to_dict()) for edge in index.call_edges
        },
        "unresolved_calls": {
            _sha(item.to_dict()) for item in index.unresolved_calls
        },
        "dependency_edges": {
            f"{source}\0{target}"
            for source, targets in index.file_dependencies.items()
            for target in targets
        },
    }


def check_verified_graph_snapshot(
    root: Path,
    current: RepositoryIndex,
    payload: Mapping[str, object],
    *,
    index_digest: str,
    limit: int = 10_000,
) -> dict[str, object]:
    """Compare a committed snapshot with the live verified graph."""
    bound = max(1, min(int(limit), 100_000))
    valid = verify_graph_snapshot_commitment(payload)
    imported: RepositoryIndex | None = None
    import_error: str | None = None
    if valid:
        try:
            imported = load_verified_graph_snapshot(root, payload)
        except (OSError, ValueError) as exc:
            import_error = str(exc)
    current_ids = _identities(current)
    snapshot_ids = _identities(imported) if imported is not None else {
        key: set() for key in current_ids
    }
    categories: dict[str, object] = {}
    total_drift = 0
    omitted = 0
    for key in sorted(current_ids):
        added = sorted(current_ids[key] - snapshot_ids[key])
        removed = sorted(snapshot_ids[key] - current_ids[key])
        total_drift += len(added) + len(removed)
        omitted += max(0, len(added) - bound) + max(0, len(removed) - bound)
        categories[key] = {
            "only_current": added[:bound],
            "only_snapshot": removed[:bound],
            "only_current_count": len(added),
            "only_snapshot_count": len(removed),
        }
    result: dict[str, object] = {
        "schema_version": VERIFIED_GRAPH_SNAPSHOT_CHECK_SCHEMA_VERSION,
        "current_index_digest": index_digest,
        "snapshot_index_digest": payload.get("index_digest") if valid else None,
        "snapshot_commitment_valid": valid,
        "snapshot_importable": imported is not None,
        "in_sync": imported is not None and total_drift == 0,
        "import_error": import_error,
        "drift": categories,
        "truncation": {"identities_omitted": omitted},
        "receipt": {
            "comparison": "set-exact-over-full-bounded-index",
            "stale_snapshot_facts_merged": 0,
            "commitment_scope": (
                "payload-excluding-command-generation-and-check-sha256"
            ),
        },
    }
    receipt = result["receipt"]
    assert isinstance(receipt, dict)
    receipt["check_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def verify_graph_snapshot_check_commitment(payload: Mapping[str, object]) -> bool:
    try:
        candidate = copy.deepcopy(dict(payload))
        candidate.pop("command", None)
        candidate.pop("generation", None)
        if candidate.get("schema_version") != VERIFIED_GRAPH_SNAPSHOT_CHECK_SCHEMA_VERSION:
            return False
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("check_sha256"))
        return hashlib.sha256(_canonical(candidate)).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "VERIFIED_GRAPH_SNAPSHOT_SCHEMA_VERSION",
    "VERIFIED_GRAPH_SNAPSHOT_CHECK_SCHEMA_VERSION",
    "build_verified_graph_snapshot",
    "check_verified_graph_snapshot",
    "load_verified_graph_snapshot",
    "verify_graph_snapshot_check_commitment",
    "verify_graph_snapshot_commitment",
]
