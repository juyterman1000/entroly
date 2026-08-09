"""Opt-in content-addressed incremental parsing for repository intelligence."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .graph import resolve_calls, resolve_imports
from .models import (
    CallEdge,
    FileRecord,
    RepositoryIndex,
    RepositoryLimits,
    Symbol,
    UnresolvedCall,
)
from .parsers import ParsedCall, ParsedFile, scan_repository

CACHE_SCHEMA_VERSION = "entroly.repository-parse-cache.v2"
INDEX_SNAPSHOT_SCHEMA_VERSION = "entroly.repository-index-snapshot.v1"
_MAX_CACHE_ENTRY_BYTES = 16 * 1024 * 1024
_MAX_INDEX_SNAPSHOT_BYTES = 256 * 1024 * 1024


@dataclass
class IncrementalCacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    corruptions: int = 0

    def diagnostic(self) -> str:
        return (
            "incremental-parse-cache "
            f"hits={self.hits} misses={self.misses} writes={self.writes} "
            f"corruptions={self.corruptions}"
        )


@dataclass
class IndexSnapshotStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    corruptions: int = 0

    def diagnostic(self) -> str:
        return (
            "persistent-index-snapshot "
            f"hits={self.hits} misses={self.misses} writes={self.writes} "
            f"corruptions={self.corruptions}"
        )


def _parser_fingerprint() -> str:
    payload: dict[str, object] = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "python": platform.python_version(),
        "tree_sitter_package": None,
        "cached_grammars": [],
        "download_opt_in": os.getenv(
            "ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD", ""
        ).strip().lower() in {"1", "true", "yes", "on"},
    }
    try:
        payload["tree_sitter_package"] = importlib.metadata.version(
            "tree-sitter-language-pack"
        )
        import tree_sitter_language_pack as pack

        downloaded = getattr(pack, "downloaded_languages", None)
        if callable(downloaded):
            payload["cached_grammars"] = sorted(str(item) for item in downloaded())
        else:
            payload["cached_grammars"] = ["bundled-or-unknown"]
    except Exception:
        payload["tree_sitter_package"] = "unavailable"
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _entry_key(path: str, source_sha256: str, parser_fingerprint: str) -> str:
    material = f"{CACHE_SCHEMA_VERSION}\0{parser_fingerprint}\0{path}\0{source_sha256}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _serialize(path: str, source_sha256: str, fingerprint: str, item: ParsedFile) -> str:
    payload: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "path": path,
        "source_sha256": source_sha256,
        "parser_fingerprint": fingerprint,
        "record": item.record.to_dict(),
        "symbols": [symbol.to_dict() for symbol in item.symbols],
        "imports": sorted(item.imports),
        "import_aliases": dict(sorted(item.import_aliases.items())),
        "calls": [vars(call).copy() for call in item.calls],
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    payload["cache_sha256"] = hashlib.sha256(canonical).hexdigest()
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _deserialize(
    rendered: str,
    *,
    path: str,
    source_sha256: str,
    fingerprint: str,
) -> ParsedFile:
    payload = json.loads(rendered)
    if not isinstance(payload, dict):
        raise ValueError("cache entry must be an object")
    expected = payload.pop("cache_sha256", None)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if not isinstance(expected, str) or hashlib.sha256(canonical).hexdigest() != expected:
        raise ValueError("cache entry commitment mismatch")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("cache schema mismatch")
    if payload.get("path") != path or payload.get("source_sha256") != source_sha256:
        raise ValueError("cache identity mismatch")
    if payload.get("parser_fingerprint") != fingerprint:
        raise ValueError("cache parser fingerprint mismatch")
    record_payload = payload.get("record")
    if not isinstance(record_payload, dict):
        raise ValueError("cache record missing")
    record_values = dict(record_payload)
    record_values["imports"] = tuple(record_values.get("imports", ()))
    record = FileRecord(**record_values)
    if record.path != path or record.sha256 != source_sha256:
        raise ValueError("cached file record mismatch")
    symbols_payload = payload.get("symbols")
    calls_payload = payload.get("calls")
    if not isinstance(symbols_payload, list) or not isinstance(calls_payload, list):
        raise ValueError("cache relationships missing")
    if any(not isinstance(item, dict) for item in (*symbols_payload, *calls_payload)):
        raise ValueError("cache relationship entry invalid")
    symbols = [Symbol(**item) for item in symbols_payload]
    calls = [ParsedCall(**item) for item in calls_payload]
    if any(symbol.path != path for symbol in symbols):
        raise ValueError("cached symbol path mismatch")
    aliases = payload.get("import_aliases")
    imports = payload.get("imports")
    if not isinstance(aliases, dict) or not isinstance(imports, list):
        raise ValueError("cache imports missing")
    return ParsedFile(
        record,
        symbols,
        {str(item) for item in imports},
        {str(key): str(value) for key, value in aliases.items()},
        calls,
    )


class ContentAddressedParseCache:
    """Fail-open cache whose entries are immutable and identity-validated."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.expanduser().resolve()
        self.fingerprint = _parser_fingerprint()
        self.stats = IncrementalCacheStats()
        self._corrupt_targets: set[Path] = set()

    def _path(self, path: str, source_sha256: str) -> Path:
        key = _entry_key(path, source_sha256, self.fingerprint)
        return self.directory / key[:2] / f"{key}.json"

    def load(self, path: str, source_sha256: str) -> ParsedFile | None:
        target = self._path(path, source_sha256)
        try:
            if not target.is_file() or target.stat().st_size > _MAX_CACHE_ENTRY_BYTES:
                self.stats.misses += 1
                return None
            rendered = target.read_text(encoding="utf-8")
            item = _deserialize(
                rendered,
                path=path,
                source_sha256=source_sha256,
                fingerprint=self.fingerprint,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            self.stats.misses += 1
            self.stats.corruptions += 1
            self._corrupt_targets.add(target)
            return None
        self.stats.hits += 1
        return item

    def store(self, path: str, source_sha256: str, item: ParsedFile) -> None:
        target = self._path(path, source_sha256)
        try:
            rendered = _serialize(path, source_sha256, self.fingerprint, item)
            if len(rendered.encode("utf-8")) > _MAX_CACHE_ENTRY_BYTES:
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target not in self._corrupt_targets:
                return
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.stem}.",
                suffix=".tmp",
                dir=target.parent,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(rendered)
                    handle.flush()
                    os.fsync(handle.fileno())
                Path(temporary_name).replace(target)
                self._corrupt_targets.discard(target)
                self.stats.writes += 1
            finally:
                temporary = Path(temporary_name)
                if temporary.exists():
                    temporary.unlink()
        except OSError:
            # Cache failure must not make repository intelligence unavailable.
            return


def _snapshot_manifest(
    parsed: dict[str, ParsedFile],
    limits: RepositoryLimits,
    parser_fingerprint: str,
) -> tuple[str, dict[str, object]]:
    manifest: dict[str, object] = {
        "schema_version": INDEX_SNAPSHOT_SCHEMA_VERSION,
        "parser_fingerprint": parser_fingerprint,
        "limits": dict(sorted(vars(limits).items())),
        "files": [
            {
                "path": path,
                "source_sha256": parsed[path].record.sha256,
                "record_sha256": hashlib.sha256(json.dumps(
                    parsed[path].record.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")).hexdigest(),
            }
            for path in sorted(parsed)
        ],
    }
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), manifest


def _snapshot_payload(
    manifest_sha256: str,
    manifest: dict[str, object],
    index: RepositoryIndex,
) -> str:
    index_payload = index.to_dict()
    index_payload["root"] = "."
    payload: dict[str, object] = {
        "schema_version": INDEX_SNAPSHOT_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "manifest": manifest,
        "index": index_payload,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    payload["snapshot_sha256"] = hashlib.sha256(canonical).hexdigest()
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _index_from_snapshot(
    rendered: str,
    *,
    root: Path,
    manifest_sha256: str,
    manifest: dict[str, object],
) -> RepositoryIndex:
    payload = json.loads(rendered)
    if not isinstance(payload, dict) or payload.get("schema_version") != INDEX_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("index snapshot schema mismatch")
    if payload.get("manifest_sha256") != manifest_sha256 or payload.get("manifest") != manifest:
        raise ValueError("index snapshot manifest mismatch")
    expected = payload.pop("snapshot_sha256", None)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if not isinstance(expected, str) or hashlib.sha256(canonical).hexdigest() != expected:
        raise ValueError("index snapshot commitment mismatch")
    raw_index = payload.get("index")
    if (
        not isinstance(raw_index, dict)
        or raw_index.get("root") != "."
        or raw_index.get("schema_version") != "entroly.repository-index.v2"
    ):
        raise ValueError("index snapshot payload missing")

    files_payload = raw_index.get("files")
    symbols_payload = raw_index.get("symbols")
    edges_payload = raw_index.get("call_edges")
    unresolved_payload = raw_index.get("unresolved_calls")
    dependencies_payload = raw_index.get("file_dependencies")
    diagnostics_payload = raw_index.get("diagnostics")
    if not all(isinstance(value, list) for value in (
        files_payload, symbols_payload, edges_payload, unresolved_payload, diagnostics_payload,
    )) or not isinstance(dependencies_payload, dict):
        raise ValueError("index snapshot relationships missing")
    if any(not isinstance(item, dict) for item in (
        *files_payload, *symbols_payload, *edges_payload, *unresolved_payload,
    )):
        raise ValueError("index snapshot relationship entry invalid")

    files: dict[str, FileRecord] = {}
    for item in files_payload:
        if not isinstance(item, dict):
            raise ValueError("invalid snapshot file record")
        values = dict(item)
        values["imports"] = tuple(values.get("imports", ()))
        record = FileRecord(**values)
        files[record.path] = record
    if len(files) != len(files_payload):
        raise ValueError("duplicate index snapshot file record")
    expected_files = {
        str(item["path"]): str(item["source_sha256"])
        for item in manifest["files"]
        if isinstance(item, dict) and "path" in item and "source_sha256" in item
    }
    if {path: record.sha256 for path, record in files.items()} != expected_files:
        raise ValueError("index snapshot file identity mismatch")
    expected_records = {
        str(item["path"]): str(item["record_sha256"])
        for item in manifest["files"]
        if isinstance(item, dict) and "path" in item and "record_sha256" in item
    }
    actual_records = {
        path: hashlib.sha256(json.dumps(
            record.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")).hexdigest()
        for path, record in files.items()
    }
    if actual_records != expected_records:
        raise ValueError("index snapshot file record mismatch")

    symbols = {
        symbol.symbol_id: symbol
        for item in symbols_payload
        for symbol in [Symbol(**item)]
    }
    if len(symbols) != len(symbols_payload) or any(
        symbol.path not in files for symbol in symbols.values()
    ):
        raise ValueError("index snapshot symbol path missing")
    edges = tuple(CallEdge(**item) for item in edges_payload)
    unresolved: list[UnresolvedCall] = []
    for item in unresolved_payload:
        values = dict(item)
        values["candidates"] = tuple(values.get("candidates", ()))
        unresolved.append(UnresolvedCall(**values))
    if any(
        edge.callee_id not in symbols
        or edge.path not in files
        or (
            edge.caller_id not in symbols
            and not edge.caller_id.endswith("::<module>::module")
        )
        for edge in edges
    ):
        raise ValueError("index snapshot call edge identity mismatch")
    if any(
        item.path not in files
        or any(candidate not in symbols for candidate in item.candidates)
        for item in unresolved
    ):
        raise ValueError("index snapshot unresolved edge identity mismatch")
    dependencies = {
        str(path): tuple(str(target) for target in targets)
        for path, targets in dependencies_payload.items()
        if isinstance(targets, list)
    }
    if set(dependencies) != set(files) or any(
        target not in files for targets in dependencies.values() for target in targets
    ):
        raise ValueError("index snapshot dependency identity mismatch")
    limits = manifest.get("limits")
    if not isinstance(limits, dict) or (
        len(files) > int(limits.get("max_files", 0))
        or len(symbols) > int(limits.get("max_symbols", 0))
        or len(edges) + len(unresolved) > int(limits.get("max_edges", 0))
    ):
        raise ValueError("index snapshot exceeds manifest limits")
    return RepositoryIndex(
        root=str(root),
        files=files,
        symbols=symbols,
        call_edges=edges,
        unresolved_calls=tuple(unresolved),
        file_dependencies=dependencies,
        diagnostics=tuple(str(item) for item in diagnostics_payload),
    )


class ContentAddressedIndexSnapshotStore:
    """Immutable, commitment-checked global graph snapshots.

    A single source change produces a new manifest identity.  This deliberately
    prefers exact whole-graph invalidation over a clever but stale partial graph.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory.expanduser().resolve() / "index-snapshots"
        self.stats = IndexSnapshotStats()
        self._corrupt_targets: set[Path] = set()

    def _path(self, manifest_sha256: str) -> Path:
        return self.directory / manifest_sha256[:2] / f"{manifest_sha256}.json"

    def load(
        self,
        *,
        root: Path,
        manifest_sha256: str,
        manifest: dict[str, object],
    ) -> RepositoryIndex | None:
        target = self._path(manifest_sha256)
        try:
            if not target.is_file() or target.stat().st_size > _MAX_INDEX_SNAPSHOT_BYTES:
                self.stats.misses += 1
                return None
            index = _index_from_snapshot(
                target.read_text(encoding="utf-8"),
                root=root,
                manifest_sha256=manifest_sha256,
                manifest=manifest,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            self.stats.misses += 1
            self.stats.corruptions += 1
            self._corrupt_targets.add(target)
            return None
        self.stats.hits += 1
        return index

    def store(
        self,
        *,
        manifest_sha256: str,
        manifest: dict[str, object],
        index: RepositoryIndex,
    ) -> None:
        target = self._path(manifest_sha256)
        try:
            rendered = _snapshot_payload(manifest_sha256, manifest, index)
            if len(rendered.encode("utf-8")) > _MAX_INDEX_SNAPSHOT_BYTES:
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target not in self._corrupt_targets:
                return
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(rendered)
                    handle.flush()
                    os.fsync(handle.fileno())
                Path(temporary_name).replace(target)
                self._corrupt_targets.discard(target)
                self.stats.writes += 1
            finally:
                temporary = Path(temporary_name)
                if temporary.exists():
                    temporary.unlink()
        except OSError:
            return


def build_repository_index_incremental(
    root: str | os.PathLike[str],
    *,
    cache_dir: str | os.PathLike[str],
    limits: RepositoryLimits | None = None,
) -> RepositoryIndex:
    """Build an index while reusing exact parser results for unchanged files."""
    policy = limits or RepositoryLimits()
    root_path = Path(root).expanduser().resolve(strict=True)
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)
    cache = ContentAddressedParseCache(Path(cache_dir))
    parsed, diagnostics = scan_repository(
        root_path,
        policy,
        load_cached=cache.load,
        store_cached=cache.store,
    )
    snapshot_store = ContentAddressedIndexSnapshotStore(Path(cache_dir))
    manifest_sha256, manifest = _snapshot_manifest(
        parsed, policy, cache.fingerprint
    )
    cached_index = snapshot_store.load(
        root=root_path,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
    )
    if cached_index is not None:
        cached_index.diagnostics = tuple(sorted(dict.fromkeys((
            *cached_index.diagnostics,
            *diagnostics,
            cache.stats.diagnostic(),
            snapshot_store.stats.diagnostic(),
        ))))
        return cached_index
    symbols = {
        symbol.symbol_id: symbol
        for path in sorted(parsed)
        for symbol in sorted(parsed[path].symbols, key=lambda item: item.symbol_id)
    }
    dependencies = resolve_imports(parsed)
    calls, unresolved_calls = resolve_calls(parsed, symbols, policy)
    if len(calls) + len(unresolved_calls) >= policy.max_edges:
        diagnostics.append("relationship limit reached; remaining evidence omitted")
    index = RepositoryIndex(
        root=str(root_path),
        files={path: parsed[path].record for path in sorted(parsed)},
        symbols=symbols,
        call_edges=calls,
        unresolved_calls=unresolved_calls,
        file_dependencies=dependencies,
        diagnostics=tuple(sorted(dict.fromkeys(diagnostics))),
    )
    snapshot_store.store(
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        index=index,
    )
    index.diagnostics = tuple(sorted(dict.fromkeys((
        *index.diagnostics,
        cache.stats.diagnostic(),
        snapshot_store.stats.diagnostic(),
    ))))
    return index


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "INDEX_SNAPSHOT_SCHEMA_VERSION",
    "ContentAddressedParseCache",
    "ContentAddressedIndexSnapshotStore",
    "IncrementalCacheStats",
    "IndexSnapshotStats",
    "build_repository_index_incremental",
]
