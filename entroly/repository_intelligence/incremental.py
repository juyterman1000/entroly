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
from .models import FileRecord, RepositoryIndex, RepositoryLimits, Symbol
from .parsers import ParsedCall, ParsedFile, scan_repository

CACHE_SCHEMA_VERSION = "entroly.repository-parse-cache.v1"
_MAX_CACHE_ENTRY_BYTES = 16 * 1024 * 1024


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
    payload = {
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
    symbols = [Symbol(**item) for item in symbols_payload if isinstance(item, dict)]
    calls = [ParsedCall(**item) for item in calls_payload if isinstance(item, dict)]
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
    symbols = {
        symbol.symbol_id: symbol
        for path in sorted(parsed)
        for symbol in sorted(parsed[path].symbols, key=lambda item: item.symbol_id)
    }
    dependencies = resolve_imports(parsed)
    calls, unresolved_calls = resolve_calls(parsed, symbols, policy)
    if len(calls) + len(unresolved_calls) >= policy.max_edges:
        diagnostics.append("relationship limit reached; remaining evidence omitted")
    diagnostics.append(cache.stats.diagnostic())
    return RepositoryIndex(
        root=str(root_path),
        files={path: parsed[path].record for path in sorted(parsed)},
        symbols=symbols,
        call_edges=calls,
        unresolved_calls=unresolved_calls,
        file_dependencies=dependencies,
        diagnostics=tuple(sorted(dict.fromkeys(diagnostics))),
    )


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "ContentAddressedParseCache",
    "IncrementalCacheStats",
    "build_repository_index_incremental",
]
