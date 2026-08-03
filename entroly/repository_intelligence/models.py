"""Value objects for deterministic repository intelligence."""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class RepositoryLimits:
    max_files: int = 20_000
    max_total_bytes: int = 256 * 1024 * 1024
    max_file_bytes: int = 2 * 1024 * 1024
    max_symbols: int = 500_000
    max_edges: int = 1_000_000

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Symbol:
    symbol_id: str
    path: str
    name: str
    qualified_name: str
    kind: str
    line_start: int
    line_end: int
    language: str
    parent_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return vars(self).copy()


@dataclass(frozen=True)
class CallEdge:
    caller_id: str
    callee_id: str
    path: str
    line: int
    confidence: str = "resolved"

    def to_dict(self) -> dict[str, object]:
        return vars(self).copy()


@dataclass(frozen=True)
class FileRecord:
    path: str
    language: str
    sha256: str
    byte_length: int
    line_count: int
    is_test: bool
    imports: tuple[str, ...] = ()
    parse_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return vars(self).copy()


@dataclass(frozen=True)
class ImpactReport:
    changed_paths: tuple[str, ...]
    impacted_paths: tuple[str, ...]
    impacted_symbols: tuple[str, ...]
    reasons: Mapping[str, tuple[str, ...]]
    truncated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "changed_paths": list(self.changed_paths),
            "impacted_paths": list(self.impacted_paths),
            "impacted_symbols": list(self.impacted_symbols),
            "reasons": {
                path: list(values) for path, values in sorted(self.reasons.items())
            },
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class TestCandidate:
    path: str
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "score": round(self.score, 6),
            "reasons": list(self.reasons),
        }


def normalize_relative(path: str | os.PathLike[str]) -> str:
    value = str(path).replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value.strip("/")


@dataclass
class RepositoryIndex:
    root: str
    files: dict[str, FileRecord] = field(default_factory=dict)
    symbols: dict[str, Symbol] = field(default_factory=dict)
    call_edges: tuple[CallEdge, ...] = ()
    file_dependencies: dict[str, tuple[str, ...]] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        grouped: dict[str, list[str]] = defaultdict(list)
        for symbol_id, symbol in self.symbols.items():
            grouped[symbol.path].append(symbol_id)
        self._symbols_by_path = {
            path: tuple(sorted(ids)) for path, ids in sorted(grouped.items())
        }
        reverse_files: dict[str, set[str]] = defaultdict(set)
        for source, dependencies in self.file_dependencies.items():
            for dependency in dependencies:
                reverse_files[dependency].add(source)
        self._reverse_file_dependencies = {
            path: tuple(sorted(values))
            for path, values in sorted(reverse_files.items())
        }
        reverse_calls: dict[str, set[str]] = defaultdict(set)
        for edge in self.call_edges:
            reverse_calls[edge.callee_id].add(edge.caller_id)
        self._reverse_calls = {
            symbol_id: tuple(sorted(values))
            for symbol_id, values in sorted(reverse_calls.items())
        }

    @property
    def test_paths(self) -> tuple[str, ...]:
        return tuple(
            path for path, record in sorted(self.files.items()) if record.is_test
        )

    def symbols_for_path(self, path: str) -> tuple[Symbol, ...]:
        ids = self._symbols_by_path.get(normalize_relative(path), ())
        return tuple(self.symbols[symbol_id] for symbol_id in ids)

    def callers_of(self, symbol_id: str) -> tuple[Symbol, ...]:
        return tuple(
            self.symbols[caller]
            for caller in self._reverse_calls.get(symbol_id, ())
            if caller in self.symbols
        )

    def dependents_of(self, path: str) -> tuple[str, ...]:
        return self._reverse_file_dependencies.get(normalize_relative(path), ())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "entroly.repository-index.v1",
            "root": self.root,
            "files": [self.files[path].to_dict() for path in sorted(self.files)],
            "symbols": [
                self.symbols[symbol_id].to_dict()
                for symbol_id in sorted(self.symbols)
            ],
            "call_edges": [edge.to_dict() for edge in self.call_edges],
            "file_dependencies": {
                path: list(self.file_dependencies[path])
                for path in sorted(self.file_dependencies)
            },
            "diagnostics": list(self.diagnostics),
        }
