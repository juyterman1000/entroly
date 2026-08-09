"""Normalize cross-language parser-registry facts into Entroly-owned evidence.

This module is deliberately small and dependency-facing.  Third-party parser
objects never escape it: repository graph, retrieval, receipts, and future flow
engines consume stable Entroly dataclasses instead.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


REGISTRY_FACTS_SCHEMA_VERSION = "entroly.registry-facts.v1"


@dataclass(frozen=True)
class RegistrySpan:
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    evidence_sha256: str


@dataclass(frozen=True)
class RegistryImport:
    source: str
    items: tuple[str, ...]
    alias: str
    is_wildcard: bool
    span: RegistrySpan


@dataclass(frozen=True)
class RegistryExport:
    name: str
    kind: str
    span: RegistrySpan


@dataclass(frozen=True)
class RegistrySymbol:
    name: str
    kind: str
    type_annotation: str
    span: RegistrySpan


@dataclass(frozen=True)
class RegistryDiagnostic:
    message: str
    severity: str
    span: RegistrySpan | None


@dataclass(frozen=True)
class RegistryFacts:
    language: str
    imports: tuple[RegistryImport, ...]
    exports: tuple[RegistryExport, ...]
    symbols: tuple[RegistrySymbol, ...]
    diagnostics: tuple[RegistryDiagnostic, ...]
    node_count: int
    complete: bool


def _kind(value: object) -> str:
    raw = getattr(value, "name", None) or str(value or "")
    text = str(raw).strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.replace("_", "-") or "unknown"


def _span(raw: bytes, value: object) -> RegistrySpan | None:
    if value is None:
        return None
    start = int(getattr(value, "start_byte", -1))
    end = int(getattr(value, "end_byte", -1))
    if not (0 <= start <= end <= len(raw)):
        return None
    return RegistrySpan(
        start_byte=start,
        end_byte=end,
        start_line=int(getattr(value, "start_line", 0)) + 1,
        end_line=int(getattr(value, "end_line", 0)) + 1,
        evidence_sha256=hashlib.sha256(raw[start:end]).hexdigest(),
    )


def extract_registry_facts(
    source: str,
    language: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    max_nodes: int = 100_000,
) -> RegistryFacts | None:
    """Return normalized parser facts, or ``None`` when safely unavailable.

    `complete=False` means the parser produced a tree larger than the caller's
    declared analysis budget.  Callers must not promote those facts to complete
    repository truth; the objects are retained only for diagnostics/research.
    """
    raw = source.encode("utf-8", errors="surrogateescape")
    if len(raw) > max_bytes or not source.strip() or not language:
        return None
    try:
        import tree_sitter_language_pack as pack
    except (ImportError, OSError):
        return None
    process = getattr(pack, "process", None)
    config_type = getattr(pack, "ProcessConfig", None)
    if not callable(process) or config_type is None:
        return None
    try:
        config = config_type(
            language=language,
            structure=False,
            imports=True,
            exports=True,
            comments=False,
            docstrings=False,
            symbols=True,
            diagnostics=True,
        )
        result: Any = process(source, config)
    except Exception:
        return None

    metrics = getattr(result, "metrics", None)
    if isinstance(result, dict):
        metrics = result.get("metrics")
    node_count = int(
        getattr(metrics, "node_count", 0)
        if metrics is not None and not isinstance(metrics, dict)
        else (metrics or {}).get("node_count", 0)
    )
    complete = node_count <= max_nodes if node_count else True

    def field(name: str) -> list[Any]:
        if isinstance(result, dict):
            value = result.get(name, ())
        else:
            value = getattr(result, name, ())
        return list(value or ())

    imports: list[RegistryImport] = []
    for item in field("imports"):
        span = _span(raw, getattr(item, "span", None))
        if span is None:
            continue
        source_name = str(getattr(item, "source", "") or "").strip()
        if not source_name:
            continue
        imports.append(RegistryImport(
            source=source_name,
            items=tuple(sorted({str(value) for value in getattr(item, "items", ()) if value})),
            alias=str(getattr(item, "alias", "") or "").strip(),
            is_wildcard=bool(getattr(item, "is_wildcard", False)),
            span=span,
        ))

    exports: list[RegistryExport] = []
    for item in field("exports"):
        span = _span(raw, getattr(item, "span", None))
        name = str(getattr(item, "name", "") or "").strip()
        if span is None or not name:
            continue
        exports.append(RegistryExport(
            name=name,
            kind=_kind(getattr(item, "kind", "named")),
            span=span,
        ))

    symbols: list[RegistrySymbol] = []
    for item in field("symbols"):
        span = _span(raw, getattr(item, "span", None))
        name = str(getattr(item, "name", "") or "").strip()
        if span is None or not name:
            continue
        symbols.append(RegistrySymbol(
            name=name,
            kind=_kind(getattr(item, "kind", "unknown")),
            type_annotation=str(getattr(item, "type_annotation", "") or "").strip(),
            span=span,
        ))

    diagnostics: list[RegistryDiagnostic] = []
    for item in field("diagnostics"):
        diagnostics.append(RegistryDiagnostic(
            message=str(getattr(item, "message", "") or "")[:1000],
            severity=_kind(getattr(item, "severity", "error")),
            span=_span(raw, getattr(item, "span", None)),
        ))

    return RegistryFacts(
        language=str(language),
        imports=tuple(sorted(imports, key=lambda value: (
            value.span.start_byte, value.source, value.items
        ))),
        exports=tuple(sorted(exports, key=lambda value: (
            value.span.start_byte, value.name
        ))),
        symbols=tuple(sorted(symbols, key=lambda value: (
            value.span.start_byte, value.name, value.kind
        ))),
        diagnostics=tuple(diagnostics),
        node_count=node_count,
        complete=complete,
    )


__all__ = [
    "REGISTRY_FACTS_SCHEMA_VERSION",
    "RegistryDiagnostic",
    "RegistryExport",
    "RegistryFacts",
    "RegistryImport",
    "RegistrySpan",
    "RegistrySymbol",
    "extract_registry_facts",
]
