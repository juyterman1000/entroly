"""Normalize cross-language parser-registry facts into Entroly-owned evidence.

This module is deliberately small and dependency-facing. Third-party parser
objects never escape it: repository graph, retrieval, receipts, and future flow
engines consume stable Entroly dataclasses instead. Structure, imports, exports,
symbols, and diagnostics are extracted in one bounded registry pass.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping


REGISTRY_FACTS_SCHEMA_VERSION = "entroly.registry-facts.v2"


@dataclass(frozen=True)
class RegistrySpan:
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    evidence_sha256: str


@dataclass(frozen=True)
class RegistryStructure:
    name: str
    kind: str
    signature: str
    span: RegistrySpan


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
    structures: tuple[RegistryStructure, ...]
    imports: tuple[RegistryImport, ...]
    exports: tuple[RegistryExport, ...]
    symbols: tuple[RegistrySymbol, ...]
    diagnostics: tuple[RegistryDiagnostic, ...]
    node_count: int
    complete: bool


def _get(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _kind(value: object) -> str:
    raw = _get(value, "name", None) or str(value or "")
    text = str(raw).strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.replace("_", "-") or "unknown"


def _span(raw: bytes, value: object) -> RegistrySpan | None:
    if value is None:
        return None
    try:
        start = int(_get(value, "start_byte", -1))
        end = int(_get(value, "end_byte", -1))
        start_line = int(_get(value, "start_line", 0)) + 1
        end_line = int(_get(value, "end_line", 0)) + 1
    except (TypeError, ValueError):
        return None
    if not (0 <= start <= end <= len(raw)):
        return None
    return RegistrySpan(
        start_byte=start,
        end_byte=end,
        start_line=start_line,
        end_line=end_line,
        evidence_sha256=hashlib.sha256(raw[start:end]).hexdigest(),
    )


def _sequence(value: object) -> list[Any]:
    if isinstance(value, (str, bytes, Mapping)):
        return []
    try:
        return list(value or ())
    except TypeError:
        return []


def extract_registry_facts(
    source: str,
    language: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    max_nodes: int = 100_000,
    max_structures: int = 50_000,
) -> RegistryFacts | None:
    """Return normalized parser facts, or ``None`` when safely unavailable.

    ``complete=False`` means the parser reported a tree larger than the caller's
    declared analysis budget or the normalized structure budget was reached.
    Callers must never treat absence from an incomplete result as negative
    evidence.
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
            structure=True,
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

    metrics = _get(result, "metrics", None)
    try:
        node_count = int(_get(metrics, "node_count", 0) or 0)
    except (TypeError, ValueError):
        node_count = 0
    complete = node_count <= max_nodes if node_count else True

    def field(name: str) -> list[Any]:
        return _sequence(_get(result, name, ()))

    structures: list[RegistryStructure] = []
    structure_budget_hit = False

    def visit_structure(item: object) -> None:
        nonlocal structure_budget_hit
        if len(structures) >= max(1, int(max_structures)):
            structure_budget_hit = True
            return
        span = _span(raw, _get(item, "span", None))
        name = str(_get(item, "name", "") or "").strip()
        if span is not None and name and len(name) <= 512 and "\n" not in name:
            signature = str(_get(item, "signature", "") or "").strip()
            structures.append(RegistryStructure(
                name=name,
                kind=_kind(_get(item, "kind", "unknown")),
                signature=signature[:2_000],
                span=span,
            ))
        for child in _sequence(_get(item, "children", ())):
            if structure_budget_hit:
                break
            visit_structure(child)

    for item in field("structure"):
        if structure_budget_hit:
            break
        visit_structure(item)
    if structure_budget_hit:
        complete = False

    imports: list[RegistryImport] = []
    for item in field("imports"):
        span = _span(raw, _get(item, "span", None))
        if span is None:
            continue
        source_name = str(_get(item, "source", "") or "").strip()
        if not source_name:
            continue
        names = tuple(sorted({
            str(value)
            for value in _sequence(_get(item, "items", ()))
            if value
        }))
        imports.append(RegistryImport(
            source=source_name,
            items=names,
            alias=str(_get(item, "alias", "") or "").strip(),
            is_wildcard=bool(_get(item, "is_wildcard", False)),
            span=span,
        ))

    exports: list[RegistryExport] = []
    for item in field("exports"):
        span = _span(raw, _get(item, "span", None))
        name = str(_get(item, "name", "") or "").strip()
        if span is None or not name:
            continue
        exports.append(RegistryExport(
            name=name,
            kind=_kind(_get(item, "kind", "named")),
            span=span,
        ))

    symbols: list[RegistrySymbol] = []
    for item in field("symbols"):
        span = _span(raw, _get(item, "span", None))
        name = str(_get(item, "name", "") or "").strip()
        if span is None or not name:
            continue
        symbols.append(RegistrySymbol(
            name=name,
            kind=_kind(_get(item, "kind", "unknown")),
            type_annotation=str(_get(item, "type_annotation", "") or "").strip(),
            span=span,
        ))

    diagnostics: list[RegistryDiagnostic] = []
    for item in field("diagnostics"):
        diagnostics.append(RegistryDiagnostic(
            message=str(_get(item, "message", "") or "")[:1000],
            severity=_kind(_get(item, "severity", "error")),
            span=_span(raw, _get(item, "span", None)),
        ))

    return RegistryFacts(
        language=str(language),
        structures=tuple(sorted(structures, key=lambda value: (
            value.span.start_byte, -value.span.end_byte, value.name, value.kind
        ))),
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
    "RegistryStructure",
    "RegistrySymbol",
    "extract_registry_facts",
]
