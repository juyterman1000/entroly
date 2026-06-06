"""Dependency detection for multi-document Context Receipts."""

from __future__ import annotations

import re
from collections import defaultdict

from .models import ContextIndex, DependencyLink, DocumentChunk

DEFINED_TERM_PATTERNS = [
    re.compile(r'["“]([^"”]{2,80})["”]\s+(?:means|shall mean|is defined as|refers to)\b', re.IGNORECASE),
    re.compile(r'\b([A-Z][A-Za-z0-9 \-]{2,80})\s+\(the\s+["“]([^"”]{2,80})["”]\)', re.IGNORECASE),
]
REFERENCE_PATTERNS = [
    ("defined_in", re.compile(r"\bas defined in\s+((?:section|clause|article|exhibit|schedule|addendum)\s+[A-Za-z0-9.\-]+)", re.IGNORECASE)),
    ("subject_to", re.compile(r"\bsubject to\s+((?:section|clause|article|exhibit|schedule|addendum)\s+[A-Za-z0-9.\-]+)", re.IGNORECASE)),
    ("pursuant_to", re.compile(r"\bpursuant to\s+((?:section|clause|article|exhibit|schedule|addendum)\s+[A-Za-z0-9.\-]+)", re.IGNORECASE)),
    ("see_reference", re.compile(r"\bsee\s+((?:section|clause|article|exhibit|schedule|addendum)\s+[A-Za-z0-9.\-]+)", re.IGNORECASE)),
    ("structural_reference", re.compile(r"\b(section|clause|article|exhibit|schedule|addendum)\s+([A-Za-z0-9.\-]+)", re.IGNORECASE)),
]


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _defined_terms(chunk: DocumentChunk) -> list[str]:
    terms: list[str] = []
    for pattern in DEFINED_TERM_PATTERNS:
        for match in pattern.finditer(chunk.text):
            term = match.group(match.lastindex or 1)
            if term:
                terms.append(_norm(term))
    return [t for t in dict.fromkeys(terms) if len(t) >= 2]


def _resolve_heading(chunks: list[DocumentChunk], label: str, source_chunk_id: str) -> DocumentChunk | None:
    normalized = _norm(label)
    if not normalized:
        return None
    best: DocumentChunk | None = None
    for chunk in chunks:
        if chunk.chunk_id == source_chunk_id:
            continue
        heading = _norm(chunk.section_heading or "")
        path = _norm(chunk.source_path)
        text_prefix = _norm(chunk.text[:300])
        if normalized and (normalized in heading or normalized in path or normalized in text_prefix):
            best = chunk
            break
    return best


def detect_dependencies(index: ContextIndex) -> list[DependencyLink]:
    """Detect obvious defined-term and document-reference dependencies."""

    chunks = index.chunks
    term_defs: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        for term in _defined_terms(chunk):
            term_defs[term].append(chunk)

    links: list[DependencyLink] = []
    seen: set[tuple[str, str | None, str, str]] = set()

    def add_link(source: DocumentChunk, target: DocumentChunk | None, relation: str, evidence: str) -> None:
        key = (source.chunk_id, target.chunk_id if target else None, relation, evidence.lower())
        if key in seen:
            return
        seen.add(key)
        links.append(
            DependencyLink(
                source_chunk_id=source.chunk_id,
                target_chunk_id=target.chunk_id if target else None,
                relation_type=relation,
                evidence=evidence.strip()[:160],
                source_document_id=source.document_id,
                target_document_id=target.document_id if target else None,
                resolved=target is not None,
                warning=None if target else f"Unresolved reference: {evidence.strip()[:80]}",
            )
        )

    for chunk in chunks:
        lower = _norm(chunk.text)
        for term, definitions in term_defs.items():
            if term in lower and all(d.chunk_id != chunk.chunk_id for d in definitions):
                add_link(chunk, definitions[0], "defined_term", term)
        for relation, pattern in REFERENCE_PATTERNS:
            for match in pattern.finditer(chunk.text):
                label = " ".join(g for g in match.groups() if g)
                target = _resolve_heading(chunks, label, chunk.chunk_id)
                add_link(chunk, target, relation, label)

    return links
