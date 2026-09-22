"""Lexical reference coverage against caller-supplied context inventories.

A mention is not evidence for the assertion surrounding it. This checker
cannot infer what a model saw, detect all entities/paraphrases, establish
truth, or establish honesty. Missing references remain missing even if
neither inventory contains them. Inventories must describe the actual run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class UncoveredReference:
    """An answer claim about an entity whose evidence was omitted."""

    entity: str
    assertion_span: str
    omitted_source: str
    omitted_chunk_id: str


@dataclass(frozen=True)
class EvidenceBoundaryReceipt:
    """Reference presence only. ``honest`` is a legacy coverage alias.

    None means no extractable references, not a verified answer.
    """

    honest: bool | None
    entities_referenced: int
    entities_sourced: int
    entities_unsourced: int
    unsourced: tuple[UncoveredReference, ...]

    @property
    def coverage_ratio(self) -> float:
        if self.entities_referenced == 0:
            return 0.0
        return self.entities_sourced / self.entities_referenced

    def to_dict(self) -> dict[str, Any]:
        return {
            "honest": self.honest,
            "scope": "lexical reference coverage only",
            "claims_verified": False,
            "entities_referenced": self.entities_referenced,
            "entities_sourced": self.entities_sourced,
            "entities_unsourced": self.entities_unsourced,
            "coverage_ratio": round(self.coverage_ratio, 4),
            "unsourced": [
                {
                    "entity": ua.entity,
                    "assertion_span": ua.assertion_span,
                    "omitted_source": ua.omitted_source,
                    "omitted_chunk_id": ua.omitted_chunk_id,
                }
                for ua in self.unsourced
            ],
        }


_PATH_RE = re.compile(r"(?:[\w.-]+/)+[\w.-]+\.[\w]+")
_SYMBOL_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
_FUNC_RE = re.compile(r"\b[a-z_]\w*\(\)")
_ENTITY_RE = re.compile(
    r"`([^`]+)`"
    r"|(?:[\w.-]+/)+[\w.-]+\.[\w]+"
    r"|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b"
    r"|\b[a-z_]\w*\(\)"
)


def _extract_entities(text: str) -> set[str]:
    """Extract referenced entities: file paths, class names, function calls, backtick refs."""
    entities: set[str] = set()
    for m in _ENTITY_RE.finditer(text):
        entity = m.group(1) or m.group(0)
        entity = entity.strip()
        if len(entity) >= 3:
            entities.add(entity)
    return entities


def _source_key(path: str) -> str:
    """Normalise a source path for matching."""
    return path.replace("\\", "/").strip()


def audit_evidence_boundary(
    answer: str,
    selected_sources: Sequence[dict[str, Any]],
    omitted_sources: Sequence[dict[str, Any]],
) -> EvidenceBoundaryReceipt:
    """Check if every entity in the answer had evidence in selected context.

    ``selected_sources`` and ``omitted_sources`` are lists of dicts with
    at minimum ``source_path`` (or ``source``) and ``chunk_id`` (or
    ``id``).  Optionally ``text`` or ``text_preview`` for content
    matching.

    "Sourced" means a lexical mention, not support for a factual claim.
    Every missing reference is unsourced; omitted-source fields are empty
    when neither inventory contains it.
    """
    entities = _extract_entities(answer)
    if not entities:
        return EvidenceBoundaryReceipt(
            honest=None,
            entities_referenced=0,
            entities_sourced=0,
            entities_unsourced=0,
            unsourced=(),
        )

    # Build coverage maps
    def _path(item: dict[str, Any]) -> str:
        return _source_key(
            str(item.get("source_path") or item.get("source") or "")
        )

    def _content(item: dict[str, Any]) -> str:
        return str(
            item.get("text") or item.get("text_preview") or item.get("content") or ""
        )

    def _chunk_id(item: dict[str, Any]) -> str:
        return str(item.get("chunk_id") or item.get("id") or "")

    def _mentions(entity: str, source: dict[str, Any]) -> bool:
        # Whole identifiers, per source: Foo must not match FooBar, and
        # fragments must not manufacture a match when concatenated.
        boundary = r"[\w./\\-]" if "/" in entity or "\\" in entity else r"[\w]"
        pattern = r"(?<!" + boundary + ")" + re.escape(entity) + r"(?!" + boundary + ")"
        return (_source_key(entity) == _path(source) or
                re.search(pattern, _content(source)) is not None)

    def _entity_in_selected(entity: str) -> bool:
        return any(_mentions(entity, source) for source in selected_sources)

    unsourced: list[UncoveredReference] = []
    sourced_count = 0

    for entity in sorted(entities):
        if _entity_in_selected(entity):
            sourced_count += 1
            continue

        # Check if entity appears in omitted context
        el = entity.lower()
        found_in_omitted = False
        for omit in omitted_sources:
            if _mentions(entity, omit):
                # Find the assertion span in the answer
                idx = answer.lower().find(el)
                span_start = max(0, idx - 30)
                span_end = min(len(answer), idx + len(entity) + 30)
                span = answer[span_start:span_end]

                unsourced.append(UncoveredReference(
                    entity=entity,
                    assertion_span=span,
                    omitted_source=str(
                        omit.get("source_path") or omit.get("source") or ""
                    ),
                    omitted_chunk_id=_chunk_id(omit),
                ))
                found_in_omitted = True
                break

        if not found_in_omitted:
            unsourced.append(UncoveredReference(
                entity=entity, assertion_span=entity,
                omitted_source='', omitted_chunk_id='',
            ))

    return EvidenceBoundaryReceipt(
        honest=len(unsourced) == 0,
        entities_referenced=len(entities),
        entities_sourced=sourced_count,
        entities_unsourced=len(unsourced),
        unsourced=tuple(unsourced),
    )
