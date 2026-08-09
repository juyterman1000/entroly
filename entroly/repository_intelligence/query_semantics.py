"""Language-neutral lexical semantics from standard Tree-sitter queries.

Tree-sitter defines fixed locals-query captures (``local.scope``,
``local.definition``, ``local.reference``) and a standard tags vocabulary
(``definition.*`` / ``reference.*`` with ``name``).  When a grammar bundles
those queries, Entroly can extract exact lexical facts from the already-parsed
syntax tree without another parse.

This is intentionally narrower than compiler semantics.  A reference is bound
only when a unique prior same-name definition exists in the nearest enclosing
captured scope.  Tags enrich roles/kinds but never create semantic bindings.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .syntax_session import SyntaxSession

QUERY_SEMANTICS_SCHEMA_VERSION = "entroly.query-semantics.v1"


@dataclass(frozen=True)
class QueryFact:
    fact_id: str
    role: str
    kind: str
    name: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    evidence_sha256: str
    scope_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "role": self.role,
            "kind": self.kind,
            "name": self.name,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "evidence_sha256": self.evidence_sha256,
            "scope_id": self.scope_id,
            "epistemic_class": "parser-query-verified",
        }


@dataclass(frozen=True)
class LexicalBinding:
    definition_id: str
    reference_id: str
    name: str
    scope_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "definition_id": self.definition_id,
            "reference_id": self.reference_id,
            "name": self.name,
            "scope_id": self.scope_id,
            "resolution": "unique-prior-definition-in-nearest-scope",
            "epistemic_class": "lexical-query-resolved",
        }


@dataclass(frozen=True)
class UnresolvedLexicalReference:
    reference_id: str
    name: str
    reason: str
    candidates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_id": self.reference_id,
            "name": self.name,
            "reason": self.reason,
            "candidates": list(self.candidates),
        }


@dataclass(frozen=True)
class QuerySemanticGraph:
    language: str
    source_sha256: str
    scopes: tuple[QueryFact, ...]
    definitions: tuple[QueryFact, ...]
    references: tuple[QueryFact, ...]
    tag_facts: tuple[QueryFact, ...]
    bindings: tuple[LexicalBinding, ...]
    unresolved: tuple[UnresolvedLexicalReference, ...]
    locals_available: bool
    tags_available: bool
    complete: bool
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": QUERY_SEMANTICS_SCHEMA_VERSION,
            "language": self.language,
            "source_sha256": self.source_sha256,
            "scopes": [item.to_dict() for item in self.scopes],
            "definitions": [item.to_dict() for item in self.definitions],
            "references": [item.to_dict() for item in self.references],
            "tag_facts": [item.to_dict() for item in self.tag_facts],
            "bindings": [item.to_dict() for item in self.bindings],
            "unresolved": [item.to_dict() for item in self.unresolved],
            "locals_available": self.locals_available,
            "tags_available": self.tags_available,
            "complete": self.complete,
            "diagnostics": list(self.diagnostics),
            "analysis_contract": {
                "binding_guarantee": "lexical-query-only",
                "binding_rule": "unique-prior-definition-in-nearest-enclosing-scope",
                "tags_may_create_bindings": False,
                "not_claimed": [
                    "type-resolution",
                    "overload-resolution",
                    "dynamic-dispatch",
                    "macro-expansion",
                    "module-or-package-binding",
                    "alias-or-heap-flow",
                    "path-feasibility",
                ],
            },
        }


def _text(raw: bytes, node: object) -> str:
    try:
        start = int(getattr(node, "start_byte"))
        end = int(getattr(node, "end_byte"))
    except (AttributeError, TypeError, ValueError):
        return ""
    if not (0 <= start < end <= len(raw)):
        return ""
    return raw[start:end].decode("utf-8", errors="surrogateescape").strip()


def _fact_id(role: str, kind: str, name: str, start: int, end: int) -> str:
    material = f"{role}\0{kind}\0{name}\0{start}\0{end}".encode("utf-8")
    return "query-fact:" + hashlib.sha256(material).hexdigest()[:24]


def _fact(
    raw: bytes,
    node: object,
    *,
    role: str,
    kind: str,
    name: str | None = None,
    scope_id: str | None = None,
) -> QueryFact | None:
    try:
        start = int(getattr(node, "start_byte"))
        end = int(getattr(node, "end_byte"))
        start_point = getattr(node, "start_point")
        end_point = getattr(node, "end_point")
    except (AttributeError, TypeError, ValueError):
        return None
    if not (0 <= start < end <= len(raw)):
        return None
    value = (name if name is not None else _text(raw, node)).strip()
    if not value or len(value) > 1_000 or "\n" in value:
        return None
    return QueryFact(
        fact_id=_fact_id(role, kind, value, start, end),
        role=role,
        kind=kind,
        name=value,
        start_byte=start,
        end_byte=end,
        start_line=int(start_point[0]) + 1,
        end_line=int(end_point[0]) + 1,
        evidence_sha256=hashlib.sha256(raw[start:end]).hexdigest(),
        scope_id=scope_id,
    )


def _query_results(
    language_object: object,
    query_source: str,
    root_node: object,
) -> tuple[dict[str, list[object]], list[tuple[int, dict[str, list[object]]]], bool] | None:
    """Execute a query across py-tree-sitter 0.23-0.26 APIs."""
    try:
        from tree_sitter import Query
        query = Query(language_object, query_source)
    except Exception:
        return None

    # py-tree-sitter >=0.25 moved execution from Query to QueryCursor.
    try:
        from tree_sitter import QueryCursor
        cursor = QueryCursor(query)
        captures = cursor.captures(root_node)
        matches = cursor.matches(root_node)
        exceeded = bool(getattr(cursor, "did_exceed_match_limit", False))
        return dict(captures), list(matches), not exceeded
    except (ImportError, TypeError, AttributeError):
        pass
    try:
        captures = query.captures(root_node)
        matches = query.matches(root_node)
        exceeded = bool(getattr(query, "did_exceed_match_limit", False))
        return dict(captures), list(matches), not exceeded
    except Exception:
        return None


def _scope_for_position(scopes: Iterable[QueryFact], start: int, end: int) -> QueryFact | None:
    containing = [
        scope
        for scope in scopes
        if scope.start_byte <= start and end <= scope.end_byte
    ]
    if not containing:
        return None
    return min(
        containing,
        key=lambda item: (item.end_byte - item.start_byte, -item.start_byte, item.fact_id),
    )


def resolve_lexical_bindings(
    scopes: tuple[QueryFact, ...],
    definitions: tuple[QueryFact, ...],
    references: tuple[QueryFact, ...],
) -> tuple[tuple[LexicalBinding, ...], tuple[UnresolvedLexicalReference, ...]]:
    """Resolve only unique prior definitions in the nearest lexical scope."""
    bindings: list[LexicalBinding] = []
    unresolved: list[UnresolvedLexicalReference] = []
    by_scope: dict[str, list[QueryFact]] = {}
    scope_map = {scope.fact_id: scope for scope in scopes}
    root_scope = "scope:root"
    for definition in definitions:
        by_scope.setdefault(definition.scope_id or root_scope, []).append(definition)

    for reference in references:
        containing = sorted(
            (
                scope
                for scope in scopes
                if scope.start_byte <= reference.start_byte
                and reference.end_byte <= scope.end_byte
            ),
            key=lambda item: (item.end_byte - item.start_byte, -item.start_byte, item.fact_id),
        )
        search_scopes = [scope.fact_id for scope in containing]
        search_scopes.append(root_scope)
        resolved = False
        for scope_id in search_scopes:
            candidates = [
                item
                for item in by_scope.get(scope_id, ())
                if item.name == reference.name and item.start_byte < reference.start_byte
            ]
            if not candidates:
                continue
            if len(candidates) == 1:
                definition = candidates[0]
                bindings.append(LexicalBinding(
                    definition_id=definition.fact_id,
                    reference_id=reference.fact_id,
                    name=reference.name,
                    scope_id=scope_id,
                ))
            else:
                unresolved.append(UnresolvedLexicalReference(
                    reference_id=reference.fact_id,
                    name=reference.name,
                    reason="ambiguous-prior-definitions",
                    candidates=tuple(sorted(item.fact_id for item in candidates)),
                ))
            resolved = True
            break
        if not resolved:
            # Later definitions are not treated as hoisted because that is a
            # language semantic claim. They remain visible as candidates only.
            later = [
                item.fact_id
                for item in definitions
                if item.name == reference.name and item.start_byte >= reference.start_byte
            ]
            unresolved.append(UnresolvedLexicalReference(
                reference_id=reference.fact_id,
                name=reference.name,
                reason="no-unique-prior-definition",
                candidates=tuple(sorted(later)[:100]),
            ))
    return (
        tuple(sorted(bindings, key=lambda item: (item.reference_id, item.definition_id))),
        tuple(sorted(unresolved, key=lambda item: (item.reference_id, item.reason, item.candidates))),
    )


def _locals_facts(
    raw: bytes,
    captures: Mapping[str, list[object]],
    *,
    max_facts: int,
) -> tuple[tuple[QueryFact, ...], tuple[QueryFact, ...], tuple[QueryFact, ...], bool]:
    ignored = {
        (int(getattr(node, "start_byte", -1)), int(getattr(node, "end_byte", -1)))
        for node in captures.get("ignore", ())
    }
    scopes: list[QueryFact] = []
    for node in captures.get("local.scope", ()):
        fact = _fact(raw, node, role="scope", kind="local")
        if fact is not None:
            scopes.append(fact)
    scopes.sort(key=lambda item: (item.start_byte, -item.end_byte, item.fact_id))

    definitions: list[QueryFact] = []
    references: list[QueryFact] = []
    complete = True
    for capture_name, role, target in (
        ("local.definition", "definition", definitions),
        ("local.reference", "reference", references),
    ):
        for node in captures.get(capture_name, ()):
            span = (
                int(getattr(node, "start_byte", -1)),
                int(getattr(node, "end_byte", -1)),
            )
            if span in ignored:
                continue
            if len(definitions) + len(references) >= max_facts:
                complete = False
                break
            scope = _scope_for_position(scopes, span[0], span[1])
            fact = _fact(
                raw,
                node,
                role=role,
                kind="local",
                scope_id=scope.fact_id if scope else None,
            )
            if fact is not None:
                target.append(fact)
        if not complete:
            break
    return (
        tuple(scopes[:max_facts]),
        tuple(sorted(definitions, key=lambda item: (item.start_byte, item.fact_id))),
        tuple(sorted(references, key=lambda item: (item.start_byte, item.fact_id))),
        complete and len(scopes) <= max_facts,
    )


def _tag_facts(
    raw: bytes,
    matches: list[tuple[int, dict[str, list[object]]]],
    *,
    max_facts: int,
) -> tuple[tuple[QueryFact, ...], bool]:
    facts: list[QueryFact] = []
    complete = True
    seen: set[str] = set()
    for _pattern, captures in matches:
        names = list(captures.get("name", ()))
        role_captures = [
            (capture, node)
            for capture, nodes in captures.items()
            if capture.startswith(("definition.", "reference."))
            for node in nodes
        ]
        for capture, entity in role_captures:
            if len(facts) >= max_facts:
                complete = False
                break
            try:
                entity_start = int(getattr(entity, "start_byte"))
                entity_end = int(getattr(entity, "end_byte"))
            except (AttributeError, TypeError, ValueError):
                continue
            contained_names = [
                node
                for node in names
                if entity_start <= int(getattr(node, "start_byte", -1))
                and int(getattr(node, "end_byte", -1)) <= entity_end
            ]
            for name_node in contained_names[:10]:
                role, _, kind = capture.partition(".")
                fact = _fact(raw, name_node, role=role, kind=kind or "unknown")
                if fact is not None and fact.fact_id not in seen:
                    seen.add(fact.fact_id)
                    facts.append(fact)
        if not complete:
            break
    return (
        tuple(sorted(facts, key=lambda item: (item.start_byte, item.role, item.kind, item.fact_id))),
        complete,
    )


def build_query_semantic_graph(
    session: SyntaxSession,
    *,
    max_facts: int = 50_000,
) -> QuerySemanticGraph | None:
    """Use bundled locals/tags queries against an existing syntax session."""
    try:
        import tree_sitter_language_pack as pack
    except (ImportError, OSError):
        return None
    get_language = getattr(pack, "get_language", None)
    if not callable(get_language):
        return None
    try:
        language_object = get_language(session.language)
    except Exception:
        return None

    fact_limit = max(1, min(int(max_facts), 500_000))
    diagnostics: list[str] = []
    locals_source = None
    tags_source = None
    get_locals = getattr(pack, "get_locals_query", None)
    get_tags = getattr(pack, "get_tags_query", None)
    if callable(get_locals):
        try:
            locals_source = get_locals(session.language)
        except Exception:
            diagnostics.append("locals-query-load-failed")
    if callable(get_tags):
        try:
            tags_source = get_tags(session.language)
        except Exception:
            diagnostics.append("tags-query-load-failed")

    scopes: tuple[QueryFact, ...] = ()
    definitions: tuple[QueryFact, ...] = ()
    references: tuple[QueryFact, ...] = ()
    tag_facts: tuple[QueryFact, ...] = ()
    complete = True
    locals_available = bool(locals_source)
    tags_available = bool(tags_source)

    if locals_source:
        result = _query_results(language_object, str(locals_source), session.tree.root_node)
        if result is None:
            complete = False
            diagnostics.append("locals-query-execution-failed")
        else:
            captures, _matches, query_complete = result
            scopes, definitions, references, extraction_complete = _locals_facts(
                session.raw, captures, max_facts=fact_limit
            )
            complete = complete and query_complete and extraction_complete

    if tags_source:
        result = _query_results(language_object, str(tags_source), session.tree.root_node)
        if result is None:
            complete = False
            diagnostics.append("tags-query-execution-failed")
        else:
            _captures, matches, query_complete = result
            tag_facts, extraction_complete = _tag_facts(
                session.raw, matches, max_facts=fact_limit
            )
            complete = complete and query_complete and extraction_complete

    if not locals_available and not tags_available:
        return None

    bindings, unresolved = resolve_lexical_bindings(scopes, definitions, references)
    return QuerySemanticGraph(
        language=session.language,
        source_sha256=session.source_sha256,
        scopes=scopes,
        definitions=definitions,
        references=references,
        tag_facts=tag_facts,
        bindings=bindings,
        unresolved=unresolved,
        locals_available=locals_available,
        tags_available=tags_available,
        complete=complete,
        diagnostics=tuple(sorted(dict.fromkeys(diagnostics))),
    )


__all__ = [
    "QUERY_SEMANTICS_SCHEMA_VERSION",
    "LexicalBinding",
    "QueryFact",
    "QuerySemanticGraph",
    "UnresolvedLexicalReference",
    "build_query_semantic_graph",
    "resolve_lexical_bindings",
]
