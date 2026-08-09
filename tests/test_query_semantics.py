from __future__ import annotations

import hashlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from entroly.repository_intelligence import query_semantics as semantics
from entroly.repository_intelligence.query_semantics import (
    QueryFact,
    resolve_lexical_bindings,
)
from entroly.repository_intelligence.syntax_session import SyntaxSession


def _fact(
    fact_id: str,
    role: str,
    name: str,
    start: int,
    end: int,
    *,
    scope_id: str | None = None,
) -> QueryFact:
    return QueryFact(
        fact_id=fact_id,
        role=role,
        kind="local",
        name=name,
        start_byte=start,
        end_byte=end,
        start_line=1,
        end_line=1,
        evidence_sha256="0" * 64,
        scope_id=scope_id,
    )


def test_nearer_later_definition_blocks_unsafe_outer_binding() -> None:
    outer = _fact("scope:outer", "scope", "scope", 0, 200)
    inner = _fact("scope:inner", "scope", "scope", 10, 100)
    outer_def = _fact("def:outer", "definition", "x", 2, 3, scope_id=outer.fact_id)
    inner_later = _fact("def:inner", "definition", "x", 50, 51, scope_id=inner.fact_id)
    reference = _fact("ref", "reference", "x", 20, 21, scope_id=inner.fact_id)

    bindings, unresolved = resolve_lexical_bindings(
        (outer, inner),
        (outer_def, inner_later),
        (reference,),
    )
    assert bindings == ()
    assert unresolved[0].reason == "nearer-scope-definition-not-prior"
    assert unresolved[0].candidates == ("def:inner",)


def test_unique_prior_definition_in_nearest_scope_binds() -> None:
    outer = _fact("scope:outer", "scope", "scope", 0, 200)
    inner = _fact("scope:inner", "scope", "scope", 10, 100)
    outer_def = _fact("def:outer", "definition", "x", 2, 3, scope_id=outer.fact_id)
    inner_def = _fact("def:inner", "definition", "x", 15, 16, scope_id=inner.fact_id)
    reference = _fact("ref", "reference", "x", 20, 21, scope_id=inner.fact_id)

    bindings, unresolved = resolve_lexical_bindings(
        (outer, inner),
        (outer_def, inner_def),
        (reference,),
    )
    assert unresolved == ()
    assert len(bindings) == 1
    assert bindings[0].definition_id == "def:inner"
    assert bindings[0].scope_id == "scope:inner"


def test_multiple_prior_definitions_stay_ambiguous() -> None:
    scope = _fact("scope", "scope", "scope", 0, 100)
    first = _fact("def:1", "definition", "x", 10, 11, scope_id=scope.fact_id)
    second = _fact("def:2", "definition", "x", 15, 16, scope_id=scope.fact_id)
    reference = _fact("ref", "reference", "x", 20, 21, scope_id=scope.fact_id)

    bindings, unresolved = resolve_lexical_bindings(
        (scope,),
        (first, second),
        (reference,),
    )
    assert bindings == ()
    assert unresolved[0].reason == "ambiguous-prior-definitions"
    assert unresolved[0].candidates == ("def:1", "def:2")


class _Node:
    def __init__(self, start: int, end: int) -> None:
        self.start_byte = start
        self.end_byte = end
        self.start_point = (0, start)
        self.end_point = (1, end)


def test_multiline_scope_capture_uses_synthetic_identity() -> None:
    raw = b"fn f() {\n  x\n}\n"
    scope = _Node(0, len(raw))
    definition = _Node(11, 12)
    captures = {
        "local.scope": [scope],
        "local.definition": [definition],
        "local.reference": [],
    }
    scopes, definitions, references, complete = semantics._locals_facts(
        raw,
        captures,
        max_facts=100,
    )
    assert complete
    assert len(scopes) == 1
    assert scopes[0].name == "scope"
    assert definitions[0].scope_id == scopes[0].fact_id
    assert references == ()


def test_tags_enrich_roles_but_do_not_bind_without_locals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "fn run() {}\n"
    raw = source.encode()
    name = _Node(3, 6)
    entity = _Node(0, len(raw) - 1)
    session = SyntaxSession(
        file_path="main.zig",
        language="zig",
        raw=raw,
        tree=SimpleNamespace(root_node=entity),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        syntax_valid=True,
    )
    fake_pack = SimpleNamespace(
        get_language=lambda language: object(),
        get_locals_query=lambda language: None,
        get_tags_query=lambda language: "tags",
    )
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", fake_pack)
    monkeypatch.setattr(
        semantics,
        "_query_results",
        lambda language, query, root: (
            {},
            [(0, {"name": [name], "definition.function": [entity]})],
            True,
        ),
    )

    graph = semantics.build_query_semantic_graph(session)
    assert graph is not None
    assert graph.locals_available is False
    assert graph.tags_available is True
    assert [(item.role, item.kind, item.name) for item in graph.tag_facts] == [
        ("definition", "function", "run")
    ]
    assert graph.bindings == ()


def test_query_cursor_api_is_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("tree_sitter")

    class Query:
        def __init__(self, language, source) -> None:
            self.language = language
            self.source = source

    class QueryCursor:
        did_exceed_match_limit = False

        def __init__(self, query) -> None:
            self.query = query

        def captures(self, root):
            return {"local.reference": [root]}

        def matches(self, root):
            return [(0, {"local.reference": [root]})]

    module.Query = Query
    module.QueryCursor = QueryCursor
    monkeypatch.setitem(sys.modules, "tree_sitter", module)
    root = object()
    result = semantics._query_results(object(), "(x) @local.reference", root)
    assert result is not None
    captures, matches, complete = result
    assert captures["local.reference"] == [root]
    assert len(matches) == 1
    assert complete


def test_legacy_query_api_is_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("tree_sitter")

    class Query:
        did_exceed_match_limit = False

        def __init__(self, language, source) -> None:
            pass

        def captures(self, root):
            return {"local.definition": [root]}

        def matches(self, root):
            return [(0, {"local.definition": [root]})]

    module.Query = Query
    monkeypatch.setitem(sys.modules, "tree_sitter", module)
    root = object()
    result = semantics._query_results(object(), "(x) @local.definition", root)
    assert result is not None
    captures, matches, complete = result
    assert captures["local.definition"] == [root]
    assert len(matches) == 1
    assert complete
