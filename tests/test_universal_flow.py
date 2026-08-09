from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from entroly.repository_intelligence import syntax_session as syntax
from entroly.repository_intelligence import universal_flow as flow


class _Node:
    def __init__(
        self,
        node_type: str,
        start: int,
        end: int,
        *,
        children: list["_Node"] | None = None,
        text_name: str = "",
    ) -> None:
        self.type = node_type
        self.start_byte = start
        self.end_byte = end
        self.start_point = (0, start)
        self.end_point = (0, end)
        self.named_children = children or []
        self.children = self.named_children
        self.is_error = False
        self._text_name = text_name

    def child_by_field_name(self, name: str):
        if name in {"function", "method", "name"} and self._text_name:
            return SimpleNamespace(
                start_byte=self.start_byte,
                end_byte=min(self.end_byte, self.start_byte + len(self._text_name)),
            )
        return None


class _Parser:
    def __init__(self, root: _Node, counter: list[int] | None = None) -> None:
        self.root = root
        self.counter = counter

    def parse(self, raw: bytes):
        if self.counter is not None:
            self.counter[0] += 1
        return SimpleNamespace(root_node=self.root)


def _scan(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    language: str,
    root: _Node,
    **kwargs: int,
):
    monkeypatch.setattr(syntax, "language_for_source", lambda *args: language)
    monkeypatch.setattr(syntax, "_get_local_parser", lambda selected: _Parser(root))
    session = syntax.build_syntax_session(source, f"main.{language}")
    assert session is not None
    return syntax.scan_syntax_session(session, **kwargs)


def test_shared_session_extracts_calls_and_control_shapes_in_one_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "if(x){foo(); while(y){bar();} return;}"
    raw = source.encode("utf-8")
    foo_start = source.index("foo")
    foo = _Node("call_expression", foo_start, foo_start + 5, text_name="foo")
    bar_start = source.index("bar")
    bar = _Node("call_expression", bar_start, bar_start + 5, text_name="bar")
    loop_start = source.index("while")
    loop = _Node("while_statement", loop_start, source.index(" return"), children=[bar])
    terminal_start = source.index("return")
    terminal = _Node("return_statement", terminal_start, terminal_start + len("return;"))
    branch = _Node("if_statement", 0, len(raw), children=[foo, loop, terminal])
    root = _Node("translation_unit", 0, len(raw), children=[branch])
    parses = [0]
    monkeypatch.setattr(syntax, "language_for_source", lambda *args: "c")
    monkeypatch.setattr(
        syntax,
        "_get_local_parser",
        lambda selected: _Parser(root, parses),
    )

    session = syntax.build_syntax_session(source, "main.c")
    assert session is not None
    scan = syntax.scan_syntax_session(session)
    assert parses == [1]
    assert [call.target for call in scan.calls] == ["foo", "bar"]
    assert [item.kind for item in scan.flow_shapes] == [
        "branch", "call", "loop", "call", "terminal"
    ]
    graph = flow.build_syntactic_flow_graph_from_scan(scan)
    assert graph.complete is True
    for node in graph.nodes:
        assert node.evidence_sha256 == hashlib.sha256(
            raw[node.start_byte:node.end_byte]
        ).hexdigest()
    payload = graph.to_dict()
    assert payload["analysis_contract"]["guarantee"] == (
        "parser-verified-control-shape-only"
    )
    assert "definition-use-binding" in payload["analysis_contract"]["not_claimed"]


def test_nested_flow_shape_edges_are_containment_not_cfg_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "if(x){while(y){foo();}}"
    raw = source.encode()
    call_start = source.index("foo")
    call = _Node("call_expression", call_start, call_start + 5, text_name="foo")
    loop_start = source.index("while")
    loop = _Node("while_expression", loop_start, len(raw) - 1, children=[call])
    branch = _Node("if_expression", 0, len(raw), children=[loop])
    root = _Node("source_file", 0, len(raw), children=[branch])
    scan = _scan(monkeypatch, source, "zig", root)

    graph = flow.build_syntactic_flow_graph_from_scan(scan)
    relations = {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges}
    loop_node = next(node for node in graph.nodes if node.kind == "loop")
    branch_node = next(node for node in graph.nodes if node.kind == "branch")
    call_node = next(node for node in graph.nodes if node.kind == "call")
    assert (branch_node.node_id, loop_node.node_id, "contains-flow-shape") in relations
    assert (loop_node.node_id, call_node.node_id, "contains-flow-shape") in relations
    assert all(edge.relation != "control-flow" for edge in graph.edges)


def test_traversal_limit_is_explicitly_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "if(x){foo();bar();baz();}"
    raw = source.encode()
    children = [
        _Node("call_expression", 6 + i * 2, 7 + i * 2, text_name="x")
        for i in range(3)
    ]
    branch = _Node("if_statement", 0, len(raw), children=children)
    root = _Node("translation_unit", 0, len(raw), children=[branch])
    scan = _scan(monkeypatch, source, "c", root, max_nodes=2)
    graph = flow.build_syntactic_flow_graph_from_scan(scan)
    assert graph.complete is False
    assert graph.nodes_visited == 2
    assert scan.calls_complete is False


def test_flow_budget_does_not_prevent_call_family_from_finishing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "if(x){foo();bar();}"
    raw = source.encode()
    first = source.index("foo")
    second = source.index("bar")
    calls = [
        _Node("call_expression", first, first + 5, text_name="foo"),
        _Node("call_expression", second, second + 5, text_name="bar"),
    ]
    branch = _Node("if_statement", 0, len(raw), children=calls)
    root = _Node("translation_unit", 0, len(raw), children=[branch])
    scan = _scan(monkeypatch, source, "c", root, max_flow_shapes=1)
    assert scan.flow_complete is False
    assert scan.calls_complete is True
    assert [call.target for call in scan.calls] == ["foo", "bar"]
    graph = flow.build_syntactic_flow_graph_from_scan(scan)
    assert graph.complete is False
    assert len(graph.nodes) == 1


def test_no_parser_means_no_synthetic_flow_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(flow, "build_syntax_session", lambda *args, **kwargs: None)
    assert flow.build_syntactic_flow_graph("thing { value }", "main.future") is None
