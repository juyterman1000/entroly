from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from entroly.repository_intelligence import universal_flow as flow


class _Node:
    def __init__(
        self,
        node_type: str,
        start: int,
        end: int,
        *,
        children: list["_Node"] | None = None,
    ) -> None:
        self.type = node_type
        self.start_byte = start
        self.end_byte = end
        self.start_point = (0, start)
        self.end_point = (0, end)
        self.named_children = children or []
        self.children = self.named_children
        self.is_error = False


class _Parser:
    def __init__(self, root: _Node) -> None:
        self.root = root

    def parse(self, raw: bytes):
        return SimpleNamespace(root_node=self.root)


def test_c_to_zig_style_control_shapes_are_parser_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "if(x){foo(); while(y){bar();} return;}"
    raw = source.encode("utf-8")
    call = _Node("call_expression", 6, 11)
    nested_call = _Node("call_expression", 22, 27)
    loop = _Node("while_statement", 13, 29, children=[nested_call])
    terminal = _Node("return_statement", 30, 37)
    branch = _Node("if_statement", 0, len(raw), children=[call, loop, terminal])
    root = _Node("translation_unit", 0, len(raw), children=[branch])

    monkeypatch.setattr(flow, "language_for_source", lambda *args: "c")
    monkeypatch.setattr(flow, "_get_local_parser", lambda language: _Parser(root))
    graph = flow.build_syntactic_flow_graph(source, "main.c")
    assert graph is not None
    assert graph.complete is True
    assert [node.kind for node in graph.nodes] == ["branch", "call", "loop", "call", "terminal"]
    assert all(node.evidence_sha256 for node in graph.nodes)
    for node in graph.nodes:
        assert node.evidence_sha256 == hashlib.sha256(
            raw[node.start_byte:node.end_byte]
        ).hexdigest()
    payload = graph.to_dict()
    assert payload["analysis_contract"]["guarantee"] == "parser-verified-control-shape-only"
    assert "definition-use-binding" in payload["analysis_contract"]["not_claimed"]


def test_nested_flow_shape_edges_are_containment_not_cfg_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "if(x){while(y){foo();}}"
    raw = source.encode()
    call = _Node("call_expression", 15, 20)
    loop = _Node("while_expression", 6, 22, children=[call])
    branch = _Node("if_expression", 0, len(raw), children=[loop])
    root = _Node("source_file", 0, len(raw), children=[branch])
    monkeypatch.setattr(flow, "language_for_source", lambda *args: "zig")
    monkeypatch.setattr(flow, "_get_local_parser", lambda language: _Parser(root))

    graph = flow.build_syntactic_flow_graph(source, "main.zig")
    assert graph is not None
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
    children = [_Node("call_expression", 6 + i * 2, 7 + i * 2) for i in range(3)]
    branch = _Node("if_statement", 0, len(raw), children=children)
    root = _Node("translation_unit", 0, len(raw), children=[branch])
    monkeypatch.setattr(flow, "language_for_source", lambda *args: "c")
    monkeypatch.setattr(flow, "_get_local_parser", lambda language: _Parser(root))

    graph = flow.build_syntactic_flow_graph(source, "main.c", max_nodes=2)
    assert graph is not None
    assert graph.complete is False
    assert graph.nodes_visited == 2


def test_flow_node_budget_is_explicitly_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "if(x){foo();bar();}"
    raw = source.encode()
    calls = [_Node("call_expression", 6, 11), _Node("call_expression", 12, 17)]
    branch = _Node("if_statement", 0, len(raw), children=calls)
    root = _Node("translation_unit", 0, len(raw), children=[branch])
    monkeypatch.setattr(flow, "language_for_source", lambda *args: "c")
    monkeypatch.setattr(flow, "_get_local_parser", lambda language: _Parser(root))

    graph = flow.build_syntactic_flow_graph(source, "main.c", max_flow_nodes=1)
    assert graph is not None
    assert graph.complete is False
    assert len(graph.nodes) == 1


def test_no_parser_means_no_synthetic_flow_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(flow, "language_for_source", lambda *args: "futurelang")
    monkeypatch.setattr(flow, "_get_local_parser", lambda language: None)
    assert flow.build_syntactic_flow_graph("thing { value }", "main.future") is None
