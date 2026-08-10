"""Source-verified interprocedural value-flow summaries for Python calls.

This module deliberately models only relationships that can be tied to a
uniquely resolved static call and exact source spans. It is not a whole-program
alias, heap, exception, or path-sensitive analysis.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .models import CallEdge, RepositoryIndex, Symbol, UnresolvedCall

INTERPROCEDURAL_FLOW_SCHEMA_VERSION = "entroly.verified-interprocedural-flow.v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


@dataclass(frozen=True)
class _ParsedSource:
    raw: bytes
    text: str
    tree: ast.Module
    offsets: tuple[int, ...]
    source_sha256: str
    parents: Mapping[ast.AST, ast.AST]

    def span(self, node: ast.AST) -> tuple[int, int]:
        line = max(1, int(getattr(node, "lineno", 1)))
        end_line = max(line, int(getattr(node, "end_lineno", line)))
        start = self.offsets[min(line - 1, len(self.offsets) - 1)]
        start += max(0, int(getattr(node, "col_offset", 0)))
        end = self.offsets[min(end_line - 1, len(self.offsets) - 1)]
        end += max(0, int(getattr(node, "end_col_offset", 0)))
        return min(start, len(self.raw)), min(max(start, end), len(self.raw))


class _ReturnVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.values: list[ast.Return] = []

    def visit_Return(self, node: ast.Return) -> None:
        self.values.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


class _SourceCache:
    def __init__(self, root: Path, index: RepositoryIndex) -> None:
        self.root = root
        self.index = index
        self.values: dict[str, _ParsedSource | None] = {}
        self.status: dict[str, str] = {}

    def get(self, path: str) -> _ParsedSource | None:
        if path in self.values:
            return self.values[path]
        record = self.index.files.get(path)
        if record is None:
            self.values[path] = None
            self.status[path] = "not-indexed"
            return None
        try:
            candidate = (self.root / path).resolve(strict=True)
            candidate.relative_to(self.root)
            raw = candidate.read_bytes()
        except (OSError, RuntimeError, ValueError):
            self.values[path] = None
            self.status[path] = "unsafe-or-unreadable"
            return None
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record.sha256:
            self.values[path] = None
            self.status[path] = "stale-index"
            return None
        try:
            text = raw.decode("utf-8", errors="surrogateescape")
            tree = ast.parse(text, filename=path, type_comments=True)
        except (SyntaxError, ValueError, MemoryError, RecursionError, UnicodeError):
            self.values[path] = None
            self.status[path] = "parse-failed"
            return None
        offsets = [0]
        running = 0
        for line in text.splitlines(keepends=True):
            running += len(line.encode("utf-8", errors="surrogateescape"))
            offsets.append(running)
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        parsed = _ParsedSource(
            raw=raw,
            text=text,
            tree=tree,
            offsets=tuple(offsets),
            source_sha256=digest,
            parents=parents,
        )
        self.values[path] = parsed
        self.status[path] = "verified"
        return parsed


def _resolve(index: RepositoryIndex, query: str) -> list[Symbol]:
    normalized = query.casefold()
    return sorted(
        (
            symbol
            for symbol in index.symbols.values()
            if normalized
            in {
                symbol.symbol_id.casefold(),
                symbol.qualified_name.casefold(),
                symbol.name.casefold(),
            }
        ),
        key=lambda item: item.symbol_id,
    )


def _function_node(parsed: _ParsedSource, symbol: Symbol) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(parsed.tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != symbol.name:
            continue
        if parsed.span(node) == (symbol.start_byte, symbol.end_byte):
            return node
    return None


def _call_node(parsed: _ParsedSource, edge: CallEdge) -> ast.Call | None:
    for node in ast.walk(parsed.tree):
        if isinstance(node, ast.Call) and parsed.span(node) == (
            edge.start_byte,
            edge.end_byte,
        ):
            return node
    return None


def _evidence_node(
    nodes: dict[str, dict[str, object]],
    parsed: _ParsedSource,
    path: str,
    node: ast.AST,
    role: str,
    *,
    symbol_id: str,
) -> str:
    start, end = parsed.span(node)
    evidence = parsed.raw[start:end]
    node_id = "flow:" + hashlib.sha256(
        f"{path}\0{start}\0{end}\0{role}\0{symbol_id}".encode("utf-8")
    ).hexdigest()
    nodes.setdefault(
        node_id,
        {
            "node_id": node_id,
            "symbol_id": symbol_id,
            "path": path,
            "role": role,
            "line_start": max(1, int(getattr(node, "lineno", 1))),
            "line_end": max(
                1,
                int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
            ),
            "start_byte": start,
            "end_byte": end,
            "source_sha256": parsed.source_sha256,
            "evidence_sha256": hashlib.sha256(evidence).hexdigest(),
            "text": evidence.decode("utf-8", errors="surrogateescape"),
            "trust": "verified-source-span",
        },
    )
    return node_id


def _parameters(function: ast.FunctionDef | ast.AsyncFunctionDef, symbol: Symbol) -> tuple[list[ast.arg], ast.arg | None, ast.arg | None]:
    positional = [*function.args.posonlyargs, *function.args.args]
    if (
        symbol.kind == "method"
        and positional
        and positional[0].arg in {"self", "cls"}
    ):
        positional = positional[1:]
    return positional, function.args.vararg, function.args.kwarg


def _argument_bindings(
    call: ast.Call,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    symbol: Symbol,
) -> Iterable[tuple[ast.AST, ast.arg, str, int | None, str | None]]:
    all_positional = [*function.args.posonlyargs, *function.args.args]
    implicit_receiver = (
        all_positional[0]
        if symbol.kind == "method"
        and all_positional
        and all_positional[0].arg in {"self", "cls"}
        else None
    )
    positional, vararg, kwarg = _parameters(function, symbol)
    named = {
        argument.arg: argument
        for argument in [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        ]
        if argument is not implicit_receiver
    }
    for position, argument in enumerate(call.args):
        if isinstance(argument, ast.Starred):
            if vararg is not None:
                yield (
                    argument.value,
                    vararg,
                    "starred-variadic-positional",
                    position,
                    None,
                )
            continue
        value = argument
        if position < len(positional):
            yield value, positional[position], "positional", position, None
        elif vararg is not None:
            yield value, vararg, "variadic-positional", position, None
    for keyword in call.keywords:
        if keyword.arg is not None and keyword.arg in named:
            yield keyword.value, named[keyword.arg], "keyword", None, keyword.arg
        elif kwarg is not None:
            yield keyword.value, kwarg, "variadic-keyword", None, keyword.arg


def _return_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Return]:
    visitor = _ReturnVisitor()
    for statement in function.body:
        visitor.visit(statement)
    return visitor.values


def _consumer_nodes(parsed: _ParsedSource, call: ast.Call) -> list[tuple[ast.AST, str]]:
    current: ast.AST = call
    parent = parsed.parents.get(current)
    if isinstance(parent, ast.Await):
        current = parent
        parent = parsed.parents.get(current)
    if isinstance(parent, ast.Assign) and parent.value is current:
        return [(target, "assignment-target") for target in parent.targets]
    if isinstance(parent, ast.AnnAssign) and parent.value is current:
        return [(parent.target, "assignment-target")]
    if isinstance(parent, ast.NamedExpr) and parent.value is current:
        return [(parent.target, "assignment-expression-target")]
    if isinstance(parent, ast.Return) and parent.value is current:
        return [(parent, "caller-return-site")]
    return []


def _verified_call_relation(
    cache: _SourceCache,
    edge: CallEdge,
) -> tuple[dict[str, object] | None, str]:
    parsed = cache.get(edge.path)
    if parsed is None:
        return None, cache.status.get(edge.path, "unavailable")
    if not 0 <= edge.start_byte <= edge.end_byte <= len(parsed.raw):
        return None, "invalid-call-range"
    evidence = parsed.raw[edge.start_byte:edge.end_byte]
    digest = hashlib.sha256(evidence).hexdigest()
    if edge.evidence_sha256 and digest != edge.evidence_sha256:
        return None, "call-evidence-mismatch"
    return {
        "caller_id": edge.caller_id,
        "callee_id": edge.callee_id,
        "kind": edge.kind,
        "resolution": edge.resolution,
        "confidence": edge.confidence,
        "path": edge.path,
        "line": edge.line,
        "start_byte": edge.start_byte,
        "end_byte": edge.end_byte,
        "source_sha256": parsed.source_sha256,
        "evidence_sha256": digest,
    }, "verified"


def _unresolved_payload(
    cache: _SourceCache,
    call: UnresolvedCall,
) -> dict[str, object]:
    parsed = cache.get(call.path)
    status = cache.status.get(call.path, "unavailable")
    payload = call.to_dict()
    payload["freshness"] = status
    if parsed is not None and 0 <= call.start_byte <= call.end_byte <= len(parsed.raw):
        evidence = parsed.raw[call.start_byte:call.end_byte]
        digest = hashlib.sha256(evidence).hexdigest()
        payload["source_sha256"] = parsed.source_sha256
        payload["evidence_verified"] = (
            not call.evidence_sha256 or digest == call.evidence_sha256
        )
    else:
        payload["evidence_verified"] = False
    return payload


def _commit(payload: dict[str, object]) -> dict[str, object]:
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt["interprocedural_flow_sha256"] = hashlib.sha256(
        _canonical(payload)
    ).hexdigest()
    return payload


def build_verified_interprocedural_flow(
    root: Path,
    index: RepositoryIndex,
    symbol_query: str,
    *,
    index_digest: str,
    direction: str = "outgoing",
    max_depth: int = 3,
    max_call_edges: int = 1_000,
    max_flow_edges: int = 10_000,
    max_nodes: int = 10_000,
) -> dict[str, object]:
    """Build bounded argument/parameter and return/result summaries."""
    root = root.expanduser().resolve(strict=True)
    query = symbol_query.strip()
    if not query or len(query) > 1_000:
        raise ValueError("symbol query must contain 1 to 1000 characters")
    if direction not in {"outgoing", "incoming", "both"}:
        raise ValueError("direction must be outgoing, incoming, or both")
    depth_limit = max(0, min(int(max_depth), 12))
    call_limit = max(1, min(int(max_call_edges), 100_000))
    flow_limit = max(1, min(int(max_flow_edges), 100_000))
    node_limit = max(1, min(int(max_nodes), 100_000))
    matches = _resolve(index, query)
    resolution = "resolved" if len(matches) == 1 else "ambiguous" if matches else "not-found"
    payload: dict[str, object] = {
        "schema_version": INTERPROCEDURAL_FLOW_SCHEMA_VERSION,
        "index_digest": index_digest,
        "query": query,
        "resolution": resolution,
        "candidates": [item.to_dict() for item in matches[:100]],
        "symbols": [],
        "call_relations": [],
        "flow_nodes": [],
        "flow_edges": [],
        "unresolved_boundary": [],
        "diagnostics": [],
        "truncation": {
            "candidates_omitted": max(0, len(matches) - 100),
            "call_edges_omitted": 0,
            "flow_edges_omitted": 0,
            "flow_nodes_omitted": 0,
            "unresolved_calls_omitted": 0,
        },
        "analysis_contract": {
            "call_binding": "existing-unique-static-index-edges-only",
            "argument_flow": "syntactic-actual-to-formal-binding",
            "return_flow": "may-reach-summary-from-each-explicit-return",
            "not_modeled": [
                "alias-and-heap-flow",
                "implicit-or-exceptional-returns",
                "mutation-and-side-effects",
                "path-conditions",
                "dynamic-dispatch-beyond-index-resolution",
                "descriptor-unbound-and-static-method-semantics",
                "reflection-macros-generated-and-external-code",
            ],
        },
        "receipt": {
            "freshness": "verified-per-used-source",
            "remote_calls": 0,
            "commitment_scope": (
                "payload-excluding-generation-command-and-interprocedural-flow-sha256"
            ),
        },
    }
    if len(matches) != 1:
        return _commit(payload)
    root_symbol = matches[0]
    if root_symbol.language != "python":
        payload["resolution"] = "unsupported-language"
        return _commit(payload)

    outgoing: dict[str, list[CallEdge]] = defaultdict(list)
    incoming: dict[str, list[CallEdge]] = defaultdict(list)
    for edge in index.call_edges:
        outgoing[edge.caller_id].append(edge)
        incoming[edge.callee_id].append(edge)
    for values in (*outgoing.values(), *incoming.values()):
        values.sort(key=lambda item: (
            item.caller_id,
            item.callee_id,
            item.path,
            item.start_byte,
            item.end_byte,
        ))

    cache = _SourceCache(root, index)
    queue = deque([(root_symbol.symbol_id, 0)])
    visited_symbols: set[str] = set()
    seen_edges: set[tuple[str, str, str, int, int]] = set()
    selected_edges: list[CallEdge] = []
    while queue:
        symbol_id, depth = queue.popleft()
        if symbol_id in visited_symbols:
            continue
        visited_symbols.add(symbol_id)
        if depth >= depth_limit:
            continue
        candidates: list[CallEdge] = []
        if direction in {"outgoing", "both"}:
            candidates.extend(outgoing.get(symbol_id, ()))
        if direction in {"incoming", "both"}:
            candidates.extend(incoming.get(symbol_id, ()))
        for edge in sorted(
            candidates,
            key=lambda item: (
                item.caller_id,
                item.callee_id,
                item.path,
                item.start_byte,
                item.end_byte,
            ),
        ):
            identity = (
                edge.caller_id,
                edge.callee_id,
                edge.path,
                edge.start_byte,
                edge.end_byte,
            )
            if identity in seen_edges:
                continue
            seen_edges.add(identity)
            if len(selected_edges) >= call_limit:
                truncation = payload["truncation"]
                assert isinstance(truncation, dict)
                truncation["call_edges_omitted"] = int(
                    truncation["call_edges_omitted"]
                ) + 1
                continue
            selected_edges.append(edge)
            neighbor = (
                edge.callee_id if edge.caller_id == symbol_id else edge.caller_id
            )
            if neighbor not in visited_symbols:
                queue.append((neighbor, depth + 1))

    nodes: dict[str, dict[str, object]] = {}
    flows: list[dict[str, object]] = []
    call_relations: list[dict[str, object]] = []
    diagnostics: set[str] = set()
    used_symbols = {root_symbol.symbol_id}
    for edge in selected_edges:
        relation, status = _verified_call_relation(cache, edge)
        if relation is None:
            diagnostics.add(f"omitted-call:{edge.path}:{edge.line}:{status}")
            continue
        call_relations.append(relation)
        used_symbols.update((edge.caller_id, edge.callee_id))
        caller = index.symbols.get(edge.caller_id)
        callee = index.symbols.get(edge.callee_id)
        if (
            caller is None
            or callee is None
            or caller.language != "python"
            or callee.language != "python"
            or callee.kind not in {"function", "method", "test"}
        ):
            diagnostics.add(f"flow-unsupported:{edge.caller_id}->{edge.callee_id}")
            continue
        caller_source = cache.get(caller.path)
        callee_source = cache.get(callee.path)
        if caller_source is None or callee_source is None:
            diagnostics.add(f"flow-source-unavailable:{edge.caller_id}->{edge.callee_id}")
            continue
        call = _call_node(caller_source, edge)
        callee_function = _function_node(callee_source, callee)
        if call is None or callee_function is None:
            diagnostics.add(f"flow-ast-node-unavailable:{edge.caller_id}->{edge.callee_id}")
            continue

        result_node = _evidence_node(
            nodes,
            caller_source,
            caller.path,
            call,
            "call-result",
            symbol_id=caller.symbol_id,
        )
        for argument, parameter, binding_kind, position, keyword in _argument_bindings(
            call, callee_function, callee
        ):
            if len(flows) >= flow_limit:
                truncation = payload["truncation"]
                assert isinstance(truncation, dict)
                truncation["flow_edges_omitted"] = int(
                    truncation["flow_edges_omitted"]
                ) + 1
                break
            argument_node = _evidence_node(
                nodes,
                caller_source,
                caller.path,
                argument,
                "actual-argument",
                symbol_id=caller.symbol_id,
            )
            parameter_node = _evidence_node(
                nodes,
                callee_source,
                callee.path,
                parameter,
                "formal-parameter",
                symbol_id=callee.symbol_id,
            )
            flows.append({
                "source": argument_node,
                "target": parameter_node,
                "kind": "argument-to-parameter",
                "binding": binding_kind,
                "position": position,
                "keyword": keyword,
                "call": {
                    "caller_id": edge.caller_id,
                    "callee_id": edge.callee_id,
                    "path": edge.path,
                    "start_byte": edge.start_byte,
                    "end_byte": edge.end_byte,
                },
                "confidence": "verified-syntactic-summary",
            })
        for return_statement in _return_nodes(callee_function):
            if len(flows) >= flow_limit:
                truncation = payload["truncation"]
                assert isinstance(truncation, dict)
                truncation["flow_edges_omitted"] = int(
                    truncation["flow_edges_omitted"]
                ) + 1
                break
            returned = return_statement.value or return_statement
            return_node = _evidence_node(
                nodes,
                callee_source,
                callee.path,
                returned,
                "return-value",
                symbol_id=callee.symbol_id,
            )
            flows.append({
                "source": return_node,
                "target": result_node,
                "kind": "return-to-call-result",
                "call": {
                    "caller_id": edge.caller_id,
                    "callee_id": edge.callee_id,
                    "path": edge.path,
                    "start_byte": edge.start_byte,
                    "end_byte": edge.end_byte,
                },
                "confidence": "may-reach-verified-return",
            })
        for consumer, consumer_kind in _consumer_nodes(caller_source, call):
            if len(flows) >= flow_limit:
                truncation = payload["truncation"]
                assert isinstance(truncation, dict)
                truncation["flow_edges_omitted"] = int(
                    truncation["flow_edges_omitted"]
                ) + 1
                break
            consumer_node = _evidence_node(
                nodes,
                caller_source,
                caller.path,
                consumer,
                consumer_kind,
                symbol_id=caller.symbol_id,
            )
            flows.append({
                "source": result_node,
                "target": consumer_node,
                "kind": "call-result-to-consumer",
                "consumer_kind": consumer_kind,
                "call": {
                    "caller_id": edge.caller_id,
                    "callee_id": edge.callee_id,
                    "path": edge.path,
                    "start_byte": edge.start_byte,
                    "end_byte": edge.end_byte,
                },
                "confidence": "verified-direct-syntactic-consumer",
            })

    # A nested call expression can be both the result of one call and an actual
    # argument to another. Preserve that value path explicitly instead of
    # leaving two role-specific nodes disconnected despite identical evidence.
    nodes_by_span: dict[tuple[str, int, int, str], list[dict[str, object]]] = (
        defaultdict(list)
    )
    for node in nodes.values():
        nodes_by_span[(
            str(node["path"]),
            int(node["start_byte"]),
            int(node["end_byte"]),
            str(node["symbol_id"]),
        )].append(node)
    for same_span in nodes_by_span.values():
        results = [item for item in same_span if item["role"] == "call-result"]
        arguments = [item for item in same_span if item["role"] == "actual-argument"]
        for result in results:
            for argument in arguments:
                if len(flows) >= flow_limit:
                    truncation = payload["truncation"]
                    assert isinstance(truncation, dict)
                    truncation["flow_edges_omitted"] = int(
                        truncation["flow_edges_omitted"]
                    ) + 1
                    continue
                flows.append({
                    "source": result["node_id"],
                    "target": argument["node_id"],
                    "kind": "call-result-to-argument",
                    "confidence": "verified-identical-expression-span",
                })

    sorted_nodes = sorted(nodes.values(), key=lambda item: str(item["node_id"]))
    if len(sorted_nodes) > node_limit:
        truncation = payload["truncation"]
        assert isinstance(truncation, dict)
        truncation["flow_nodes_omitted"] = len(sorted_nodes) - node_limit
        kept = {str(item["node_id"]) for item in sorted_nodes[:node_limit]}
        sorted_nodes = sorted_nodes[:node_limit]
        before = len(flows)
        flows = [
            item
            for item in flows
            if str(item["source"]) in kept and str(item["target"]) in kept
        ]
        truncation["flow_edges_omitted"] = int(
            truncation["flow_edges_omitted"]
        ) + before - len(flows)

    all_unresolved = [
        _unresolved_payload(cache, call)
        for call in index.unresolved_calls
        if call.caller_id in used_symbols
    ]
    all_unresolved.sort(
        key=lambda item: (
            str(item["caller_id"]), str(item["path"]), int(item["start_byte"])
        )
    )
    unresolved = all_unresolved[:call_limit]
    truncation = payload["truncation"]
    assert isinstance(truncation, dict)
    truncation["unresolved_calls_omitted"] = max(
        0, len(all_unresolved) - call_limit
    )
    payload.update({
        "root_symbol_id": root_symbol.symbol_id,
        "symbols": [
            index.symbols[symbol_id].to_dict()
            for symbol_id in sorted(used_symbols)
            if symbol_id in index.symbols
        ],
        "call_relations": sorted(
            call_relations,
            key=lambda item: (
                str(item["caller_id"]),
                str(item["callee_id"]),
                str(item["path"]),
                int(item["start_byte"]),
            ),
        ),
        "flow_nodes": sorted_nodes,
        "flow_edges": sorted(
            flows,
            key=lambda item: (
                str(item["source"]), str(item["target"]), str(item["kind"])
            ),
        ),
        "unresolved_boundary": unresolved,
        "diagnostics": sorted(diagnostics),
    })
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt.update({
        "verified_source_count": sum(
            status == "verified" for status in cache.status.values()
        ),
        "call_relation_count": len(call_relations),
        "flow_node_count": len(sorted_nodes),
        "flow_edge_count": len(flows),
        "unresolved_boundary_count": len(unresolved),
    })
    return _commit(payload)


def verify_interprocedural_flow_commitment(payload: Mapping[str, object]) -> bool:
    """Verify the detached receipt without trusting labels or a workspace."""
    try:
        candidate = copy.deepcopy(dict(payload))
        candidate.pop("generation", None)
        candidate.pop("command", None)
        if candidate.get("schema_version") != INTERPROCEDURAL_FLOW_SCHEMA_VERSION:
            return False
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("interprocedural_flow_sha256"))
        return hashlib.sha256(_canonical(candidate)).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "INTERPROCEDURAL_FLOW_SCHEMA_VERSION",
    "build_verified_interprocedural_flow",
    "verify_interprocedural_flow_commitment",
]
