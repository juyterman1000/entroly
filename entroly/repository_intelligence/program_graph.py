"""Source-verified Python control flow and reaching-definition evidence."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .models import RepositoryIndex, Symbol

PROGRAM_GRAPH_SCHEMA_VERSION = "entroly.verified-program-graph.v1"
_COMPOUND_TYPES = (
    ast.Try,
    getattr(ast, "TryStar", ast.Try),
    ast.Match,
    ast.With,
    ast.AsyncWith,
)


@dataclass
class _FlowNode:
    node_id: str
    kind: str
    label: str
    ast_node: ast.AST | None
    occurrences: list[dict[str, object]] = field(default_factory=list)


class _OccurrenceVisitor(ast.NodeVisitor):
    def __init__(self, span) -> None:
        self._span = span
        self.items: list[dict[str, object]] = []

    @staticmethod
    def _attribute_name(node: ast.AST) -> str:
        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
        return ""

    def _add(self, node: ast.AST, name: str, role: str) -> None:
        start, end, evidence = self._span(node)
        self.items.append({
            "occurrence_id": f"{start}:{end}:{role}:{name}",
            "name": name,
            "role": role,
            "start_byte": start,
            "end_byte": end,
            "line": max(1, int(getattr(node, "lineno", 1))),
            "evidence_sha256": evidence,
        })

    def visit_Name(self, node: ast.Name) -> None:
        role = "definition" if isinstance(node.ctx, (ast.Store, ast.Del)) else "use"
        self._add(node, node.id, role)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = self._attribute_name(node)
        if name:
            role = "definition" if isinstance(node.ctx, (ast.Store, ast.Del)) else "use"
            self._add(node, name, role)
            return
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            self._add(node.target, node.target.id, "use")
            self._add(node.target, node.target.id, "definition")
        elif isinstance(node.target, ast.Attribute):
            name = self._attribute_name(node.target)
            if name:
                self._add(node.target, name, "use")
                self._add(node.target, name, "definition")
        self.visit(node.value)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


class _ProgramGraphBuilder:
    def __init__(self, raw: bytes, line_offsets: list[int], limit: int) -> None:
        self.raw = raw
        self.line_offsets = line_offsets
        self.limit = limit
        self.nodes: dict[str, _FlowNode] = {}
        self.control_edges: set[tuple[str, str, str]] = set()
        self.diagnostics: set[str] = set()
        self.truncated = False
        self._sequence = 0
        self.entry_id = self._synthetic("entry")
        self.exit_id = self._synthetic("exit")

    def _byte_range(self, node: ast.AST) -> tuple[int, int]:
        line = max(1, int(getattr(node, "lineno", 1)))
        end_line = max(line, int(getattr(node, "end_lineno", line)))
        start = self.line_offsets[min(line - 1, len(self.line_offsets) - 1)]
        start += max(0, int(getattr(node, "col_offset", 0)))
        end = self.line_offsets[min(end_line - 1, len(self.line_offsets) - 1)]
        end += max(0, int(getattr(node, "end_col_offset", 0)))
        return min(start, len(self.raw)), min(max(start, end), len(self.raw))

    def _span(self, node: ast.AST) -> tuple[int, int, str]:
        start, end = self._byte_range(node)
        return start, end, hashlib.sha256(self.raw[start:end]).hexdigest()

    def _synthetic(self, label: str) -> str:
        node_id = f"synthetic:{label}"
        self.nodes[node_id] = _FlowNode(node_id, "synthetic", label, None)
        return node_id

    def _node(
        self,
        node: ast.AST,
        kind: str,
        label: str,
        *,
        occurrence_roots: Iterable[ast.AST] = (),
    ) -> str:
        start, end = self._byte_range(node)
        node_id = f"source:{start}:{end}:{kind}:{self._sequence}"
        self._sequence += 1
        if len(self.nodes) >= self.limit:
            self.truncated = True
            return self.exit_id
        visitor = _OccurrenceVisitor(self._span)
        for root in occurrence_roots:
            visitor.visit(root)
        self.nodes[node_id] = _FlowNode(
            node_id,
            kind,
            label,
            node,
            sorted(visitor.items, key=lambda item: str(item["occurrence_id"])),
        )
        return node_id

    def _edge(self, source: str, target: str, kind: str) -> None:
        if source != target or kind == "loop-back":
            self.control_edges.add((source, target, kind))

    def build_sequence(
        self,
        statements: list[ast.stmt],
        follow: str,
        *,
        break_target: str | None = None,
        continue_target: str | None = None,
    ) -> str:
        current = follow
        for statement in reversed(statements):
            current = self.build_statement(
                statement,
                current,
                break_target=break_target,
                continue_target=continue_target,
            )
        return current

    def build_statement(
        self,
        statement: ast.stmt,
        follow: str,
        *,
        break_target: str | None,
        continue_target: str | None,
    ) -> str:
        if isinstance(statement, ast.If):
            predicate = self._node(
                statement.test,
                "branch",
                "if",
                occurrence_roots=(statement.test,),
            )
            true_entry = self.build_sequence(
                statement.body,
                follow,
                break_target=break_target,
                continue_target=continue_target,
            )
            false_entry = self.build_sequence(
                statement.orelse,
                follow,
                break_target=break_target,
                continue_target=continue_target,
            ) if statement.orelse else follow
            self._edge(predicate, true_entry, "true")
            self._edge(predicate, false_entry, "false")
            return predicate

        if isinstance(statement, (ast.While, ast.For, ast.AsyncFor)):
            expression = statement.test if isinstance(statement, ast.While) else statement.iter
            roots: list[ast.AST] = [expression]
            if isinstance(statement, (ast.For, ast.AsyncFor)):
                roots.append(statement.target)
            loop = self._node(
                expression,
                "loop",
                type(statement).__name__.lower(),
                occurrence_roots=roots,
            )
            false_entry = self.build_sequence(
                statement.orelse,
                follow,
                break_target=break_target,
                continue_target=continue_target,
            ) if statement.orelse else follow
            body_entry = self.build_sequence(
                statement.body,
                loop,
                break_target=false_entry,
                continue_target=loop,
            )
            self._edge(loop, body_entry, "loop-body")
            self._edge(loop, false_entry, "loop-exit")
            return loop

        if isinstance(statement, ast.Break):
            node_id = self._node(statement, "jump", "break")
            self._edge(node_id, break_target or follow, "break")
            if break_target is None:
                self.diagnostics.add("break-outside-modeled-loop")
            return node_id

        if isinstance(statement, ast.Continue):
            node_id = self._node(statement, "jump", "continue")
            self._edge(node_id, continue_target or follow, "continue")
            if continue_target is None:
                self.diagnostics.add("continue-outside-modeled-loop")
            return node_id

        if isinstance(statement, (ast.Return, ast.Raise)):
            roots = tuple(
                item
                for item in (getattr(statement, "value", None), getattr(statement, "exc", None))
                if isinstance(item, ast.AST)
            )
            node_id = self._node(statement, "terminal", type(statement).__name__.lower(), occurrence_roots=roots)
            self._edge(node_id, self.exit_id, "return" if isinstance(statement, ast.Return) else "raise")
            return node_id

        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node_id = self._node(statement, "declaration", getattr(statement, "name", "declaration"))
            self._edge(node_id, follow, "fallthrough")
            return node_id

        if isinstance(statement, (ast.With, ast.AsyncWith)):
            roots: list[ast.AST] = []
            for item in statement.items:
                roots.append(item.context_expr)
                if item.optional_vars:
                    roots.append(item.optional_vars)
            header = self._node(
                statement,
                "with",
                type(statement).__name__.lower(),
                occurrence_roots=roots,
            )
            body_entry = self.build_sequence(
                statement.body,
                follow,
                break_target=break_target,
                continue_target=continue_target,
            )
            self._edge(header, body_entry, "enter-with")
            return header

        if isinstance(statement, ast.Match):
            predicate = self._node(
                statement.subject,
                "branch",
                "match",
                occurrence_roots=(statement.subject,),
            )
            for position, case in enumerate(statement.cases):
                case_entry = self.build_sequence(
                    case.body,
                    follow,
                    break_target=break_target,
                    continue_target=continue_target,
                )
                self._edge(predicate, case_entry, f"case:{position}")
            self._edge(predicate, follow, "no-match")
            self.diagnostics.add("match-pattern-bindings-omitted")
            return predicate

        if isinstance(statement, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            final_entry = self.build_sequence(
                statement.finalbody,
                follow,
                break_target=break_target,
                continue_target=continue_target,
            ) if statement.finalbody else follow
            else_entry = self.build_sequence(
                statement.orelse,
                final_entry,
                break_target=break_target,
                continue_target=continue_target,
            ) if statement.orelse else final_entry
            header = self._node(statement, "try", type(statement).__name__.lower())
            body_entry = self.build_sequence(
                statement.body,
                else_entry,
                break_target=break_target,
                continue_target=continue_target,
            )
            self._edge(header, body_entry, "try-body")
            for position, handler in enumerate(statement.handlers):
                evidence_node = handler.type if handler.type is not None else statement
                handler_node = self._node(
                    evidence_node,
                    "exception-handler",
                    f"handler:{position}",
                    occurrence_roots=(handler.type,) if handler.type is not None else (),
                )
                handler_entry = self.build_sequence(
                    handler.body,
                    final_entry,
                    break_target=break_target,
                    continue_target=continue_target,
                )
                self._edge(header, handler_node, "may-exception")
                self._edge(handler_node, handler_entry, "handler-body")
                if handler.name:
                    self.diagnostics.add("exception-binding-omitted")
            return header

        if isinstance(statement, _COMPOUND_TYPES):
            self.diagnostics.add(f"conservative-compound:{type(statement).__name__}")
            node_id = self._node(statement, "compound", type(statement).__name__.lower())
            self._edge(node_id, follow, "conservative-fallthrough")
            return node_id

        node_id = self._node(
            statement,
            "statement",
            type(statement).__name__.lower(),
            occurrence_roots=(statement,),
        )
        self._edge(node_id, follow, "fallthrough")
        return node_id

    def add_parameters(self, arguments: ast.arguments) -> None:
        values = [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ]
        if arguments.vararg:
            values.append(arguments.vararg)
        if arguments.kwarg:
            values.append(arguments.kwarg)
        visitor = _OccurrenceVisitor(self._span)
        for argument in values:
            start, end, evidence = self._span(argument)
            visitor.items.append({
                "occurrence_id": f"{start}:{end}:definition:{argument.arg}",
                "name": argument.arg,
                "role": "definition",
                "start_byte": start,
                "end_byte": end,
                "line": max(1, int(getattr(argument, "lineno", 1))),
                "evidence_sha256": evidence,
            })
        self.nodes[self.entry_id].occurrences = visitor.items


def _find_symbol_node(tree: ast.AST, symbol: Symbol) -> ast.AST | None:
    expected = {"function": ast.FunctionDef, "method": ast.FunctionDef, "test": ast.FunctionDef}
    target_type = expected.get(symbol.kind)
    for node in ast.walk(tree):
        if target_type and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == symbol.name and int(node.lineno) == symbol.line_start:
                return node
        if symbol.kind == "class" and isinstance(node, ast.ClassDef):
            if node.name == symbol.name and int(node.lineno) == symbol.line_start:
                return node
    return None


def _payload_node(builder: _ProgramGraphBuilder, node: _FlowNode) -> dict[str, object]:
    if node.ast_node is None:
        return {
            "node_id": node.node_id,
            "kind": node.kind,
            "label": node.label,
            "trust": "synthetic-control-node",
            "occurrences": node.occurrences,
        }
    start, end, evidence = builder._span(node.ast_node)
    return {
        "node_id": node.node_id,
        "kind": node.kind,
        "label": node.label,
        "start_byte": start,
        "end_byte": end,
        "line_start": max(1, int(getattr(node.ast_node, "lineno", 1))),
        "line_end": max(1, int(getattr(node.ast_node, "end_lineno", getattr(node.ast_node, "lineno", 1)))),
        "evidence_sha256": evidence,
        "trust": "verified-source-span",
        "occurrences": node.occurrences,
    }


def _data_flow(
    nodes: dict[str, _FlowNode],
    control_edges: set[tuple[str, str, str]],
    entry_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    predecessors: dict[str, set[str]] = defaultdict(set)
    for source, target, _kind in control_edges:
        predecessors[target].add(source)
    gen: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    uses: dict[str, list[dict[str, object]]] = defaultdict(list)
    occurrence_lookup: dict[str, dict[str, object]] = {}
    for node_id, node in nodes.items():
        for occurrence in node.occurrences:
            occurrence_id = str(occurrence["occurrence_id"])
            occurrence_lookup[occurrence_id] = occurrence
            name = str(occurrence["name"])
            if occurrence["role"] == "definition":
                gen[node_id][name].add(occurrence_id)
            else:
                uses[node_id].append(occurrence)

    incoming: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    outgoing: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    outgoing[entry_id] = defaultdict(set, {
        name: set(ids) for name, ids in gen.get(entry_id, {}).items()
    })
    ordered = sorted(nodes)
    changed = True
    iterations = 0
    while changed and iterations <= max(4, len(nodes) * 2):
        changed = False
        iterations += 1
        for node_id in ordered:
            if node_id == entry_id:
                continue
            merged: dict[str, set[str]] = defaultdict(set)
            for predecessor in predecessors.get(node_id, ()):
                for name, definitions in outgoing.get(predecessor, {}).items():
                    merged[name].update(definitions)
            produced = {name: set(values) for name, values in merged.items()}
            for name, definitions in gen.get(node_id, {}).items():
                produced[name] = set(definitions)
            if dict(incoming[node_id]) != dict(merged) or dict(outgoing[node_id]) != produced:
                incoming[node_id] = defaultdict(set, merged)
                outgoing[node_id] = defaultdict(set, produced)
                changed = True

    definite_in: dict[str, set[str]] = defaultdict(set)
    definite_out: dict[str, set[str]] = defaultdict(set)
    definite_out[entry_id] = set(gen.get(entry_id, {}))
    changed = True
    iterations = 0
    while changed and iterations <= max(4, len(nodes) * 2):
        changed = False
        iterations += 1
        for node_id in ordered:
            if node_id == entry_id:
                continue
            parents = predecessors.get(node_id, set())
            parent_sets = [definite_out[parent] for parent in parents]
            incoming_names = set.intersection(*parent_sets) if parent_sets else set()
            produced_names = incoming_names | set(gen.get(node_id, {}))
            if definite_in[node_id] != incoming_names or definite_out[node_id] != produced_names:
                definite_in[node_id] = incoming_names
                definite_out[node_id] = produced_names
                changed = True

    edges: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    for node_id in ordered:
        for use in uses.get(node_id, ()):
            definitions = sorted(incoming[node_id].get(str(use["name"]), ()))
            if not definitions:
                unresolved.append({**use, "reason": "no-reaching-repository-definition"})
                continue
            for definition_id in definitions:
                edges.append({
                    "source_occurrence": definition_id,
                    "target_occurrence": use["occurrence_id"],
                    "name": use["name"],
                    "kind": "reaching-definition",
                    "confidence": (
                        "must-reach"
                        if len(definitions) == 1 and str(use["name"]) in definite_in[node_id]
                        else "may-reach"
                    ),
                })
    edges.sort(key=lambda item: (str(item["target_occurrence"]), str(item["source_occurrence"])))
    unresolved.sort(key=lambda item: str(item["occurrence_id"]))
    return edges, unresolved


def _finish(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt["program_graph_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def build_verified_program_graph(
    root: Path,
    index: RepositoryIndex,
    symbol_query: str,
    *,
    index_digest: str,
    limit: int = 1_000,
) -> dict[str, object]:
    """Build a source-verified intraprocedural graph for one Python symbol."""
    root = root.expanduser().resolve(strict=True)
    query = symbol_query.strip()
    if not query or len(query) > 1_000:
        raise ValueError("symbol query must contain 1 to 1000 characters")
    matches = sorted(
        (
            symbol for symbol in index.symbols.values()
            if query.lower() in {
                symbol.symbol_id.lower(), symbol.qualified_name.lower(), symbol.name.lower()
            }
        ),
        key=lambda symbol: symbol.symbol_id,
    )
    resolution = "resolved" if len(matches) == 1 else "ambiguous" if matches else "not-found"
    base: dict[str, object] = {
        "schema_version": PROGRAM_GRAPH_SCHEMA_VERSION,
        "index_digest": index_digest,
        "query": query,
        "resolution": resolution,
        "candidates": [symbol.to_dict() for symbol in matches[:100]],
        "nodes": [],
        "control_edges": [],
        "data_edges": [],
        "unresolved_uses": [],
        "diagnostics": [],
        "truncated": len(matches) > 100,
        "receipt": {
            "freshness": "not-applicable",
            "remote_calls": 0,
            "commitment_scope": "payload-excluding-generation-command-and-program-graph-sha256",
        },
    }
    if len(matches) != 1:
        return _finish(base)
    symbol = matches[0]
    if symbol.language != "python" or symbol.kind not in {"function", "method", "test"}:
        base["resolution"] = "unsupported-symbol-kind"
        return _finish(base)
    record = index.files.get(symbol.path)
    try:
        candidate = (root / symbol.path).resolve(strict=True)
        candidate.relative_to(root)
        raw = candidate.read_bytes()
    except (OSError, RuntimeError, ValueError):
        base["resolution"] = "unsafe-or-unreadable"
        return _finish(base)
    if record is None or hashlib.sha256(raw).hexdigest() != record.sha256:
        base["resolution"] = "stale-index"
        return _finish(base)
    try:
        text = raw.decode("utf-8", errors="surrogateescape")
        tree = ast.parse(text, filename=symbol.path, type_comments=True)
    except (SyntaxError, ValueError, MemoryError, RecursionError, UnicodeError):
        base["resolution"] = "parse-failed"
        return _finish(base)
    target = _find_symbol_node(tree, symbol)
    if not isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
        base["resolution"] = "symbol-node-not-found"
        return _finish(base)
    offsets = [0]
    running = 0
    for line in text.splitlines(keepends=True):
        running += len(line.encode("utf-8", errors="surrogateescape"))
        offsets.append(running)
    builder = _ProgramGraphBuilder(raw, offsets, max(16, min(int(limit), 10_000)))
    builder.add_parameters(target.args)
    body_entry = builder.build_sequence(target.body, builder.exit_id)
    builder._edge(builder.entry_id, body_entry, "enter")
    data_edges, unresolved = _data_flow(
        builder.nodes,
        builder.control_edges,
        builder.entry_id,
    )
    base.update({
        "root_symbol_id": symbol.symbol_id,
        "source_sha256": record.sha256,
        "nodes": [
            _payload_node(builder, builder.nodes[node_id]) for node_id in sorted(builder.nodes)
        ],
        "control_edges": [
            {"source": source, "target": target_id, "kind": kind}
            for source, target_id, kind in sorted(builder.control_edges)
        ],
        "data_edges": data_edges,
        "unresolved_uses": unresolved,
        "diagnostics": sorted(builder.diagnostics),
        "truncated": builder.truncated,
        "receipt": {
            "freshness": "verified-against-indexed-source-sha256",
            "source_node_count": sum(node.ast_node is not None for node in builder.nodes.values()),
            "control_edge_count": len(builder.control_edges),
            "data_edge_count": len(data_edges),
            "unresolved_use_count": len(unresolved),
            "remote_calls": 0,
            "commitment_scope": "payload-excluding-generation-command-and-program-graph-sha256",
        },
    })
    return _finish(base)


def verify_program_graph_commitment(payload: dict[str, object]) -> bool:
    """Verify a program-graph receipt without workspace access."""
    try:
        candidate = copy.deepcopy(payload)
        candidate.pop("generation", None)
        candidate.pop("command", None)
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("program_graph_sha256"))
        canonical = json.dumps(
            candidate,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "PROGRAM_GRAPH_SCHEMA_VERSION",
    "build_verified_program_graph",
    "verify_program_graph_commitment",
]
