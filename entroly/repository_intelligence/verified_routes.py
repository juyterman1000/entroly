"""Freshness-checked HTTP route intelligence with explicit uncertainty.

Python framework routes are recovered from the language AST, including static
router/blueprint mounts.  Other supported languages use bounded, disclosed
framework patterns.  Every result carries a source commitment; dynamic paths
and ambiguous handler bindings are omitted instead of guessed.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .models import RepositoryIndex, Symbol

VERIFIED_ROUTES_SCHEMA_VERSION = "entroly.verified-http-routes.v1"

_HTTP_METHODS = {
    "get", "post", "put", "patch", "delete", "head", "options", "trace",
}
_PYTHON_ROUTE_OWNERS = {"app", "api", "router", "bp", "blueprint"}
_PARAMETER = re.compile(r"(?::[A-Za-z_][\w-]*|<[^>]+>|\{[^}]+\})")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _join_path(prefix: str, route: str) -> str:
    if not prefix:
        return route or "/"
    if not route:
        return prefix or "/"
    return "/" + "/".join(
        part for part in (prefix.strip("/"), route.strip("/")) if part
    )


def _normalize_path(path: str) -> str:
    value = "/" + path.lstrip("/")
    value = re.sub(r"/{2,}", "/", value)
    value = _PARAMETER.sub("{param}", value)
    return value.rstrip("/") or "/"


def _dotted(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _literal_string(node: ast.AST | None, constants: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string(node.left, constants)
        right = _literal_string(node.right, constants)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.JoinedStr):
        pieces: list[str] = []
        for value in node.values:
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                return None
            pieces.append(value.value)
        return "".join(pieces)
    return None


def _string_values(node: ast.AST | None, constants: Mapping[str, str]) -> list[str]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_literal_string(item, constants) for item in node.elts]
        return [item for item in values if item is not None]
    single = _literal_string(node, constants)
    return [single] if single is not None else []


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def _node_bytes(text: str, node: ast.AST) -> tuple[int, int]:
    lines = text.splitlines(keepends=True)
    line = max(1, int(getattr(node, "lineno", 1)))
    end_line = max(line, int(getattr(node, "end_lineno", line)))
    start = sum(len(item.encode("utf-8")) for item in lines[: line - 1])
    start += int(getattr(node, "col_offset", 0))
    end = sum(len(item.encode("utf-8")) for item in lines[: end_line - 1])
    end += int(getattr(node, "end_col_offset", getattr(node, "col_offset", 0)))
    return start, max(start, end)


@dataclass(frozen=True)
class _Route:
    method: str
    path: str
    owner: str
    handler: str
    framework: str
    line: int
    start_byte: int
    end_byte: int
    extraction: str


@dataclass(frozen=True)
class _Mount:
    parent: str
    child: str
    prefix: str
    line: int


class _PythonAnalyzer(ast.NodeVisitor):
    def __init__(self, text: str) -> None:
        self.text = text
        self.constants: dict[str, str] = {}
        self.objects: dict[str, str] = {}
        self.prefixes: dict[str, str] = {}
        self.routes: list[_Route] = []
        self.mounts: list[_Mount] = []
        self.omissions: Counter[str] = Counter()

    def analyze(self, tree: ast.Module) -> None:
        self._definitions(tree)
        self.visit(tree)

    def _definitions(self, tree: ast.Module) -> None:
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                literal = _literal_string(value, self.constants)
                if literal is not None:
                    self.constants[target.id] = literal
                if not isinstance(value, ast.Call):
                    continue
                constructor = _dotted(value.func).rsplit(".", 1)[-1]
                framework = {
                    "FastAPI": "fastapi", "APIRouter": "fastapi",
                    "Flask": "flask", "Blueprint": "flask",
                    "Quart": "quart", "Sanic": "sanic",
                    "Starlette": "starlette", "Falcon": "falcon",
                }.get(constructor)
                if framework:
                    self.objects[target.id] = framework
                    prefix_node = _keyword(value, "prefix") or _keyword(value, "url_prefix")
                    prefix = _literal_string(prefix_node, self.constants)
                    if prefix is not None:
                        self.prefixes[target.id] = prefix

    def _framework(self, owner: str) -> str | None:
        root = owner.split(".", 1)[0]
        if root in self.objects:
            return self.objects[root]
        return "python-web" if root in _PYTHON_ROUTE_OWNERS else None

    def _add_decorator(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(
                decorator.func, ast.Attribute
            ):
                continue
            owner = _dotted(decorator.func.value)
            framework = self._framework(owner)
            action = decorator.func.attr.lower()
            if framework is None or action not in _HTTP_METHODS | {
                "route", "api_route", "websocket",
            }:
                continue
            path = _literal_string(
                decorator.args[0] if decorator.args else _keyword(decorator, "path"),
                self.constants,
            )
            if path is None:
                self.omissions["dynamic-path"] += 1
                continue
            if action in _HTTP_METHODS:
                methods = [action.upper()]
            elif action == "websocket":
                methods = ["WEBSOCKET"]
            else:
                methods = [item.upper() for item in _string_values(
                    _keyword(decorator, "methods"), self.constants
                )] or ["GET"]
            start, end = _node_bytes(self.text, decorator)
            for method in methods:
                self.routes.append(_Route(
                    method, path, owner, node.name, framework,
                    decorator.lineno, start, end, "python-ast-decorator",
                ))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._add_decorator(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._add_decorator(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = _dotted(node.func)
        action = name.rsplit(".", 1)[-1]
        owner = name.rsplit(".", 1)[0] if "." in name else ""
        if action in {"include_router", "register_blueprint"} and node.args:
            child = _dotted(node.args[0])
            prefix_node = _keyword(node, "prefix") or _keyword(node, "url_prefix")
            prefix = _literal_string(prefix_node, self.constants) or ""
            if owner and child:
                self.mounts.append(_Mount(owner, child, prefix, node.lineno))
        elif action == "mount" and len(node.args) >= 2:
            prefix = _literal_string(node.args[0], self.constants)
            child = _dotted(node.args[1])
            if owner and child and prefix is not None:
                self.mounts.append(_Mount(owner, child, prefix, node.lineno))
        elif action in {"add_api_route", "add_route"} and node.args:
            framework = self._framework(owner)
            path = _literal_string(node.args[0], self.constants)
            if framework and path is not None:
                handler_node = node.args[1] if len(node.args) > 1 else _keyword(node, "endpoint")
                handler = _dotted(handler_node)
                methods = [item.upper() for item in _string_values(
                    _keyword(node, "methods"), self.constants
                )] or ["*"]
                start, end = _node_bytes(self.text, node)
                for method in methods:
                    self.routes.append(_Route(
                        method, path, owner, handler, framework, node.lineno,
                        start, end, "python-ast-registration",
                    ))
            elif framework:
                self.omissions["dynamic-path"] += 1
        elif name in {"path", "django.urls.path", "re_path", "django.urls.re_path"}:
            path = _literal_string(node.args[0] if node.args else None, self.constants)
            if path is not None:
                handler = _dotted(node.args[1]) if len(node.args) > 1 else ""
                start, end = _node_bytes(self.text, node)
                self.routes.append(_Route(
                    "*", "/" + path.lstrip("/"), "urlpatterns", handler,
                    "django", node.lineno, start, end, "python-ast-urlpattern",
                ))
            else:
                self.omissions["dynamic-path"] += 1
        elif name in {"Route", "starlette.routing.Route"} and node.args:
            path = _literal_string(node.args[0], self.constants)
            if path is not None:
                handler = _dotted(node.args[1]) if len(node.args) > 1 else ""
                methods = [item.upper() for item in _string_values(
                    _keyword(node, "methods"), self.constants
                )] or ["GET"]
                start, end = _node_bytes(self.text, node)
                for method in methods:
                    self.routes.append(_Route(
                        method, path, "routes", handler, "starlette", node.lineno,
                        start, end, "python-ast-route-object",
                    ))
        self.generic_visit(node)


_PATTERNS: tuple[tuple[str, tuple[str, ...], re.Pattern[str]], ...] = (
    ("javascript-router", ("javascript", "typescript"), re.compile(
        r"(?m)\b(?:app|router|server)\s*\.\s*(get|post|put|patch|delete|head|options|all)\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*([\w.$]+)"
    )),
    ("go-router", ("go",), re.compile(
        r"(?m)\b(?:r|router|e|app)\s*\.\s*(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|Any)\s*\(\s*[`\"]([^`\"]+)[`\"]\s*,\s*([\w.]+)"
    )),
    ("aspnet-minimal", ("c_sharp",), re.compile(
        r"(?m)\bapp\s*\.\s*Map(Get|Post|Put|Patch|Delete|Methods)\s*\(\s*\"([^\"]+)\"\s*,\s*([\w.]+)"
    )),
    ("rails", ("ruby",), re.compile(
        r"(?m)^\s*(get|post|put|patch|delete)\s+['\"]([^'\"]+)['\"](?:\s*,\s*to:\s*['\"]([^'\"]+)['\"])?"
    )),
    ("spring", ("java", "kotlin"), re.compile(
        r"(?m)@(Get|Post|Put|Patch|Delete)Mapping\s*\(\s*(?:value\s*=\s*)?[\"']([^\"']+)[\"']"
    )),
    ("axum", ("rust",), re.compile(
        r"(?m)\.route\s*\(\s*\"([^\"]+)\"\s*,\s*(?:axum::routing::|routing::)?(get|post|put|patch|delete|head|options|any)\s*\(\s*([\w:]+)"
    )),
    ("actix-or-rocket", ("rust",), re.compile(
        r"(?m)^\s*#\[(get|post|put|patch|delete|head|options)\s*\(\s*\"([^\"]+)\""
    )),
)


def _pattern_routes(path: str, language: str, text: str) -> list[_Route]:
    routes: list[_Route] = []
    for framework, languages, pattern in _PATTERNS:
        if language not in languages:
            continue
        for match in pattern.finditer(text):
            selected_framework = framework
            if framework == "javascript-router":
                if re.search(r"(?:from\s+|require\s*\(\s*)['\"]hono(?:/|['\"])", text):
                    selected_framework = "hono"
                elif re.search(
                    r"(?:from\s+|require\s*\(\s*)['\"]express(?:/|['\"])", text
                ):
                    selected_framework = "express"
            groups = match.groups()
            if framework == "axum":
                route_path, method, handler = groups
            elif framework == "aspnet-minimal":
                method, route_path, handler = groups
            elif framework == "spring":
                method, route_path = groups
                handler = ""
            elif framework == "actix-or-rocket":
                method, route_path = groups
                handler = ""
            else:
                method, route_path = groups[:2]
                handler = groups[2] if len(groups) > 2 and groups[2] else ""
            line = text.count("\n", 0, match.start()) + 1
            start = len(text[: match.start()].encode("utf-8"))
            end = len(text[: match.end()].encode("utf-8"))
            routes.append(_Route(
                method.upper().replace("MAP", "") if method else "*",
                route_path, "", handler.rsplit(".", 1)[-1], selected_framework,
                line, start, end, "bounded-framework-pattern",
            ))
    if language in {"javascript", "typescript"} and "/api/" in path.replace("\\", "/"):
        route_path = _next_route(path)
        pattern = re.compile(
            r"(?m)^\s*export\s+(?:async\s+)?function\s+(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s*\("
        )
        for match in pattern.finditer(text):
            start = len(text[: match.start()].encode("utf-8"))
            end = len(text[: match.end()].encode("utf-8"))
            routes.append(_Route(
                match.group(1), route_path, "", match.group(1), "nextjs",
                text.count("\n", 0, match.start()) + 1, start, end,
                "bounded-framework-pattern",
            ))
    return routes


def _next_route(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    index = parts.index("api")
    route = "/" + "/".join(parts[index:])
    route = re.sub(r"/route\.(?:js|ts|jsx|tsx)$", "", route)
    return re.sub(r"\[([^]]+)\]", r"{\1}", route)


def _mount_paths(
    owner: str,
    mounts: Iterable[_Mount],
    prefixes: Mapping[str, str],
    *,
    limit: int = 64,
) -> tuple[list[tuple[str, list[dict[str, object]]]], bool]:
    parents: dict[str, list[_Mount]] = defaultdict(list)
    for mount in mounts:
        parents[mount.child].append(mount)
    results: list[tuple[str, list[dict[str, object]]]] = []
    stack = [(owner, prefixes.get(owner, ""), [], frozenset({owner}))]
    truncated = False
    while stack:
        current, prefix, chain, seen = stack.pop()
        candidates = sorted(parents.get(current, ()), key=lambda item: (
            item.parent, item.prefix, item.line
        ))
        if not candidates:
            results.append((prefix, chain))
            if len(results) >= limit:
                truncated = bool(stack)
                break
            continue
        for mount in reversed(candidates):
            if mount.parent in seen:
                continue
            next_prefix = _join_path(
                prefixes.get(mount.parent, ""), _join_path(mount.prefix, prefix)
            )
            next_chain = [{
                "parent": mount.parent,
                "child": mount.child,
                "prefix": mount.prefix,
                "line": mount.line,
            }, *chain]
            stack.append((
                mount.parent, next_prefix, next_chain, seen | {mount.parent}
            ))
    return results or [(prefixes.get(owner, ""), [])], truncated


def _handler_binding(
    index: RepositoryIndex,
    path: str,
    handler: str,
    line: int,
) -> dict[str, object]:
    name = handler.rsplit(".", 1)[-1]
    if not name:
        return {"name": "", "status": "unresolved", "candidates": []}
    candidates = [
        symbol for symbol in index.symbols_for_path(path) if symbol.name == name
    ]
    exact = [symbol for symbol in candidates if symbol.line_start >= line]
    if len(exact) == 1:
        candidates = exact
    if len(candidates) == 1:
        symbol = candidates[0]
        return {
            "name": name,
            "status": "resolved",
            "symbol_id": symbol.symbol_id,
            "qualified_name": symbol.qualified_name,
            "candidates": [],
        }
    return {
        "name": name,
        "status": "ambiguous" if candidates else "unresolved",
        "candidates": [symbol.symbol_id for symbol in sorted(
            candidates, key=lambda item: item.symbol_id
        )],
    }


def _verified_sources(
    root: Path,
    index: RepositoryIndex,
) -> tuple[dict[str, bytes], Counter[str]]:
    verified: dict[str, bytes] = {}
    omissions: Counter[str] = Counter()
    for path, record in sorted(index.files.items()):
        try:
            candidate = (root / path).resolve(strict=True)
            candidate.relative_to(root)
            raw = candidate.read_bytes()
        except (OSError, RuntimeError, ValueError):
            omissions["unsafe-or-unreadable"] += 1
            continue
        if hashlib.sha256(raw).hexdigest() != record.sha256:
            omissions["stale-source"] += 1
            continue
        verified[path] = raw
    return verified, omissions


def build_verified_routes(
    root: Path,
    index: RepositoryIndex,
    *,
    index_digest: str,
    method: str | None = None,
    path_prefix: str | None = None,
    max_routes: int = 10_000,
    max_conflicts: int = 1_000,
) -> dict[str, object]:
    """Build static HTTP endpoint evidence without inventing dynamic values."""
    root = root.expanduser().resolve(strict=True)
    route_limit = max(1, min(int(max_routes), 100_000))
    conflict_limit = max(1, min(int(max_conflicts), 10_000))
    wanted_method = method.upper() if method else None
    verified, omissions = _verified_sources(root, index)
    extracted: list[tuple[str, _Route, list[_Mount], Mapping[str, str], bool]] = []
    sources: dict[str, str] = {}

    for path, raw in sorted(verified.items()):
        record = index.files[path]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            omissions["non-utf8"] += 1
            continue
        sources[path] = record.sha256
        if record.language == "python":
            try:
                tree = ast.parse(text, filename=path, type_comments=True)
            except (SyntaxError, ValueError, MemoryError, RecursionError):
                omissions["python-parse-error"] += 1
                continue
            analyzer = _PythonAnalyzer(text)
            analyzer.analyze(tree)
            omissions.update(analyzer.omissions)
            extracted.extend(
                (path, route, analyzer.mounts, analyzer.prefixes, False)
                for route in analyzer.routes
            )
        else:
            extracted.extend(
                (path, route, [], {}, True)
                for route in _pattern_routes(path, record.language, text)
            )

    routes: list[dict[str, object]] = []
    mount_truncated = 0
    for path, route, mounts, prefixes, heuristic in extracted:
        paths, truncated = _mount_paths(route.owner, mounts, prefixes)
        mount_truncated += int(truncated)
        raw = verified[path]
        evidence = raw[route.start_byte:route.end_byte]
        for prefix, mount_chain in paths:
            full_path = _join_path(prefix, route.path)
            if wanted_method and route.method != wanted_method:
                continue
            if path_prefix and not full_path.startswith(path_prefix):
                continue
            binding = _handler_binding(index, path, route.handler, route.line)
            identity = {
                "method": route.method,
                "path": full_path,
                "handler": binding,
                "source": [path, route.start_byte, route.end_byte],
                "mount_chain": mount_chain,
            }
            routes.append({
                "route_id": "route:" + _sha(identity)[:24],
                "method": route.method,
                "path": full_path,
                "normalized_path": _normalize_path(full_path),
                "framework": route.framework,
                "owner": route.owner,
                "handler": binding,
                "source": {
                    "path": path,
                    "line": route.line,
                    "start_byte": route.start_byte,
                    "end_byte": route.end_byte,
                    "source_sha256": index.files[path].sha256,
                    "evidence_sha256": hashlib.sha256(evidence).hexdigest(),
                },
                "mount_chain": mount_chain,
                "extraction": route.extraction,
                "confidence": "heuristic-static" if heuristic else "exact-static",
            })
    routes.sort(key=lambda item: (
        str(item["normalized_path"]), str(item["method"]), str(item["route_id"])
    ))
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for route in routes:
        grouped[(str(route["method"]), str(route["normalized_path"]))].append(route)
    all_conflicts = [
        {
            "method": key[0],
            "normalized_path": key[1],
            "kind": "parameter-shape-or-exact-collision",
            "route_ids": [str(item["route_id"]) for item in values],
            "conclusion": "review-required-not-defect-proof",
        }
        for key, values in sorted(grouped.items())
        if len(values) > 1
    ]
    total = len(routes)
    payload: dict[str, object] = {
        "schema_version": VERIFIED_ROUTES_SCHEMA_VERSION,
        "index_digest": index_digest,
        "filters": {"method": wanted_method, "path_prefix": path_prefix},
        "sources": sources,
        "routes": routes[:route_limit],
        "conflicts": all_conflicts[:conflict_limit],
        "policy": {
            "python": "language-ast-with-static-literal-and-mount-resolution",
            "other_languages": "bounded-disclosed-framework-patterns",
            "dynamic_values": "omit-rather-than-guess",
            "handler_binding": "same-file-symbol-identity-or-explicit-ambiguity",
            "conflicts": "normalized-shape-review-signal-not-defect-proof",
        },
        "truncation": {
            "routes_omitted": max(0, total - route_limit),
            "conflicts_omitted": max(0, len(all_conflicts) - conflict_limit),
            "mount_expansions_truncated": mount_truncated,
        },
        "receipt": {
            "freshness": "verified-against-indexed-source-sha256",
            "verified_file_count": len(verified),
            "route_count_before_output_limit": total,
            "source_manifest_sha256": _sha(sources),
            "omissions_by_reason": dict(sorted(omissions.items())),
            "remote_calls": 0,
            "commitment_scope": (
                "payload-excluding-generation-command-and-routes-sha256"
            ),
        },
    }
    canonical = _canonical(payload)
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt["routes_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def verify_routes_commitment(payload: Mapping[str, object]) -> bool:
    """Verify a detached route-intelligence receipt."""
    try:
        candidate = copy.deepcopy(dict(payload))
        candidate.pop("generation", None)
        candidate.pop("command", None)
        if candidate.get("schema_version") != VERIFIED_ROUTES_SCHEMA_VERSION:
            return False
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("routes_sha256"))
        return hashlib.sha256(_canonical(candidate)).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "VERIFIED_ROUTES_SCHEMA_VERSION",
    "build_verified_routes",
    "verify_routes_commitment",
]
