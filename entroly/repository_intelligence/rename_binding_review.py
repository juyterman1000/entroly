"""Compiler-derived Python rename counterevidence, not semantic certification.

Unknown scopes and runtime effects remain unproven. No reviewed code is executed.
"""
from __future__ import annotations

import ast
import hashlib
import re
import symtable
import sys

MAX_REVIEW_SOURCE_BYTES = 2 * 1024 * 1024
MAX_RETAINED_CONFLICTS = 64
MAX_RETAINED_UNKNOWN_REASONS = 32
MAX_PREVIEW_REVIEW_FILES = 128


def _symbol(table, name):
    try:
        return table.lookup(name)
    except KeyError:
        return None


def _bound(table, name):
    symbol = _symbol(table, name)
    # Assignment to a nonlocal/free symbol does not create a local cell.
    return symbol is not None and symbol.is_local()


def _tables(root):
    result = {}

    def visit(table, path):
        result[path] = table
        for index, child in enumerate(table.get_children()):
            visit(child, path + (index,))

    visit(root, ())
    return result


def _owner(tables, path, name):
    table = tables[path]
    symbol = _symbol(table, name)
    if symbol is None:
        return None
    if path == () or symbol.is_global():
        return ("scope", ()) if _bound(tables[()], name) else ("external-global", name)
    if symbol.is_free() or symbol.is_nonlocal():
        if name == "__class__":
            return None
        parent = path[:-1]
        while parent:
            candidate = tables[parent]
            if candidate.get_type() == "function" and _bound(candidate, name):
                return "scope", parent
            parent = parent[:-1]
        return None
    if symbol.is_local():
        return "scope", path
    return None


class _Locations(ast.NodeVisitor):
    def __init__(self, raw, tables, names):
        self.raw = raw
        self.tables = tables
        self.names = names
        self.path = ()
        self.loads = []
        self.bindings = []
        self.foreign = set()
        self.unknown = []
        self.offsets = [0]
        for line in raw.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))

    def span(self, node, name):
        start = self.offsets[node.lineno - 1] + node.col_offset
        end = start + len(name.encode("utf-8"))
        return (start, end) if self.raw[start:end] == name.encode("utf-8") else None

    def record(self, node, name, role, span=None):
        if name not in self.names:
            return
        span = span or self.span(node, name)
        if span is None:
            self.unknown.append("unmapped-identifier")
            return
        item = {"name": name, "span": span, "scope": self.path,
                "line": node.lineno, "role": role}
        (self.loads if role == "load" else self.bindings).append(item)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.record(node, node.id, "load")
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.record(node, node.id, "binding-write")

    def unsupported(self, node):
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in self.names:
                self.unknown.append("unsupported-scope:" + type(node).__name__)

    def definition(self, node):
        if node.name in self.names:
            start = self.offsets[node.lineno - 1] + node.col_offset
            pattern = rb"(?:async[ \t]+)?(?:def|class)[ \t]+(" + re.escape(node.name.encode()) + rb")\b"
            match = re.match(pattern, self.raw[start:])
            if match:
                self.record(node, node.name, "definition", (start + match.start(1), start + match.end(1)))
            else:
                self.unknown.append("unsupported-definition-header")

    def visit_FunctionDef(self, node):
        self.definition(node)
        # Defaults/decorators are evaluated in the enclosing scope. Deferred
        # annotations absent from the compiler table become unknown, not safe.
        for item in (*node.decorator_list, *node.args.defaults,
                     *(value for value in node.args.kw_defaults if value is not None)):
            self.visit(item)
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)
        children = [
            path for path, table in self.tables.items()
            if len(path) == len(self.path) + 1 and path[:-1] == self.path
            and table.get_type() == "function" and table.get_lineno() == node.lineno
            and table.get_name() == node.name
        ]
        if len(children) != 1 or getattr(node, "type_params", ()):
            self.unsupported(node)
            return
        previous, self.path = self.path, children[0]
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        arguments += [value for value in (node.args.vararg, node.args.kwarg) if value is not None]
        for argument in arguments:
            self.record(argument, argument.arg, "parameter")
        for statement in node.body:
            self.visit(statement)
        self.path = previous

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.definition(node)
        self.unsupported(node)

    visit_Lambda = unsupported
    visit_ListComp = unsupported
    visit_SetComp = unsupported
    visit_DictComp = unsupported
    visit_GeneratorExp = unsupported

    def visit_ImportFrom(self, node):
        for alias in node.names:
            span = self.span(alias, alias.name)
            if alias.asname is None:
                self.record(alias, alias.name, "import-binding", span)
            else:
                if span is not None:
                    self.foreign.add(span)
                end = self.offsets[alias.end_lineno - 1] + alias.end_col_offset
                start = end - len(alias.asname.encode())
                if self.raw[start:end] == alias.asname.encode():
                    self.record(alias, alias.asname, "import-alias", (start, end))

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            span = self.span(alias, name)
            if alias.asname is not None:
                end = self.offsets[alias.end_lineno - 1] + alias.end_col_offset
                start = end - len(name.encode())
                span = (start, end) if self.raw[start:end] == name.encode() else None
            self.record(alias, name, "import-alias" if alias.asname else "import-binding", span)
            if name in self.names:
                # Mapping a module import is counterevidence for overwrites;
                # module-import migration itself remains outside this review.
                self.unknown.append("unsupported-module-import-binding")

    def visit_ExceptHandler(self, node):
        if node.name in self.names:
            # Exception targets are implicit writes followed by implicit
            # deletion. They must not be mistaken for the imported global.
            self.unknown.append("unsupported-exception-binding")
        self.generic_visit(node)

    def visit_MatchAs(self, node):
        if node.name in self.names:
            self.unknown.append("unsupported-pattern-binding")
        self.generic_visit(node)

    visit_MatchStar = visit_MatchAs

    def visit_MatchMapping(self, node):
        if node.rest in self.names:
            self.unknown.append("unsupported-pattern-binding")
        self.generic_visit(node)


def python_module_binding_evidence(path: str, raw: bytes, tree: ast.AST, name: str) -> dict:
    """Locate compiler-owned module loads/writes without executing source.

    Shared with rename discovery so scope and shadowing rules cannot drift
    from the independent staged binding review. Unsupported scopes stay unknown.
    """
    tables = _tables(symtable.symtable(raw.decode("utf-8"), path, "exec"))
    locations = _Locations(raw, tables, {name})
    locations.visit(tree)
    return {
        "loads": [item["span"] for item in locations.loads
                  if _owner(tables, item["scope"], name) == ("scope", ())],
        "writes": [item for item in locations.bindings
                   if _owner(tables, item["scope"], name) == ("scope", ())],
        "unknown": locations.unknown,
    }


def review_python_rename_bindings(path: str, before: bytes, after: bytes, changes: list[dict]):
    before_sha = hashlib.sha256(before).hexdigest()
    result = {"schema_version": "entroly.python-rename-binding-review.v1", "path": path,
              "source_sha256": before_sha, "after_sha256": hashlib.sha256(after).hexdigest(),
              "compiler": sys.implementation.cache_tag, "conflicts": [], "unknown": [],
              "semantic_status": "not-checked", "checked_edits": 0}
    if max(len(before), len(after)) > MAX_REVIEW_SOURCE_BYTES:
        result["unknown"].append("source-size-limit")
        result["status"] = "unknown"
        return _finish_review(result)
    pairs = {(item["old_identifier"], item["new_identifier"]) for item in changes}
    if len(pairs) != 1 or not path.endswith((".py", ".pyi", ".pyw")):
        result["unknown"].append("unsupported-edit-kind")
        result["status"] = "unknown"
        return _finish_review(result)
    old, new = next(iter(pairs))
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", new) or old == new:
        result["unknown"].append("unsupported-identifier")
        result["status"] = "unknown"
        return _finish_review(result)
    edits = {(int(item["start_byte"]), int(item["end_byte"])): item for item in changes}
    if len(edits) != len(changes):
        raise ValueError("duplicate edit spans")
    staged, cursor = before, len(before)
    for (start, end), item in sorted(edits.items(), reverse=True):
        if not (0 <= start < end <= cursor) or before[start:end] != old.encode():
            raise ValueError("invalid or overlapping edit preimage")
        staged = staged[:start] + new.encode() + staged[end:]
        cursor = start
    if staged != after:
        raise ValueError("after source does not match the exact edits")
    try:
        tree = ast.parse(before)
        # AST/symbol-table construction alone can accept invalid assignments
        # such as a function named __debug__. Compile, but never execute, the
        # staged module before treating its binding evidence as reviewable.
        compile(after, path, "exec", dont_inherit=True)
        before_tables = _tables(symtable.symtable(before.decode("utf-8"), path, "exec"))
        after_tables = _tables(symtable.symtable(after.decode("utf-8"), path, "exec"))
    except (SyntaxError, UnicodeError, ValueError, RecursionError) as exc:
        result["unknown"].append("compiler-rejected:" + type(exc).__name__)
        result["status"] = "unknown"
        return _finish_review(result)
    aligned = before_tables.keys() == after_tables.keys() and all(
        before_tables[key].get_type() == after_tables[key].get_type()
        and before_tables[key].get_lineno() == after_tables[key].get_lineno()
        and after_tables[key].get_name() in (before_tables[key].get_name(), new)
        for key in before_tables
    )
    if not aligned:
        result["unknown"].append("scope-origin-alignment-unproven")
        result["status"] = "unknown"
        return _finish_review(result)
    locations = _Locations(before, before_tables, {old, new})
    locations.visit(tree)
    result["unknown"].extend(locations.unknown)
    recognized = set(locations.foreign) & set(edits)

    def conflict(kind, item, **extra):
        start, end = item["span"]
        result["conflicts"].append({
            "kind": kind, "line": item["line"], "scope": item["scope"],
            "start_byte": start, "end_byte": end,
            "identifier": item["name"],
            "evidence_sha256": hashlib.sha256(before[start:end]).hexdigest(),
            **extra,
        })

    renamed_slots = set()
    for item in locations.bindings:
        if item["name"] != old or item["span"] not in edits:
            continue
        recognized.add(item["span"])
        if not _bound(before_tables[item["scope"]], old):
            result["unknown"].append("non-local-binding-write")
            continue
        renamed_slots.add(item["scope"])
        if _bound(before_tables[item["scope"]], new):
            witnesses = [other for other in locations.bindings
                         if other["name"] == new and other["scope"] == item["scope"]]
            conflict("existing-binding-merged", item,
                     existing_binding_ranges=[other["span"] for other in witnesses])
    for item in locations.bindings:
        if item["name"] == old and item["scope"] in renamed_slots and item["span"] not in edits:
            conflict("old-binding-write-left-behind", item)
    for item in locations.loads:
        edited = item["span"] in edits
        if edited:
            recognized.add(item["span"])
        target = new if edited else item["name"]
        origin_before = _owner(before_tables, item["scope"], item["name"])
        origin_after = _owner(after_tables, item["scope"], target)
        if origin_before is None or origin_after is None:
            result["unknown"].append("unresolved-load-owner")
        elif edited and (origin_before[0] != "scope" or origin_after[0] != "scope"):
            result["unknown"].append("external-load-owner")
            if origin_before != origin_after and origin_before[0] == "scope":
                conflict("renamed-load-lost-binding", item,
                         before_owner=origin_before, after_owner=origin_after)
        elif origin_before != origin_after:
            conflict("load-owner-changed" if edited else "unmodified-load-owner-changed", item,
                     before_owner=origin_before, after_owner=origin_after)
        elif edited and origin_before[0] == "scope" and origin_before[1] not in renamed_slots:
            # Two names can belong to one function while referring to distinct
            # parameters/locals. Equal scope paths are insufficient unless the
            # corresponding binding was actually renamed.
            original_bindings = [binding for binding in locations.bindings
                                 if binding["name"] == old
                                 and binding["scope"] == origin_before[1]]
            if original_bindings:
                conflict("load-target-binding-not-renamed", item,
                         before_owner=origin_before, after_owner=origin_after)
            else:
                result["unknown"].append("unmapped-load-binding-rename")
    result["checked_edits"] = len(recognized)
    result["unknown"].extend("unmapped-edit-location" for span in edits if span not in recognized)
    result["unknown"] = sorted(set(result["unknown"]))
    result["status"] = "conflict" if result["conflicts"] else (
        "unknown" if result["unknown"] else "no-conflict-in-checked-bindings"
    )
    return _finish_review(result)

def _finish_review(result):
    count = len(result["conflicts"])
    result["conflict_count"] = count
    result["omitted_conflict_count"] = max(0, count - MAX_RETAINED_CONFLICTS)
    result["conflicts"] = result["conflicts"][:MAX_RETAINED_CONFLICTS]
    reasons = sorted(set(result["unknown"]))
    result["omitted_unknown_reason_count"] = max(0, len(reasons) - MAX_RETAINED_UNKNOWN_REASONS)
    result["unknown"] = reasons[:MAX_RETAINED_UNKNOWN_REASONS]
    return result


def stage_identifier_changes(before: bytes, changes: list[dict]) -> bytes:
    """Render an exact no-write preview; apply separately rechecks preimages."""
    spans = [(int(item["start_byte"]), int(item["end_byte"])) for item in changes]
    if len(set(spans)) != len(spans):
        raise ValueError("duplicate edit spans")
    result, cursor = before, len(before)
    for item in sorted(changes, key=lambda value: int(value["start_byte"]), reverse=True):
        start, end = int(item["start_byte"]), int(item["end_byte"])
        old, new = str(item["old_identifier"]).encode(), str(item["new_identifier"]).encode()
        if not (0 <= start < end <= cursor) or before[start:end] != old:
            raise ValueError("invalid or overlapping edit preimage")
        result = result[:start] + new + result[end:]
        cursor = start
    return result


def summarize_binding_reviews(files: list[dict], *, omitted_files: int = 0) -> dict:
    """Summarize bounded preview evidence without issuing a safety certificate."""
    conflicts = sum(item.get("conflict_count", len(item["conflicts"])) for item in files)
    unknown = omitted_files > 0 or not files or any(item["status"] == "unknown" for item in files)
    return {
        "schema_version": "entroly.rename-binding-review.v1",
        "status": "conflict" if conflicts else "unknown" if unknown else "no-conflict-in-checked-bindings",
        "semantic_status": "not-checked",
        "conflict_count": conflicts,
        "reviewed_file_count": len(files),
        "omitted_file_count": omitted_files,
        "files": files,
    }


def reject_rename_binding_conflicts(originals: dict[str, bytes], updated: dict[str, bytes],
                                    grouped: dict[str, list[dict]]) -> None:
    """Recompute from staged bytes; never trust a caller's preview verdict."""
    for path in sorted(updated):
        review = review_python_rename_bindings(path, originals[path], updated[path], grouped[path])
        if review["conflict_count"]:
            first = review["conflicts"][0]
            raise ValueError(
                f"rename binding conflict in {path}:{first['line']}: {first['kind']}; "
                "use a different name or an explicit binding-preserving edit"
            )
        if "compiler-rejected:SyntaxError" in review["unknown"]:
            raise ValueError(f"rename staged Python compiler check failed in {path}")

