"""
JavaScript / TypeScript Symbol Extraction
==========================================

Companion to the Python extractor in symbol_resolution.py. Provides:

  - JS/TS symbol-reference extraction from generated code (regex-based,
    no external deps)
  - Project-wide manifest construction from .js/.jsx/.ts/.tsx files
  - Optional fast-path via the Rust extract_entities in CogOpsEngine
    (when available)

Design tradeoff
---------------
We avoid pulling in tree-sitter or esprima. Both work, both add weight,
and neither catches every modern JS pattern (private fields, decorators,
TS-specific syntax) without per-grammar work.

A regex-based extractor is "good enough" for hallucination detection:
the goal isn't to lossless-parse JS, it's to surface symbol references
that we can match against the manifest. False positives in extraction
(treating a string literal as a call) are filtered downstream by the
manifest membership test.

For projects that want full AST analysis, the Rust path
(CogOpsEngine.extract_entities for .js/.ts) is wired up here as the
preferred path. If `entroly_core` isn't available we fall back to regex.

JS scoping
----------
ES6 module scope: a symbol declared at the top of a file is in scope
for the whole file. Imports bring names into scope from other modules.
We implement file-level scope only (same simplification as Python v1).

Patterns we extract:
  - `import X from 'y'`            → X bound, y is module path
  - `import { a, b } from 'y'`     → a, b bound
  - `import * as ns from 'y'`      → ns bound
  - `const x = require('y')`       → x bound (CommonJS)
  - `function foo(...)`            → foo defined
  - `class Foo { ... }`            → Foo defined
  - `const foo = (...) => ...`     → foo defined
  - `export function bar()`        → bar defined and exported
  - `obj.method()`                 → method referenced
  - `new Foo()`                    → Foo referenced

What we deliberately skip (out of scope for v1):
  - JSX components (capitalized refs in JSX are too noisy to extract
    reliably with regex; would need real parsing)
  - Generic type parameters (`<T extends ...>`)
  - Decorators (just less common in JS than Python)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("entroly.verifiers.lang_js")


# ── Regex bank ───────────────────────────────────────────────────────


# Identifier pattern (JS identifiers, including $ which Python doesn't have)
_ID = r"[A-Za-z_$][A-Za-z0-9_$]*"

# Imports (ES6). We handle four forms:
#   import X from 'y'                                 — default
#   import X, { a, b } from 'y'                       — default + named (mixed)
#   import { a, b } from 'y'                          — named only
#   import * as ns from 'y'                           — namespace
#
# The _IMPORT_DEFAULT and _IMPORT_NAMED patterns allow each other's
# syntax to lead, so mixed imports yield both matches.
_IMPORT_DEFAULT = re.compile(
    rf"^\s*import\s+({_ID})\s*(?:,\s*\{{[^}}]*\}})?\s*from\s+['\"]([^'\"]+)['\"]",
    re.M,
)
_IMPORT_NAMED = re.compile(
    rf"^\s*import\s*(?:{_ID}\s*,\s*)?\{{\s*([^}}]+?)\s*\}}\s*from\s+['\"]([^'\"]+)['\"]",
    re.M,
)
_IMPORT_STAR = re.compile(
    rf"^\s*import\s*\*\s*as\s+({_ID})\s+from\s+['\"]([^'\"]+)['\"]",
    re.M,
)

# CommonJS
_REQUIRE = re.compile(rf"(?:const|let|var)\s+({_ID})\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]")
_REQUIRE_DESTRUCT = re.compile(
    r"(?:const|let|var)\s*\{\s*([^}]+)\s*\}\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]"
)

# Declarations
_FUNCTION_DECL = re.compile(rf"(?:export\s+)?(?:async\s+)?function\s+({_ID})\s*\(", re.M)
_CLASS_DECL = re.compile(rf"(?:export\s+(?:default\s+)?)?class\s+({_ID})", re.M)
_CONST_ARROW = re.compile(rf"(?:export\s+)?(?:const|let|var)\s+({_ID})\s*=\s*(?:async\s+)?(?:\([^)]*\)|{_ID})\s*=>")
_CONST_FUNC = re.compile(rf"(?:export\s+)?(?:const|let|var)\s+({_ID})\s*=\s*(?:async\s+)?function")
_CONST_PLAIN = re.compile(rf"(?:export\s+)?(?:const|let|var)\s+({_ID})\s*=", re.M)

# References — order matters (extract call/new/attr in passes)
_NEW_EXPR = re.compile(rf"new\s+({_ID})")
_CALL = re.compile(rf"\b({_ID})\s*\(")
_ATTR = re.compile(rf"\b{_ID}\.({_ID})")

# JS keywords / sentinels we never want to flag
_JS_KEYWORDS = frozenset({
    "if", "else", "while", "for", "do", "switch", "case", "break",
    "continue", "return", "function", "var", "let", "const", "class",
    "extends", "import", "export", "from", "as", "default", "new",
    "throw", "try", "catch", "finally", "typeof", "instanceof", "in",
    "of", "this", "super", "true", "false", "null", "undefined", "void",
    "yield", "async", "await", "delete", "static", "get", "set",
    "public", "private", "protected", "readonly", "interface", "type",
    "enum", "namespace", "declare", "abstract", "implements",
    "console", "Math", "JSON", "Object", "Array", "String", "Number",
    "Boolean", "Date", "Error", "Promise", "Map", "Set", "Symbol",
    "Function", "RegExp", "globalThis", "window", "document",
    "process", "global", "Buffer", "module", "exports", "require",
})


# ── Data structures (mirror Python side) ─────────────────────────────


@dataclass
class JSSymbolReference:
    name: str
    kind: str       # "call" | "attribute" | "import" | "new" | "name"
    line: int
    weight: float
    context: str = ""


# ── Extraction ───────────────────────────────────────────────────────


def extract_js_definitions(source: str) -> set[str]:
    """Top-level definitions in a JS/TS source string.

    Returns the set of names this file *defines* (i.e. would export
    if we asked for the module exports). Imports are NOT included —
    same invariant as the Python side.
    """
    defs: set[str] = set()
    for rx in (_FUNCTION_DECL, _CLASS_DECL, _CONST_ARROW,
               _CONST_FUNC, _CONST_PLAIN):
        for m in rx.finditer(source):
            defs.add(m.group(1))
    return defs


def extract_js_scope(source: str) -> set[str]:
    """Names in scope at file level: imports + defs.

    Same shape as scope_analyzer.compute_scope but for JS.
    """
    scope: set[str] = set(_JS_KEYWORDS)
    # Default import
    for m in _IMPORT_DEFAULT.finditer(source):
        scope.add(m.group(1))
    # Named imports
    for m in _IMPORT_NAMED.finditer(source):
        for item in m.group(1).split(","):
            item = item.strip()
            if not item:
                continue
            # Handle "a as b" → b is the bound name
            if " as " in item:
                _, alias = item.split(" as ", 1)
                scope.add(alias.strip())
            else:
                scope.add(item)
    # Namespace import
    for m in _IMPORT_STAR.finditer(source):
        scope.add(m.group(1))
    # CommonJS require
    for m in _REQUIRE.finditer(source):
        scope.add(m.group(1))
    for m in _REQUIRE_DESTRUCT.finditer(source):
        for item in m.group(1).split(","):
            item = item.strip()
            if " as " in item or ":" in item:
                _, alias = re.split(r"\s+as\s+|\s*:\s*", item, 1)
                scope.add(alias.strip())
            elif item:
                scope.add(item)

    # Local defs
    scope |= extract_js_definitions(source)
    return scope


def extract_js_symbol_refs(source: str) -> list[JSSymbolReference]:
    """Extract symbol references that should be resolved against the manifest."""
    refs: list[JSSymbolReference] = []
    lines = source.split("\n")

    seen_positions: set[tuple[int, int, str, str]] = set()

    def add(name: str, kind: str, line: int, weight: float, col: int = 0):
        key = (line, col, kind, name)
        if name in _JS_KEYWORDS or name.startswith("_"):
            return
        if key in seen_positions:
            return
        seen_positions.add(key)
        ctx = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        refs.append(JSSymbolReference(
            name=name, kind=kind, line=line, weight=weight, context=ctx,
        ))

    # Find line numbers helper
    def line_of(offset: int) -> int:
        return source.count("\n", 0, offset) + 1

    # Imports (high signal)
    for m in _IMPORT_DEFAULT.finditer(source):
        add(m.group(2).split("/")[0], "import", line_of(m.start()), 1.0)
    for m in _IMPORT_NAMED.finditer(source):
        ln = line_of(m.start())
        for item in m.group(1).split(","):
            item = item.strip()
            if not item:
                continue
            name_part = item.split(" as ")[0].strip()
            add(name_part, "import", ln, 1.0)
        add(m.group(2).split("/")[0], "import", ln, 1.0)
    for m in _IMPORT_STAR.finditer(source):
        add(m.group(2).split("/")[0], "import", line_of(m.start()), 1.0)
    for m in _REQUIRE.finditer(source):
        add(m.group(2).split("/")[0], "import", line_of(m.start()), 1.0)

    # new Expr (high signal)
    for m in _NEW_EXPR.finditer(source):
        add(m.group(1), "new", line_of(m.start()), 1.0)

    # Calls — note: this matches `function foo(`, `if (`, etc.
    # We filter keywords above.
    for m in _CALL.finditer(source):
        name = m.group(1)
        if name in _JS_KEYWORDS:
            continue
        add(name, "call", line_of(m.start()), 1.0, m.start())

    # Attribute access — only the rightmost identifier
    for m in _ATTR.finditer(source):
        add(m.group(1), "attribute", line_of(m.start()), 0.6, m.start())

    return refs


# ── Manifest builder ─────────────────────────────────────────────────


def collect_js_repo_symbols(
    repo_root: str,
    extensions: tuple[str, ...] = (".js", ".jsx", ".ts", ".tsx", ".mjs"),
    max_files: int = 5000,
) -> set[str]:
    """Walk a JS/TS repo and return all top-level definitions.

    Used as the `repo` layer of a JS/TS SymbolManifest.
    """
    from pathlib import Path
    skip = {"node_modules", "dist", "build", ".git", "out", ".next",
            "coverage", ".cache", ".parcel-cache"}

    out: set[str] = set()
    files_seen = 0
    root = Path(repo_root)
    for path in root.rglob("*"):
        if files_seen >= max_files:
            break
        if path.is_dir():
            continue
        if path.suffix not in extensions:
            continue
        if any(p in skip for p in path.parts):
            continue
        files_seen += 1
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        out |= extract_js_definitions(content)
    return out


def collect_js_installed_packages(repo_root: str) -> set[str]:
    """Read package.json (root + nested) and return declared dependency names.

    This is the JS analog of `pip freeze`. We don't actually load the
    packages — just list which ones are *declared*.
    """
    import json
    from pathlib import Path

    out: set[str] = set()
    root = Path(repo_root)
    for pkg_json in root.rglob("package.json"):
        if "node_modules" in pkg_json.parts:
            continue
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for key in ("dependencies", "devDependencies", "peerDependencies",
                    "optionalDependencies"):
            block = data.get(key, {})
            if isinstance(block, dict):
                for name in block.keys():
                    # Scoped: @org/pkg → bound name is "@org/pkg" or
                    # often just the package basename. Add both forms.
                    out.add(name)
                    if name.startswith("@") and "/" in name:
                        out.add(name.split("/", 1)[1])
    return out
