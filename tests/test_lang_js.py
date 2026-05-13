"""
Tests for JS / TS symbol extraction (entroly.verifiers.lang_js).

The module is the JavaScript / TypeScript companion to symbol_resolution.py.
It is intentionally regex-based (no tree-sitter dep) — these tests pin its
behavior on a handful of canonical JS/TS patterns so the regex set can be
maintained without quietly breaking the manifest used by the hallucination
verifier.

Coverage targets:
  - ES6 imports (default, named, namespace)
  - CommonJS require (plain + destructured)
  - function / class / const-arrow / const-function declarations
  - reference extraction (calls + identifier reads)
  - module scope detection (which names are bound from the file's
    own definitions + imports)
"""
from __future__ import annotations

from entroly.verifiers import (
    JSSymbolReference,
    collect_js_repo_symbols,
    extract_js_definitions,
    extract_js_scope,
    extract_js_symbol_refs,
)


# ── Definition extraction ────────────────────────────────────────────


def test_function_decl():
    defs = extract_js_definitions("function parseUrl(s) { return s; }")
    assert "parseUrl" in defs


def test_async_function_decl():
    defs = extract_js_definitions("async function fetchUser(id) { return id; }")
    assert "fetchUser" in defs


def test_class_decl():
    defs = extract_js_definitions("class Cache { constructor() {} }")
    assert "Cache" in defs


def test_export_default_class():
    defs = extract_js_definitions("export default class Engine { }")
    assert "Engine" in defs


def test_const_arrow():
    defs = extract_js_definitions("const handler = (req) => req.body;")
    assert "handler" in defs


def test_const_function_expression():
    defs = extract_js_definitions("const parse = function(s) { return s; }")
    assert "parse" in defs


def test_multiple_definitions_one_file():
    src = """
    function alpha() {}
    class Beta {}
    const gamma = () => null;
    const delta = function() {};
    """
    defs = extract_js_definitions(src)
    assert {"alpha", "Beta", "gamma", "delta"}.issubset(defs)


# ── Scope (definitions + imports) ────────────────────────────────────


def test_import_default_in_scope():
    scope = extract_js_scope("import React from 'react';")
    assert "React" in scope


def test_import_named_in_scope():
    scope = extract_js_scope("import { useState, useEffect } from 'react';")
    assert "useState" in scope
    assert "useEffect" in scope


def test_import_namespace_in_scope():
    scope = extract_js_scope("import * as fs from 'fs';")
    assert "fs" in scope


def test_require_in_scope():
    scope = extract_js_scope("const path = require('path');")
    assert "path" in scope


def test_require_destructured_in_scope():
    scope = extract_js_scope("const { readFile, writeFile } = require('fs');")
    assert "readFile" in scope
    assert "writeFile" in scope


def test_definitions_also_in_scope():
    """File-level scope includes both imports AND own definitions."""
    src = """
    import React from 'react';
    function MyComponent() {}
    """
    scope = extract_js_scope(src)
    assert "React" in scope
    assert "MyComponent" in scope


# ── Reference extraction ─────────────────────────────────────────────


def test_function_call_reference():
    refs = extract_js_symbol_refs("parseUrl('http://example.com');")
    names = [r.name for r in refs]
    assert "parseUrl" in names


def test_kinds_are_marked():
    refs = extract_js_symbol_refs("React.createElement('div');")
    # Each reference must carry a kind label the verifier can use
    assert all(isinstance(r, JSSymbolReference) for r in refs)
    assert all(r.kind for r in refs)


def test_reference_carries_line_number():
    src = "// line 1\n// line 2\nparseUrl('x');"
    refs = extract_js_symbol_refs(src)
    parse_url = next((r for r in refs if r.name == "parseUrl"), None)
    assert parse_url is not None
    assert parse_url.line == 3


# ── Repository-level collection ──────────────────────────────────────


def test_collect_js_repo_symbols_walks_directory(tmp_path):
    """collect_js_repo_symbols should find definitions across multiple
    JS/TS files in a project tree."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "utils.js").write_text(
        "export function hashUrl(u) { return u; }\n"
        "export const cacheKey = (s) => s.toLowerCase();\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "main.ts").write_text(
        "import { hashUrl } from './utils';\n"
        "export class Router {}\n",
        encoding="utf-8",
    )
    symbols = collect_js_repo_symbols(str(tmp_path))
    assert "hashUrl" in symbols
    assert "cacheKey" in symbols
    assert "Router" in symbols


def test_collect_js_repo_symbols_ignores_node_modules(tmp_path):
    """node_modules is the largest source of irrelevant symbols.
    Skipping it is the load-bearing part of repo-scale extraction."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "real.js").write_text(
        "export function realFunction() {}\n", encoding="utf-8",
    )
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text(
        "export function junkFromDep() {}\n", encoding="utf-8",
    )
    symbols = collect_js_repo_symbols(str(tmp_path))
    assert "realFunction" in symbols
    assert "junkFromDep" not in symbols, (
        "node_modules symbols leaked into the manifest — would poison "
        "the hallucination verifier with thousands of irrelevant matches"
    )
