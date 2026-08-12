from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from entroly.semantic_resolution import extract_blocks
from entroly.tree_sitter_support import (
    LANGUAGE_BY_SUFFIX,
    _get_local_parser,
    extract_structural_calls,
    extract_structural_spans,
)


def test_language_map_covers_at_least_twenty_seven_languages() -> None:
    # This is only the no-registry fallback now. Runtime breadth comes from the
    # universal language registry and is tested separately.
    assert len(set(LANGUAGE_BY_SUFFIX.values())) >= 27


@pytest.mark.parametrize(
    ("path", "source", "names"),
    [
        ("sample.py", "class Cart:\n    def total(self):\n        return 1\n", {"Cart", "total"}),
        ("sample.rs", "pub struct Cart { n: u32 }\nimpl Cart { pub fn total(&self) -> u32 { self.n } }\n", {"Cart", "total"}),
        ("sample.ts", "export class Cart { total(): number { return 1; } }\n", {"Cart", "total"}),
        ("sample.c", "int total(int price) { return price; }\n", {"total"}),
        # Kotlin's grammar names declarations positionally, with no `name`
        # field, and the pack's structure processing returns the class alone.
        # Both the top-level function and the class's own methods must appear.
        (
            "sample.kt",
            "fun total(price: Int): Int { return price }\n"
            "class Cart {\n    fun add() {}\n    fun clear() {}\n}\n",
            {"total", "Cart", "add", "clear"},
        ),
        # Perl's sub is `subroutine_declaration_statement`, which the suffix
        # heuristic skips; Haskell names a definition with a bare `function`.
        ("sample.pl", "sub total { return 1 }\nsub add { }\n", {"total", "add"}),
        (
            "sample.hs",
            "total :: Int -> Int\ntotal x = x\n\ndata Cart = Empty\n",
            {"total", "Cart"},
        ),
        # Go dropped every `type` -- its name lives in `type_spec`, and the
        # pack structure omitted it. Both the func and the type must appear.
        ("sample.go", "package m\nfunc total() {}\ntype Cart struct{}\n", {"total", "Cart"}),
        # Dart's pack structure returned the class alone, no functions.
        ("sample.dart", "int total(int x){return x;}\nclass Cart{ void add(){} }\n", {"total", "Cart", "add"}),
        # Erlang defines a function as `function_clause`s.
        ("sample.erl", "total(X) -> X.\nadd() -> total(1).\n", {"total", "add"}),
        # Solidity's contract is `contract_declaration`, outside the hint set.
        ("sample.sol", "contract Cart {\n    function total() public {}\n}\n", {"Cart", "total"}),
        # Zig names a function positionally in `FnProto` (a bare IDENTIFIER).
        ("sample.zig", "pub fn total() void {}\nfn add() void {}\n", {"total", "add"}),
        # Protobuf `message` names via `message_name`, no `name` field.
        ("sample.proto", "message Cart {\n    string id = 1;\n}\n", {"Cart"}),
        # R binds a function by assignment; the name is left of the arrow, and
        # the anonymous `function_definition` must not surface as `function`.
        ("sample.R", "total <- function(x) x + 1\nrunner <- function() total(1)\n", {"total", "runner"}),
        # SQL names a created object in `object_reference`.
        (
            "sample.sql",
            "CREATE FUNCTION add_tax(p int) RETURNS int AS $$ SELECT 1 $$ LANGUAGE SQL;\n",
            {"add_tax"},
        ),
        # Svelte/Vue keep the script as opaque text; it is re-parsed as JS and
        # the spans are shifted back to file coordinates.
        ("sample.svelte", "<script>\nfunction total(){}\nclass Cart{}\n</script>\n<div/>\n", {"total", "Cart"}),
        ("sample.vue", "<script>\nfunction total(){}\n</script>\n<template><div/></template>\n", {"total"}),
    ],
)
def test_parser_backed_spans_are_exact(path: str, source: str, names: set[str]) -> None:
    pytest.importorskip("tree_sitter_language_pack")
    spans = extract_structural_spans(source, path)
    assert spans is not None
    assert names <= {span.name for span in spans}
    for span in spans:
        assert span.source in source
        assert source.splitlines()[span.start_line - 1].strip()
        assert span.start_line <= span.end_line


def test_r_function_binding_does_not_leak_the_keyword() -> None:
    """R's anonymous `function_definition` must not surface as `function`.

    A function acquires a name only through assignment, so the binding is the
    declaration; the bare definition node carries the keyword `function` and
    was emitted as a symbol before the binding predicate and suppression.
    """
    pytest.importorskip("tree_sitter_language_pack")
    spans = extract_structural_spans("helper <- function(x) x + 1\nx <- 5\n", "a.R")
    assert spans is not None
    names = {span.name for span in spans}
    assert names == {"helper"}, names


def test_embedded_script_spans_recover_exact_file_bytes() -> None:
    """A Svelte span's byte range must address the real bytes in the file.

    The script is parsed in isolation and its spans are shifted by the script
    block's offset; an off-by-one there would still name the symbol but point
    at the wrong bytes, which the byte-fidelity contract forbids.
    """
    pytest.importorskip("tree_sitter_language_pack")
    source = "<p>hi</p>\n<script>\nfunction total(x){ return x }\n</script>\n"
    spans = extract_structural_spans(source, "a.svelte")
    assert spans is not None and spans
    raw = source.encode("utf-8")
    for span in spans:
        assert raw[span.start_byte:span.end_byte].decode("utf-8") == span.source


def test_kotlin_function_name_is_not_the_return_type_or_a_parameter() -> None:
    """Guards the exact defect: the name resolved to the return type.

    `fun helper(x: Int): Int` has no `name` field, so the generic resolver
    descended into the return type and returned `Int`, and a naive fix then
    pulled in the parameter `x`. The Kotlin-scoped resolver takes the first
    direct `simple_identifier`, which is the name and cannot be either.
    """
    pytest.importorskip("tree_sitter_language_pack")
    spans = extract_structural_spans("fun helper(x: Int): Int { return x }\n", "a.kt")
    assert spans is not None
    names = {span.name for span in spans}
    assert names == {"helper"}, names


def test_dart_function_body_does_not_invent_a_returned_identifier() -> None:
    """A function body is not a declaration even when its type contains that word."""
    pytest.importorskip("tree_sitter_language_pack")
    spans = extract_structural_spans(
        "int total(int x) { return x; }\nclass Cart { void add() {} }\n",
        "a.dart",
    )
    assert spans is not None
    names = {span.name for span in spans}
    assert names == {"total", "Cart", "add"}, names



def test_semantic_resolution_uses_parser_spans_when_available() -> None:
    pytest.importorskip("tree_sitter_language_pack")
    source = "export class Cart {\n  total(): number { return 1; }\n}\n"
    blocks = extract_blocks(source, "cart.ts")
    assert {block.name for block in blocks} >= {"Cart", "total"}
    assert all(block.start_line <= block.end_line for block in blocks)


def test_pack_v1_never_downloads_without_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake = SimpleNamespace(
        downloaded_languages=lambda: ["python"],
        get_parser=lambda language: calls.append(language),
    )
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", fake)
    monkeypatch.delenv("ENTROLY_AIR_GAP", raising=False)
    monkeypatch.delenv("ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD", raising=False)
    assert _get_local_parser("rust") is None
    assert calls == []


def test_default_deny_allows_proven_locally_loadable_grammar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    parser = object()
    fake = SimpleNamespace(
        has_language=lambda language: language == "python",
        get_parser=lambda language: calls.append(language) or parser,
    )
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", fake)
    monkeypatch.delenv("ENTROLY_AIR_GAP", raising=False)
    monkeypatch.delenv("ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD", raising=False)

    assert _get_local_parser("python") is parser
    assert calls == ["python"]
    assert _get_local_parser("rust") is None
    assert calls == ["python"]


def test_explicit_download_opt_in_allows_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    parser = object()
    fake = SimpleNamespace(
        downloaded_languages=lambda: ["python"],
        get_parser=lambda language: calls.append(language) or parser,
    )
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", fake)
    monkeypatch.delenv("ENTROLY_AIR_GAP", raising=False)
    monkeypatch.setenv("ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD", "1")
    assert _get_local_parser("rust") is parser
    assert calls == ["rust"]


def test_explicit_no_download_uses_only_cached_grammars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake = SimpleNamespace(
        downloaded_languages=lambda: ["python"],
        get_parser=lambda language: calls.append(language),
    )
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", fake)
    monkeypatch.delenv("ENTROLY_AIR_GAP", raising=False)
    monkeypatch.setenv("ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD", "0")
    assert _get_local_parser("rust") is None
    assert calls == []


def test_air_gap_overrides_explicit_download_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake = SimpleNamespace(
        downloaded_languages=lambda: ["python"],
        get_parser=lambda language: calls.append(language),
    )
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", fake)
    monkeypatch.setenv("ENTROLY_AIR_GAP", "1")
    monkeypatch.setenv("ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD", "1")
    assert _get_local_parser("rust") is None
    assert calls == []


def test_unknown_or_unavailable_parser_fails_open() -> None:
    assert extract_structural_spans("hello", "notes.txt") is None


@pytest.mark.parametrize(
    ("path", "source", "target"),
    [
        ("sample.py", "def run():\n    return helper()\n", "helper"),
        ("sample.rs", "fn run() { helper(); }\n", "helper"),
        ("sample.ts", "function run() { return client.send(); }\n", "client.send"),
        ("sample.go", "func run() { client.Send() }\n", "client.Send"),
        ("Sample.java", "class Sample { void run() { client.send(); } }\n", "send"),
    ],
)
def test_parser_backed_calls_have_exact_evidence(
    path: str,
    source: str,
    target: str,
) -> None:
    pytest.importorskip("tree_sitter_language_pack")
    calls = extract_structural_calls(source, path)
    assert calls is not None
    call = next(item for item in calls if item.target == target)
    raw = source.encode("utf-8")
    assert raw[call.start_byte:call.end_byte]
    assert len(call.evidence_sha256) == 64
