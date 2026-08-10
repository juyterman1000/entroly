"""Precision of the line-oriented reader used when no grammar is available.

This path is the only call analysis Rust and the JS family get on a base
install, so its false positives are not a corner case -- they are what those
users see. It used to read a function's own declaration as a call to itself:
the two-line TypeScript fixture below produced three edges, of which one was
real, and all three claimed confidence "resolved".
"""

from __future__ import annotations

import pytest

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.parsers import _parse_conservative


def _targets(source: str, language: str) -> list[str]:
    return [
        call.target
        for call in _parse_conservative("x", source, source.encode("utf-8"), language).calls
    ]


@pytest.mark.parametrize(
    ("label", "language", "source", "expected"),
    [
        (
            "declaration on its own line is not a call",
            "typescript",
            "function helper(): number { return 1; }\n"
            "export function run(): number { return helper(); }\n",
            ["helper"],
        ),
        (
            "declaration nested mid-line is not a call",
            # No line-anchored pattern matches `fn run` here, because the line
            # starts with `impl`. This shipped a fabricated `<module> -> run`.
            "rust",
            "fn helper() {}\nstruct Worker;\nimpl Worker { fn run(&self) { helper(); } }\n",
            ["helper"],
        ),
    ],
)
def test_declarations_are_not_read_as_calls(
    label: str, language: str, source: str, expected: list[str]
) -> None:
    assert _targets(source, language) == expected, label


@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("typescript", "function f(): number { return f(); }\n"),
        ("rust", "fn f() -> i32 { f() }\n"),
    ],
)
def test_single_line_recursion_survives_declaration_suppression(
    language: str, source: str
) -> None:
    """Suppressing by name rather than by position would silently delete this.

    Both the declaration and a real recursive call are the same identifier on
    the same line, so this is the case that distinguishes a positional rule
    from a lazy `name == enclosing_symbol` one.
    """
    assert _targets(source, language) == ["f"]


def test_class_method_shorthand_is_a_known_false_positive() -> None:
    """Pinned limitation, not an endorsement.

    `run() { ... }` declares a method, but JS and TS omit any declaration
    keyword there, so it is lexically identical to a call. Without a grammar
    this is not decidable, and suppressing it would need a rule broad enough
    to start deleting real calls. It is recorded here so the boundary is
    visible, and such edges are reported as heuristic rather than resolved.
    """
    assert _targets("class A { run() { helper(); } }\n", "typescript") == ["run", "helper"]


def test_conservative_edges_are_not_labelled_as_resolved(tmp_path, monkeypatch) -> None:
    """A lexical guess must not be indistinguishable from a verified edge.

    These edges were reported with confidence "resolved", exactly like an
    AST-verified Python edge, because the builder never consulted the backend
    that produced the call.

    The grammar is stubbed out rather than skipped on absence, so this runs
    identically whether or not tree-sitter is installed -- the fallback is
    reached the same way a base install reaches it.
    """
    monkeypatch.setattr(
        "entroly.repository_intelligence.parsers.extract_structural_spans",
        lambda *args, **kwargs: None,
    )
    (tmp_path / "index.ts").write_text(
        "function helper(): number { return 1; }\n"
        "export function run(): number { return helper(); }\n",
        encoding="utf-8",
    )
    edges = build_repository_index(tmp_path).call_edges
    assert edges, "fixture produced no call edges"
    assert [edge.callee_id.split("::")[1] for edge in edges] == ["helper"]
    assert {edge.confidence for edge in edges} == {"heuristic-static"}
