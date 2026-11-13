"""Guard against false exact-recovery claims in the dogfood evidence harness."""
from __future__ import annotations

import ast

from benchmarks.agentic_arms import Arm, BuiltContext, Fragment, build_closed_loop
from benchmarks.intensive_dogfood_trials import CASES, ROOT, recover_omitted_for_query, source_corpus


def test_source_corpus_keeps_entire_original_definition() -> None:
    corpus = source_corpus()
    assert len(corpus) == len(CASES)
    assert len({f.source for f in corpus}) == len(corpus)
    extended = 0
    for fragment, (relative_path, symbol, _) in zip(corpus, CASES):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        nodes = [n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == symbol]
        assert len(nodes) == 1
        original = ast.get_source_segment(text, nodes[0])
        assert fragment.content == original
        extended += len(original) > 2700
    assert extended >= 7, "Regression fixture should expose the old silent truncation"


def test_retrieval_uses_query_not_oracle_target() -> None:
    a = Fragment(source="pkg/payments.py::reserve", content="def reserve():\n    return True")
    b = Fragment(source="pkg/billing.py::settle", content="def settle():\n    return True")
    assert recover_omitted_for_query("How does reserve prevent duplicate billing?", [a, b]) == [a]
    assert recover_omitted_for_query("The receipt is missing", [a, b]) == []


def test_ambiguous_symbol_names_refuse_recovery() -> None:
    a = Fragment("pkg/a.py::parse", "def parse(x): return x")
    b = Fragment("pkg/b.py::parse", "def parse(y): return y")
    assert recover_omitted_for_query("How does parse work?", [a, b]) == []
    assert recover_omitted_for_query("Explain parse and verify", [a, Fragment("pkg/c.py::verify", "def verify(): pass")]) == []


def test_recovered_content_must_match_omitted_digest() -> None:
    original = Fragment("pkg/api.py::validate", "def validate():\n    return True\n")
    selected = Fragment(original.source, "def validate(): ...")
    first = BuiltContext(arm=Arm.COMPRESS, text=selected.content, included=[selected], omitted=[original])
    wrong = Fragment(original.source, "def validate():\n    return False\n")
    result = build_closed_loop(first, recovered=[wrong])
    assert result.included == [selected]
    assert result.omitted == [original]
    restored = build_closed_loop(first, recovered=recover_omitted_for_query("inspect validate", first.omitted))
    assert restored.included == [original]
    assert restored.omitted == []
