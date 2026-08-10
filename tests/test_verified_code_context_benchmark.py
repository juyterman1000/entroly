from __future__ import annotations

import pytest

from benchmarks.verified_code_context import run_benchmark, verify
from entroly.tree_sitter_support import extract_structural_spans

# The preregistered matrix demands gold_edge_recall == 1 across Python, Rust,
# TypeScript, Go and Java. Only Python is reachable from the standard library;
# the other four need tree-sitter, which the base `pip install entroly` job does
# not have. Without it the conservative reader emits no Go or Java symbols at
# all and names Rust methods `run` rather than `Worker.run`, so recall lands at
# 0.4 -- correct fail-closed degradation, not a regression.
#
# Probe the capability instead of the package: `extract_structural_spans`
# returns None exactly when `_parse_parser_backed` would fall back, so this
# skips when and only when the contract under test is unreachable. It stays
# fully enforced wherever a parser exists.
_PROBES = (
    ("probe.go", "package main\n\nfunc helper() {}\n"),
    ("probe.java", "class Sample { void helper() {} }\n"),
    ("probe.rs", "fn helper() {}\n"),
    ("probe.ts", "function helper() {}\n"),
)
_UNPARSEABLE = tuple(
    path for path, source in _PROBES if extract_structural_spans(source, path) is None
)


@pytest.mark.skipif(
    bool(_UNPARSEABLE),
    reason=f"structural parsing unavailable for {', '.join(_UNPARSEABLE)}; "
    "the preregistered matrix is a parser-backed contract",
)
def test_verified_code_context_preregistered_matrix() -> None:
    payload = run_benchmark()
    assert verify(payload)
    assert payload["errors"] == []


def test_every_emitted_call_edge_carries_recoverable_evidence() -> None:
    """No edge may be asserted without evidence, parser or not.

    This holds on the base install too, so it is deliberately left ungated.
    The conservative reader used to build edges on the ParsedCall defaults --
    a (0, 0) span with an empty digest -- which meant a TypeScript edge
    validated against zero bytes while claiming a call. Recall may degrade
    without a parser; evidence may not.
    """
    payload = run_benchmark()
    metrics = payload["metrics"]
    assert metrics["edge_evidence_validity"] == 1.0
    assert metrics["fragment_evidence_validity"] == 1.0
    assert metrics["symbol_graph_evidence_validity"] == 1
    assert payload["errors"] == []
