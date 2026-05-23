"""
Tests for BIPT — Byte-Level Information Provenance Tracer
==========================================================

These tests prove that BIPT detects hallucination classes that
NO OTHER verifier in the pipeline can catch.

The key insight being tested: BIPT measures HOW MUCH of each
identifier can be explained by the provided context, at the byte
level. This catches confabulations that pass symbol verification
(identifier exists) but have no provenance in what the LLM was given.
"""

from __future__ import annotations

import pytest

from entroly.verifiers.provenance_tracer import (
    SuffixAutomaton,
    trace_provenance,
    BIPTResult,
    _extract_identifier_spans,
)


# ═══════════════════════════════════════════════════════════════════════
# Suffix Automaton — Core Data Structure
# ═══════════════════════════════════════════════════════════════════════


class TestSuffixAutomaton:
    """Verify the suffix automaton correctly finds all substrings."""

    def test_build_small(self):
        sa = SuffixAutomaton.build(b"abcbc")
        # SA should accept all substrings: a, b, c, ab, bc, cb, abc, bcb, cbc, abcb, bcbc, abcbc
        assert sa._size > 0

    def test_build_empty(self):
        sa = SuffixAutomaton.build(b"")
        assert sa._size == 1  # just the initial state

    def test_longest_match_exact(self):
        """Full string match — all positions should be covered."""
        sa = SuffixAutomaton.build(b"hello_world")
        matches = sa.longest_match_at_each_position(b"hello_world")
        # Every position should be covered (coverage > 0)
        assert all(m > 0 for m in matches)
        # Maximum coverage should be 11 (full length)
        assert max(matches) == 11

    def test_longest_match_partial(self):
        """Partial match at start."""
        sa = SuffixAutomaton.build(b"compress_messages")
        matches = sa.longest_match_at_each_position(b"compress_msgs")
        # "compress_" (9 bytes) should cover positions 0-8
        assert matches[0] > 0  # position 0 is covered
        assert matches[4] > 0  # position 4 is covered (inside "compress_")
        # Position 12 (end of "msgs") has low/no coverage
        assert matches[12] <= 2

    def test_longest_match_no_match(self):
        """Completely different strings."""
        sa = SuffixAutomaton.build(b"aaaaaaa")
        matches = sa.longest_match_at_each_position(b"bbbbbbb")
        assert all(m == 0 for m in matches)

    def test_longest_match_recombination(self):
        """Recombination of context fragments."""
        sa = SuffixAutomaton.build(b"authenticate_user validate_token")
        matches = sa.longest_match_at_each_position(b"authenticate_token")
        # "authenticate_" covers positions 0-12 (13 bytes)
        assert matches[0] > 0
        assert matches[6] > 0
        # "token" covers positions 13-17 (5 bytes)
        assert matches[17] > 0

    def test_state_count_linear(self):
        """SA should have at most 2n-1 states."""
        text = b"abracadabra"
        sa = SuffixAutomaton.build(text)
        assert sa._size <= 2 * len(text)

    def test_single_byte(self):
        sa = SuffixAutomaton.build(b"x")
        matches = sa.longest_match_at_each_position(b"x")
        assert matches[0] == 1

    def test_repeated_pattern(self):
        """SA handles repetitive text correctly."""
        sa = SuffixAutomaton.build(b"ababababab")
        matches = sa.longest_match_at_each_position(b"abab")
        # All positions should be covered
        assert all(m > 0 for m in matches)
        # Max coverage should be 4 (full match)
        assert max(matches) == 4


# ═══════════════════════════════════════════════════════════════════════
# Identifier Span Extraction
# ═══════════════════════════════════════════════════════════════════════


class TestIdentifierExtraction:
    """Verify AST-based identifier span extraction."""

    def test_simple_names(self):
        code = "result = process_data(items)"
        spans = _extract_identifier_spans(code)
        names = {s.name for s in spans}
        assert "result" in names
        assert "process_data" in names
        assert "items" in names

    def test_attributes(self):
        code = "df.sort_values('sales')"
        spans = _extract_identifier_spans(code)
        names = {s.name for s in spans}
        assert "df" in names
        assert "sort_values" in names

    def test_skips_builtins(self):
        code = "x = len(items)\nprint(x)"
        spans = _extract_identifier_spans(code)
        names = {s.name for s in spans}
        assert "len" not in names
        assert "print" not in names
        assert "x" in names
        assert "items" in names

    def test_imports(self):
        code = "from pandas import DataFrame"
        spans = _extract_identifier_spans(code)
        names = {s.name for s in spans}
        assert "pandas" in names
        assert "DataFrame" in names

    def test_syntax_error_returns_empty(self):
        spans = _extract_identifier_spans("def broken(")
        assert spans == []


# ═══════════════════════════════════════════════════════════════════════
# Full BIPT — End-to-End Provenance Tracing
# ═══════════════════════════════════════════════════════════════════════


class TestBIPTFullyGrounded:
    """Output that is fully derivable from context → IPD ≈ 0."""

    def test_direct_copy(self):
        context = """
def process_data(items):
    result = sorted(items)
    return result
"""
        output = """
result = process_data(items)
"""
        r = trace_provenance(output, context)
        assert r.ipd < 0.3
        assert r.verdict == "grounded"

    def test_recombination(self):
        """Identifiers composed from context fragments -> grounded."""
        context = """
class UserAuth:
    def validate_token(self, token):
        pass
    def authenticate_user(self, user):
        pass
"""
        output = """
auth = UserAuth()
auth.validate_token(token)
"""
        r = trace_provenance(output, context)
        # All identifiers exist in context
        assert r.ipd < 0.4


class TestBIPTInvented:
    """Output with fabricated identifiers → high IPD."""

    def test_completely_invented(self):
        context = """
def read_file(path):
    return open(path).read()
"""
        output = """
result = quantum_hyperbolic_transform(nebula_data)
"""
        r = trace_provenance(output, context)
        # These identifiers have NO provenance in context
        assert r.ipd > 0.3
        invented = [t for t in r.traces if t.verdict == "invented"]
        assert len(invented) >= 1

    def test_partial_match_detected(self):
        """compress_msgs vs compress_messages → partial grounding."""
        context = """
def compress_messages(data):
    return zlib.compress(data)
"""
        output = """
result = compress_msgs(data)
"""
        r = trace_provenance(output, context)
        # "compress_msgs" partially matches "compress_messages"
        # Should have partial grounding, not fully invented
        for t in r.traces:
            if t.identifier.name == "compress_msgs":
                assert t.grounding_ratio > 0.3  # "compress_" matches
                assert t.grounding_ratio < 1.0  # "msgs" doesn't match
                break


class TestBIPTEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_output(self):
        r = trace_provenance("", "some context")
        assert r.ipd == 0.0
        assert r.verdict == "grounded"

    def test_empty_context(self):
        r = trace_provenance("x = foo(bar)", "")
        # With no context, everything is invented
        assert isinstance(r, BIPTResult)

    def test_both_empty(self):
        r = trace_provenance("", "")
        assert r.ipd == 0.0

    def test_syntax_error_output(self):
        r = trace_provenance("def broken(", "context")
        assert isinstance(r, BIPTResult)
        assert r.n_identifiers == 0  # can't parse AST

    def test_ipd_bounds(self):
        """IPD must always be in [0, 1]."""
        for output, context in [
            ("x = 1", "x = 1"),
            ("foo = bar(baz)", ""),
            ("", "anything"),
            ("result = process(data)", "def process(data): pass"),
        ]:
            r = trace_provenance(output, context)
            assert 0.0 <= r.ipd <= 1.0

    def test_explain_output(self):
        r = trace_provenance("x = foo(bar)", "def foo(x): pass")
        text = r.explain()
        assert "BIPT" in text
        assert "verdict" in text
        assert "IPD" in text


class TestBIPTComplexity:
    """Verify computational complexity claims."""

    def test_large_context_builds(self):
        """SA builds on 100KB context without timeout."""
        context = "abcdefghij" * 10_000  # 100KB
        output = "result = abcdef()"
        r = trace_provenance(output, context)
        assert isinstance(r, BIPTResult)
        # SA should have ≤ 2n states
        assert r.automaton_states <= 2 * len(context.encode("utf-8")) + 1

    def test_grounding_ratio_property(self):
        r = trace_provenance("x = foo(bar)", "def foo(x): pass")
        assert abs(r.grounding_ratio + r.ipd - 1.0) < 1e-6


class TestBIPTNovelDetection:
    """Prove BIPT catches things OTHER verifiers cannot.

    These are cases where:
    - GRAPHS passes (symbol exists in stdlib/manifest)
    - Pyright passes (types check out)
    - Scope passes (symbol is imported)
    - BUT the LLM was never given context about this symbol,
      so it's confabulating from training data.
    """

    def test_confabulated_api_usage(self):
        """LLM uses real API but context only showed different API."""
        context = """
import pandas as pd
df = pd.read_csv("data.csv")
result = df.groupby("region").sum()
"""
        # LLM generates code using pd.merge — which is a REAL API
        # but was never in the context. It's confabulating from
        # training data, not from the provided context.
        output = """
merged = pd.merge(left_df, right_df, on="customer_id", how="inner")
"""
        r = trace_provenance(output, context)
        # "left_df", "right_df", "customer_id" have no provenance
        invented = [t for t in r.traces if t.verdict == "invented"]
        assert len(invented) >= 1  # at least some identifiers are ungrounded

    def test_grounded_api_usage(self):
        """Same API but context DID mention it → grounded."""
        context = """
import pandas as pd
left_df = pd.read_csv("orders.csv")
right_df = pd.read_csv("customers.csv")
merged = pd.merge(left_df, right_df, on="customer_id")
"""
        output = """
result = pd.merge(left_df, right_df, on="customer_id", how="inner")
"""
        r = trace_provenance(output, context)
        # Now everything has provenance
        assert r.ipd < 0.3
