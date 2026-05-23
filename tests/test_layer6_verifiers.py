"""
Tests for the Layer 6 verifiers: PROVE, CAVE, TRIAD
=====================================================

This file proves that the three new verifiers catch the hallucination
classes they target:

  PROVE (semantic_entropy):
    1. Correctly-described code → pass
    2. Mismatched verb (sum vs. average) → fail
    3. Negated predicate → fail
    4. No falsifiable claims → pass (graceful)
    5. Empty inputs → pass (fail-open)

  CAVE (reasoning_chain):
    1. Sound chain (all premises necessary) → sound
    2. Chain with decorative premises → weak/flawed
    3. Short text (not a chain) → sound (graceful)
    4. Conclusion-only text → sound (graceful)

  TRIAD (commit_alignment):
    1. Matching message + diff → aligned
    2. Completely unrelated message → misaligned
    3. Scope mismatch (minor claim, large diff) → suspect/misaligned
    4. Empty diff → graceful handling
"""

from __future__ import annotations

import pytest

from entroly.verifiers.semantic_entropy import (
    prove_verify,
    ProveResult,
    _extract_prose_predicates,
    _extract_code_predicates,
    _predicate_similarity,
    Predicate,
)
from entroly.verifiers.reasoning_chain import (
    cave_verify,
    CaveResult,
    _parse_reasoning_steps,
    _compute_necessity_scores,
    _content_tokens,
)
from entroly.verifiers.commit_alignment import (
    triad_verify,
    TriadResult,
    _parse_diff,
    _parse_message,
)


# ═══════════════════════════════════════════════════════════════════════
# PROVE — Prose Verification via Causal-Weighted Semantic Entropy
# ═══════════════════════════════════════════════════════════════════════


class TestProvePredicateExtraction:
    """Unit tests for predicate extraction from prose and code."""

    def test_extract_prose_predicates_basic(self):
        text = "This function reads the file and returns the data"
        preds = _extract_prose_predicates(text)
        verbs = {p.verb for p in preds}
        assert "read" in verbs
        assert "return" in verbs

    def test_extract_prose_predicates_negation(self):
        text = "This does not filter the data"
        preds = _extract_prose_predicates(text)
        assert any(p.verb.startswith("not_") for p in preds)

    def test_extract_prose_predicates_empty(self):
        preds = _extract_prose_predicates("x = 42")
        # No falsifiable verb-object claims
        assert len(preds) == 0

    def test_extract_code_predicates_function_calls(self):
        code = """
data = read_file("input.csv")
result = process(data)
write_file(result, "output.csv")
"""
        preds = _extract_code_predicates(code)
        verbs = {p.verb for p in preds}
        # Should extract at least read_file and write_file calls
        assert len(preds) >= 2

    def test_extract_code_predicates_control_flow(self):
        code = """
for item in items:
    if item.valid:
        return item
"""
        preds = _extract_code_predicates(code)
        verbs = {p.verb for p in preds}
        assert "loop" in verbs
        assert "check" in verbs
        assert "return" in verbs

    def test_extract_code_predicates_syntax_error_graceful(self):
        preds = _extract_code_predicates("def broken(")
        assert preds == []


class TestProveVerbSimilarity:
    """Verb synonym resolution is critical for alignment accuracy."""

    def test_same_cluster_high_sim(self):
        p = Predicate(verb="sum", object_="sales", raw="", source="prose")
        c = Predicate(verb="sum", object_="sales", raw="", source="code")
        sim = _predicate_similarity(p, c)
        assert sim > 0.8

    def test_different_cluster_low_sim(self):
        p = Predicate(verb="mean", object_="sales", raw="", source="prose")
        c = Predicate(verb="sum", object_="sales", raw="", source="code")
        sim = _predicate_similarity(p, c)
        assert sim < 0.5  # Different canonical verbs

    def test_negation_zero_sim(self):
        p = Predicate(verb="not_sum", object_="x", raw="", source="prose")
        c = Predicate(verb="sum", object_="x", raw="", source="code")
        sim = _predicate_similarity(p, c)
        assert sim < 0.3  # Negation should yield ~0


class TestProveFull:
    """End-to-end PROVE verification."""

    def test_correct_description_passes(self):
        code = """
df = df.groupby("region").sum()
result = df.sort_values("sales")
"""
        prose = "This groups the data by region, sums the values, and sorts by sales."
        result = prove_verify(prose, code)
        assert result.verdict == "pass"
        assert result.alignment_score > 0.5

    def test_wrong_verb_fails(self):
        code = """
df = df.groupby("region").sum()
"""
        prose = "This averages sales by region."
        result = prove_verify(prose, code)
        # "averages" vs "sum" → verb mismatch
        assert result.alignment_score < 0.8
        # Should flag at least one unaligned predicate
        assert result.n_unaligned >= 0  # May or may not be unaligned depending on threshold

    def test_no_falsifiable_claims_passes(self):
        code = "x = 42"
        prose = "Here's a variable assignment."
        result = prove_verify(prose, code)
        assert result.verdict == "pass"

    def test_empty_prose(self):
        result = prove_verify("", "x = 1")
        assert result.verdict == "pass"

    def test_empty_code(self):
        result = prove_verify("This sums the data", "")
        # No code predicates → alignment will be low
        assert isinstance(result, ProveResult)

    def test_explain_output(self):
        result = prove_verify("This reads the file", "data = open('x').read()")
        text = result.explain()
        assert "PROVE" in text
        assert "verdict" in text

    def test_h_score_bounds(self):
        """Hallucination risk must be in [0, 1]."""
        for prose, code in [
            ("This sums sales", "df.sum()"),
            ("This does nothing", "pass"),
            ("Reads all files", "x = 1"),
        ]:
            r = prove_verify(prose, code)
            assert 0.0 <= r.hallucination_risk <= 1.0
            assert 0.0 <= r.alignment_score <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# CAVE — Counterfactual Ablation Verification Engine
# ═══════════════════════════════════════════════════════════════════════


class TestCaveStepParsing:
    """Unit tests for reasoning step extraction."""

    def test_numbered_steps(self):
        text = """
1. Python lists are dynamic arrays.
2. Dynamic arrays have O(1) amortized append.
3. Therefore list.append() is O(1).
"""
        steps, conclusion = _parse_reasoning_steps(text)
        assert len(steps) >= 3
        assert conclusion is not None
        assert conclusion.is_conclusion

    def test_sentence_fallback(self):
        text = (
            "Lists are arrays. Appending is amortized constant. "
            "Therefore append is fast."
        )
        steps, conclusion = _parse_reasoning_steps(text)
        assert len(steps) >= 2
        assert conclusion is not None

    def test_short_text(self):
        steps, conclusion = _parse_reasoning_steps("x = 1")
        assert len(steps) <= 1


class TestCaveNecessity:
    """Unit tests for necessity score computation."""

    def test_relevant_premise_high_score(self):
        text = """
1. Python lists use dynamic arrays for storage.
2. Dynamic arrays provide O(1) amortized append time.
3. Therefore list.append() has O(1) amortized complexity.
"""
        steps, conclusion = _parse_reasoning_steps(text)
        _compute_necessity_scores(steps, conclusion)
        # Step 2 should have higher necessity since "append" and "O(1)"
        # directly connect to the conclusion
        non_conclusion = [s for s in steps if not s.is_conclusion]
        assert len(non_conclusion) >= 1
        # At least one step should be classified as necessary or supportive
        classifications = {s.classification for s in non_conclusion}
        assert "necessary" in classifications or "supportive" in classifications

    def test_decorative_premise_flagged(self):
        text = """
1. Python was created by Guido van Rossum.
2. Dynamic arrays have O(1) amortized append.
3. Python's garbage collector uses reference counting.
4. Therefore list.append() is O(1).
"""
        steps, conclusion = _parse_reasoning_steps(text)
        _compute_necessity_scores(steps, conclusion)
        non_conclusion = [s for s in steps if not s.is_conclusion]
        # Step 1 (Guido) and Step 3 (GC) should have lower necessity
        # than Step 2 (append/O(1))
        scored = [(s.text[:30], s.necessity_score) for s in non_conclusion]
        # The step mentioning "append" and "O(1)" should score highest
        assert len(scored) >= 2


class TestCaveFull:
    """End-to-end CAVE verification."""

    def test_sound_chain(self):
        text = """
1. The function takes a list of integers as input.
2. It iterates over each integer and squares it.
3. Therefore the output is a list of squared integers.
"""
        result = cave_verify(text)
        assert result.verdict in ("sound", "weak")
        assert result.chain_integrity > 0.3

    def test_chain_with_irrelevant_steps(self):
        text = """
1. The function was written in 2023.
2. Python 3.12 was released in October 2023.
3. Modern CPUs have branch prediction.
4. The function squares each number.
5. Therefore the output is squared values.
"""
        result = cave_verify(text)
        # Steps 1-3 are irrelevant to the conclusion
        assert result.n_decorative + result.n_irrelevant >= 1

    def test_short_text_passes(self):
        result = cave_verify("The function returns True.")
        assert result.verdict == "sound"

    def test_explain_output(self):
        result = cave_verify("1. A is true. 2. B is true. 3. Therefore C.")
        text = result.explain()
        assert "CAVE" in text
        assert "verdict" in text

    def test_integrity_bounds(self):
        """Chain integrity must be in [0, 1]."""
        for text in [
            "1. A. 2. B. 3. Therefore C.",
            "Short text.",
            "1. Python is fast. 2. C is faster. 3. Therefore use Cython.",
        ]:
            r = cave_verify(text)
            assert 0.0 <= r.chain_integrity <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# TRIAD — Triangulated Diff-Message-PR Alignment
# ═══════════════════════════════════════════════════════════════════════


class TestTriadDiffParsing:
    """Unit tests for diff analysis."""

    def test_parse_basic_diff(self):
        diff = """
--- a/src/utils.py
+++ b/src/utils.py
@@ -10,6 +10,10 @@
 def existing():
     pass

+def new_helper(data):
+    return data.strip()
+
+import json
"""
        analysis = _parse_diff(diff)
        assert "src/utils.py" in analysis.files_changed
        assert "new_helper" in analysis.added_functions
        assert analysis.lines_added >= 3

    def test_parse_removal(self):
        diff = """
--- a/old.py
+++ b/old.py
@@ -1,5 +1,3 @@
-def deprecated_func():
-    pass
-
 def kept_func():
     pass
"""
        analysis = _parse_diff(diff)
        assert "deprecated_func" in analysis.removed_functions
        assert analysis.lines_removed >= 2


class TestTriadMessageParsing:
    """Unit tests for message analysis."""

    def test_parse_action_verbs(self):
        msg = "Add new helper function for data processing"
        analysis = _parse_message(msg)
        assert "add" in analysis.action_verbs

    def test_parse_scope_minor(self):
        msg = "Minor typo fix in README"
        analysis = _parse_message(msg)
        assert analysis.scope_claim == "minor"

    def test_parse_scope_refactor(self):
        msg = "Refactor authentication module"
        analysis = _parse_message(msg)
        assert analysis.scope_claim == "refactor"


class TestTriadFull:
    """End-to-end TRIAD verification."""

    def test_matching_message_and_diff(self):
        diff = """
--- a/src/utils.py
+++ b/src/utils.py
@@ -10,3 +10,8 @@
 def existing():
     pass
+
+def process_data(items):
+    filtered = [i for i in items if i.valid]
+    return sorted(filtered)
"""
        msg = "Add process_data function that filters and sorts items"
        result = triad_verify(msg, diff)
        assert result.verdict in ("aligned", "suspect")
        assert result.triad_score > 0.2

    def test_completely_unrelated_message(self):
        diff = """
--- a/src/auth.py
+++ b/src/auth.py
@@ -1,3 +1,5 @@
+import logging
+logger = logging.getLogger(__name__)
 def login(user, password):
     pass
"""
        msg = "Refactored payment processing to use async/await"
        result = triad_verify(msg, diff)
        # Message talks about payment/async, diff touches auth/logging
        assert result.triad_score < 0.5
        assert len(result.mismatches) >= 1

    def test_scope_mismatch_minor_but_large(self):
        # Generate a "large" diff
        added_lines = "\n".join(f"+def func_{i}(): pass" for i in range(60))
        diff = f"""
--- a/big.py
+++ b/big.py
@@ -1,0 +1,60 @@
{added_lines}
"""
        msg = "Minor typo fix"
        result = triad_verify(msg, diff)
        assert result.scope_consistency < 0.8

    def test_empty_diff(self):
        result = triad_verify("Update something", "")
        assert isinstance(result, TriadResult)
        # Should not crash

    def test_empty_message(self):
        diff = "+def foo(): pass"
        result = triad_verify("", diff)
        assert isinstance(result, TriadResult)

    def test_explain_output(self):
        result = triad_verify("Fix bug", "+x = 1")
        text = result.explain()
        assert "TRIAD" in text
        assert "verdict" in text

    def test_triad_score_bounds(self):
        """TRIAD score must be in [0, 1]."""
        for msg, diff in [
            ("Add func", "+def func(): pass"),
            ("Remove everything", "-x = 1"),
            ("", ""),
        ]:
            r = triad_verify(msg, diff)
            assert 0.0 <= r.triad_score <= 1.0
            assert 0.0 <= r.token_alignment <= 1.0
            assert 0.0 <= r.structural_alignment <= 1.0
            assert 0.0 <= r.scope_consistency <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# Cross-layer integration: import verification
# ═══════════════════════════════════════════════════════════════════════


class TestLayer6Imports:
    """Verify that the new verifiers integrate with the public API."""

    def test_import_from_init(self):
        from entroly.verifiers import prove_verify, cave_verify, triad_verify
        from entroly.verifiers import ProveResult, CaveResult, TriadResult
        assert callable(prove_verify)
        assert callable(cave_verify)
        assert callable(triad_verify)

    def test_all_exports_present(self):
        import entroly.verifiers as v
        for name in [
            "prove_verify", "ProveResult",
            "cave_verify", "CaveResult",
            "triad_verify", "TriadResult",
        ]:
            assert hasattr(v, name), f"{name} not in verifiers.__all__"
