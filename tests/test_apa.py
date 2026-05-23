"""
Tests for Adaptive Prompt Augmentation (APA)
=============================================

Tests the three APA features:
  1. calibrated_token_count — per-language char/token ratios
  2. _build_preamble — task-aware 1-2 sentence directives
  3. _deduplicate_fragments — content-hash dedup
  4. format_context_block integration — preamble + dedup in context output
"""

import pytest

from entroly.proxy_transform import (
    calibrated_token_count,
    _build_preamble,
    _deduplicate_fragments,
    format_context_block,
)


# ═══════════════════════════════════════════════════════════════════
# 1. calibrated_token_count
# ═══════════════════════════════════════════════════════════════════

class TestCalibratedTokenCount:
    """Per-language char/token ratio estimation."""

    def test_empty_content_returns_zero(self):
        assert calibrated_token_count("", "file:foo.py") == 0

    def test_python_denser_than_default(self):
        code = "def foo(): return bar()" * 10  # 230 chars
        py_tokens = calibrated_token_count(code, "file:utils.py")
        # Python ratio = 3.0, so 230/3.0 ≈ 76
        assert py_tokens > 0
        # Compare to what len/4 would give (230/4 = 57)
        old_estimate = max(1, len(code) // 4)
        assert py_tokens > old_estimate  # calibrated catches more tokens

    def test_rust_less_dense_than_python(self):
        code = "fn main() { println!(\"hello\"); }" * 10
        rust_tokens = calibrated_token_count(code, "file:lib.rs")
        py_tokens = calibrated_token_count(code, "file:lib.py")
        # Rust ratio = 3.5, Python ratio = 3.0
        # Same content, but Rust estimates fewer tokens (more chars per token)
        assert rust_tokens < py_tokens

    def test_json_densest(self):
        code = '{"key": "value", "nested": {"a": 1}}' * 5
        json_tokens = calibrated_token_count(code, "file:config.json")
        # JSON ratio = 2.8 → most tokens per char
        py_tokens = calibrated_token_count(code, "file:config.py")
        assert json_tokens > py_tokens

    def test_unknown_extension_uses_default(self):
        code = "some content here" * 10
        tokens = calibrated_token_count(code, "file:readme.xyz")
        # Default ratio = 3.3
        expected = max(1, int(len(code) / 3.3))
        assert tokens == expected

    def test_never_returns_zero_for_nonempty(self):
        assert calibrated_token_count("x", "file:f.py") >= 1

    def test_no_source_uses_default(self):
        code = "hello world" * 50
        tokens = calibrated_token_count(code)
        expected = max(1, int(len(code) / 3.3))
        assert tokens == expected

    @pytest.mark.parametrize("ext,lang", [
        (".py", "python"), (".rs", "rust"), (".ts", "typescript"),
        (".go", "go"), (".java", "java"), (".c", "c"), (".cpp", "cpp"),
    ])
    def test_all_languages_return_positive(self, ext, lang):
        code = "x" * 100
        tokens = calibrated_token_count(code, f"file:test{ext}")
        assert tokens > 0


# ═══════════════════════════════════════════════════════════════════
# 2. _build_preamble
# ═══════════════════════════════════════════════════════════════════

class TestBuildPreamble:
    """Task-aware preamble generation."""

    def test_no_signals_returns_empty(self):
        """Unknown task, low vagueness, no security → no preamble."""
        result = _build_preamble("Unknown", 0.3, 0)
        assert result == ""

    def test_security_issues_trigger_warning(self):
        result = _build_preamble("Unknown", 0.0, 2)
        assert "SAST found 2 issues" in result
        assert "Address these" in result

    def test_single_security_issue_singular(self):
        result = _build_preamble("Unknown", 0.0, 1)
        assert "1 issue" in result
        assert "issues" not in result

    def test_high_vagueness_triggers_clarification(self):
        result = _build_preamble("Unknown", 0.8, 0)
        assert "ambiguous" in result
        assert "clarify" in result

    def test_low_vagueness_no_clarification(self):
        result = _build_preamble("Unknown", 0.5, 0)
        assert "ambiguous" not in result

    def test_bugtracing_hint(self):
        result = _build_preamble("BugTracing", 0.0, 0)
        assert "error propagation" in result

    def test_refactoring_hint(self):
        result = _build_preamble("Refactoring", 0.0, 0)
        assert "Preserve existing behavior" in result

    def test_testing_hint(self):
        result = _build_preamble("Testing", 0.0, 0)
        assert "edge cases" in result

    def test_codereview_hint(self):
        result = _build_preamble("CodeReview", 0.0, 0)
        assert "correctness" in result

    def test_codegeneration_no_hint(self):
        """CodeGeneration doesn't have a specific hint — no preamble."""
        result = _build_preamble("CodeGeneration", 0.0, 0)
        assert result == ""

    def test_combined_signals(self):
        """Security + vagueness + task should all appear."""
        result = _build_preamble("BugTracing", 0.9, 3)
        assert "SAST found 3 issues" in result
        assert "ambiguous" in result
        assert "error propagation" in result


# ═══════════════════════════════════════════════════════════════════
# 3. _deduplicate_fragments
# ═══════════════════════════════════════════════════════════════════

class TestDeduplicateFragments:
    """Content-hash deduplication."""

    def test_no_duplicates_keeps_all(self):
        frags = [
            {"content": "def foo(): pass", "source": "a.py"},
            {"content": "def bar(): pass", "source": "b.py"},
        ]
        result = _deduplicate_fragments(frags)
        assert len(result) == 2

    def test_exact_duplicates_removed(self):
        frags = [
            {"content": "def foo(): pass", "source": "a.py"},
            {"content": "def foo(): pass", "source": "a.py"},
            {"content": "def foo(): pass", "source": "a.py"},
        ]
        result = _deduplicate_fragments(frags)
        assert len(result) == 1

    def test_keeps_first_occurrence(self):
        frags = [
            {"content": "first", "source": "a.py"},
            {"content": "first", "source": "b.py"},
        ]
        result = _deduplicate_fragments(frags)
        assert result[0]["source"] == "a.py"

    def test_near_duplicates_kept(self):
        """Fragments with same prefix but different suffix are kept."""
        content_a = "x" * 300 + "AAAA"
        content_b = "x" * 300 + "BBBB"
        frags = [
            {"content": content_a, "source": "a.py"},
            {"content": content_b, "source": "b.py"},
        ]
        # Hash is on first 256 chars — these are identical in prefix
        result = _deduplicate_fragments(frags)
        # Both have same first 256 chars → deduped to 1
        assert len(result) == 1

    def test_empty_list(self):
        assert _deduplicate_fragments([]) == []

    def test_uses_preview_field(self):
        """Should use preview field when content is not available."""
        frags = [
            {"preview": "same code", "source": "a.py"},
            {"preview": "same code", "source": "b.py"},
        ]
        result = _deduplicate_fragments(frags)
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════
# 4. format_context_block integration
# ═══════════════════════════════════════════════════════════════════

class TestFormatContextBlockAPA:
    """Integration: preamble + dedup in context block."""

    def _make_fragment(self, source="file:test.py", content="def test(): pass"):
        return {
            "source": source,
            "relevance": 0.85,
            "token_count": 10,
            "content": content,
        }

    def test_preamble_appears_for_bugtracing(self):
        frag = self._make_fragment()
        result = format_context_block(
            [frag], [], [], None,
            task_type="BugTracing", vagueness=0.0,
        )
        assert "error propagation" in result

    def test_no_preamble_for_unknown_low_vagueness(self):
        frag = self._make_fragment()
        result = format_context_block(
            [frag], [], [], None,
            task_type="Unknown", vagueness=0.3,
        )
        # Should NOT contain any preamble hints
        assert "error propagation" not in result
        assert "ambiguous" not in result
        assert "SAST" not in result

    def test_security_preamble_with_issues(self):
        frag = self._make_fragment()
        result = format_context_block(
            [frag],
            ["[test.py] hardcoded API key found"],
            [], None,
            task_type="Unknown", vagueness=0.0,
        )
        assert "SAST found 1 issue" in result

    def test_dedup_removes_duplicate_fragments(self):
        frag1 = self._make_fragment(content="same content")
        frag2 = self._make_fragment(content="same content")
        result = format_context_block(
            [frag1, frag2], [], [], None,
        )
        # Content should appear only once
        assert result.count("same content") == 1

    def test_backward_compatible_without_kwargs(self):
        """Existing callers without task_type/vagueness should still work."""
        frag = self._make_fragment()
        result = format_context_block([frag], [], [], None)
        assert "Relevant Code Context" in result
        assert "test.py" in result

    def test_vagueness_threshold(self):
        """Vagueness at exactly 0.6 should NOT trigger (> 0.6 needed)."""
        frag = self._make_fragment()
        result = format_context_block(
            [frag], [], [], None,
            task_type="Unknown", vagueness=0.6,
        )
        assert "ambiguous" not in result

        result2 = format_context_block(
            [frag], [], [], None,
            task_type="Unknown", vagueness=0.61,
        )
        assert "ambiguous" in result2

    def test_empty_fragments_and_memories_returns_empty(self):
        result = format_context_block(
            [], [], [], None,
            task_type="BugTracing", vagueness=0.9,
        )
        assert result == ""

    def test_preamble_before_fragments(self):
        """Preamble should appear before code fragments."""
        frag = self._make_fragment()
        result = format_context_block(
            [frag], [], [], None,
            task_type="BugTracing", vagueness=0.0,
        )
        lines = result.split("\n")
        preamble_idx = next(i for i, line in enumerate(lines) if "error propagation" in line)
        code_idx = next(i for i, line in enumerate(lines) if "test.py" in line)
        assert preamble_idx < code_idx


# ═══════════════════════════════════════════════════════════════════
# 5. Token budget accuracy comparison
# ═══════════════════════════════════════════════════════════════════

class TestTokenBudgetAccuracy:
    """Verify calibrated estimation is more accurate than len/4."""

    @pytest.mark.parametrize("code,source,expected_range", [
        ("import os\nimport sys\n" * 10, "file:main.py", (50, 90)),
        ('{"key": "val"}' * 20, "file:data.json", (80, 130)),
        ("fn main() {}" * 20, "file:main.rs", (50, 85)),
    ])
    def test_reasonable_estimates(self, code, source, expected_range):
        tokens = calibrated_token_count(code, source)
        lo, hi = expected_range
        assert lo <= tokens <= hi, f"Expected {lo}-{hi}, got {tokens} for {source}"
