"""
Tests for Response Distillation — Output-Side Token Optimization
=================================================================

Validates the output-side token optimizer that strips LLM filler
(pleasantries, hedging, meta-commentary) while preserving all
code blocks and technical content.
"""

import pytest

from entroly.proxy_transform import (
    distill_response,
    distill_response_sse_chunk,
)


class TestDistillResponse:
    """Output-side token compression — strip filler, preserve code."""

    def test_preserves_code_blocks(self):
        """Code blocks must NEVER be modified."""
        text = (
            "Sure! Here's the fix:\n\n"
            "```python\n"
            "def fix_bug():\n"
            "    return 42\n"
            "```\n\n"
            "Hope this helps!"
        )
        result, orig, comp = distill_response(text, mode="full")
        # Code block must be intact
        assert "def fix_bug():" in result
        assert "return 42" in result
        assert "```python" in result
        # Filler should be stripped
        assert "Sure!" not in result
        assert "Hope this helps" not in result
        # Token count should decrease
        assert comp < orig

    def test_strips_pleasantries(self):
        """Pleasantries carry zero technical content."""
        text = (
            "I'd be happy to help! Let me take a look at your code.\n"
            "The issue is in the authentication module.\n"
            "Feel free to ask if you have more questions!"
        )
        result, orig, comp = distill_response(text, mode="full")
        assert "happy to help" not in result
        assert "Let me take a look" not in result
        assert "Feel free to ask" not in result
        # Technical content preserved
        assert "authentication module" in result
        assert comp < orig

    def test_strips_hedging(self):
        """Hedging adds noise without information."""
        text = (
            "I think the problem is in the database connection. "
            "It seems like the timeout is too short."
        )
        result, _, _ = distill_response(text, mode="full")
        assert "I think " not in result
        assert "It seems like " not in result
        # Content preserved
        assert "database connection" in result
        assert "timeout" in result

    def test_verbose_connectors_simplified(self):
        """Verbose connectors are replaced with terse equivalents."""
        text = "In order to fix this, due to the fact that the API changed."
        result, _, _ = distill_response(text, mode="full")
        assert "In order to" not in result
        assert "Due to the fact that" not in result
        assert "To" in result or "to" in result
        assert "Because" in result or "because" in result

    def test_lite_mode_less_aggressive(self):
        """Lite mode only removes pleasantries, not verbose connectors."""
        text = (
            "Sure! I think the issue is in the parser module. "
            "In order to fix this you need to update the regex and the tokenizer configuration."
        )
        result, _, _ = distill_response(text, mode="lite")
        assert "Sure!" not in result
        # Lite doesn't strip verbose connectors
        assert "In order to" in result
        # Core content preserved
        assert "parser" in result

    def test_ultra_mode_strips_articles(self):
        """Ultra mode also removes articles and filler words."""
        text = "The issue is just a simple fix in the configuration."
        result, _, _ = distill_response(text, mode="ultra")
        # Articles and filler words stripped
        assert result.count("the ") < text.lower().count("the ")

    def test_empty_input(self):
        """Empty input should pass through."""
        result, orig, comp = distill_response("", mode="full")
        assert result == ""
        assert orig == 0
        assert comp == 0

    def test_short_input_passthrough(self):
        """Very short input should pass through unchanged."""
        text = "Fix the bug."
        result, orig, comp = distill_response(text, mode="full")
        assert result == text
        assert orig == comp

    def test_multiblock_preservation(self):
        """Multiple code blocks should all be preserved."""
        text = (
            "Here's how to fix it:\n\n"
            "```python\nprint('hello')\n```\n\n"
            "And the test:\n\n"
            "```python\nassert True\n```\n\n"
            "Let me know if you need anything else!"
        )
        result, _, _ = distill_response(text, mode="full")
        assert "print('hello')" in result
        assert "assert True" in result
        assert "Let me know" not in result

    def test_pure_filler_lines_removed(self):
        """Entire lines that are pure filler should be dropped."""
        text = (
            "Sure, I'd be happy to help you with that!\n"
            "The error is on line 42.\n"
            "To summarize: the variable is undefined."
        )
        result, _, _ = distill_response(text, mode="full")
        assert "happy to help" not in result
        assert "To summarize" not in result
        assert "line 42" in result

    def test_returns_token_counts(self):
        """Should return accurate original and compressed counts."""
        text = (
            "I'd be happy to help! Let me take a look.\n"
            "The bug is in the parser. "
            "In order to fix it, you need to update the regex.\n"
            "Hope this helps! Let me know if you need anything else."
        )
        _, orig, comp = distill_response(text, mode="full")
        assert orig > 0
        assert comp > 0
        assert comp < orig
        savings = (orig - comp) / orig * 100
        assert savings > 10  # Should save at least 10%


class TestDistillSSEChunk:
    """Streaming chunk distillation — lightweight pattern matching."""

    def test_strips_filler_in_chunk(self):
        """Should strip filler from individual chunks."""
        chunk = "Sure! The issue is..."
        result = distill_response_sse_chunk(chunk, mode="full")
        assert "Sure!" not in result

    def test_preserves_code_chunks(self):
        """Code block chunks must pass through."""
        chunk = "```python\ndef foo():\n```"
        result = distill_response_sse_chunk(chunk, mode="full")
        assert result == chunk

    def test_short_chunks_passthrough(self):
        """Very short chunks should pass through."""
        result = distill_response_sse_chunk("OK", mode="full")
        assert result == "OK"
