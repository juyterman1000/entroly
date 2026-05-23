"""
Tests for Entropic Context Compression (ECC) — Python integration.

Tests format_hierarchical_context() and the proxy config flag.
Rust-level hierarchical tests are in entroly-core/src/hierarchical.rs.
"""

import pytest
from entroly.proxy_transform import (
    format_hierarchical_context,
    format_context_block,
    _build_preamble,
)
from entroly.proxy_config import ProxyConfig


# ═══════════════════════════════════════════════════════════════
# format_hierarchical_context tests
# ═══════════════════════════════════════════════════════════════

class TestFormatHierarchicalContext:
    """Tests for format_hierarchical_context()."""

    def _make_hcc_result(self, **overrides):
        """Helper to build an HCC result dict."""
        base = {
            "status": "compressed",
            "level1_map": "auth.py → AuthService, login\ndb.py → Database, connect\nconfig.py",
            "level1_tokens": 15,
            "level2_cluster": "## auth.py\nclass AuthService:\n    ...\ndef login(user):\n    ...",
            "level2_tokens": 50,
            "level3_count": 1,
            "level3_tokens": 100,
            "level3_fragments": [
                {
                    "id": "f1",
                    "source": "file:src/auth/login.py",
                    "token_count": 100,
                    "content": "class AuthService:\n    def login(self, user, pwd):\n        # ... implementation",
                    "preview": "class AuthService:...",
                }
            ],
            "coverage": {
                "level1_files": 3,
                "level2_cluster_files": 2,
                "level3_full_files": 1,
            },
            "total_tokens": 165,
            "budget_utilization": 0.033,
            "cluster_ids": ["f1", "f2"],
        }
        base.update(overrides)
        return base

    def test_basic_formatting(self):
        """Should produce non-empty output with all 3 levels."""
        result = self._make_hcc_result()
        text = format_hierarchical_context(result, [], [], None)

        assert "Codebase Overview" in text
        assert "auth.py" in text
        assert "Related Code Structure" in text
        assert "Full Code" in text
        assert "AuthService" in text
        assert "--- End Context ---" in text

    def test_empty_result_returns_empty(self):
        """Should return empty string for empty HCC result."""
        result = {"status": "empty"}
        text = format_hierarchical_context(result, [], [], None)
        assert text == ""

    def test_includes_security_warnings(self):
        """Security issues should appear in the output."""
        result = self._make_hcc_result()
        text = format_hierarchical_context(
            result, ["[auth.py] SQL injection risk"], [], None,
        )
        assert "Security Warnings" in text
        assert "SQL injection risk" in text

    def test_includes_ltm_memories(self):
        """LTM memories should appear in the output."""
        result = self._make_hcc_result()
        memories = [{"retention": 0.85, "content": "User prefers type hints"}]
        text = format_hierarchical_context(result, [], memories, None)
        assert "Cross-Session Memory" in text
        assert "type hints" in text

    def test_includes_refinement_info(self):
        """Query refinement should appear in the output."""
        result = self._make_hcc_result()
        refinement = {
            "original": "fix auth",
            "refined": "fix authentication login flow",
            "vagueness": 0.65,
        }
        text = format_hierarchical_context(result, [], [], refinement)
        assert "Query refined" in text
        assert "fix auth" in text

    def test_includes_preamble_when_warranted(self):
        """Preamble should appear when signals warrant it."""
        result = self._make_hcc_result()
        text = format_hierarchical_context(
            result,
            ["[auth.py] hardcoded secret"],  # security issue → preamble
            [],
            None,
            task_type="BugTracing",
            vagueness=0.7,  # high vagueness → preamble
        )
        assert "SAST found" in text
        assert "ambiguous" in text
        assert "error propagation" in text

    def test_l1_shows_file_count(self):
        """L1 header should show the file count."""
        result = self._make_hcc_result()
        text = format_hierarchical_context(result, [], [], None)
        assert "3 files" in text

    def test_l2_shows_cluster_count(self):
        """L2 header should show the cluster size."""
        result = self._make_hcc_result()
        text = format_hierarchical_context(result, [], [], None)
        assert "2 connected files" in text

    def test_l3_shows_fragment_count(self):
        """L3 header should show the fragment count."""
        result = self._make_hcc_result()
        text = format_hierarchical_context(result, [], [], None)
        assert "1 fragments" in text

    def test_l3_infers_language(self):
        """L3 code fences should have the correct language."""
        result = self._make_hcc_result()
        text = format_hierarchical_context(result, [], [], None)
        assert "```python" in text

    def test_no_l2_if_empty(self):
        """If L2 cluster is empty, that section should be omitted."""
        result = self._make_hcc_result(level2_cluster="", level2_tokens=0)
        text = format_hierarchical_context(result, [], [], None)
        assert "Related Code Structure" not in text

    def test_no_l3_if_empty(self):
        """If L3 fragments are empty, that section should be omitted."""
        result = self._make_hcc_result(level3_fragments=[], level3_count=0)
        text = format_hierarchical_context(result, [], [], None)
        assert "Full Code" not in text


# ═══════════════════════════════════════════════════════════════
# Config tests
# ═══════════════════════════════════════════════════════════════

class TestHCCConfig:
    """Tests for the HCC config flag."""

    def test_flag_exists_and_defaults_true(self):
        config = ProxyConfig()
        assert hasattr(config, "enable_hierarchical_compression")
        assert config.enable_hierarchical_compression is True

    def test_flag_can_be_disabled(self):
        config = ProxyConfig(enable_hierarchical_compression=False)
        assert config.enable_hierarchical_compression is False


# ═══════════════════════════════════════════════════════════════
# Integration: format_hierarchical_context vs format_context_block
# ═══════════════════════════════════════════════════════════════

class TestHCCVsFlat:
    """Verify HCC and flat formatters produce compatible output."""

    def test_both_have_start_end_markers(self):
        """Both formatters should have the same start/end markers."""
        # Flat
        frags = [{"source": "test.py", "relevance": 0.5, "token_count": 10, "content": "x=1"}]
        flat_text = format_context_block(frags, [], [], None)
        assert "--- Relevant Code Context" in flat_text
        assert "--- End Context ---" in flat_text

        # HCC
        hcc_result = {
            "status": "compressed",
            "level1_map": "test.py",
            "level1_tokens": 5,
            "level2_cluster": "",
            "level2_tokens": 0,
            "level3_count": 0,
            "level3_tokens": 0,
            "level3_fragments": [],
            "coverage": {"level1_files": 1, "level2_cluster_files": 0, "level3_full_files": 0},
            "total_tokens": 5,
        }
        hcc_text = format_hierarchical_context(hcc_result, [], [], None)
        assert "--- Relevant Code Context" in hcc_text
        assert "--- End Context ---" in hcc_text

    def test_hcc_has_more_structure(self):
        """HCC output should have explicit level headers."""
        hcc_result = {
            "status": "compressed",
            "level1_map": "file1.py\nfile2.py",
            "level1_tokens": 5,
            "level2_cluster": "## file1.py\ndef foo(): ...",
            "level2_tokens": 10,
            "level3_count": 1,
            "level3_tokens": 50,
            "level3_fragments": [
                {"source": "file1.py", "token_count": 50, "content": "def foo(): return 1"}
            ],
            "coverage": {"level1_files": 2, "level2_cluster_files": 1, "level3_full_files": 1},
            "total_tokens": 65,
        }
        text = format_hierarchical_context(hcc_result, [], [], None)
        assert "Codebase Overview" in text
        assert "Related Code Structure" in text
        assert "Full Code" in text
