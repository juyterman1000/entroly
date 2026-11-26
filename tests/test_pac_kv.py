"""Tests for PAC-KV (Provable Attention & Cache-Aligned Compression Engine).

Covers provider page-grid quantization, Economic Break-Even Gate,
Paged Merkle Trees, and CacheAligner integration.
"""

from __future__ import annotations

import pytest

from entroly.cache_aligner import CacheAligner
from entroly.pac_kv import (
    PROVIDER_GRIDS,
    CachePage,
    KVDemandBudgeter,
    PACKVEngine,
    PagedMerkleTree,
    get_provider_grid,
)


class TestProviderCacheGrids:
    """Verify hardware and billing grid configurations."""

    def test_anthropic_grid(self):
        grid = get_provider_grid("anthropic")
        assert grid.page_size == 1024
        assert grid.min_cache_tokens == 1024
        assert grid.read_discount == 0.90
        assert grid.read_factor == pytest.approx(0.10)

    def test_openai_grid(self):
        grid = get_provider_grid("openai")
        assert grid.page_size == 128
        assert grid.min_cache_tokens == 1024
        assert grid.read_discount == 0.50

    def test_deepseek_grid(self):
        grid = get_provider_grid("deepseek")
        assert grid.page_size == 64
        assert grid.min_cache_tokens == 64
        assert grid.read_discount == 0.85

    def test_vllm_grid(self):
        grid = get_provider_grid("vllm")
        assert grid.page_size == 16
        assert grid.read_discount == 1.0

    def test_generic_fallback(self):
        grid = get_provider_grid("unknown_provider_xyz")
        assert grid.page_size == 1024
        assert grid.name == "generic"


class TestPagedMerkleTree:
    """Verify virtual page slicing and Merkle proof generation."""

    def test_empty_text_merkle(self):
        tree = PagedMerkleTree.from_text("", page_token_size=128)
        assert len(tree.pages) == 0
        assert len(tree.root_hash) == 64

    def test_page_slicing_and_digests(self):
        # Generate multi-page text
        paragraphs = [f"Paragraph {i}: This is structured context content for page testing.\n" * 15 for i in range(10)]
        full_text = "\n".join(paragraphs)

        tree = PagedMerkleTree.from_text(full_text, page_token_size=128)
        assert len(tree.pages) > 1
        for i, page in enumerate(tree.pages):
            assert page.page_index == i
            assert len(page.digest) == 64

    def test_merkle_root_determinism(self):
        text = "System instructions: Act as an expert security auditor.\n" * 20
        tree1 = PagedMerkleTree.from_text(text, page_token_size=64)
        tree2 = PagedMerkleTree.from_text(text, page_token_size=64)
        assert tree1.root_hash == tree2.root_hash

    def test_common_prefix_pages(self):
        page_content = "Word " * 150 + "\n"  # roughly ~150 tokens
        text1 = page_content * 5
        text2 = (page_content * 3) + ("Different content here.\n" * 20)

        tree1 = PagedMerkleTree.from_text(text1, page_token_size=128)
        tree2 = PagedMerkleTree.from_text(text2, page_token_size=128)

        # First 3 pages should match byte-for-byte
        common = tree1.common_prefix_pages(tree2)
        assert common >= 2


class TestPACKVEngine:
    """Verify the core PAC-KV engine and Economic Break-Even Gate."""

    def test_cold_start_quantizes_prefix(self):
        engine = PACKVEngine()
        prefix = "System: You are an autonomous coding assistant.\n" * 30
        tail = "User: Fix the bug in auth.py"

        result = engine.process(prefix, tail, provider="anthropic", is_warm=False)
        assert result.is_cache_hit is False
        assert result.receipt.economic_decision in ("page_quantized_aligned", "page_quantized_compressed")
        assert result.receipt.pages_total > 0
        assert result.receipt.merkle_root is not None

    def test_economic_break_even_gate_locks_warm_cache(self):
        """When a prefix is warm, PAC-KV must freeze it to preserve 90% cache discount."""
        engine = PACKVEngine()
        # Large prefix that exceeds Anthropic's 1024-token minimum under the
        # real o200k tokenizer too (x60 was 780 tokens; it passed only on the
        # chars/4 fallback estimate).
        prefix = "SYSTEM_INSTRUCTIONS_MODULE: Define all tool specifications and architecture rules.\n" * 160
        tail = "USER: Please review the latest PR."

        # Warm turn with an active cache lease
        result = engine.process(
            prefix,
            tail,
            provider="anthropic",
            is_warm=True,
            base_price_per_million=3.0,
            cached_read_price_per_million=0.30,  # 90% off
        )

        # Must trigger cache-locked optimization
        assert result.is_cache_hit is True
        assert result.receipt.economic_decision == "cache_locked_tail_compressed"
        assert result.receipt.pages_cached > 0
        assert result.receipt.pages_dirty == 0
        assert result.receipt.projected_dollar_savings > 0.0
        assert result.receipt.effective_discount > 0.50

    def test_aatc_tail_compression_under_tight_budget(self):
        """When budget is constrained, AATC compresses the middle tail while preserving ends."""
        engine = PACKVEngine()
        prefix = "SYSTEM ANCHOR: Base policy.\n" * 40
        tail = (
            "USER QUERY PRIMACY: Start here.\n"
            + ("INTERMEDIATE LOG: Verbose status message line.\n" * 80)
            + "USER QUERY RECENCY: End here."
        )

        # Strict target budget
        result = engine.process(
            prefix,
            tail,
            provider="openai",
            target_budget=400,
            is_warm=True,
        )

        assert result.receipt.aligned_tokens_total <= 450
        # Primacy and recency should be preserved in tail
        assert "USER QUERY PRIMACY" in result.aligned_tail
        assert "USER QUERY RECENCY" in result.aligned_tail


class TestCacheAlignerIntegration:
    """Verify CacheAligner with upgraded PAC-KV capabilities."""

    def test_backward_compatible_align(self):
        aligner = CacheAligner()
        ctx = "Standard context string to test backwards compatibility."

        aligned, hit = aligner.align("client_1", ctx)
        assert aligned == ctx
        assert hit is False

        # Second call with identical context hits cache
        aligned2, hit2 = aligner.align("client_1", ctx)
        assert aligned2 == ctx
        assert hit2 is True

        stats = aligner.stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_align_prompt_flow(self):
        aligner = CacheAligner(default_provider="anthropic")
        # x120 was 960 o200k tokens, under the 1024 minimum; x300 clears it.
        prefix = "System: Core guidelines and tool schemas.\n" * 300
        tail = "User: Execute step 1."

        # Turn 1: Cold start
        res1 = aligner.align_prompt("agent_session_1", prefix, tail, provider="anthropic")
        assert res1.is_cache_hit is False
        assert res1.receipt.provider == "anthropic"

        # Turn 2: Warm prefix, new user turn
        new_tail = "User: Execute step 2 with additional parameters."
        res2 = aligner.align_prompt("agent_session_1", prefix, new_tail, provider="anthropic")
        assert res2.is_cache_hit is True
        assert res2.receipt.economic_decision == "cache_locked_tail_compressed"
        assert res2.receipt.cached_prefix_tokens > 0

        stats = aligner.stats()
        assert stats["cache_hits"] == 1
        assert stats["projected_dollar_savings"] > 0.0

    def test_invalidate(self):
        aligner = CacheAligner()
        aligner.align("client_x", "context")
        aligner.invalidate("client_x")
        stats = aligner.stats()
        assert stats["active_clients"] == 0


class TestKVDemandBudgeter:
    def test_shared_prefix_pages_deduplicated_in_pool_union(self):
        # Pool of 20 pages, ceiling 0.85 => safe_page_limit = 17 pages.
        budgeter = KVDemandBudgeter(pool_pages=20, utilization_ceiling=0.85, ttl_seconds=60.0)
        shared_anchor = [f"anchor_p{i}" for i in range(10)]

        # Three swarm agents share 10 system-prompt pages + 2 private pages each.
        # Naive sum = 36 pages (> 17), but true union = 10 + 2*3 = 16 <= 17!
        r1 = budgeter.admit("agent_1", shared_anchor + ["a1_0", "a1_1"], page_size=128, now=10.0)
        r2 = budgeter.admit("agent_2", shared_anchor + ["a2_0", "a2_1"], page_size=128, now=11.0)
        r3 = budgeter.admit("agent_3", shared_anchor + ["a3_0", "a3_1"], page_size=128, now=12.0)

        assert r1.admitted is True and r1.decision == "admitted_cold"
        assert r2.admitted is True and r2.shared_prefix_pages == 10 and r2.marginal_new_pages == 2
        assert r3.admitted is True and r3.projected_union_pages == 16

        # Fourth agent needs 3 private pages => projected union = 19 > 17 => deferred
        # with exact safe token budget = (10 shared + 1 slack) * 128 = 1408 tokens.
        r4 = budgeter.admit("agent_4", shared_anchor + ["a4_0", "a4_1", "a4_2"], page_size=128, now=13.0)
        assert r4.admitted is False
        assert r4.decision == "defer_or_compress_tail"
        assert r4.recommended_token_budget == (10 + 1) * 128
        # Refused admission must not mutate resident union.
        assert budgeter.resident_union_pages(now=13.0) == 16

    def test_ttl_expiry_and_warm_growth_reclaim_pool_capacity(self):
        budgeter = KVDemandBudgeter(pool_pages=10, utilization_ceiling=0.80, ttl_seconds=30.0)
        # safe_page_limit = 8
        assert budgeter.admit("s1", [f"p{i}" for i in range(6)], now=0.0).admitted is True
        # Warm update replacing pages stays admitted_warm.
        warm = budgeter.admit("s1", [f"p{i}" for i in range(7)], now=10.0)
        assert warm.admitted is True and warm.decision == "admitted_warm"

        # New session needing 3 pages blocked at t=20 (7 + 3 = 10 > 8),
        # then admitted at t=45 after s1's 30s lease from t=10 expires at t=40.
        assert budgeter.admit("s2", ["q0", "q1", "q2"], now=20.0).admitted is False
        after_ttl = budgeter.admit("s2", ["q0", "q1", "q2"], now=45.0)
        assert after_ttl.admitted is True
        assert after_ttl.projected_union_pages == 3
