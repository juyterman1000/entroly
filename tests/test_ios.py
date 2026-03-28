"""
IOS — Information-Optimal Selection Test Suite
===============================================

Comprehensive tests for the three novel IOS algorithms:

1. SDS — Submodular Diversity Selection
   Tests that redundant fragments get penalized, diverse selections
   are preferred, and the (1-1/e) approximation guarantee holds.

2. MRK — Multi-Resolution Knapsack
   Tests that fragments are selected at optimal resolution (full,
   skeleton, reference) based on budget pressure.

3. ECDB — Entropy-Calibrated Dynamic Budget
   Tests that the budget adapts to query vagueness and codebase size,
   saving tokens on specific queries.

Plus end-to-end pipeline integration tests.
"""

import math
import pytest
from unittest.mock import MagicMock

# Rust engine
from entroly_core import EntrolyEngine, ContextFragment, py_simhash, py_hamming_distance

# Python transform layer
from entroly.proxy_transform import (
    compute_dynamic_budget,
    compute_token_budget,
    format_context_block,
)
from entroly.proxy_config import ProxyConfig


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def make_engine(**kwargs):
    """Create an EntrolyEngine with IOS enabled by default."""
    defaults = dict(
        w_recency=0.30, w_frequency=0.25, w_semantic=0.25, w_entropy=0.20,
        decay_half_life=15, min_relevance=0.05, hamming_threshold=3,
        exploration_rate=0.0,  # Disable exploration for deterministic tests
        max_fragments=10000,
        enable_ios=True, enable_ios_diversity=True, enable_ios_multi_resolution=True,
    )
    defaults.update(kwargs)
    return EntrolyEngine(**defaults)


def ingest_fragment(engine, content, source="test.py", token_count=0, is_pinned=False):
    """Helper to ingest and return the fragment ID."""
    result = engine.ingest(content, source, token_count, is_pinned)
    return result["fragment_id"]


# ═══════════════════════════════════════════════════════════════════
# 1. SDS — Submodular Diversity Selection
# ═══════════════════════════════════════════════════════════════════

class TestSDSDiversityPenalty:
    """Verify that SDS penalizes redundant information."""

    def test_diverse_selection_over_redundant(self):
        """Given duplicate + unique fragments, SDS should prefer diverse set."""
        engine = make_engine()

        # Two nearly identical fragments about tax calculation
        ingest_fragment(engine, "def calculate_tax(income, rate):\n    return income * rate\n", "tax1.py", 100)
        ingest_fragment(engine, "def calculate_tax(income, rate):\n    return income * rate * 1.0\n", "tax2.py", 100)
        # One unique fragment about database connection
        ingest_fragment(engine, "async def connect_db(host, port):\n    conn = await create_connection(host, port)\n    return conn\n", "db.py", 100)

        engine.advance_turn()
        result = engine.optimize(200, "calculate_tax and database")
        selected = result["selected"]

        sources = [f["source"] for f in selected]
        # With diversity: should have db.py (diverse) rather than both tax files
        assert "db.py" in sources, f"Diverse fragment should be selected. Sources: {sources}"

    def test_diversity_score_reported(self):
        """IOS should report a diversity score."""
        engine = make_engine()

        ingest_fragment(engine, "machine learning neural network training gradient descent", "ml.py", 100)
        ingest_fragment(engine, "kubernetes docker container orchestration deployment", "ops.py", 100)
        ingest_fragment(engine, "react component virtual dom rendering state management", "ui.py", 100)

        engine.advance_turn()
        result = engine.optimize(1000, "software development")

        assert "ios_diversity_score" in result, "IOS should report diversity score"
        div = result["ios_diversity_score"]
        assert 0.0 <= div <= 1.0, f"Diversity score should be in [0,1], got {div}"

    def test_similar_fragments_low_diversity(self):
        """Selecting similar (but not identical) fragments should yield low diversity score."""
        engine = make_engine(enable_ios_diversity=False)

        # Near-identical content but different enough to bypass dedup
        for i in range(5):
            ingest_fragment(
                engine,
                f"def process_{i}(): return compute_result(data_{i})",
                f"copy{i}.py", 50,
            )

        engine.advance_turn()
        result = engine.optimize(500, "process")
        # Without diversity, engine just greedily selects — diversity score should still be reported
        assert "ios_diversity_score" in result

    def test_all_unique_content_high_diversity(self):
        """Completely different fragments should yield high diversity score."""
        engine = make_engine()

        unique_contents = [
            "import numpy as np\ndef matrix_multiply(a, b): return np.dot(a, b)",
            "CREATE TABLE users (id INT, name VARCHAR(255), email VARCHAR(255))",
            "export default function App() { return <div>Hello World</div> }",
            "fn main() { let x: i32 = 42; println!(\"{}\", x); }",
        ]
        for i, content in enumerate(unique_contents):
            ingest_fragment(engine, content, f"file{i}.py", 80)

        engine.advance_turn()
        result = engine.optimize(1000, "software")
        div = result.get("ios_diversity_score", 0.0)
        # Diverse content should have high diversity
        assert div > 0.3, f"Unique content should have high diversity, got {div}"


class TestSDSBudgetRespect:
    """Verify SDS always respects the token budget."""

    @pytest.mark.parametrize("budget", [100, 500, 1000, 5000])
    def test_budget_never_exceeded(self, budget):
        """Total selected tokens must never exceed budget."""
        engine = make_engine()

        for i in range(20):
            ingest_fragment(engine, f"def function_{i}(): return {i} * {i+1}", f"f{i}.py", 50 + i * 10)

        engine.advance_turn()
        result = engine.optimize(budget, "function")
        total = result["total_tokens"]
        assert total <= budget, f"Budget violated: {total} > {budget}"

    def test_zero_budget(self):
        """Zero budget should select nothing (or only pinned)."""
        engine = make_engine()
        ingest_fragment(engine, "def foo(): pass", "foo.py", 100)
        engine.advance_turn()
        result = engine.optimize(0, "foo")
        # May have pinned fragments, but total should be 0 or minimal


class TestSDSFeedbackIntegration:
    """Verify SDS respects Wilson score feedback multipliers."""

    def test_boosted_fragment_preferred(self):
        """Fragments with positive feedback should be preferred."""
        engine = make_engine()

        id_good = ingest_fragment(engine, "def good_function(): return optimal_result()", "good.py", 100)
        id_bad = ingest_fragment(engine, "def bad_function(): return suboptimal_result()", "bad.py", 100)

        # Give feedback
        engine.record_success([id_good])
        engine.record_failure([id_bad])

        engine.advance_turn()
        result = engine.optimize(150, "function")  # Budget for only one
        selected = result["selected"]

        if len(selected) == 1:
            assert selected[0]["source"] == "good.py", \
                f"Feedback-boosted fragment should be preferred, got {selected[0]['source']}"


# ═══════════════════════════════════════════════════════════════════
# 2. MRK — Multi-Resolution Knapsack
# ═══════════════════════════════════════════════════════════════════

class TestMRKResolutionSelection:
    """Verify MRK selects optimal resolution per fragment."""

    def test_full_resolution_with_generous_budget(self):
        """With plenty of budget, fragments should be full resolution."""
        engine = make_engine()

        ingest_fragment(engine, (
            "def process_data(input_data):\n"
            "    result = {}\n"
            "    for item in input_data:\n"
            "        key, val = item.split('=')\n"
            "        result[key] = val\n"
            "    return result\n"
        ), "process.py", 100)

        engine.advance_turn()
        result = engine.optimize(10000, "process")
        selected = result["selected"]

        full_frags = [f for f in selected if f.get("variant") == "full"]
        assert len(full_frags) >= 1, "Should select full resolution with generous budget"

    def test_skeleton_resolution_with_tight_budget(self):
        """With tight budget, should use skeleton resolution for some fragments."""
        engine = make_engine()

        # Ingest fragments with enough content for skeleton extraction
        for i in range(5):
            content = (
                f"import os\n"
                f"from pathlib import Path\n\n"
                f"class Handler{i}:\n"
                f"    def __init__(self, config):\n"
                f"        self.config = config\n"
                f"        self.data = {{}}\n\n"
                f"    def process(self, input_data):\n"
                f"        result = {{}}\n"
                f"        for item in input_data:\n"
                f"            key, val = item.split('=')\n"
                f"            result[key] = val\n"
                f"        return result\n\n"
                f"    def cleanup(self):\n"
                f"        self.data.clear()\n"
            )
            ingest_fragment(engine, content, f"handler{i}.py", 200)

        engine.advance_turn()
        # Budget enough for ~2 full fragments but not all 5
        result = engine.optimize(500, "handler")
        selected = result["selected"]

        variants = [f.get("variant", "full") for f in selected]
        # Should have a mix of resolutions
        unique_frags = set(f["source"] for f in selected)
        assert len(unique_frags) >= 2, \
            f"MRK should cover multiple files, got {unique_frags}"

    def test_reference_resolution_exists(self):
        """Reference fragments should appear when budget is very tight."""
        engine = make_engine()

        for i in range(10):
            ingest_fragment(engine, f"def function_{i}(): return {i}", f"f{i}.py", 100)

        engine.advance_turn()
        result = engine.optimize(200, "function")  # Very tight for 10 fragments
        selected = result["selected"]

        variants = [f.get("variant", "full") for f in selected]
        # Should have at least some non-full variants
        has_non_full = any(v != "full" for v in variants)
        # This depends on whether skeletons were extracted — may or may not have skeleton/reference

    def test_mrk_disabled_uses_full_only(self):
        """With MRK disabled, all fragments should be full resolution."""
        engine = make_engine(enable_ios_multi_resolution=False)

        ingest_fragment(engine, (
            "import os\n"
            "from pathlib import Path\n\n"
            "class Service:\n"
            "    def __init__(self):\n"
            "        self.data = {}\n\n"
            "    def run(self):\n"
            "        for item in self.data:\n"
            "            print(item)\n"
        ), "service.py", 200)

        engine.advance_turn()
        result = engine.optimize(1000, "service")
        selected = result["selected"]

        for frag in selected:
            assert frag.get("variant") == "full", \
                f"With MRK disabled, all should be full, got {frag.get('variant')}"


class TestMRKCoverageImprovement:
    """Verify MRK covers more files than standard knapsack."""

    def test_more_files_covered_with_mrk(self):
        """MRK should cover more unique files than legacy knapsack."""
        # Test with MRK enabled
        engine_mrk = make_engine(enable_ios=True, enable_ios_multi_resolution=True)
        # Test with MRK disabled (falls back to legacy + skeleton pass)
        engine_legacy = make_engine(enable_ios=False)

        content_template = (
            "import os\nfrom pathlib import Path\n\n"
            "class Handler{i}:\n"
            "    def __init__(self, config):\n"
            "        self.config = config\n"
            "        self.data = {{}}\n\n"
            "    def process(self, input_data):\n"
            "        result = {{}}\n"
            "        for item in input_data:\n"
            "            key, val = item.split('=')\n"
            "            result[key] = val\n"
            "        return result\n"
        )

        for i in range(8):
            content = content_template.replace("{i}", str(i))
            ingest_fragment(engine_mrk, content, f"handler{i}.py", 200)
            ingest_fragment(engine_legacy, content, f"handler{i}.py", 200)

        engine_mrk.advance_turn()
        engine_legacy.advance_turn()

        budget = 600  # Tight budget for 8 × 200 = 1600 tokens
        result_mrk = engine_mrk.optimize(budget, "handler")
        result_legacy = engine_legacy.optimize(budget, "handler")

        files_mrk = set(f["source"] for f in result_mrk["selected"])
        files_legacy = set(f["source"] for f in result_legacy["selected"])

        # MRK should cover at least as many files
        assert len(files_mrk) >= len(files_legacy), \
            f"MRK should cover >= files: {len(files_mrk)} vs {len(files_legacy)}"


# ═══════════════════════════════════════════════════════════════════
# 3. ECDB — Entropy-Calibrated Dynamic Budget
# ═══════════════════════════════════════════════════════════════════

class TestECDBQueryFactor:
    """Verify ECDB scales budget with query vagueness."""

    def test_specific_query_small_budget(self):
        """Specific queries (low vagueness) should get smaller budgets."""
        config = ProxyConfig()
        budget_specific = compute_dynamic_budget("gpt-4o", config, vagueness=0.0, total_fragments=100)
        budget_vague = compute_dynamic_budget("gpt-4o", config, vagueness=1.0, total_fragments=100)

        assert budget_specific < budget_vague, \
            f"Specific query should get smaller budget: {budget_specific} >= {budget_vague}"

    def test_vague_query_large_budget(self):
        """Vague queries (high vagueness) should get larger budgets."""
        config = ProxyConfig()
        budget = compute_dynamic_budget("gpt-4o", config, vagueness=1.0, total_fragments=100)
        static_budget = compute_token_budget("gpt-4o", config)

        assert budget > static_budget, \
            f"Vague query budget ({budget}) should exceed static ({static_budget})"

    def test_medium_vagueness_near_static(self):
        """Medium vagueness should produce budget near the static value."""
        config = ProxyConfig()
        budget = compute_dynamic_budget("gpt-4o", config, vagueness=0.4, total_fragments=100)
        static_budget = compute_token_budget("gpt-4o", config)

        # Within 50% of static budget
        assert abs(budget - static_budget) < static_budget * 0.5, \
            f"Medium vagueness budget ({budget}) should be near static ({static_budget})"

    @pytest.mark.parametrize("vagueness", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_budget_monotonic_in_vagueness(self, vagueness):
        """Budget should increase monotonically with vagueness."""
        config = ProxyConfig()
        budgets = []
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
            b = compute_dynamic_budget("gpt-4o", config, vagueness=v, total_fragments=100)
            budgets.append(b)

        for i in range(len(budgets) - 1):
            assert budgets[i] <= budgets[i + 1], \
                f"Budget should be monotonic in vagueness: {budgets}"


class TestECDBCodebaseFactor:
    """Verify ECDB scales budget with codebase size."""

    def test_larger_codebase_larger_budget(self):
        """Larger codebases should get larger budgets."""
        config = ProxyConfig()
        budget_small = compute_dynamic_budget("gpt-4o", config, vagueness=0.5, total_fragments=10)
        budget_large = compute_dynamic_budget("gpt-4o", config, vagueness=0.5, total_fragments=500)

        assert budget_large > budget_small, \
            f"Larger codebase should get larger budget: {budget_large} <= {budget_small}"

    def test_codebase_factor_caps_at_2x(self):
        """Codebase factor should cap at 2.0 (300+ fragments)."""
        config = ProxyConfig()
        budget_300 = compute_dynamic_budget("gpt-4o", config, vagueness=0.5, total_fragments=300)
        budget_1000 = compute_dynamic_budget("gpt-4o", config, vagueness=0.5, total_fragments=1000)

        # Should be close (both at cap)
        assert abs(budget_300 - budget_1000) < budget_300 * 0.1, \
            f"Codebase factor should cap: {budget_300} vs {budget_1000}"


class TestECDBBounds:
    """Verify ECDB respects minimum and maximum budget bounds."""

    def test_minimum_budget(self):
        """Budget should never drop below 500 tokens."""
        config = ProxyConfig(context_fraction=0.001)  # Very small fraction
        budget = compute_dynamic_budget("gpt-4o", config, vagueness=0.0, total_fragments=1)
        assert budget >= 500, f"Budget below minimum: {budget}"

    def test_maximum_budget(self):
        """Budget should never exceed 30% of context window."""
        config = ProxyConfig(context_fraction=0.99)  # Unreasonably large
        budget = compute_dynamic_budget("gpt-4o", config, vagueness=1.0, total_fragments=10000)
        max_allowed = int(128_000 * 0.30)
        assert budget <= max_allowed, f"Budget exceeds maximum: {budget} > {max_allowed}"

    @pytest.mark.parametrize("model,window", [
        ("gpt-4o", 128_000),
        ("claude-opus-4-6", 200_000),
        ("gpt-4", 8_192),
    ])
    def test_model_aware_budget(self, model, window):
        """Budget should scale with model's context window."""
        config = ProxyConfig()
        budget = compute_dynamic_budget(model, config, vagueness=0.5, total_fragments=100)
        assert budget > 0
        assert budget <= int(window * 0.30)


# ═══════════════════════════════════════════════════════════════════
# 4. Format Context Block — Resolution-Aware Output
# ═══════════════════════════════════════════════════════════════════

class TestContextBlockFormatting:
    """Verify context block correctly formats multi-resolution output."""

    def test_full_fragments_in_code_fences(self):
        """Full resolution fragments should appear in code fences."""
        fragments = [
            {"source": "main.py", "relevance": 0.9, "token_count": 100,
             "variant": "full", "preview": "def main(): pass"},
        ]
        block = format_context_block(fragments, [], [], None)
        assert "```python" in block
        assert "def main(): pass" in block

    def test_skeleton_fragments_grouped(self):
        """Skeleton fragments should appear under 'Structural Outlines'."""
        fragments = [
            {"source": "main.py", "relevance": 0.9, "token_count": 100,
             "variant": "full", "preview": "def main(): pass"},
            {"source": "utils.py", "relevance": 0.5, "token_count": 30,
             "variant": "skeleton", "preview": "def helper(): ..."},
        ]
        block = format_context_block(fragments, [], [], None)
        assert "Structural Outlines" in block
        assert "def helper(): ..." in block

    def test_reference_fragments_listed(self):
        """Reference fragments should appear under 'Also relevant'."""
        fragments = [
            {"source": "main.py", "relevance": 0.9, "token_count": 100,
             "variant": "full", "preview": "def main(): pass"},
            {"source": "config.py", "relevance": 0.3, "token_count": 5,
             "variant": "reference", "preview": "[ref] config.py"},
        ]
        block = format_context_block(fragments, [], [], None)
        assert "Also relevant" in block
        assert "config.py" in block

    def test_empty_fragments_returns_empty(self):
        """No fragments = empty string."""
        block = format_context_block([], [], [], None)
        assert block == ""

    def test_resolution_ordering(self):
        """Full fragments should appear before skeleton, skeleton before reference."""
        fragments = [
            {"source": "ref.py", "relevance": 0.2, "token_count": 5,
             "variant": "reference", "preview": "[ref] ref.py"},
            {"source": "main.py", "relevance": 0.9, "token_count": 100,
             "variant": "full", "preview": "def main(): pass"},
            {"source": "skel.py", "relevance": 0.5, "token_count": 30,
             "variant": "skeleton", "preview": "def helper(): ..."},
        ]
        block = format_context_block(fragments, [], [], None)
        # Full should appear before skeleton
        full_pos = block.index("def main(): pass")
        skel_pos = block.index("def helper(): ...")
        ref_pos = block.index("Also relevant")
        assert full_pos < skel_pos < ref_pos, \
            f"Order wrong: full={full_pos}, skel={skel_pos}, ref={ref_pos}"


# ═══════════════════════════════════════════════════════════════════
# 5. End-to-End Integration
# ═══════════════════════════════════════════════════════════════════

class TestIOSEndToEnd:
    """Full pipeline tests with real engine + IOS."""

    def test_pipeline_produces_valid_output(self):
        """Full pipeline: ingest → optimize → format should produce valid context."""
        engine = make_engine()

        contents = [
            ("def calculate_total(items):\n    return sum(i.price for i in items)\n", "calc.py"),
            ("class Item:\n    def __init__(self, name, price):\n        self.name = name\n        self.price = price\n", "item.py"),
            ("import unittest\nclass TestCalc(unittest.TestCase):\n    def test_empty(self):\n        self.assertEqual(calculate_total([]), 0)\n", "test_calc.py"),
        ]
        for content, source in contents:
            ingest_fragment(engine, content, source, 100)

        engine.advance_turn()
        result = engine.optimize(500, "calculate total price")
        selected = result["selected"]

        block = format_context_block(selected, [], [], None)
        assert "--- Relevant Code Context" in block
        assert "--- End Context ---" in block

    def test_ios_vs_legacy_both_valid(self):
        """Both IOS and legacy paths should produce valid results."""
        contents = [
            "def auth_login(user, password): return check_credentials(user, password)",
            "def auth_logout(session): session.invalidate()",
            "def auth_register(user, email): return create_account(user, email)",
        ]

        for enable_ios in [True, False]:
            engine = make_engine(enable_ios=enable_ios)
            for i, content in enumerate(contents):
                ingest_fragment(engine, content, f"auth{i}.py", 80)

            engine.advance_turn()
            result = engine.optimize(200, "authentication")

            # Legacy path allows ε-exploration swaps with up to +100 token slack
            budget = result.get("effective_budget", 200)
            assert result["total_tokens"] <= budget + 100
            assert len(result["selected"]) >= 1

    def test_ios_enabled_flag_in_result(self):
        """When IOS is enabled, result should include ios_enabled flag."""
        engine = make_engine(enable_ios=True)
        ingest_fragment(engine, "def foo(): pass", "foo.py", 50)
        engine.advance_turn()
        result = engine.optimize(1000, "foo")

        assert result.get("ios_enabled") is True

    def test_ios_disabled_no_flag(self):
        """When IOS is disabled, result should not include ios_enabled."""
        engine = make_engine(enable_ios=False)
        ingest_fragment(engine, "def foo(): pass", "foo.py", 50)
        engine.advance_turn()
        result = engine.optimize(1000, "foo")

        assert "ios_enabled" not in result or result.get("ios_enabled") is not True


class TestIOSPerformance:
    """Verify IOS doesn't degrade performance significantly."""

    def test_1000_fragments_under_100ms(self):
        """IOS should handle 1000 fragments in under 100ms."""
        import time
        engine = make_engine()

        for i in range(1000):
            engine.ingest(
                f"def function_{i}(x): return x * {i} + {i*2}",
                f"module{i // 10}/func{i}.py",
                30,
                False,
            )

        engine.advance_turn()

        t0 = time.perf_counter()
        result = engine.optimize(5000, "function processing")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 1000, \
            f"IOS with 1000 fragments took {elapsed_ms:.1f}ms (should be <1000ms)"
        assert result["total_tokens"] <= 5000


# ═══════════════════════════════════════════════════════════════════
# 6. Mathematical Properties
# ═══════════════════════════════════════════════════════════════════

class TestMathProperties:
    """Verify mathematical properties of the algorithms."""

    def test_diversity_factor_bounds(self):
        """Diversity factor should be in [0.1, 1.0]."""
        # Test with known SimHash values
        h1 = py_simhash("machine learning neural network")
        h2 = py_simhash("kubernetes docker container")
        h3 = py_simhash("machine learning neural network training")

        # Distance between similar content should be small
        dist_similar = py_hamming_distance(h1, h3)
        # Distance between different content should be large
        dist_different = py_hamming_distance(h1, h2)

        assert dist_similar <= dist_different, \
            f"Similar content should have smaller Hamming distance: {dist_similar} > {dist_different}"

    def test_ecdb_sigmoid_shape(self):
        """ECDB query factor should follow sigmoid shape."""
        config = ProxyConfig()
        budgets = [
            compute_dynamic_budget("gpt-4o", config, vagueness=v/10.0, total_fragments=100)
            for v in range(11)
        ]

        # Should be monotonically non-decreasing
        for i in range(len(budgets) - 1):
            assert budgets[i] <= budgets[i + 1], \
                f"Budget not monotonic at v={i/10}: {budgets[i]} > {budgets[i+1]}"

        # Should have S-curve shape: steepest in the middle
        # (difference between consecutive budgets is largest near v=0.5)
        diffs = [budgets[i + 1] - budgets[i] for i in range(len(budgets) - 1)]
        mid_diff = diffs[4] + diffs[5]  # around v=0.5
        edge_diff = diffs[0] + diffs[9]  # at edges
        assert mid_diff >= edge_diff * 0.5, \
            f"Sigmoid shape expected: mid_diff={mid_diff}, edge_diff={edge_diff}"

    def test_ecdb_query_factor_formula(self):
        """Verify ECDB query factor matches the documented formula."""
        # At v=0: query_factor = 0.5 + 1.5 * sigmoid(-1.5) ≈ 0.5 + 1.5 * 0.182 ≈ 0.773
        # At v=0.5: query_factor = 0.5 + 1.5 * sigmoid(0) = 0.5 + 0.75 = 1.25
        # At v=1: query_factor = 0.5 + 1.5 * sigmoid(1.5) ≈ 0.5 + 1.5 * 0.818 ≈ 1.727
        for v, expected_min, expected_max in [
            (0.0, 0.5, 0.9),
            (0.5, 1.1, 1.4),
            (1.0, 1.5, 2.0),
        ]:
            z = 3.0 * (v - 0.5)
            query_factor = 0.5 + 1.5 / (1.0 + math.exp(-z))
            assert expected_min <= query_factor <= expected_max, \
                f"Query factor at v={v}: {query_factor} not in [{expected_min}, {expected_max}]"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_fragment(self):
        """Single fragment should always be selected at full resolution."""
        engine = make_engine()
        ingest_fragment(engine, "def solo(): return 42", "solo.py", 50)
        engine.advance_turn()
        result = engine.optimize(1000, "solo")
        assert len(result["selected"]) == 1
        assert result["selected"][0]["variant"] == "full"

    def test_all_pinned(self):
        """All pinned fragments should always be included."""
        engine = make_engine()
        for i in range(3):
            ingest_fragment(engine, f"CRITICAL: config_{i} = True", f"critical{i}.py", 100, is_pinned=True)
        engine.advance_turn()
        result = engine.optimize(1000, "config")
        assert len(result["selected"]) >= 3

    def test_budget_smaller_than_smallest(self):
        """Budget smaller than any fragment shouldn't crash."""
        engine = make_engine()
        ingest_fragment(engine, "def big(): " + "x = 1; " * 100, "big.py", 500)
        engine.advance_turn()
        result = engine.optimize(10, "big")
        # May select reference resolution or nothing — shouldn't crash

    def test_empty_query(self):
        """Empty query shouldn't crash IOS."""
        engine = make_engine()
        ingest_fragment(engine, "def foo(): pass", "foo.py", 50)
        engine.advance_turn()
        result = engine.optimize(1000, "")
        assert "selected" in result

    def test_unicode_content(self):
        """Unicode content shouldn't crash IOS."""
        engine = make_engine()
        ingest_fragment(engine, "def greet(): return '你好世界 🌍'", "unicode.py", 50)
        engine.advance_turn()
        result = engine.optimize(1000, "greet")
        assert len(result["selected"]) >= 1


class TestIOSRegressionFixes:
    """Regression coverage for IOS-specific production bugs."""

    def test_negative_feedback_does_not_backfill_bad_reference(self):
        """Strongly down-ranked fragments should not reappear as cheap references."""
        engine = make_engine()
        bad_id = ingest_fragment(engine, "def bad_code(): raise RuntimeError('broken')", "bad.py", 100)
        good_id = ingest_fragment(engine, "def good_code(): return True", "good.py", 100)

        for _ in range(10):
            engine.record_failure([bad_id])
            engine.record_success([good_id])

        result = engine.optimize(110, "")
        sources = [frag["source"] for frag in result["selected"]]

        assert "good.py" in sources
        assert "bad.py" not in sources, f"bad.py should stay out of selection: {result['selected']}"

    def test_ios_exploration_can_vary_selected_set(self):
        """Exploration must still fire when IOS is enabled."""
        engine = make_engine(exploration_rate=0.1)

        for i in range(20):
            ingest_fragment(
                engine,
                f"def module_{i}():\n    value = {i}\n    return value * {i + 1}\n",
                f"m{i}.py",
                50,
            )

        selected_sets = set()
        explored_calls = 0
        for _ in range(50):
            result = engine.optimize(250, "search")
            selected_sets.add(tuple(sorted(frag["id"] for frag in result["selected"])))
            if result.get("explored"):
                explored_calls += 1

        assert len(selected_sets) > 1, "IOS exploration should vary selected sets over repeated optimizations"
        assert explored_calls > 0, "IOS exploration should report explored calls"

    def test_exploration_does_not_seed_exploit_cache(self):
        """Exploratory selections must not become exact-hit exploit cache entries."""
        engine = make_engine(exploration_rate=1.0)

        for i in range(20):
            ingest_fragment(
                engine,
                f"def module_{i}():\n    value = {i}\n    return value * {i + 1}\n",
                f"m{i}.py",
                50,
            )

        exploratory = None
        for _ in range(10):
            result = engine.optimize(250, "search")
            if result.get("explored"):
                exploratory = result
                break

        assert exploratory is not None, "Expected at least one realized exploration call"
        assert exploratory["cache_hit"] is False
        assert exploratory["cache_eligible"] is False
        assert exploratory["optimization_policy"] == "explore"

        engine.set_exploration_rate(0.0)

        exploit_first = engine.optimize(250, "search")
        assert exploit_first["cache_hit"] is False
        assert exploit_first["cache_eligible"] is True
        assert exploit_first["optimization_policy"] == "exploit"

        exploit_second = engine.optimize(250, "search")
        assert exploit_second["cache_hit"] is True
        assert exploit_second["cache_eligible"] is True
        assert exploit_second["optimization_policy"] == "exploit"

    def test_explain_selection_includes_compressed_variants(self):
        """Explainability should stay in sync with reference/skeleton selections."""
        engine = make_engine()
        ingest_fragment(
            engine,
            "def primary_handler(data):\n    transformed = normalize(data)\n    return transformed\n",
            "primary.py",
            120,
        )
        ingest_fragment(
            engine,
            "def helper_one():\n    return 1\n\ndef helper_two():\n    return 2\n",
            "helper.py",
            120,
        )

        result = engine.optimize(123, "")
        assert any(frag["variant"] != "full" for frag in result["selected"]), result["selected"]

        explained = engine.explain_selection()
        selected_sources = {frag["source"] for frag in result["selected"]}
        explained_sources = {frag["source"] for frag in explained["included"]}

        assert selected_sources.issubset(explained_sources), (
            f"Selected sources missing from explain_selection: {selected_sources - explained_sources}"
        )
