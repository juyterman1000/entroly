"""
EGTC v2 — Comprehensive Test Suite
====================================

Tests for Entropy-Gap Temperature Calibration v2:
  T-01  FISHER BASE              H^(1/4) base temperature computation
  T-02  SIGMOID CORRECTION       non-Fisher signal modulation
  T-03  MONOTONICITY             vagueness↑ → τ↑, sufficiency↑ → τ↓
  T-04  BOUNDS                   τ always in [τ_min, τ_max]
  T-05  TASK TYPE ORDERING       BugTracing < Unknown < Exploration
  T-06  DISPERSION EFFECT        high entropy dispersion → lower τ
  T-07  TRAJECTORY CONVERGENCE   temperature decays across turns
  T-08  TRAJECTORY MATH          exact convergence formula verification
  T-09  USER OVERRIDE            explicit temperature is never overwritten
  T-10  CONTEXT INJECTION        OpenAI and Anthropic injection correctness
  T-11  PROVIDER DETECTION       path/header-based provider detection
  T-12  MESSAGE EXTRACTION       user message extraction from various formats
  T-13  TOKEN BUDGET             model-aware context budget computation
  T-14  CONTEXT BLOCK FORMAT     formatted output structure
  T-15  TUNABLE COEFFICIENTS     custom alpha/gamma/epsilon/fisher_scale
  T-16  RUST ENTROPY SCORE       entropy_score present in optimize() output
  T-17  VAGUENESS ALWAYS PRESENT query_analysis in optimize_context result
  T-18  END-TO-END PIPELINE      full proxy pipeline with real Rust engine
  T-19  AUTOTUNE CONFIG          tuning_config.json has egtc section
  T-20  EDGE CASES               empty inputs, NaN guards, extreme values
"""

import json
import math
import os
import sys
from pathlib import Path

import pytest

# Ensure the entroly package is importable
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from entroly.proxy_transform import (
    apply_temperature,
    apply_trajectory_convergence,
    compute_optimal_temperature,
    compute_token_budget,
    detect_provider,
    extract_model,
    extract_user_message,
    format_context_block,
    inject_context_anthropic,
    inject_context_openai,
)
from entroly.proxy_config import ProxyConfig, context_window_for_model


# ═══════════════════════════════════════════════════════════════════════
# T-01: Fisher Base Temperature
# ═══════════════════════════════════════════════════════════════════════

class TestFisherBase:
    """The Fisher base τ_fisher = (H_c + ε)^(1/4) × scale."""

    def test_fisher_increases_with_entropy(self):
        """Higher mean context entropy → higher Fisher base → higher τ."""
        tau_low = compute_optimal_temperature(0.0, [0.1, 0.1], 0.5, "Unknown")
        tau_mid = compute_optimal_temperature(0.0, [0.5, 0.5], 0.5, "Unknown")
        tau_high = compute_optimal_temperature(0.0, [0.9, 0.9], 0.5, "Unknown")
        assert tau_low < tau_mid < tau_high, (
            f"Fisher base should increase with H_c: {tau_low} < {tau_mid} < {tau_high}"
        )

    def test_fisher_fourth_root_shape(self):
        """Verify the H^(1/4) relationship holds approximately."""
        # With all other signals zeroed out (v=0, s=0, Unknown task),
        # the correction factor is constant, so τ ∝ (H_c + ε)^(1/4)
        tau_a = compute_optimal_temperature(0.0, [0.2], 0.0, "Unknown")
        tau_b = compute_optimal_temperature(0.0, [0.8], 0.0, "Unknown")

        # Predicted ratio from Fisher: ((0.8+0.01)/(0.2+0.01))^(1/4)
        predicted_ratio = ((0.8 + 0.01) / (0.2 + 0.01)) ** 0.25
        actual_ratio = tau_b / tau_a
        # Should be close (exact when correction is constant)
        assert abs(actual_ratio - predicted_ratio) < 0.01, (
            f"Fisher ratio mismatch: actual={actual_ratio:.4f}, predicted={predicted_ratio:.4f}"
        )

    def test_fisher_zero_entropy_doesnt_collapse(self):
        """H_c=0 should not produce τ=0 due to epsilon guard."""
        tau = compute_optimal_temperature(0.0, [0.0, 0.0], 0.5, "Unknown")
        assert tau >= 0.15, f"Zero entropy should produce valid τ: τ={tau}"

    def test_fisher_scale_parameter(self):
        """Custom fisher_scale adjusts the base proportionally."""
        tau_default = compute_optimal_temperature(
            0.3, [0.5], 0.5, "Unknown", fisher_scale=0.55
        )
        tau_high = compute_optimal_temperature(
            0.3, [0.5], 0.5, "Unknown", fisher_scale=0.75
        )
        assert tau_high > tau_default, (
            f"Higher fisher_scale should raise τ: {tau_high} vs {tau_default}"
        )


# ═══════════════════════════════════════════════════════════════════════
# T-02: Sigmoid Correction
# ═══════════════════════════════════════════════════════════════════════

class TestSigmoidCorrection:
    """Sigmoid correction modulates τ for non-Fisher signals."""

    def test_correction_range(self):
        """Correction factor should map to [0.3, 1.7]."""
        # Extreme precision: BugTracing, high sufficiency, no vagueness
        tau_min_corr = compute_optimal_temperature(0.0, [0.5], 1.0, "BugTracing")
        # Extreme exploration: Exploration, high vagueness, no sufficiency
        tau_max_corr = compute_optimal_temperature(1.0, [0.5], 0.0, "Exploration")
        # Both should be within bounds
        assert 0.15 <= tau_min_corr <= 0.95
        assert 0.15 <= tau_max_corr <= 0.95
        # Max should be significantly higher than min
        assert tau_max_corr > tau_min_corr * 1.5, (
            f"Correction range too narrow: {tau_min_corr} → {tau_max_corr}"
        )

    def test_no_double_counting_entropy(self):
        """Changing H_c should only affect Fisher base, not sigmoid z."""
        # If entropy were in both stages, the effect would be amplified.
        # With two-stage, the ratio between different H_c values should
        # match the Fisher prediction exactly (correction is H_c-independent).
        tau_a = compute_optimal_temperature(0.3, [0.3], 0.5, "Unknown")
        tau_b = compute_optimal_temperature(0.3, [0.7], 0.5, "Unknown")
        ratio = tau_b / tau_a
        fisher_ratio = ((0.7 + 0.01) / (0.3 + 0.01)) ** 0.25
        assert abs(ratio - fisher_ratio) < 0.02, (
            f"Double-counting suspected: ratio={ratio:.4f}, fisher={fisher_ratio:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# T-03: Monotonicity Properties
# ═══════════════════════════════════════════════════════════════════════

class TestMonotonicity:
    """Critical monotonicity properties that must always hold."""

    def test_vagueness_monotone_increasing(self):
        """More vague query → higher temperature (explore more)."""
        taus = [
            compute_optimal_temperature(v, [0.5, 0.5], 0.5, "Unknown")
            for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ]
        for i in range(len(taus) - 1):
            assert taus[i] <= taus[i + 1], (
                f"Vagueness monotonicity violated at v={0.2*i:.1f}: "
                f"τ[{i}]={taus[i]} > τ[{i+1}]={taus[i+1]}"
            )

    def test_sufficiency_monotone_decreasing(self):
        """More sufficient context → lower temperature (more constrained)."""
        taus = [
            compute_optimal_temperature(0.3, [0.5, 0.5], s, "Unknown")
            for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ]
        for i in range(len(taus) - 1):
            assert taus[i] >= taus[i + 1], (
                f"Sufficiency monotonicity violated at s={0.2*i:.1f}: "
                f"τ[{i}]={taus[i]} < τ[{i+1}]={taus[i+1]}"
            )

    def test_entropy_monotone_increasing(self):
        """Higher context entropy → higher Fisher base → higher τ."""
        taus = [
            compute_optimal_temperature(0.3, [h], 0.5, "Unknown")
            for h in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ]
        for i in range(len(taus) - 1):
            assert taus[i] <= taus[i + 1], (
                f"Entropy monotonicity violated at h={0.2*i:.1f}: "
                f"τ[{i}]={taus[i]} > τ[{i+1}]={taus[i+1]}"
            )


# ═══════════════════════════════════════════════════════════════════════
# T-04: Bounds
# ═══════════════════════════════════════════════════════════════════════

class TestBounds:
    """Temperature must always be in [0.15, 0.95]."""

    @pytest.mark.parametrize("vagueness", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("sufficiency", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("task_type", [
        "BugTracing", "Unknown", "Exploration", "Documentation",
    ])
    def test_bounds_parametric(self, vagueness, sufficiency, task_type):
        tau = compute_optimal_temperature(
            vagueness, [0.3, 0.7], sufficiency, task_type
        )
        assert 0.15 <= tau <= 0.95, (
            f"Out of bounds: τ={tau} for v={vagueness}, s={sufficiency}, "
            f"task={task_type}"
        )

    def test_extreme_inputs(self):
        """Extreme/adversarial inputs stay bounded."""
        # All signals pushing temperature up
        tau_max = compute_optimal_temperature(1.0, [1.0] * 100, 0.0, "Exploration")
        assert tau_max <= 0.95

        # All signals pushing temperature down
        tau_min = compute_optimal_temperature(0.0, [0.0] * 100, 1.0, "BugTracing")
        assert tau_min >= 0.15

    def test_negative_inputs_clamped(self):
        """Negative inputs should be clamped, not cause errors."""
        tau = compute_optimal_temperature(-1.0, [-0.5, -0.3], -1.0, "Unknown")
        assert 0.15 <= tau <= 0.95

    def test_above_one_inputs_clamped(self):
        """Inputs > 1.0 should be clamped."""
        tau = compute_optimal_temperature(5.0, [2.0, 3.0], 5.0, "Unknown")
        assert 0.15 <= tau <= 0.95


# ═══════════════════════════════════════════════════════════════════════
# T-05: Task Type Ordering
# ═══════════════════════════════════════════════════════════════════════

class TestTaskTypeOrdering:
    """Task types should produce τ in expected relative order."""

    def test_precision_tasks_lower_than_creative(self):
        """BugTracing < Refactoring < Unknown < CodeGeneration < Exploration."""
        base_args = ([0.5, 0.5], 0.5)
        tasks = ["BugTracing", "Refactoring", "Unknown", "CodeGeneration", "Exploration"]
        taus = [compute_optimal_temperature(0.3, *base_args, t) for t in tasks]
        for i in range(len(taus) - 1):
            assert taus[i] <= taus[i + 1], (
                f"Task ordering violated: {tasks[i]}({taus[i]}) > "
                f"{tasks[i+1]}({taus[i+1]})"
            )

    def test_unknown_task_type_fallback(self):
        """Unknown/unseen task types use neutral bias (0.0)."""
        tau_unknown = compute_optimal_temperature(0.3, [0.5], 0.5, "Unknown")
        tau_novel = compute_optimal_temperature(0.3, [0.5], 0.5, "NeverSeenBefore")
        assert tau_unknown == tau_novel


# ═══════════════════════════════════════════════════════════════════════
# T-06: Dispersion Effect
# ═══════════════════════════════════════════════════════════════════════

class TestDispersion:
    """High entropy dispersion → model must be selective → lower τ."""

    def test_high_dispersion_lowers_temperature(self):
        """Heterogeneous fragments (mix of boilerplate + complex) → lower τ."""
        # Uniform entropy: no dispersion
        tau_uniform = compute_optimal_temperature(
            0.3, [0.5, 0.5, 0.5, 0.5], 0.5, "Unknown"
        )
        # High dispersion: mix of low and high entropy
        tau_dispersed = compute_optimal_temperature(
            0.3, [0.1, 0.9, 0.1, 0.9], 0.5, "Unknown"
        )
        assert tau_dispersed < tau_uniform, (
            f"Dispersion should lower τ: uniform={tau_uniform}, dispersed={tau_dispersed}"
        )

    def test_single_fragment_no_dispersion(self):
        """Single fragment has no dispersion (D=0)."""
        # With 1 fragment, dispersion contribution is 0
        tau_single = compute_optimal_temperature(0.3, [0.5], 0.5, "Unknown")
        tau_uniform = compute_optimal_temperature(0.3, [0.5, 0.5], 0.5, "Unknown")
        # Should be equal since same mean and zero dispersion in both
        assert abs(tau_single - tau_uniform) < 0.001


# ═══════════════════════════════════════════════════════════════════════
# T-07: Trajectory Convergence
# ═══════════════════════════════════════════════════════════════════════

class TestTrajectoryConvergence:
    """Temperature should decay across conversation turns."""

    def test_turn_zero_unchanged(self):
        """At turn 0, temperature is unmodified."""
        tau = apply_trajectory_convergence(0.5, 0)
        assert tau == 0.5

    def test_monotone_decreasing(self):
        """Temperature decreases monotonically with turn count."""
        tau_base = 0.6
        taus = [apply_trajectory_convergence(tau_base, t) for t in range(50)]
        for i in range(len(taus) - 1):
            assert taus[i] >= taus[i + 1], (
                f"Trajectory not monotone at turn {i}: {taus[i]} < {taus[i+1]}"
            )

    def test_converges_to_c_min(self):
        """At high turn counts, τ → c_min × τ_base."""
        tau_base = 0.8
        tau_100 = apply_trajectory_convergence(tau_base, 100)
        tau_500 = apply_trajectory_convergence(tau_base, 500)
        expected_steady = tau_base * 0.6  # c_min default
        assert abs(tau_500 - expected_steady) < 0.01, (
            f"Should converge to {expected_steady:.3f}, got {tau_500}"
        )

    def test_never_below_tau_min(self):
        """Even with extreme convergence, τ >= τ_min."""
        tau = apply_trajectory_convergence(0.2, 10000, c_min=0.1, lam=1.0)
        assert tau >= 0.15

    def test_custom_convergence_rate(self):
        """Faster lambda converges faster."""
        tau_slow = apply_trajectory_convergence(0.5, 5, lam=0.03)
        tau_fast = apply_trajectory_convergence(0.5, 5, lam=0.15)
        assert tau_fast < tau_slow, (
            f"Faster λ should converge faster: slow={tau_slow}, fast={tau_fast}"
        )


# ═══════════════════════════════════════════════════════════════════════
# T-08: Trajectory Math
# ═══════════════════════════════════════════════════════════════════════

class TestTrajectoryMath:
    """Verify the exact convergence formula."""

    def test_half_convergence_at_ln2_over_lambda(self):
        """Half-convergence should occur at turn ≈ ln(2)/λ."""
        lam = 0.07
        c_min = 0.6
        half_turn = math.log(2) / lam  # ≈ 9.9

        tau_base = 0.8
        tau_half = apply_trajectory_convergence(tau_base, int(round(half_turn)))

        # At half-convergence: factor = 1 - 0.5*(1-c_min) = 1 - 0.2 = 0.8
        expected = tau_base * (1.0 - 0.5 * (1.0 - c_min))
        assert abs(tau_half - expected) < 0.02, (
            f"Half-convergence mismatch: got {tau_half}, expected ~{expected:.3f}"
        )

    def test_exact_formula(self):
        """Verify the exact formula at a specific turn."""
        tau = 0.7
        turn = 20
        c_min = 0.6
        lam = 0.07

        # Expected: convergence = 1 - (1-0.6)*(1-exp(-0.07*20))
        convergence = 1.0 - (1.0 - c_min) * (1.0 - math.exp(-lam * turn))
        expected = max(0.15, round(tau * convergence, 4))
        actual = apply_trajectory_convergence(tau, turn, c_min=c_min, lam=lam)
        assert actual == expected, f"Formula mismatch: {actual} != {expected}"


# ═══════════════════════════════════════════════════════════════════════
# T-09: User Override
# ═══════════════════════════════════════════════════════════════════════

class TestUserOverride:
    """User-set temperature must never be overwritten."""

    def test_explicit_temperature_preserved(self):
        body = {"model": "gpt-4o", "temperature": 0.7}
        result = apply_temperature(body, 0.3)
        assert result["temperature"] == 0.7

    def test_no_temperature_gets_injected(self):
        body = {"model": "gpt-4o"}
        result = apply_temperature(body, 0.42)
        assert result["temperature"] == 0.42

    def test_temperature_zero_is_explicit(self):
        """temperature=0 is a valid explicit choice (greedy decoding)."""
        body = {"model": "gpt-4o", "temperature": 0}
        result = apply_temperature(body, 0.5)
        assert result["temperature"] == 0

    def test_original_body_not_mutated(self):
        """apply_temperature should deepcopy, not mutate in place."""
        body = {"model": "gpt-4o", "messages": []}
        result = apply_temperature(body, 0.5)
        assert "temperature" not in body
        assert result["temperature"] == 0.5


# ═══════════════════════════════════════════════════════════════════════
# T-10: Context Injection
# ═══════════════════════════════════════════════════════════════════════

class TestContextInjection:
    """OpenAI and Anthropic context injection correctness."""

    def test_openai_new_system_message(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}
        result = inject_context_openai(body, "CONTEXT")
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "CONTEXT"
        assert result["messages"][1]["content"] == "hello"

    def test_openai_prepend_to_existing_system(self):
        body = {"messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
        ]}
        result = inject_context_openai(body, "CONTEXT")
        assert result["messages"][0]["content"] == "CONTEXT\n\nYou are helpful"

    def test_anthropic_new_system(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}
        result = inject_context_anthropic(body, "CONTEXT")
        assert result["system"] == "CONTEXT"

    def test_anthropic_prepend_to_existing(self):
        body = {"system": "Be precise", "messages": []}
        result = inject_context_anthropic(body, "CONTEXT")
        assert result["system"] == "CONTEXT\n\nBe precise"

    def test_anthropic_system_content_blocks(self):
        body = {
            "system": [{"type": "text", "text": "existing"}],
            "messages": [],
        }
        result = inject_context_anthropic(body, "CONTEXT")
        assert isinstance(result["system"], list)
        assert result["system"][0] == {"type": "text", "text": "CONTEXT"}
        assert result["system"][1] == {"type": "text", "text": "existing"}

    def test_injection_doesnt_mutate_original(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        inject_context_openai(body, "CONTEXT")
        assert len(body["messages"]) == 1  # original unchanged


# ═══════════════════════════════════════════════════════════════════════
# T-11: Provider Detection
# ═══════════════════════════════════════════════════════════════════════

class TestProviderDetection:

    def test_anthropic_by_path(self):
        assert detect_provider("/v1/messages", {}) == "anthropic"

    def test_openai_by_path(self):
        assert detect_provider("/v1/chat/completions", {}) == "openai"

    def test_anthropic_by_api_key_header(self):
        assert detect_provider("/v1/chat/completions", {"x-api-key": "sk-ant-..."}) == "anthropic"

    def test_openai_with_authorization(self):
        assert detect_provider("/v1/chat/completions", {"authorization": "Bearer sk-..."}) == "openai"

    def test_both_headers_prefers_openai(self):
        """When both x-api-key and authorization present, it's OpenAI."""
        headers = {"x-api-key": "key", "authorization": "Bearer sk"}
        assert detect_provider("/v1/chat/completions", headers) == "openai"


# ═══════════════════════════════════════════════════════════════════════
# T-12: Message Extraction
# ═══════════════════════════════════════════════════════════════════════

class TestMessageExtraction:

    def test_simple_string_content(self):
        body = {"messages": [{"role": "user", "content": "fix the bug"}]}
        assert extract_user_message(body, "openai") == "fix the bug"

    def test_content_blocks(self):
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]}]}
        assert extract_user_message(body, "anthropic") == "hello world"

    def test_last_user_message(self):
        body = {"messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ]}
        assert extract_user_message(body, "openai") == "second"

    def test_no_user_message(self):
        body = {"messages": [{"role": "system", "content": "be helpful"}]}
        assert extract_user_message(body, "openai") == ""

    def test_empty_messages(self):
        assert extract_user_message({"messages": []}, "openai") == ""
        assert extract_user_message({}, "openai") == ""


# ═══════════════════════════════════════════════════════════════════════
# T-13: Token Budget
# ═══════════════════════════════════════════════════════════════════════

class TestTokenBudget:

    def test_gpt4o_budget(self):
        config = ProxyConfig(context_fraction=0.15)
        budget = compute_token_budget("gpt-4o", config)
        assert budget == int(128_000 * 0.15)

    def test_claude_opus_budget(self):
        config = ProxyConfig(context_fraction=0.10)
        budget = compute_token_budget("claude-opus-4-6", config)
        assert budget == int(200_000 * 0.10)

    def test_unknown_model_uses_default(self):
        config = ProxyConfig(context_fraction=0.15)
        budget = compute_token_budget("future-model-xyz", config)
        assert budget == int(128_000 * 0.15)

    def test_prefix_matching(self):
        """gpt-4o-2024-08-06 should match gpt-4o."""
        window = context_window_for_model("gpt-4o-2024-08-06")
        assert window == 128_000


# ═══════════════════════════════════════════════════════════════════════
# T-14: Context Block Format
# ═══════════════════════════════════════════════════════════════════════

class TestContextBlockFormat:

    def test_empty_fragments_returns_empty(self):
        assert format_context_block([], [], [], None) == ""

    def test_fragments_formatted(self):
        frags = [{"source": "main.py", "relevance": 0.85, "token_count": 50,
                  "preview": "def main():\n    pass"}]
        result = format_context_block(frags, [], [], None)
        assert "main.py" in result
        assert "relevance: 0.85" in result
        assert "```python" in result
        assert "def main():" in result

    def test_security_warnings_included(self):
        frags = [{"source": "a.py", "relevance": 0.5, "token_count": 10,
                  "preview": "x=1"}]
        result = format_context_block(frags, ["[a.py] hardcoded password"], [], None)
        assert "Security Warnings" in result
        assert "hardcoded password" in result

    def test_ltm_memories_included(self):
        mems = [{"retention": 0.8, "content": "User prefers tabs over spaces"}]
        result = format_context_block([], [], mems, None)
        assert "Cross-Session Memory" in result
        assert "tabs over spaces" in result

    def test_refinement_info_included(self):
        frags = [{"source": "a.py", "relevance": 0.5, "token_count": 10,
                  "preview": "x=1"}]
        ref = {"original": "fix it", "refined": "fix auth bug in login.py",
               "vagueness": 0.7}
        result = format_context_block(frags, [], [], ref)
        assert "Query refined" in result
        assert "fix it" in result
        assert "fix auth bug" in result

    def test_language_inference(self):
        """File extensions should map to correct language tags."""
        for ext, lang in [(".rs", "rust"), (".ts", "typescript"), (".go", "go")]:
            frags = [{"source": f"file{ext}", "relevance": 0.5,
                      "token_count": 10, "preview": "code"}]
            result = format_context_block(frags, [], [], None)
            assert f"```{lang}" in result, f"Expected {lang} for {ext}"


# ═══════════════════════════════════════════════════════════════════════
# T-15: Tunable Coefficients
# ═══════════════════════════════════════════════════════════════════════

class TestTunableCoefficients:
    """EGTC coefficients can be overridden for autotune."""

    def test_higher_alpha_increases_vagueness_effect(self):
        tau_low_alpha = compute_optimal_temperature(
            0.8, [0.5], 0.5, "Unknown", alpha=0.5
        )
        tau_high_alpha = compute_optimal_temperature(
            0.8, [0.5], 0.5, "Unknown", alpha=3.0
        )
        # Higher alpha → vagueness pushes τ up more
        assert tau_high_alpha > tau_low_alpha

    def test_higher_gamma_increases_sufficiency_effect(self):
        tau_low_gamma = compute_optimal_temperature(
            0.3, [0.5], 0.8, "Unknown", gamma=0.3
        )
        tau_high_gamma = compute_optimal_temperature(
            0.3, [0.5], 0.8, "Unknown", gamma=2.5
        )
        # Higher gamma → sufficiency pushes τ down more
        assert tau_high_gamma < tau_low_gamma

    def test_higher_eps_d_increases_dispersion_effect(self):
        # High dispersion fragments
        entropies = [0.1, 0.9, 0.1, 0.9]
        tau_low_eps = compute_optimal_temperature(
            0.3, entropies, 0.5, "Unknown", eps_d=0.1
        )
        tau_high_eps = compute_optimal_temperature(
            0.3, entropies, 0.5, "Unknown", eps_d=1.5
        )
        # Higher eps_d → dispersion pushes τ down more
        assert tau_high_eps < tau_low_eps


# ═══════════════════════════════════════════════════════════════════════
# T-16: Rust entropy_score in optimize() output
# ═══════════════════════════════════════════════════════════════════════

class TestRustEntropyScore:
    """entropy_score must be present in selected fragment dicts."""

    def test_entropy_score_present(self):
        import entroly_core
        engine = entroly_core.EntrolyEngine()
        engine.ingest(
            "def calculate_tax(income, rate):\n    return income * rate",
            "tax.py", 20, False,
        )
        result = engine.optimize(5000, "calculate tax")
        selected = result.get("selected", [])
        assert len(selected) > 0, "Should select at least one fragment"
        for frag in selected:
            assert "entropy_score" in frag, (
                f"entropy_score missing from fragment: {list(frag.keys())}"
            )
            assert isinstance(frag["entropy_score"], float)
            assert 0.0 <= frag["entropy_score"] <= 1.0, (
                f"entropy_score out of [0,1]: {frag['entropy_score']}"
            )

    def test_entropy_score_not_default(self):
        """entropy_score should reflect actual content, not a fixed 0.5."""
        import entroly_core
        engine = entroly_core.EntrolyEngine()
        # High-entropy content (diverse characters)
        engine.ingest(
            "fn complex_parser(input: &[u8]) -> Result<AST, ParseError> {\n"
            "    let mut stack: Vec<Token> = Vec::with_capacity(256);\n"
            "    for &byte in input { stack.push(Token::from_byte(byte)); }\n"
            "    Ok(AST::build(&stack)?)\n}",
            "parser.rs", 50, False,
        )
        # Low-entropy content (repetitive)
        engine.ingest(
            "import os\nimport sys\nimport json\nimport time\nimport logging\n"
            "import pathlib\nimport typing\nimport dataclasses\n",
            "imports.py", 20, False,
        )
        result = engine.optimize(5000, "parse")
        selected = result.get("selected", [])
        scores = {f["source"]: f["entropy_score"] for f in selected}
        # Parser code should have different entropy from pure imports
        if "parser.rs" in scores and "imports.py" in scores:
            assert scores["parser.rs"] != scores["imports.py"], (
                f"Entropy scores should differ: {scores}"
            )


# ═══════════════════════════════════════════════════════════════════════
# T-17: Vagueness Always Present
# ═══════════════════════════════════════════════════════════════════════

class TestVaguenessAlwaysPresent:
    """query_analysis should be in optimize_context result regardless
    of whether refinement triggers."""

    def test_specific_query_has_analysis(self):
        from entroly.server import EntrolyEngine
        engine = EntrolyEngine()
        engine.ingest_fragment(
            "def calculate_tax(income, rate):\n    return income * rate",
            "tax.py",
        )
        # Very specific query — should NOT trigger refinement
        result = engine.optimize_context(5000, "calculate_tax function in tax.py")
        assert "query_analysis" in result, (
            f"query_analysis missing from result keys: {list(result.keys())}"
        )
        analysis = result["query_analysis"]
        assert "vagueness_score" in analysis
        assert isinstance(analysis["vagueness_score"], float)

    def test_vague_query_has_analysis(self):
        from entroly.server import EntrolyEngine
        engine = EntrolyEngine()
        engine.ingest_fragment("x = 1", "a.py")
        result = engine.optimize_context(5000, "fix it")
        assert "query_analysis" in result
        vagueness = result["query_analysis"]["vagueness_score"]
        # "fix it" should have higher vagueness than a specific query
        assert vagueness > 0.0, f"Vagueness should be > 0 for 'fix it': {vagueness}"


# ═══════════════════════════════════════════════════════════════════════
# T-18: End-to-End Pipeline
# ═══════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    """Full pipeline: Rust engine → optimize → EGTC → temperature."""

    def test_full_pipeline(self):
        import entroly_core
        from entroly.proxy_transform import compute_optimal_temperature

        engine = entroly_core.EntrolyEngine()
        engine.ingest(
            "def process_payment(amount, currency):\n"
            "    rate = get_exchange_rate(currency)\n"
            "    fee = calculate_fee(amount, rate)\n"
            "    return charge(amount + fee, currency)\n",
            "payment.py", 40, False,
        )
        engine.ingest(
            "def validate_input(data):\n"
            "    if not isinstance(data, dict):\n"
            "        raise ValueError('Expected dict')\n"
            "    return True\n",
            "validator.py", 30, False,
        )

        result = engine.optimize(5000, "process payment")
        selected = result.get("selected", [])

        # Extract signals
        entropies = [f["entropy_score"] for f in selected]
        tokens_used = sum(f["token_count"] for f in selected)
        sufficiency = min(1.0, tokens_used / 5000)

        tau = compute_optimal_temperature(
            vagueness=0.2,
            fragment_entropies=entropies,
            sufficiency=sufficiency,
            task_type="BugTracing",
        )

        # Should produce a reasonable temperature
        assert 0.15 <= tau <= 0.95
        # BugTracing with moderate context should be on the lower end
        assert tau < 0.5, f"BugTracing should produce low τ, got {tau}"

        # With trajectory convergence
        tau_converged = apply_trajectory_convergence(tau, 20)
        assert tau_converged < tau, "Trajectory should lower τ"
        assert tau_converged >= 0.15


# ═══════════════════════════════════════════════════════════════════════
# T-19: Autotune Config
# ═══════════════════════════════════════════════════════════════════════

class TestAutotuneConfig:
    """tuning_config.json should have the egtc section."""

    def test_egtc_section_exists(self):
        tc_path = REPO / "entroly" / "tuning_config.json"
        assert tc_path.exists(), f"tuning_config.json not found at {tc_path}"
        with open(tc_path) as f:
            config = json.load(f)
        assert "egtc" in config, f"Missing egtc section. Keys: {list(config.keys())}"

    def test_egtc_has_all_coefficients(self):
        tc_path = REPO / "entroly" / "tuning_config.json"
        with open(tc_path) as f:
            egtc = json.load(f)["egtc"]
        required = ["alpha", "gamma", "epsilon", "fisher_scale",
                     "trajectory_c_min", "trajectory_lambda"]
        for key in required:
            assert key in egtc, f"Missing egtc.{key}"
            assert isinstance(egtc[key], (int, float)), (
                f"egtc.{key} should be numeric, got {type(egtc[key])}"
            )

    def test_egtc_values_in_bounds(self):
        tc_path = REPO / "entroly" / "tuning_config.json"
        with open(tc_path) as f:
            egtc = json.load(f)["egtc"]
        assert 0.5 <= egtc["alpha"] <= 3.0
        assert 0.3 <= egtc["gamma"] <= 2.5
        assert 0.1 <= egtc["epsilon"] <= 1.5
        assert 0.3 <= egtc["fisher_scale"] <= 0.8
        assert 0.3 <= egtc["trajectory_c_min"] <= 0.9
        assert 0.02 <= egtc["trajectory_lambda"] <= 0.15


# ═══════════════════════════════════════════════════════════════════════
# T-20: Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_empty_fragment_entropies(self):
        """No fragments → still produces valid temperature."""
        tau = compute_optimal_temperature(0.5, [], 0.0, "Unknown")
        assert 0.15 <= tau <= 0.95

    def test_single_fragment(self):
        tau = compute_optimal_temperature(0.3, [0.6], 0.5, "CodeGeneration")
        assert 0.15 <= tau <= 0.95

    def test_many_fragments(self):
        """100 fragments should work without performance issues."""
        entropies = [i / 100.0 for i in range(100)]
        tau = compute_optimal_temperature(0.4, entropies, 0.7, "Refactoring")
        assert 0.15 <= tau <= 0.95

    def test_all_zero_entropies(self):
        tau = compute_optimal_temperature(0.0, [0.0, 0.0, 0.0], 1.0, "BugTracing")
        assert 0.15 <= tau <= 0.95

    def test_all_one_entropies(self):
        tau = compute_optimal_temperature(1.0, [1.0, 1.0, 1.0], 0.0, "Exploration")
        assert 0.15 <= tau <= 0.95

    def test_trajectory_negative_turn(self):
        """Negative turn count should be treated as zero."""
        tau = apply_trajectory_convergence(0.5, -5)
        assert tau == 0.5

    def test_deterministic(self):
        """Same inputs → same output (no randomness in EGTC)."""
        args = (0.35, [0.5, 0.6, 0.7], 0.65, "Testing")
        tau1 = compute_optimal_temperature(*args)
        tau2 = compute_optimal_temperature(*args)
        assert tau1 == tau2

    def test_proxy_config_loads_egtc(self):
        """ProxyConfig.from_env() should load EGTC values from tuning_config.json."""
        config = ProxyConfig.from_env()
        assert config.fisher_scale == 0.55
        assert config.trajectory_c_min == 0.6
        assert config.trajectory_lambda == 0.07
