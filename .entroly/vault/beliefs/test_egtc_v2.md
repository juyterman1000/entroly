---
claim_id: 25708686-5512-41da-80c3-3be4a61bb624
entity: test_egtc_v2
status: inferred
confidence: 0.75
sources:
  - tests\test_egtc_v2.py:59
  - tests\test_egtc_v2.py:108
  - tests\test_egtc_v2.py:143
  - tests\test_egtc_v2.py:187
  - tests\test_egtc_v2.py:229
  - tests\test_egtc_v2.py:254
  - tests\test_egtc_v2.py:284
  - tests\test_egtc_v2.py:329
  - tests\test_egtc_v2.py:365
  - tests\test_egtc_v2.py:396
last_checked: 2026-04-23T03:07:07.931692+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_egtc_v2

**Language:** python
**Lines of code:** 842

## Types
- `class TestFisherBase()` — The Fisher base τ_fisher = (H_c + ε)^(1/4) × scale.
- `class TestSigmoidCorrection()` — Sigmoid correction modulates τ for non-Fisher signals.
- `class TestMonotonicity()` — Critical monotonicity properties that must always hold.
- `class TestBounds()` — Temperature must always be in [0.15, 0.95].
- `class TestTaskTypeOrdering()` — Task types should produce τ in expected relative order.
- `class TestDispersion()` — High entropy dispersion → model must be selective → lower τ.
- `class TestTrajectoryConvergence()` — Temperature should decay across conversation turns.
- `class TestTrajectoryMath()` — Verify the exact convergence formula.
- `class TestUserOverride()` — User-set temperature must never be overwritten.
- `class TestContextInjection()` — OpenAI and Anthropic context injection correctness.
- `class TestProviderDetection()`
- `class TestMessageExtraction()`
- `class TestTokenBudget()`
- `class TestContextBlockFormat()`
- `class TestTunableCoefficients()` — EGTC coefficients can be overridden for autotune.
- `class TestRustEntropyScore()` — entropy_score must be present in selected fragment dicts.
- `class TestVaguenessAlwaysPresent()` — query_analysis should be in optimize_context result regardless of whether refinement triggers.
- `class TestEndToEnd()` — Full pipeline: Rust engine → optimize → EGTC → temperature.
- `class TestAutotuneConfig()` — tuning_config.json should have the egtc section.
- `class TestEdgeCases()` — No fragments → still produces valid temperature.

## Functions
- `def test_fisher_increases_with_entropy(self)` — Higher mean context entropy → higher Fisher base → higher τ.
- `def test_fisher_fourth_root_shape(self)` — Verify the H^(1/4) relationship holds approximately.
- `def test_fisher_zero_entropy_doesnt_collapse(self)` — H_c=0 should not produce τ=0 due to epsilon guard.
- `def test_fisher_scale_parameter(self)` — Custom fisher_scale adjusts the base proportionally.
- `def test_correction_range(self)` — Correction factor should map to [0.3, 1.7].
- `def test_no_double_counting_entropy(self)` — Changing H_c should only affect Fisher base, not sigmoid z.
- `def test_vagueness_monotone_increasing(self)` — More vague query → higher temperature (explore more).
- `def test_sufficiency_monotone_decreasing(self)` — More sufficient context → lower temperature (more constrained).
- `def test_entropy_monotone_increasing(self)` — Higher context entropy → higher Fisher base → higher τ.
- `def test_bounds_parametric(self, vagueness, sufficiency, task_type)`
- `def test_extreme_inputs(self)` — Extreme/adversarial inputs stay bounded.
- `def test_negative_inputs_clamped(self)` — Negative inputs should be clamped, not cause errors.
- `def test_above_one_inputs_clamped(self)` — Inputs > 1.0 should be clamped.
- `def test_precision_tasks_lower_than_creative(self)` — BugTracing < Refactoring < Unknown < CodeGeneration < Exploration.
- `def test_unknown_task_type_fallback(self)` — Unknown/unseen task types use neutral bias (0.0).
- `def test_high_dispersion_lowers_temperature(self)` — Heterogeneous fragments (mix of boilerplate + complex) → lower τ.
- `def test_single_fragment_no_dispersion(self)` — Single fragment has no dispersion (D=0).
- `def test_turn_zero_unchanged(self)` — At turn 0, temperature is unmodified.
- `def test_monotone_decreasing(self)` — Temperature decreases monotonically with turn count.
- `def test_converges_to_c_min(self)` — At high turn counts, τ → c_min × τ_base.
- `def test_never_below_tau_min(self)` — Even with extreme convergence, τ >= τ_min.
- `def test_custom_convergence_rate(self)` — Faster lambda converges faster.
- `def test_half_convergence_at_ln2_over_lambda(self)` — Half-convergence should occur at turn ≈ ln(2)/λ.
- `def test_exact_formula(self)` — Verify the exact formula at a specific turn.
- `def test_explicit_temperature_preserved(self)`
- `def test_no_temperature_gets_injected(self)`
- `def test_temperature_zero_is_explicit(self)` — temperature=0 is a valid explicit choice (greedy decoding).
- `def test_original_body_not_mutated(self)` — apply_temperature should deepcopy, not mutate in place.
- `def test_openai_new_system_message(self)`
- `def test_openai_prepend_to_existing_system(self)`
- `def test_anthropic_new_system(self)`
- `def test_anthropic_prepend_to_existing(self)`
- `def test_anthropic_system_content_blocks(self)`
- `def test_injection_doesnt_mutate_original(self)`
- `def test_anthropic_by_path(self)`
- `def test_openai_by_path(self)`
- `def test_anthropic_by_api_key_header(self)`
- `def test_openai_with_authorization(self)`
- `def test_both_headers_prefers_openai(self)` — When both x-api-key and authorization present, it's OpenAI.
- `def test_simple_string_content(self)`
- `def test_content_blocks(self)`
- `def test_last_user_message(self)`
- `def test_no_user_message(self)`
- `def test_empty_messages(self)`
- `def test_gpt4o_budget(self)`
- `def test_claude_opus_budget(self)`
- `def test_unknown_model_uses_default(self)`
- `def test_prefix_matching(self)` — gpt-4o-2024-08-06 should match gpt-4o.
- `def test_empty_fragments_returns_empty(self)`
- `def test_fragments_formatted(self)`
- `def test_security_warnings_included(self)`
- `def test_ltm_memories_included(self)`
- `def test_refinement_info_included(self)`
- `def test_language_inference(self)` — File extensions should map to correct language tags.
- `def test_higher_alpha_increases_vagueness_effect(self)`
- `def test_higher_gamma_increases_sufficiency_effect(self)`
- `def test_higher_eps_d_increases_dispersion_effect(self)`
- `def test_entropy_score_present(self)`
- `def test_entropy_score_not_default(self)` — entropy_score should reflect actual content, not a fixed 0.5.
- `def test_specific_query_has_analysis(self)`
- `def test_vague_query_has_analysis(self)`
- `def test_full_pipeline(self)`
- `def test_egtc_section_exists(self)`
- `def test_egtc_has_all_coefficients(self)`
- `def test_egtc_values_in_bounds(self)`
- `def test_empty_fragment_entropies(self)` — No fragments → still produces valid temperature.
- `def test_single_fragment(self)`
- `def test_many_fragments(self)` — 100 fragments should work without performance issues.
- `def test_all_zero_entropies(self)`
- `def test_all_one_entropies(self)`
- `def test_trajectory_negative_turn(self)` — Negative turn count should be treated as zero.
- `def test_deterministic(self)` — Same inputs → same output (no randomness in EGTC).
- `def test_proxy_config_loads_egtc(self)` — ProxyConfig.from_env() should load EGTC values from tuning_config.json.

## Dependencies
- `entroly.proxy_config`
- `entroly.proxy_transform`
- `json`
- `math`
- `os`
- `pathlib`
- `pytest`
- `sys`

## Key Invariants
- TestMonotonicity: Critical monotonicity properties that must always hold.
- TestBounds: Temperature must always be in [0.15, 0.95].
- TestDispersion: High entropy dispersion → model must be selective → lower τ.
- TestUserOverride: User-set temperature must never be overwritten.
- TestRustEntropyScore: entropy_score must be present in selected fragment dicts.
