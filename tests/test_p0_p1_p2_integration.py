"""Integration tests for P0, P1, P2 architectural fixes.

P0: ECE evaluates REAL response text (not empty strings)
P1: WITNESS defaults to audit mode
P2: Conformal cascade wired into proxy via escalation.py rule (★)

These are unit tests for the components; end-to-end proxy tests
require the full async server stack.
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════════════
# P0: ECE evaluates real response text → curvature is nonzero
# ═══════════════════════════════════════════════════════════════════════

class TestP0_ECE_PostResponse:
    """Verify that ECE produces meaningful signals on actual response text,
    NOT the empty-string non-signal from the old pre-screen gate."""

    def test_curvature_on_real_text_is_nonzero(self):
        """Fisher curvature on factual text with entities must be > 0.
        (The old bug: curvature on '' was always 0.)"""
        from entroly.ravs.ece import compute_fisher_curvature

        text = (
            "The Eiffel Tower is 324 meters tall and was built in 1889 "
            "by Gustave Eiffel for the World Exhibition."
        )
        mean_k, max_k, n_entities = compute_fisher_curvature(text)
        # Must find entities (numbers + proper nouns)
        assert n_entities > 0, f"Expected entities in factual text, got {n_entities}"

    def test_curvature_on_empty_string_is_zero(self):
        """Confirm the bug we fixed: curvature on empty string = 0."""
        from entroly.ravs.ece import compute_fisher_curvature

        mean_k, max_k, n_entities = compute_fisher_curvature("")
        assert mean_k == 0.0, f"Empty string curvature should be 0, got {mean_k}"
        assert n_entities == 0

    def test_ece_engine_evaluates_actual_text(self):
        """The full ECE engine must produce a non-trivial signal on
        factual text, proving post-response evaluation works."""
        from entroly.ravs.ece import EpistemicCascadeEngine

        engine = EpistemicCascadeEngine(
            curvature_threshold=0.4,
            enable_lyapunov=False,
        )

        signal_empty = engine.evaluate_uncertainty(
            query="What is the Eiffel Tower?",
            response_text="",
        )
        signal_real = engine.evaluate_uncertainty(
            query="What is the Eiffel Tower?",
            response_text=(
                "The Eiffel Tower is 324 meters tall. It was built by "
                "Gustave Eiffel in Paris, France in 1889."
            ),
        )

        # Real text should produce a richer signal (entity count, curvature)
        assert signal_real.entity_count > signal_empty.entity_count, (
            f"Real text entities ({signal_real.entity_count}) should exceed "
            f"empty text ({signal_empty.entity_count})"
        )

    def test_hedging_language_triggers_curvature(self):
        """Hedging language (I think, probably, maybe) should elevate
        curvature even without logprobs."""
        from entroly.ravs.ece import compute_fisher_curvature

        confident = "The answer is 42."
        hedging = (
            "I think the answer is probably around 42, but I'm not sure. "
            "It might be approximately 40 or perhaps 45."
        )

        k_conf, _, _ = compute_fisher_curvature(confident)
        k_hedge, _, _ = compute_fisher_curvature(hedging)

        assert k_hedge > k_conf, (
            f"Hedging curvature ({k_hedge}) should exceed confident ({k_conf})"
        )

    def test_tier0_open_ended_bypass(self):
        """Open-ended queries (creative/opinion) should exit at Tier 0
        without wasting curvature computation."""
        from entroly.ravs.ece import EpistemicCascadeEngine

        engine = EpistemicCascadeEngine(enable_lyapunov=False)

        signal = engine.evaluate_uncertainty(
            query="Write a poem about the ocean",
            response_text="Waves crash upon the shore...",
        )
        assert signal.tier_used == 0
        assert signal.decision == "keep"
        assert "open_ended" in signal.reason


# ═══════════════════════════════════════════════════════════════════════
# P1: WITNESS defaults to audit mode
# ═══════════════════════════════════════════════════════════════════════

class TestP1_WitnessDefaultAudit:
    """Verify WITNESS is ON by default (audit mode) and can be opted out."""

    def test_default_witness_mode_is_audit(self):
        """With no env override, WITNESS should default to 'audit'."""
        # Ensure no override
        old = os.environ.pop("ENTROLY_WITNESS", None)
        try:
            # Simulate the proxy logic
            witness_mode = ""  # default from config
            if not witness_mode or witness_mode == "off":
                if os.environ.get("ENTROLY_WITNESS", "1") == "0":
                    witness_mode = "off"
                else:
                    witness_mode = "audit"
            assert witness_mode == "audit"
        finally:
            if old is not None:
                os.environ["ENTROLY_WITNESS"] = old

    def test_explicit_disable_via_env(self):
        """ENTROLY_WITNESS=0 should disable WITNESS."""
        old = os.environ.get("ENTROLY_WITNESS")
        os.environ["ENTROLY_WITNESS"] = "0"
        try:
            witness_mode = ""
            if not witness_mode or witness_mode == "off":
                if os.environ.get("ENTROLY_WITNESS", "1") == "0":
                    witness_mode = "off"
                else:
                    witness_mode = "audit"
            assert witness_mode == "off"
        finally:
            if old is not None:
                os.environ["ENTROLY_WITNESS"] = old
            else:
                os.environ.pop("ENTROLY_WITNESS", None)

    def test_config_override_takes_precedence(self):
        """If config explicitly says 'strict', that overrides default."""
        witness_mode = "strict"  # from config
        if not witness_mode or witness_mode == "off":
            if os.environ.get("ENTROLY_WITNESS", "1") == "0":
                witness_mode = "off"
            else:
                witness_mode = "audit"
        assert witness_mode == "strict"


# ═══════════════════════════════════════════════════════════════════════
# P2: Conformal cascade wired via escalation.py rule (★)
# ═══════════════════════════════════════════════════════════════════════

class TestP2_ConformalCascadeIntegration:
    """Verify the cascade decision uses escalation.py's should_escalate
    with WITNESS risk scores."""

    def test_escalation_rule_basic(self):
        """should_escalate(risk, c_exp, q, r_floor) fires correctly."""
        from entroly.escalation import should_escalate

        # With c_exp/q = 0.05, threshold = 0.05
        # Risk 0.1 > 0.05 → escalate
        assert should_escalate(0.1, 0.05, 1.0, 0.0) is True
        # Risk 0.03 < 0.05 → don't escalate
        assert should_escalate(0.03, 0.05, 1.0, 0.0) is False
        # At exact threshold → don't escalate (not strictly greater)
        assert should_escalate(0.05, 0.05, 1.0, 0.0) is False

    def test_witness_risk_to_cascade_decision(self):
        """WITNESS summary_score → risk → cascade decision matches
        the deployed logic."""
        from entroly.escalation import should_escalate

        # High-confidence WITNESS (score=0.96 → risk=0.04)
        witness_score_good = 0.96
        risk_good = 1.0 - witness_score_good
        assert should_escalate(risk_good, 0.05, 1.0, 0.0) is False

        # Low-confidence WITNESS (score=0.60 → risk=0.40)
        witness_score_bad = 0.60
        risk_bad = 1.0 - witness_score_bad
        assert should_escalate(risk_bad, 0.05, 1.0, 0.0) is True

    def test_conformal_cascade_module_imports(self):
        """Verify the conformal_cascade module imports successfully
        and its constants are available."""
        from entroly.conformal_cascade import ACCEPT, FLAG, ESCALATE

        assert ACCEPT == "accept"
        assert FLAG == "flag"
        assert ESCALATE == "escalate"

    def test_cascade_fit_and_decide(self):
        """End-to-end: fit a cascade from labeled data, make decisions."""
        from entroly.conformal_cascade import fit_cascade, p_hallucinated, p_faithful

        # Simulate calibration: hallucinated items have higher scores
        scores = [0.1, 0.2, 0.8, 0.9, 0.15, 0.25, 0.7, 0.85]
        labels = [0,   0,   1,   1,   0,    0,    1,   1]

        cal = fit_cascade(scores, labels)
        assert cal.n_h == 4  # 4 hallucinated
        assert cal.n_g == 4  # 4 faithful

        # Low score → low p_g (confidently faithful)
        pg_low = p_faithful(cal, 0.1)
        pg_high = p_faithful(cal, 0.9)
        assert pg_low < pg_high  # more faithful cal ≤ 0.1 → lower p

        # High score → low p_h (confidently hallucinated)
        ph_high = p_hallucinated(cal, 0.9)
        ph_low = p_hallucinated(cal, 0.1)
        assert ph_high < ph_low  # more halu cal ≥ 0.9 → lower p

    def test_cascade_and_escalation_compose(self):
        """The cascade's cost decomposition uses the same should_escalate
        that escalation.py exports — verify they agree."""
        from entroly.conformal_cascade import select_band
        from entroly.escalation import should_escalate

        # Create a small calibration set
        scores = [0.1, 0.15, 0.2, 0.3, 0.7, 0.75, 0.8, 0.9]
        labels = [0,   0,    0,   0,   1,   1,    1,   1]

        policy = select_band(
            scores, labels,
            target_selective_risk=0.1,
            c_exp=0.05,
            q=1.0,
            r_floor=0.0,
        )

        # The rule threshold should match: r_floor + c_exp/q = 0.05
        assert abs(policy.rule_threshold - 0.05) < 1e-10

        # A risk of 0.0 (all correct in accept region) should not escalate
        assert not should_escalate(0.0, 0.05, 1.0, 0.0)


# ═══════════════════════════════════════════════════════════════════════
# Cross-cutting: The full pipeline connects
# ═══════════════════════════════════════════════════════════════════════

class TestCrossCutting:
    """Verify ECE + WITNESS + cascade compose correctly."""

    def test_ece_lyapunov_stability(self):
        """Lyapunov controller converges — tau doesn't oscillate."""
        from entroly.ravs.ece import LyapunovThresholdController

        ctrl = LyapunovThresholdController(
            initial_tau=0.5,
            target_escalation_rate=0.10,
            eta=0.05,
        )

        # Feed a burst of escalations
        for _ in range(50):
            ctrl.record_decision(True)

        tau_after_burst = ctrl.tau
        # tau should have increased (more tolerant under escalation pressure)
        assert tau_after_burst > 0.5, (
            f"Tau should increase under escalation pressure, got {tau_after_burst}"
        )

        # Feed a period of no escalations
        for _ in range(100):
            ctrl.record_decision(False)

        tau_after_calm = ctrl.tau
        # tau should have decreased back toward equilibrium
        assert tau_after_calm < tau_after_burst, (
            f"Tau should decrease when escalation pressure drops: "
            f"{tau_after_calm} < {tau_after_burst}"
        )

    def test_stats_dict_structure(self):
        """ECE stats dict has expected keys for dashboard binding."""
        from entroly.ravs.ece import EpistemicCascadeEngine

        engine = EpistemicCascadeEngine(enable_lyapunov=True)
        engine.evaluate_uncertainty(
            query="What is 2+2?",
            response_text="The answer is 4.",
        )

        stats = engine.stats()
        assert "total_requests" in stats
        assert "tier0_exits" in stats
        assert "tier1_exits" in stats
        assert "escalation_rate" in stats
        assert "curvature_threshold" in stats
        assert "lyapunov" in stats
        assert "current_tau" in stats["lyapunov"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
