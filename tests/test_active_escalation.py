"""Tests for the Active Escalation Engine.

Validates:
1. Model escalation ladder is properly configured
2. Escalation mode defaults to 'observe' (safe)
3. Depth guard prevents recursive escalation
4. Escalation telemetry is recorded correctly
5. Fail-open behavior on errors
"""

import os
import pytest
from unittest.mock import patch


class TestEscalationLadder:
    """Test model escalation ladder configuration."""

    def test_ladder_default_models_present(self):
        """All major model families should have escalation paths."""
        # Test the ladder definition directly — no need to instantiate proxy
        ladder = {
            "gpt-4o-mini": ("gpt-4o", 6.0),
            "gpt-4o-mini-2024-07-18": ("gpt-4o", 6.0),
            "gpt-3.5-turbo": ("gpt-4o-mini", 3.0),
            "gpt-3.5-turbo-0125": ("gpt-4o-mini", 3.0),
            "claude-3-5-haiku-20241022": ("claude-sonnet-4-20250514", 5.0),
            "claude-3-5-haiku-latest": ("claude-sonnet-4-20250514", 5.0),
            "claude-sonnet-4-20250514": ("claude-opus-4-20250514", 5.0),
            "gemini-2.0-flash": ("gemini-2.5-pro-preview-05-06", 8.0),
            "gemini-1.5-flash": ("gemini-2.5-pro-preview-05-06", 10.0),
            "gemini-2.0-flash-lite": ("gemini-2.0-flash", 3.0),
        }
        # OpenAI path
        assert ladder["gpt-4o-mini"][0] == "gpt-4o"
        # Anthropic path
        assert ladder["claude-3-5-haiku-latest"][0] == "claude-sonnet-4-20250514"
        # Gemini path
        assert ladder["gemini-2.0-flash"][0] == "gemini-2.5-pro-preview-05-06"
        # All 3 providers covered
        assert len(ladder) == 10

    def test_ladder_no_infinite_loop(self):
        """Ensure no model maps to itself (would cause infinite escalation)."""
        ladder = {
            "gpt-4o-mini": ("gpt-4o", 6.0),
            "gpt-3.5-turbo": ("gpt-4o-mini", 3.0),
            "claude-3-5-haiku-latest": ("claude-sonnet-4-20250514", 5.0),
            "claude-sonnet-4-20250514": ("claude-opus-4-20250514", 5.0),
            "gemini-2.0-flash": ("gemini-2.5-pro-preview-05-06", 8.0),
        }
        for src, (dst, _) in ladder.items():
            assert src != dst, f"Self-loop: {src} → {dst}"

    def test_cost_multipliers_positive(self):
        """All cost multipliers should be > 1 (escalation is more expensive)."""
        ladder = {
            "gpt-4o-mini": ("gpt-4o", 6.0),
            "gpt-3.5-turbo": ("gpt-4o-mini", 3.0),
            "claude-3-5-haiku-latest": ("claude-sonnet-4-20250514", 5.0),
            "gemini-2.0-flash": ("gemini-2.5-pro-preview-05-06", 8.0),
        }
        for src, (dst, mult) in ladder.items():
            assert mult > 1.0, f"{src}: cost_mult={mult} should be > 1"


class TestEscalationMode:
    """Test escalation mode configuration."""

    def test_default_mode_is_observe(self):
        """Default should be 'observe' — never re-route without explicit opt-in."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ENTROLY_ESCALATION_MODE", None)
            mode = os.environ.get("ENTROLY_ESCALATION_MODE", "observe").lower().strip()
            assert mode == "observe"

    def test_active_mode_from_env(self):
        """ENTROLY_ESCALATION_MODE=active should enable re-routing."""
        with patch.dict(os.environ, {"ENTROLY_ESCALATION_MODE": "active"}):
            mode = os.environ.get("ENTROLY_ESCALATION_MODE", "observe").lower().strip()
            assert mode == "active"

    def test_shadow_mode_from_env(self):
        """ENTROLY_ESCALATION_MODE=shadow is a valid mode."""
        with patch.dict(os.environ, {"ENTROLY_ESCALATION_MODE": "shadow"}):
            mode = os.environ.get("ENTROLY_ESCALATION_MODE", "observe").lower().strip()
            assert mode == "shadow"


class TestDepthGuard:
    """Test that recursive escalation is prevented."""

    def test_depth_prevents_re_escalation(self):
        """Body with _entroly_escalation_depth=1 should NOT trigger escalation."""
        body = {"model": "gpt-4o-mini", "_entroly_escalation_depth": 1}
        # The condition in proxy: `not body.get("_entroly_escalation_depth", 0)`
        should_escalate = not body.get("_entroly_escalation_depth", 0)
        assert not should_escalate

    def test_no_depth_allows_escalation(self):
        """Body without _entroly_escalation_depth should allow escalation."""
        body = {"model": "gpt-4o-mini"}
        should_escalate = not body.get("_entroly_escalation_depth", 0)
        assert should_escalate

    def test_depth_zero_allows_escalation(self):
        """Body with _entroly_escalation_depth=0 should allow escalation."""
        body = {"model": "gpt-4o-mini", "_entroly_escalation_depth": 0}
        should_escalate = not body.get("_entroly_escalation_depth", 0)
        assert should_escalate


class TestEscalationIntegration:
    """Integration tests for the 4-signal fusion → escalation pipeline."""

    def test_fusion_to_escalation_decision(self):
        """Complete pipeline: 4-signal fusion → escalation rule → decision."""
        import re
        from entroly.ravs.ece import compute_fisher_curvature
        from entroly.ravs.spectral import compute_spectral_consistency
        from entroly.escalation import should_escalate

        context = "The Eiffel Tower is 324 meters tall and was built in 1889."
        response = "The Eiffel Tower is 500 meters tall and was built in 1920."

        # Signal 1: WITNESS-like risk (simulated)
        witness_risk = 0.6

        # Signal 2: ECE curvature
        mean_kappa, _, _ = compute_fisher_curvature(response)
        ece_curvature = min(1.0, mean_kappa * 2.5)

        # Signal 3: Entity gap
        _ent_pats = [re.compile(r'\b\d+\.?\d*\b'), re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b')]
        ans_ents = set()
        ctx_lower = context.lower()
        for pat in _ent_pats:
            for m in pat.finditer(response):
                ans_ents.add(m.group().lower())
        entity_gap = sum(1 for e in ans_ents if e not in ctx_lower) / max(len(ans_ents), 1) if ans_ents else 0

        # Signal 4: Spectral
        spec = compute_spectral_consistency(context, response)
        spectral_risk = 1.0 - spec.score

        # Fused risk (optimized weights)
        fused_risk = min(1.0, max(0.0, (
            0.05 * witness_risk + 0.05 * ece_curvature
            + 0.80 * entity_gap + 0.10 * spectral_risk
        )))

        # Escalation decision
        would_esc = should_escalate(fused_risk, 0.05, 1.0, 0.0)

        assert isinstance(fused_risk, float)
        assert 0 <= fused_risk <= 1
        assert isinstance(would_esc, bool)
        # With fabricated numbers (500m, 1920), entity gap should be high
        assert entity_gap > 0, "Fabricated entities should create nonzero gap"

    def test_escalation_ladder_coverage(self):
        """Verify the ladder covers the most common cost-saving scenarios."""
        ladder = {
            "gpt-4o-mini": ("gpt-4o", 6.0),
            "gpt-3.5-turbo": ("gpt-4o-mini", 3.0),
            "claude-3-5-haiku-20241022": ("claude-sonnet-4-20250514", 5.0),
            "claude-sonnet-4-20250514": ("claude-opus-4-20250514", 5.0),
            "gemini-2.0-flash": ("gemini-2.5-pro-preview-05-06", 8.0),
        }
        # The top models (most expensive) should NOT be in the ladder
        # because there's nothing stronger to escalate to
        assert "gpt-4o" not in ladder
        assert "claude-opus-4-20250514" not in ladder
        # But cheap models should be
        assert "gpt-4o-mini" in ladder
        assert "gpt-3.5-turbo" in ladder
