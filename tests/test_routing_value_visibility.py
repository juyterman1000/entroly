"""Routing value must be recorded, and available value must stay separate.

Two defects sit behind this file. The dashboard has read `routing_saved_usd`
since it was written -- there is a KPI card labelled "Model-Routing Saved" --
and no production path ever incremented it, so the panel read $0 however much
traffic RAVS moved. And because routing is off until the user authorises model
substitution, a disabled router measured nothing at all, leaving that
authorisation to be given blind.
"""

from __future__ import annotations

import pytest

from entroly.value_tracker import ValueTracker, routing_delta_usd


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    return ValueTracker(tmp_path)


class TestPricing:
    def test_a_flagship_to_cheap_swap_is_worth_money(self):
        assert routing_delta_usd(
            "claude-opus-4-20250514", "claude-haiku-4-5", 100_000) > 0

    def test_an_unpriced_model_yields_nothing_rather_than_a_guess(self):
        """Two default rates differ by zero; reporting that would be invented."""
        assert routing_delta_usd("mystery-model", "claude-haiku-4-5", 100_000) == 0.0
        assert routing_delta_usd("claude-opus-4-20250514", "mystery", 100_000) == 0.0

    def test_routing_to_a_dearer_model_is_not_a_saving(self):
        assert routing_delta_usd(
            "claude-haiku-4-5", "claude-opus-4-20250514", 100_000) == 0.0

    def test_missing_model_names_are_not_priced(self):
        assert routing_delta_usd("", "claude-haiku-4-5", 100_000) == 0.0


class TestCapturedAndAvailableNeverMix:
    def test_a_captured_saving_lands_in_the_saved_field(self, tracker):
        tracker.record_routing_saving(0.42, chosen_model="claude-haiku-4-5")
        lifetime = tracker.get_lifetime()

        assert lifetime["routing_saved_usd"] == pytest.approx(0.42)
        assert lifetime.get("routing_available_usd", 0.0) == 0.0

    def test_an_available_saving_never_touches_the_saved_field(self, tracker):
        """This is money not earned. Summing it into a total would be a lie."""
        tracker.record_routing_opportunity(1.50, chosen_model="claude-haiku-4-5")
        lifetime = tracker.get_lifetime()

        assert lifetime["routing_available_usd"] == pytest.approx(1.50)
        assert lifetime.get("routing_saved_usd", 0.0) == 0.0, (
            "unrealised value must never be presented as captured"
        )

    def test_opportunities_are_counted(self, tracker):
        tracker.record_routing_opportunity(0.10, chosen_model="claude-haiku-4-5")
        tracker.record_routing_opportunity(0.20, chosen_model="claude-haiku-4-5")

        assert tracker.get_lifetime()["routing_opportunities"] == 2

    def test_non_positive_amounts_are_ignored(self, tracker):
        tracker.record_routing_opportunity(0.0)
        tracker.record_routing_opportunity(-5.0)

        assert tracker.get_lifetime().get("routing_available_usd", 0.0) == 0.0


class TestShadowMeasuresWithoutSwapping:
    def test_a_disabled_router_still_evaluates_in_shadow(self):
        """route() short-circuits when disabled; shadow_route() must not."""
        from entroly.ravs.router import BayesianRouter

        router = BayesianRouter(enabled=False)
        live = router.route("claude-opus-4-20250514", "summarise this")
        shadow = router.shadow_route("claude-opus-4-20250514", "summarise this")

        assert live.reason == "bayesian_router_disabled"
        assert shadow.reason != "bayesian_router_disabled", (
            "the user would be authorising routing with no evidence of its worth"
        )

    def test_shadow_still_refuses_without_evidence(self):
        """Bypassing the enable check must not bypass the safety gates."""
        from entroly.ravs.router import BayesianRouter

        shadow = BayesianRouter(enabled=False).shadow_route(
            "claude-opus-4-20250514", "summarise this")

        assert shadow.use_original is True, "no captured data must mean no recommendation"

    def test_shadow_defers_to_the_live_path_when_enabled(self):
        from entroly.ravs.router import BayesianRouter

        router = BayesianRouter(enabled=True)
        assert router.shadow_route("claude-opus-4-20250514", "x").reason != (
            "bayesian_router_disabled"
        )


class TestProxyWiring:
    def test_the_swap_site_records_its_saving(self):
        """Regression: the swap happened and nothing counted it."""
        import inspect

        from entroly import proxy

        source = inspect.getsource(proxy)
        assert "_record_routing_value" in source
        assert source.count("_record_routing_value") >= 3, (
            "expected the helper plus both the captured and shadow call sites"
        )

    def test_body_token_estimation_survives_malformed_input(self):
        from entroly.proxy import PromptCompilerProxy

        estimate = PromptCompilerProxy._estimate_body_tokens
        assert estimate({}) == 0
        assert estimate({"messages": None}) == 0
        assert estimate({"messages": [{"content": "abcd" * 10}]}) == 10
        assert estimate({"messages": [{"content": [{"text": "abcd" * 10}]}]}) == 10
