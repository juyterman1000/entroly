"""An unrecognised model is not a cheap model, and saying so disabled the router.

`_lookup_tier` returned None for every current Claude id -- `claude-opus-4-8`
does not start with `claude-3-opus`, which is what the prefix rule derived from
the one opus entry in the table -- and `route` read None as "already cheap". The
flagship id most callers now send therefore returned immediately, the router
never engaged, and `stats()` reported `est_savings_usd: 0.0` forever with
nothing distinguishing "nothing to save" from "never looked".

Two separate defects, fixed together because either alone leaves the feature
silently dead:

- an exact table is a treadmill, so families are matched as a fallback;
- "unrecognised" is reported as itself rather than as "cheap".
"""

from __future__ import annotations

import tempfile

import pytest

from entroly.ravs.router import BayesianRouter, _lookup_tier


@pytest.fixture
def router():
    return BayesianRouter(log_path=tempfile.mkdtemp() + "/decisions.jsonl")


class TestCurrentModelsResolve:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-opus-4-1-20250805",
            "claude-sonnet-4-5-20250929",
        ],
    )
    def test_current_flagships_are_flagships(self, model):
        tier = _lookup_tier(model)
        assert tier is not None, f"{model} resolved to nothing; the router is inert for it"
        assert tier["tier"] == "flagship"
        assert tier.get("cheap_alt"), "a flagship with no cheaper sibling can never route"

    @pytest.mark.parametrize(
        "model",
        ["claude-haiku-4-5-20251001", "claude-3-5-haiku-20241022", "gpt-4o-mini", "o3-mini"],
    )
    def test_cheap_models_are_not_promoted_to_flagship(self, model):
        # Ordering matters: the broad flagship patterns would swallow these if
        # the cheap patterns were not matched first, and the router would then
        # try to "save money" by swapping a cheap model for another cheap one.
        assert _lookup_tier(model)["tier"] == "cheap"

    def test_exact_entries_are_preferred_over_family_inference(self):
        exact = _lookup_tier("claude-3-5-sonnet-20241022")
        assert exact.get("inferred") is not True
        assert exact["cost_per_m"] == 3.0

    def test_family_matches_are_marked_inferred(self):
        assert _lookup_tier("claude-opus-4-8")["inferred"] is True


class TestUnknownIsNotCheap:
    def test_unrecognised_model_says_so(self, router):
        decision = router.route("llama-3-70b", "write a helper")
        assert "unrecognised" in decision.reason
        assert "cheap" not in decision.reason, (
            "calling an unknown model cheap asserts something unknown, and hides "
            "a permanently disengaged router behind a plausible-looking reason"
        )

    def test_unrecognised_models_are_counted_in_stats(self, router):
        router.route("llama-3-70b", "write a helper")
        router.route("llama-3-70b", "write another helper")
        router.route("some-vendor-model", "write a helper")

        counts = router.stats()["unrecognised_models"]
        assert counts["llama-3-70b"] == 2
        assert counts["some-vendor-model"] == 1

    def test_a_genuinely_cheap_model_still_reports_already_cheap(self, router):
        decision = router.route("claude-haiku-4-5-20251001", "write a helper")
        assert "already_cheap" in decision.reason
        assert "unrecognised" not in decision.reason

    def test_current_flagship_reaches_the_routing_logic(self, router):
        """The regression that matters: opus must get past the tier gate.

        With no observations the router must still decline -- that is
        fail-closed and correct -- but it has to decline for a reason that
        proves it looked, not because it mistook a flagship for a cheap model.
        """
        decision = router.route("claude-opus-4-8", "write a helper function")

        assert decision.use_original is True, "no data yet, so it must not swap"
        assert "already_cheap" not in decision.reason
        assert "unrecognised" not in decision.reason


class TestSavingsHonesty:
    def test_unpriced_swaps_are_reported_separately(self, router):
        stats = router.stats()
        # Both keys must exist even at zero, so a reader can tell "saved
        # nothing" from "saved something that could not be priced".
        assert "unpriced_swaps" in stats
        assert "unrecognised_models" in stats
        assert stats["est_savings_usd"] == 0.0
        assert stats["unpriced_swaps"] == 0
