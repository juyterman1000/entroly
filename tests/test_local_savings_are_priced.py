"""Measured local token reductions must carry a value, and must not be confused
with invoice-verified savings.

The receipt printed a hardcoded `$0.0000` for local operations while displaying
the measured token count beside it. Input avoided is input not bought, and
tokens have a public rate, so reporting nothing understated real work: an agent
saving ~28k tokens per request through the SDK or MCP banked nothing on any
surface a user looks at.

The opposite error is worse. Provider-bound savings are checked against
observed usage; local ones cannot be. The two therefore stay in separate
fields, and the local figure carries its basis in its name.
"""

from __future__ import annotations

import pytest

from entroly.value_tracker import ValueTracker, estimate_cost


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    return ValueTracker(data_dir=tmp_path)


class TestLocalReductionsArePriced:
    def test_measured_local_tokens_get_a_dollar_figure(self, tracker):
        tracker.record(tokens_saved=28_130, source="sdk")

        local = tracker.get_value_receipt()["local_operations"]
        assert local["tokens_reduced"] == 28_130
        assert local["modeled_value_at_list_usd"] > 0.0, (
            "a measured reduction reported no value; the token count beside it "
            "proves work was done"
        )

    def test_the_figure_matches_the_catalog_rate(self, tracker):
        tracker.record(tokens_saved=100_000, source="mcp")

        local = tracker.get_value_receipt()["local_operations"]
        expected = round(estimate_cost(100_000, model="", kind="input"), 6)
        assert local["modeled_value_at_list_usd"] == expected, (
            "the local figure must be the catalog rate applied to the measured "
            "tokens, not an independent estimate that could drift"
        )

    def test_the_basis_is_named_in_the_payload(self, tracker):
        tracker.record(tokens_saved=1_000, source="npm")
        local = tracker.get_value_receipt()["local_operations"]

        assert local["pricing_basis"] == "default_catalog_input_rate"
        # The evidence string is what a reader sees when they ask where the
        # number came from, so it must state both halves: that it is priced,
        # and that it is not an invoice.
        assert "replacement-cost" in local["evidence"]
        assert "not an invoice" in local["evidence"]
        assert "measured" in local["evidence"]


class TestVerifiedAndModelledStaySeparate:
    def test_local_value_never_enters_the_provider_total(self, tracker):
        tracker.record(tokens_saved=500_000, source="sdk")

        receipt = tracker.get_value_receipt()
        assert receipt["provider_path"]["modeled_input_cost_avoided_usd"] == 0.0, (
            "local replacement-cost value leaked into the provider-bound total, "
            "which is the number checked against observed usage"
        )
        assert receipt["local_operations"]["modeled_value_at_list_usd"] > 0.0

    def test_dollar_claimed_stays_zero_for_local(self, tracker):
        # This field has always meant "verified against observed provider
        # usage". Repurposing it would silently change the meaning of a number
        # already published in receipts.
        tracker.record(tokens_saved=250_000, source="local")
        assert tracker.get_value_receipt()["local_operations"]["dollar_claimed_usd"] == 0.0

    def test_provider_traffic_still_prices_into_the_provider_bucket(self, tracker):
        tracker.record(tokens_saved=10_000, model="gpt-4o", source="proxy")

        receipt = tracker.get_value_receipt()
        assert receipt["provider_path"]["input_tokens_reduced"] == 10_000
        assert receipt["provider_path"]["modeled_input_cost_avoided_usd"] > 0.0
        assert receipt["local_operations"]["tokens_reduced"] == 0
