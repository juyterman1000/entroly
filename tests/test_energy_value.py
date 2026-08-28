"""Electricity avoided must be derivable by hand from the numbers it reports.

Token reduction removes prefill work, and prefill work is joules. The figure is
modeled rather than measured -- Entroly runs locally and cannot instrument a
provider's accelerators -- so its only defence is that every input is stated and
the arithmetic is checkable. These tests recompute it independently rather than
asserting a constant, which would pass even if the formula drifted.
"""

from __future__ import annotations

import pytest

from entroly.energy_value import (
    EnergyAssumptions,
    energy_for_tokens,
    scale_energy,
)


def _by_hand(tokens: int, a: EnergyAssumptions) -> float:
    flops = 2 * a.model_params_billions * 1e9 * tokens
    seconds = flops / (a.accelerator_peak_tflops * 1e12 * a.model_flops_utilization)
    return seconds * a.accelerator_tdp_watts / 3_600_000.0


class TestArithmetic:
    @pytest.mark.parametrize("tokens", [1_000, 1_000_000, 14_065_000])
    def test_matches_an_independent_computation(self, tokens):
        a = EnergyAssumptions()
        result = energy_for_tokens(tokens, a)
        assert result["kwh_avoided"] == pytest.approx(_by_hand(tokens, a), rel=1e-6)

    def test_scales_linearly_in_tokens(self):
        one = energy_for_tokens(1_000_000)["kwh_avoided"]
        ten = energy_for_tokens(10_000_000)["kwh_avoided"]
        assert ten == pytest.approx(one * 10, rel=1e-9), (
            "prefill is linear in prompt length; a non-linear result means the "
            "model no longer describes what it claims to"
        )

    def test_zero_tokens_is_zero_energy(self):
        assert energy_for_tokens(0)["kwh_avoided"] == 0.0

    def test_negative_tokens_cannot_manufacture_energy(self):
        assert energy_for_tokens(-5_000)["kwh_avoided"] == 0.0


class TestHonesty:
    def test_result_declares_it_is_not_measured(self):
        result = energy_for_tokens(1_000_000)
        assert result["measured"] is False
        assert result["basis"] == "prefill_only"

    def test_every_assumption_is_reported(self):
        result = energy_for_tokens(1_000)
        # A reader who disagrees must be able to see, and substitute, each
        # input. Reporting kWh alone would be unfalsifiable.
        for key in ("model_params_billions", "accelerator_peak_tflops",
                    "model_flops_utilization", "accelerator_tdp_watts"):
            assert key in result["assumptions"]

    def test_intermediates_are_exposed_for_checking(self):
        result = energy_for_tokens(1_000_000)
        assert result["petaflops_avoided"] > 0
        assert result["accelerator_seconds_avoided"] > 0

    def test_no_carbon_is_derived(self):
        # kWh is arithmetic. Emissions need a grid-intensity factor that varies
        # by region, hour and methodology; attaching one would put a
        # contestable number next to a checkable one.
        result = energy_for_tokens(1_000_000)
        assert not any("co2" in k.lower() or "carbon" in k.lower() for k in result)


class TestOverrides:
    def test_a_smaller_model_avoids_less_energy(self):
        big = energy_for_tokens(1_000_000, EnergyAssumptions(model_params_billions=70))
        small = energy_for_tokens(1_000_000, EnergyAssumptions(model_params_billions=7))
        assert small["kwh_avoided"] == pytest.approx(big["kwh_avoided"] / 10, rel=1e-9)

    def test_environment_overrides_are_honoured(self, monkeypatch):
        monkeypatch.setenv("ENTROLY_ENERGY_MODEL_PARAMS_B", "8")
        assert EnergyAssumptions.from_env().model_params_billions == 8.0

    def test_a_nonsense_override_falls_back_rather_than_crashing(self, monkeypatch):
        monkeypatch.setenv("ENTROLY_ENERGY_TDP_WATTS", "not-a-number")
        assert EnergyAssumptions.from_env().accelerator_tdp_watts == 700.0

    def test_zero_override_is_rejected(self, monkeypatch):
        # A zero would divide by zero or zero out the result silently.
        monkeypatch.setenv("ENTROLY_ENERGY_MFU", "0")
        assert EnergyAssumptions.from_env().model_flops_utilization == 0.40


class TestProjection:
    def test_projection_is_labelled_as_such(self):
        base = energy_for_tokens(1_000_000)
        projected = scale_energy(base, 365)
        assert projected["projected"] is True
        assert projected["projection_multiplier"] == 365
        assert projected["kwh_avoided"] == pytest.approx(base["kwh_avoided"] * 365, rel=1e-6)


class TestReceiptIntegration:
    def test_energy_appears_without_being_asked_for(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
        from entroly.value_tracker import ValueTracker

        tracker = ValueTracker(data_dir=tmp_path)
        tracker.record(tokens_saved=1_000_000, source="sdk")

        receipt = tracker.get_value_receipt()
        assert "energy" in receipt, (
            "a figure that only appears behind a flag is a figure nobody sees"
        )
        assert receipt["energy"]["kwh_avoided"] > 0.0

    def test_energy_counts_both_channels(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
        from entroly.value_tracker import ValueTracker

        tracker = ValueTracker(data_dir=tmp_path)
        tracker.record(tokens_saved=500_000, source="sdk")
        tracker.record(tokens_saved=500_000, model="gpt-4o", source="proxy")

        # Prefill is avoided wherever the tokens were removed, so the energy
        # figure spans both channels even though their dollar treatment differs.
        assert tracker.get_value_receipt()["energy"]["tokens_saved"] == 1_000_000


class TestEnergyReachesTheDashboard:
    """Deriving kWh is worthless if the number never leaves the module.

    The energy figure is computed on read from token totals rather than
    accumulated on write, so these pin the two properties that choice buys:
    it always agrees with the tokens beside it, and a failure to derive it
    costs the kWh line rather than the dollar totals.
    """

    def _tracker(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
        from entroly.value_tracker import ValueTracker

        return ValueTracker()

    def test_lifetime_carries_energy(self, tmp_path, monkeypatch):
        tracker = self._tracker(tmp_path, monkeypatch)
        tracker.record(tokens_saved=250_000, model="claude-sonnet-4",
                       source="provider")

        energy = tracker.get_lifetime().get("energy")
        assert energy, "the dashboard cannot render a kWh line that isn't sent"
        assert energy["kwh_avoided"] > 0
        assert energy["measured"] is False, "modeled figures must say so"

    def test_energy_spans_every_token_lane(self, tmp_path, monkeypatch):
        """Provider, local and unclassified tokens all avoid the same prefill."""
        tracker = self._tracker(tmp_path, monkeypatch)
        tracker.record(tokens_saved=100_000, model="claude-sonnet-4",
                       source="provider")
        tracker.record(tokens_saved=60_000, source="mcp")
        tracker.record(tokens_saved=40_000, source="unclassified")

        lifetime = tracker.get_lifetime()
        counted = (
            lifetime.get("provider_tokens_saved", 0)
            + lifetime.get("local_tokens_reduced", 0)
            + lifetime.get("unclassified_tokens_reduced", 0)
        )
        assert lifetime["energy"]["tokens_saved"] == counted, (
            "energy must be derived from the same tokens shown as dollars"
        )

    def test_a_failure_to_derive_energy_does_not_lose_the_totals(
        self, tmp_path, monkeypatch
    ):
        tracker = self._tracker(tmp_path, monkeypatch)
        tracker.record(tokens_saved=250_000, model="claude-sonnet-4",
                       source="provider")

        def explode(*_a, **_k):
            raise RuntimeError("no")

        monkeypatch.setattr("entroly.energy_value.energy_for_tokens", explode)

        lifetime = tracker.get_lifetime()
        assert lifetime.get("provider_tokens_saved") == 250_000, (
            "a missing kWh line must not take the dollar figures with it"
        )
