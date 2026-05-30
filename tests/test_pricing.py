"""
Tests for entroly/value_tracker.py pricing — output-aware + locally overridable.

Locks the production contract of estimate_cost / _pricing / pricing_provenance:
  - backward compatibility (positional input pricing unchanged)
  - output-aware rates (output > input)
  - longest-prefix + alias matching
  - unknown-model fallback to default
  - local override file (ENTROLY_PRICING_FILE) with fail-open on bad JSON
  - provenance reporting
  - record_belief_conditioning lifetime counter
"""
from __future__ import annotations

import json

import pytest

from entroly import value_tracker as vt


@pytest.fixture(autouse=True)
def _reset_pricing():
    # Pricing is process-cached; isolate every test from override leakage.
    vt.reset_pricing_cache()
    yield
    vt.reset_pricing_cache()


def test_input_rate_backward_compatible():
    # Existing positional callers must keep the exact input rate.
    assert vt.estimate_cost(1000, "gpt-4o") == pytest.approx(0.0025)
    assert vt.estimate_cost(1000, "claude-sonnet-4") == pytest.approx(0.003)


def test_output_aware_costs_more_than_input():
    out = vt.estimate_cost(1000, "gpt-4o", "output")
    inp = vt.estimate_cost(1000, "gpt-4o", "input")
    assert out == pytest.approx(0.01)
    assert out > inp, "output tokens must price higher than input"


def test_longest_prefix_no_eating():
    # 'gpt-4o' must not eat 'gpt-4o-mini'.
    assert vt.estimate_cost(1000, "gpt-4o-mini") == pytest.approx(0.00015)
    assert vt.estimate_cost(1000, "gpt-4o-2024-08-06") == pytest.approx(0.0025)


def test_alias_maps_to_canonical_rate():
    assert vt.estimate_cost(1000, "claude-3-opus") == pytest.approx(0.015)


def test_unknown_model_falls_back_to_default():
    assert vt.estimate_cost(1000, "totally-unknown-model-xyz") == pytest.approx(0.003)


def test_zero_tokens_is_zero_cost():
    assert vt.estimate_cost(0, "gpt-4o") == 0.0
    assert vt.estimate_cost(0, "gpt-4o", "output") == 0.0


def test_local_override_applied(tmp_path, monkeypatch):
    pf = tmp_path / "pricing.json"
    pf.write_text(json.dumps({
        "as_of": "2099-01",
        "models": {"gpt-4o": {"input": 0.001, "output": 0.002}},
        "default": {"input": 0.009, "output": 0.05},
    }), encoding="utf-8")
    monkeypatch.setenv("ENTROLY_PRICING_FILE", str(pf))
    vt.reset_pricing_cache()

    assert vt.estimate_cost(1000, "gpt-4o") == pytest.approx(0.001)
    assert vt.estimate_cost(1000, "gpt-4o", "output") == pytest.approx(0.002)
    assert vt.estimate_cost(1000, "unknown") == pytest.approx(0.009)

    prov = vt.pricing_provenance()
    assert prov["as_of"] == "2099-01"
    assert prov["source"] == str(pf)


def test_override_failopen_on_bad_json(tmp_path, monkeypatch):
    pf = tmp_path / "pricing.json"
    pf.write_text("{ this is not valid json", encoding="utf-8")
    monkeypatch.setenv("ENTROLY_PRICING_FILE", str(pf))
    vt.reset_pricing_cache()
    # Must NOT crash; falls back to bundled rates.
    assert vt.estimate_cost(1000, "gpt-4o") == pytest.approx(0.0025)
    assert vt.pricing_provenance()["source"] == "bundled"


def test_bundled_provenance_when_no_override():
    prov = vt.pricing_provenance()
    assert prov["source"] == "bundled"
    assert prov["as_of"] == vt._PRICING_AS_OF


def test_record_belief_conditioning_counter(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    tracker = vt.ValueTracker()
    base = tracker.get_lifetime().get("beliefs_conditioned_fragments", 0)
    tracker.record_belief_conditioning(3, source="test")
    tracker.record_belief_conditioning(2, source="test")
    lt = tracker.get_lifetime()
    assert lt["beliefs_conditioned_fragments"] == base + 5
    assert lt["belief_conditioning_passes"] == 2
    # n<=0 is a no-op (fail-safe).
    tracker.record_belief_conditioning(0, source="test")
    assert tracker.get_lifetime()["belief_conditioning_passes"] == 2
