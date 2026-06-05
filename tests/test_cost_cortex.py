"""Cost Cortex: price-aware budget clamps (single pricing source) + the
recoverability ledger. Guards the control-plane invariants:

  * clamping can only REDUCE Entroly's injected context, never grow spend;
  * a cheap long-context model can't silently permit a runaway injection;
  * prices come from value_tracker (no second table, no invented numbers);
  * a cut is only "lossless" if it carries a recovery handle.
"""
from __future__ import annotations

from entroly.cost_cortex import (
    CostBudget,
    ContextDecision,
    ContextLedger,
    Decision,
    ProviderPrice,
    clamp_injected_budget,
)


def test_price_comes_from_value_tracker_known_model():
    p = ProviderPrice.for_model("gpt-4o")
    assert p.input_per_1k == 0.0025          # value_tracker's gpt-4o input rate
    assert p.output_per_1k > p.input_per_1k   # output priced higher
    assert p.input_usd(40_000) == 0.10        # linear, single source


def test_unknown_model_falls_back_not_invented():
    p = ProviderPrice.for_model("some-unreleased-model-x")
    assert p.input_per_1k > 0                  # documented default, never 0/None


def test_hard_cap_clamps_runaway_long_context_injection():
    # The dangerous default: a cheap long-context model permitting ~629K tokens.
    clamped, reason = CostBudget("gemini-1.5-pro", hard_token_cap=256_000).clamp(629_000)
    assert clamped == 256_000
    assert "hard cap" in reason


def test_within_cap_is_untouched():
    clamped, reason = CostBudget("gpt-4o", hard_token_cap=256_000).clamp(50_000)
    assert clamped == 50_000
    assert reason == "within budget"


def test_dollar_ceiling_clamps_by_real_price():
    # gpt-4o input = $0.0025/1k -> $0.10 ceiling allows 40,000 tokens.
    clamped, reason = CostBudget("gpt-4o", dollar_ceiling=0.10, hard_token_cap=0).clamp(100_000)
    assert clamped == 40_000
    assert "ceiling" in reason


def test_dollar_ceiling_disabled_is_noop():
    clamped, _ = CostBudget("gpt-4o", dollar_ceiling=None, hard_token_cap=0).clamp(100_000)
    assert clamped == 100_000


def test_clamp_can_only_reduce_never_inflate_spend():
    """Compliance invariant: the cortex can shrink Entroly's injected context
    but must never raise it above what was requested."""
    for requested in (0, 10, 1_000, 50_000, 5_000_000):
        clamped, _ = CostBudget("gpt-4o", dollar_ceiling=0.5, hard_token_cap=256_000).clamp(requested)
        assert 0 <= clamped <= max(0, requested)


def test_helper_matches_costbudget():
    a, _ = clamp_injected_budget("gpt-4o", 100_000, dollar_ceiling=0.10)
    b, _ = CostBudget("gpt-4o", dollar_ceiling=0.10).clamp(100_000)
    assert a == b


def test_ledger_recoverability_and_savings():
    led = ContextLedger()
    led.record(ContextDecision("a.py", Decision.EXACT, 100, 100))
    led.record(ContextDecision("b.py", Decision.SKELETON, 400, 80, handle="ccr:b"))
    led.record(ContextDecision("c.log", Decision.DIGEST, 900, 30))            # no handle -> lossy
    led.record(ContextDecision("secret", Decision.BLOCKED, 50, 0, reason="api key"))

    s = led.summary()
    assert s["tokens_before"] == 1450
    assert s["tokens_after"] == 210
    assert s["tokens_saved"] == 1240
    # EXACT + handled-SKELETON are recoverable; BLOCKED is excluded from lossy;
    # only the un-handled DIGEST counts as lossy.
    assert s["lossy_units"] == 1
    assert s["fully_recoverable"] is False

    # add a handle to the digest -> now fully recoverable
    led.decisions[2].handle = "ccr:c"
    assert led.summary()["fully_recoverable"] is True


def test_blocked_unit_is_not_recoverable_and_not_lossy():
    d = ContextDecision("secret", Decision.BLOCKED, 80, 0)
    assert d.recoverable is False
    led = ContextLedger([d])
    assert led.summary()["lossy_units"] == 0  # blocked is a compliance choice, not quality loss
