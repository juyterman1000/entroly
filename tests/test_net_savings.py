"""Red-team tests for the conservative net saving on Traffic Receipts.

Each test is one way a savings number can lie, pinned shut:

* the provider's own cache discount credited as Entroly's saving;
* content Entroly adds (injected context, rewrites) left uncharged;
* a cache prefix Entroly breaks left uncharged;
* a token conversion or rate choice that happens to favour Entroly;
* a missing input reported as a zero saving instead of no claim.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from entroly import net_savings as ns

ROOT = Path(__file__).resolve().parent.parent
BUNDLED = ROOT / "entroly" / "data" / "anthropic_pricing.json"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("ENTROLY_PRICING_CATALOG", raising=False)
    monkeypatch.delenv("ENTROLY_BUNDLED_PRICING", raising=False)
    ns._reset_catalog_cache_for_tests()
    ns.SEEN.clear()
    yield
    ns._reset_catalog_cache_for_tests()
    ns.SEEN.clear()


def _rates(model: str = "claude-opus-4-6") -> ns.ModelRates:
    resolved = ns.resolve_rates("anthropic", model)
    assert resolved is not None
    return resolved[0]


def _body(*tool_outputs: str, injected: str | None = None) -> dict:
    system = [{"type": "text", "text": "You are a coding agent."}]
    if injected:
        system.append({"type": "text", "text": injected})
    messages: list[dict] = [{"role": "user", "content": "fix the failing test"}]
    for index, output in enumerate(tool_outputs):
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"t{index}", "name": "bash", "input": {"cmd": "pytest"}}
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{index}", "content": output}],
            }
        )
    return {"model": "claude-opus-4-6", "system": system, "messages": messages}


def _estimate(raw_body: dict, out_body: dict, *, provider_tokens: int, interference: int = 0,
              model: str = "claude-opus-4-6", cache_read: int = 0) -> ns.NetSavingEstimate:
    raw = ns.prompt_inventory(raw_body)
    out = ns.prompt_inventory(out_body)
    removed, added = ns.inventory_delta(raw, out)
    seen, new = ns.SEEN.classify_and_remember(removed, raw, out)
    rates, source = ns.resolve_rates("anthropic", model)
    return ns.estimate_net_saving(
        raw=raw, outbound=out, removed_seen_bytes=seen, removed_new_bytes=new,
        added_bytes=added, interference_bytes=interference,
        uncached_input_tokens=provider_tokens - cache_read, cache_read_tokens=cache_read,
        cache_write_tokens=0, rates=rates, pricing_source=source,
    )


# ── zero when nothing changed; never the provider's cache discount ──────


def test_unchanged_request_is_exactly_zero() -> None:
    body = _body("PASSED\n" * 500)
    est = _estimate(body, json.loads(json.dumps(body)), provider_tokens=40_000)
    assert est.micro_usd == 0
    assert est.details()["removed_new_tokens"] == 0


def test_provider_cache_hit_alone_is_not_entroly_saving() -> None:
    """A fully cached, untouched request saves the client money -- not Entroly."""
    body = _body("x" * 20_000)
    ns.SEEN.classify_and_remember((), ns.prompt_inventory(body), ns.prompt_inventory(body))
    assert _estimate(body, body, provider_tokens=90_000).micro_usd == 0


# ── what Entroly removed ───────────────────────────────────────────────


def test_new_content_removed_is_priced_at_the_uncached_rate() -> None:
    log = "tests/test_a.py::test_x PASSED\n" * 400
    raw = _body(log)
    out = _body("[compressed] 400 passed")
    est = _estimate(raw, out, provider_tokens=2_000)

    assert est.removed_new_tokens > 0 and est.removed_seen_tokens == 0
    expected_floor = (
        Decimal(est.removed_new_tokens) * _rates().base.input
        - Decimal(est.added_tokens) * _rates().base.cache_write_max
    )
    assert est.micro_usd == int(expected_floor.to_integral_value(rounding="ROUND_FLOOR"))
    assert est.micro_usd > 0


def test_previously_sent_content_removed_is_priced_as_a_cache_read() -> None:
    log = "tests/test_a.py::test_x PASSED\n" * 400
    first = _body(log)
    # Turn 1 forwarded the log untouched, so the provider may now cache it.
    ns.SEEN.classify_and_remember((), ns.prompt_inventory(first), ns.prompt_inventory(first))
    later = _estimate(first, _body("[compressed] 400 passed"), provider_tokens=2_000)

    assert later.removed_seen_tokens > 0 and later.removed_new_tokens == 0
    fresh = _estimate(_body(log + "fresh"), _body("[compressed] 400 passed"), provider_tokens=2_000)
    assert later.micro_usd < fresh.micro_usd  # read rate is 10% of input


def test_seen_set_is_shared_across_conversations() -> None:
    """Subagent forks resend parent history; the provider cache is shared."""
    parent = _body("y" * 9_000)
    ns.SEEN.classify_and_remember((), ns.prompt_inventory(parent), ns.prompt_inventory(parent))
    fork_est = _estimate(parent, _body("short"), provider_tokens=500)
    assert fork_est.removed_new_tokens == 0


# ── what Entroly added or broke ─────────────────────────────────────────


def test_injected_context_that_grows_the_request_is_a_negative_saving() -> None:
    raw = _body("ok")
    out = _body("ok", injected="def helper():\n    return 1\n" * 300)
    est = _estimate(raw, out, provider_tokens=3_000)
    assert est.added_tokens > 0
    assert est.micro_usd < 0


def test_cache_prefix_broken_by_entroly_is_charged() -> None:
    raw = _body("a" * 4_000)
    out = _body("b" * 1_000)
    clean = _estimate(raw, out, provider_tokens=5_000)
    ns.SEEN.clear()
    broken = _estimate(raw, out, provider_tokens=5_000, interference=4_000)
    assert broken.interference_tokens > 0
    tier = _rates().base
    pool_left = 5_000 - min(broken.added_tokens, 5_000)
    charged = min(broken.interference_tokens, pool_left)
    charge = Decimal(charged) * (tier.cache_write_max - tier.cache_read)
    assert charged > 0
    assert abs((clean.micro_usd - broken.micro_usd) - int(charge)) <= 1


def test_rewritten_block_counts_as_removed_and_added() -> None:
    raw = ns.prompt_inventory(_body("one two three " * 100))
    out = ns.prompt_inventory(_body("one two three"))
    removed, added = ns.inventory_delta(raw, out)
    assert len(removed) == 1
    assert added > 0


# ── every unknown resolved against Entroly ──────────────────────────────


def test_unknown_write_ttl_is_priced_at_the_one_hour_rate() -> None:
    tier = _rates("claude-sonnet-4-6").base
    assert tier.cache_write_max == Decimal("6")  # 1h, not the 5m $3.75


def test_token_conversion_uses_the_unfavourable_ratio_on_each_side() -> None:
    raw = _body("r" * 8_000)
    out = _body("o" * 2_000)
    est = _estimate(raw, out, provider_tokens=10_000)
    out_inv = ns.prompt_inventory(out)
    assert out_inv.surface_bytes > out_inv.unit_bytes
    removed, added = ns.inventory_delta(ns.prompt_inventory(raw), out_inv)
    removed_bytes = sum(length for _, length in removed)
    assert est.removed_new_tokens == removed_bytes * 10_000 // out_inv.surface_bytes
    assert est.added_tokens == -(-added * 10_000 // out_inv.unit_bytes)


def test_haiku_5_5_long_context_tier_applies_to_the_actual_request_only() -> None:
    rates = _rates("claude-haiku-5-5")
    assert rates.tier(100_000) is rates.base
    assert rates.tier(100_001) is rates.long_context
    raw = _body("z" * 6_000)
    out = _body("z", injected="w" * 500)
    est = _estimate(raw, out, provider_tokens=150_000, model="claude-haiku-5-5")
    assert est.long_context_tier is True
    expected = (
        Decimal(est.removed_new_tokens) * rates.base.input
        - Decimal(est.added_tokens) * rates.long_context.cache_write_max
    )
    assert est.micro_usd == int(expected.to_integral_value(rounding="ROUND_FLOOR"))


def test_estimate_is_deterministic_integer_microdollars() -> None:
    raw, out = _body("q" * 3_333), _body("q")
    first = _estimate(raw, out, provider_tokens=7_777)
    ns.SEEN.clear()
    second = _estimate(raw, out, provider_tokens=7_777)
    assert first == second
    assert isinstance(first.micro_usd, int)


# ── catalog provenance ──────────────────────────────────────────────────


def test_bundled_catalog_is_dated_sourced_and_matches_published_rates() -> None:
    doc = json.loads(BUNDLED.read_text(encoding="utf-8"))
    assert doc["verified_at"] == "2026-10-31"
    assert doc["source_url"].startswith("https://platform.claude.com/")
    expect = {  # id: (input, 1h write, read) per MTok
        "claude-opus-5-5": ("4", "8", "0.20"),
        "claude-sonnet-5-5": ("2", "4", "0.10"),
        "claude-haiku-5-5": ("0.10", "0.20", "0.01"),
        "claude-fable-5-1": ("10", "20", "0.25"),
        "claude-opus-4-6": ("5", "10", "0.50"),
        "claude-haiku-4-5-20251001": ("1", "2", "0.10"),
    }
    for model, (inp, w1h, read) in expect.items():
        rates = _rates(model).base
        assert (rates.input, rates.cache_write_max, rates.cache_read) == (
            Decimal(inp), Decimal(w1h), Decimal(read)
        ), model


def test_bundled_catalog_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_BUNDLED_PRICING", "0")
    ns._reset_catalog_cache_for_tests()
    assert ns.resolve_rates("anthropic", "claude-opus-4-6") is None


def test_operator_catalog_overrides_the_bundled_one(monkeypatch, tmp_path) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps({
        "source": "negotiated-2026",
        "models": {"anthropic:claude-opus-4-6": {
            "input_per_million": 4, "output_per_million": 20, "cache_read_per_million": 0.4,
        }},
    }), encoding="utf-8")
    monkeypatch.setenv("ENTROLY_PRICING_CATALOG", str(path))
    ns._reset_catalog_cache_for_tests()
    rates, source = ns.resolve_rates("anthropic", "claude-opus-4-6")
    assert source == "negotiated-2026"
    # No write rate given: priced at input, matching the usage ledger.
    assert rates.base.cache_write_max == Decimal("4")
    assert ns.resolve_rates("anthropic", "claude-sonnet-4-6") is None


def test_unknown_model_has_no_price_rather_than_a_default() -> None:
    assert ns.resolve_rates("anthropic", "claude-unreleased-9") is None
    assert ns.resolve_rates("openai", "gpt-5.6-sol") is None


# ── content-blind ───────────────────────────────────────────────────────


def test_inventory_holds_no_prompt_content() -> None:
    secret = "sk-ant-secret-should-never-appear-in-accounting"
    inv = ns.prompt_inventory(_body(secret * 3))
    assert secret not in repr(inv)
    assert all(len(digest) == 32 for digest, _ in inv.units)


def test_seen_set_is_bounded() -> None:
    seen = ns.SeenUnits(capacity=3)
    inv = ns.prompt_inventory(_body("a", "b", "c", "d"))
    seen.classify_and_remember((), inv, inv)
    assert len(seen._digests) == 3


def test_content_entroly_already_forwarded_is_charged_as_the_cache_read_it_was() -> None:
    """Turn 2 re-sends turn 1's compressed form; the provider reads it from cache.

    Charging it at the write rate turned every follow-up turn into a false loss.
    Only the forwarded tokens the provider did not serve from cache can have
    cost more than a read.
    """
    log = "tests/test_a.py::test_x PASSED\n" * 400
    compressed = _body("[compressed] 400 passed")
    turn1 = _estimate(_body(log), compressed, provider_tokens=2_000)
    assert turn1.micro_usd > 0

    # Turn 2: same history, 30 new tokens uncached, everything else read.
    turn2 = _estimate(_body(log), compressed, provider_tokens=2_000, cache_read=1_970)
    assert turn2.removed_seen_tokens > 0
    tier = _rates().base
    expected = (
        Decimal(turn2.removed_seen_tokens) * tier.cache_read
        - Decimal(min(turn2.added_tokens, 30)) * tier.cache_write_max
        - Decimal(max(0, turn2.added_tokens - 30)) * tier.cache_read
    )
    assert turn2.micro_usd == int(expected.to_integral_value(rounding="ROUND_FLOOR"))
    assert turn2.micro_usd > 0


def test_charges_never_exceed_what_the_provider_billed_above_read_rate() -> None:
    raw, out = _body("ok"), _body("ok", injected="i" * 20_000)
    all_read = _estimate(raw, out, provider_tokens=6_000, cache_read=6_000)
    ns.SEEN.clear()
    none_read = _estimate(raw, out, provider_tokens=6_000)
    assert none_read.micro_usd < all_read.micro_usd <= 0
