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
              model: str = "claude-opus-4-6", cache_read: int = 0, cache_write: int = 0,
              cache_write_1h: int | None = None,
              provider: str = "anthropic") -> ns.NetSavingEstimate:
    raw = ns.prompt_inventory(raw_body)
    out = ns.prompt_inventory(out_body)
    removed, added = ns.inventory_delta(raw, out)
    seen, new = ns.SEEN.classify_and_remember(removed, raw, out)
    rates, source = ns.resolve_rates(provider, model)
    return ns.estimate_net_saving(
        raw=raw, outbound=out, removed_seen_bytes=seen, removed_new_bytes=new,
        added_bytes=added, interference_bytes=interference,
        uncached_input_tokens=provider_tokens - cache_read - cache_write,
        cache_read_tokens=cache_read, cache_write_tokens=cache_write,
        cache_write_1h_tokens=cache_write_1h, rates=rates, pricing_source=source,
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
        - Decimal(est.added_tokens) * _rates().base.input  # pool is uncached input here
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
    charge = Decimal(charged) * (tier.input - tier.cache_read)
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
        - Decimal(est.added_tokens) * rates.long_context.input
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


# ── more than one provider ──────────────────────────────────────────────

DATA = ROOT / "entroly" / "data"
PROVIDER_TABLES = {
    "anthropic_pricing.json": ("anthropic", "https://platform.claude.com/"),
    "openai_pricing.json": ("openai", "https://developers.openai.com/"),
    "google_pricing.json": ("gemini", "https://ai.google.dev/"),
}


def _floor(value: Decimal) -> int:
    from decimal import ROUND_FLOOR

    return int(value.to_integral_value(rounding=ROUND_FLOOR))


def _provider_rates(provider: str, model: str) -> ns.ModelRates:
    resolved = ns.resolve_rates(provider, model)
    assert resolved is not None, (provider, model)
    return resolved[0]


@pytest.mark.parametrize("name", sorted(PROVIDER_TABLES))
def test_every_bundled_table_is_dated_and_links_the_providers_own_page(name) -> None:
    import datetime

    provider, domain = PROVIDER_TABLES[name]
    doc = json.loads((DATA / name).read_text(encoding="utf-8"))
    datetime.date.fromisoformat(doc["verified_at"])
    assert doc["source"].startswith("bundled:") and doc["source"].endswith(doc["verified_at"])
    assert doc["source_url"].startswith(domain)
    assert doc["models"]
    for key, entry in doc["models"].items():
        assert key.startswith(f"{provider}:"), key
        # A per-model link, where present, is the provider's page too.
        assert str(entry.get("source_url", doc["source_url"])).startswith(domain), key
        assert ns.resolve_rates(provider, key.partition(":")[2])[1] == doc["source"]


def test_bundled_tables_cover_exactly_the_shipped_files() -> None:
    assert sorted(path.name for path in ns._BUNDLED_TABLES) == sorted(PROVIDER_TABLES)


def test_openai_rates_match_the_published_pages() -> None:
    expect = {  # id: (input, cached input, dearest write) per MTok
        "gpt-6-astra": ("10", "1", "25"),
        "gpt-6.1-sol": ("2", "0.10", "5"),
        "gpt-6.1-terra": ("1", "0.05", "2.50"),
        "gpt-6-luna": ("0.10", "0.01", "0.25"),
        "gpt-4.1": ("2", "0.50", "2"),
        "gpt-4.1-mini": ("0.40", "0.10", "0.40"),
        "gpt-4o": ("2.50", "1.25", "2.50"),
        "gpt-4o-mini": ("0.15", "0.075", "0.15"),
    }
    for model, (inp, read, write) in expect.items():
        rates = _provider_rates("openai", model).base
        assert (rates.input, rates.cache_read, rates.cache_write_max) == (
            Decimal(inp), Decimal(read), Decimal(write)
        ), model


def test_gemini_rates_match_the_published_page() -> None:
    expect = {  # id: (input, cached, long input, long cached) per MTok
        "gemini-3.5-flash": ("1.50", "0.15", None, None),
        "gemini-3.1-pro-preview": ("2", "0.20", "4", "0.40"),
        "gemini-3.1-flash-lite": ("0.25", "0.025", None, None),
        "gemini-3-flash-preview": ("0.50", "0.05", None, None),
        "gemini-2.5-pro": ("1.25", "0.125", "2.50", "0.25"),
        "gemini-2.5-flash": ("0.30", "0.03", None, None),
        "gemini-2.5-flash-lite": ("0.10", "0.01", None, None),
    }
    for model, (inp, read, long_inp, long_read) in expect.items():
        rates = _provider_rates("gemini", model)
        assert (rates.base.input, rates.base.cache_read) == (Decimal(inp), Decimal(read)), model
        # No write premium is published: a write costs what input costs.
        assert rates.base.cache_write_max == rates.base.input, model
        if long_inp is None:
            assert rates.long_context is None, model
        else:
            assert rates.tier(200_000) is rates.base, model
            long = rates.tier(200_001)
            assert (long.input, long.cache_read) == (Decimal(long_inp), Decimal(long_read)), model


def test_a_dated_alias_is_an_exact_id_never_a_prefix() -> None:
    assert _provider_rates("openai", "gpt-4o-2024-08-06") is _provider_rates("openai", "gpt-4o")
    # A different snapshot was sold at a different price; it has none here.
    assert ns.resolve_rates("openai", "gpt-4o-2024-05-13") is None
    assert ns.resolve_rates("openai", "gpt-4o-audio-preview") is None
    assert ns.resolve_rates("gemini", "gemini-2.5-pro-preview-06-05") is None
    # A model is priced under its own provider only.
    assert ns.resolve_rates("openai", "gemini-2.5-pro") is None
    assert ns.resolve_rates("gemini", "gpt-4o") is None


def test_duplicate_model_in_one_table_is_rejected() -> None:
    entry = {"input_per_million": 1, "cache_read_per_million": 0.1}
    with pytest.raises(ValueError, match="duplicate"):
        ns.catalog_from_mapping(
            {"models": {"openai:a": {**entry, "aliases": ["b"]}, "openai:b": entry}},
            source="t",
        )


def test_one_unreadable_table_removes_only_that_provider(monkeypatch, tmp_path) -> None:
    broken = tmp_path / "openai_pricing.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(
        ns, "_BUNDLED_TABLES",
        (DATA / "anthropic_pricing.json", broken, DATA / "google_pricing.json"),
    )
    ns._reset_catalog_cache_for_tests()
    assert ns.resolve_rates("openai", "gpt-4o") is None
    assert ns.resolve_rates("anthropic", "claude-opus-4-6") is not None
    assert ns.resolve_rates("gemini", "gemini-2.5-pro") is not None


@pytest.mark.parametrize("bad_rate", ["NaN", "sNaN", "Infinity", "-Infinity", "not-a-number"])
def test_invalid_operator_catalog_rate_is_unpriced_not_an_exception(
    monkeypatch, tmp_path, bad_rate
) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps({
        "source": "operator-invalid",
        "models": {
            "openai:gpt-4o": {
                "input_per_million": bad_rate,
                "cache_read_per_million": "0.50",
            },
        },
    }), encoding="utf-8")
    monkeypatch.setenv("ENTROLY_PRICING_CATALOG", str(path))
    ns._reset_catalog_cache_for_tests()
    # A broken operator catalog must never produce nonfinite dollar claims or
    # propagate decimal.InvalidOperation into the proxy's receipt handling.
    assert ns.resolve_rates("openai", "gpt-4o") is None
    assert ns.resolve_rates("anthropic", "claude-opus-4-6") is None


@pytest.mark.parametrize("bad_rate", ["NaN", "Infinity", "not-a-number"])
def test_bad_bundled_rate_isolated_to_its_table(
    monkeypatch, tmp_path, bad_rate
) -> None:
    bad = tmp_path / "openai_pricing.json"
    bad.write_text(json.dumps({
        "models": {
            "openai:gpt-4o": {
                "input_per_million": "2.50",
                "cache_read_per_million": bad_rate,
            },
        },
    }), encoding="utf-8")
    monkeypatch.setattr(
        ns, "_BUNDLED_TABLES",
        (DATA / "anthropic_pricing.json", bad, DATA / "google_pricing.json"),
    )
    ns._reset_catalog_cache_for_tests()
    assert ns.resolve_rates("openai", "gpt-4o") is None
    assert ns.resolve_rates("anthropic", "claude-opus-4-6") is not None
    assert ns.resolve_rates("gemini", "gemini-2.5-pro") is not None


@pytest.mark.parametrize("payload", [[], 42, {"models": []}])
def test_malformed_operator_catalog_structure_is_unpriced(
    monkeypatch, tmp_path, payload
) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("ENTROLY_PRICING_CATALOG", str(path))
    ns._reset_catalog_cache_for_tests()
    assert ns.resolve_rates("openai", "gpt-4o") is None


@pytest.mark.parametrize("field", ["priced_up_to_input_tokens", "long_context"])
def test_nonfinite_catalog_token_limit_is_unpriced(
    monkeypatch, tmp_path, field
) -> None:
    entry = {
        "input_per_million": "2",
        "cache_read_per_million": "0.5",
    }
    if field == "long_context":
        entry[field] = {
            "threshold_input_tokens": float("inf"),
            "input_per_million": "4",
            "cache_read_per_million": "1",
        }
    else:
        entry[field] = float("inf")
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps({
        "models": {"openai:gpt-4o": entry},
    }), encoding="utf-8")
    monkeypatch.setenv("ENTROLY_PRICING_CATALOG", str(path))
    ns._reset_catalog_cache_for_tests()
    assert ns.resolve_rates("openai", "gpt-4o") is None


def test_malformed_long_context_shape_disables_that_catalog(monkeypatch, tmp_path) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps({
        "models": {"gemini:gemini-2.5-pro": {
            "input_per_million": "1.25",
            "cache_read_per_million": "0.125",
            "long_context": "invalid",
        }},
    }), encoding="utf-8")
    monkeypatch.setenv("ENTROLY_PRICING_CATALOG", str(path))
    ns._reset_catalog_cache_for_tests()
    assert ns.resolve_rates("gemini", "gemini-2.5-pro") is None


def test_a_table_repricing_another_tables_model_is_dropped_whole(monkeypatch, tmp_path) -> None:
    clash = tmp_path / "clash.json"
    clash.write_text(json.dumps({"source": "clash", "models": {
        "anthropic:claude-opus-4-6": {"input_per_million": 0, "cache_read_per_million": 0},
        "openai:gpt-new": {"input_per_million": 1, "cache_read_per_million": 1},
    }}), encoding="utf-8")
    monkeypatch.setattr(ns, "_BUNDLED_TABLES", (DATA / "anthropic_pricing.json", clash))
    ns._reset_catalog_cache_for_tests()
    rates, source = ns.resolve_rates("anthropic", "claude-opus-4-6")
    assert rates.base.input == Decimal("5") and source.startswith("bundled:anthropic")
    assert ns.resolve_rates("openai", "gpt-new") is None


@pytest.mark.parametrize(
    ("provider", "model"), [("openai", "gpt-4o"), ("openai", "gpt-6-astra"), ("gemini", "gemini-2.5-pro")]
)
def test_provider_cache_hit_is_not_entroly_saving_on_any_provider(provider, model) -> None:
    body = _body("x" * 4_000)
    estimate = _estimate(body, body, provider_tokens=50_000, cache_read=48_000,
                         provider=provider, model=model)
    assert estimate.micro_usd == 0


def test_openai_removed_content_is_priced_at_the_published_input_rate() -> None:
    raw, out = _body("y" * 40_000), _body("y" * 400)
    estimate = _estimate(raw, out, provider_tokens=1_000, provider="openai", model="gpt-4o")
    assert estimate.removed_new_tokens > 0 and estimate.added_tokens > 0
    # Removed at $2.50, the rewritten stub charged at $2.50: no write premium
    # is published for this model, so nothing dearer than input exists.
    assert estimate.micro_usd == _floor(
        Decimal(estimate.removed_new_tokens) * Decimal("2.50")
        - Decimal(estimate.added_tokens) * Decimal("2.50")
    )
    assert estimate.pricing_source.startswith("bundled:openai-pricing@")


def test_unreported_cache_writes_are_charged_as_the_dearest_write() -> None:
    """OpenAI lists 30-minute and 12-hour write prices for this model, and the
    usage Entroly reads does not say which tokens were written. What Entroly
    added is charged $25/MTok; what it removed is still credited at $10."""
    raw, out = _body("z" * 400), _body("z" * 400, injected="context " * 600)
    estimate = _estimate(raw, out, provider_tokens=2_000, provider="openai", model="gpt-6-astra")
    assert estimate.removed_new_tokens == 0 and estimate.added_tokens > 0
    assert estimate.micro_usd == _floor(-Decimal(estimate.added_tokens) * Decimal("25"))

    ns.SEEN.clear()
    raw, out = _body("q" * 40_000), _body()
    credit = _estimate(raw, out, provider_tokens=500, provider="openai", model="gpt-6-astra")
    assert credit.added_tokens == 0 and credit.removed_seen_tokens == 0
    assert credit.micro_usd == _floor(Decimal(credit.removed_new_tokens) * Decimal("10"))


def test_broken_prefix_on_an_unreported_write_model_costs_the_dearest_write() -> None:
    body = _body("w" * 4_000)
    estimate = _estimate(body, body, provider_tokens=10_000, interference=2_000,
                         provider="openai", model="gpt-6-astra")
    assert estimate.interference_tokens > 0
    # (12-hour write - cache read) per token of prefix that should have been read.
    assert estimate.micro_usd == _floor(
        -Decimal(estimate.interference_tokens) * (Decimal("25") - Decimal("1"))
    )


def test_request_above_the_verified_size_has_no_price() -> None:
    rates = _provider_rates("openai", "gpt-6-astra")
    assert rates.priced_up_to == 200_000
    assert rates.covers(200_000) and not rates.covers(200_001)
    # A model with one published tier is priced at any size.
    assert _provider_rates("openai", "gpt-4.1").covers(900_000)
    assert _provider_rates("anthropic", "claude-opus-4-6").covers(900_000)


def test_gemini_long_prompt_charges_added_content_at_the_long_tier_only() -> None:
    raw, out = _body("g" * 400), _body("g" * 400, injected="context " * 600)
    small = _estimate(raw, out, provider_tokens=100_000, provider="gemini", model="gemini-2.5-pro")
    ns.SEEN.clear()
    large = _estimate(raw, out, provider_tokens=300_000, provider="gemini", model="gemini-2.5-pro")
    assert not small.long_context_tier and large.long_context_tier
    assert small.micro_usd == _floor(-Decimal(small.added_tokens) * Decimal("1.25"))
    assert large.micro_usd == _floor(-Decimal(large.added_tokens) * Decimal("2.50"))


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
        - Decimal(min(turn2.added_tokens, 30)) * tier.input
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


# ── exact cache-write pricing ───────────────────────────────────────────


def test_charges_draw_from_the_dearest_pool_tokens_first() -> None:
    """Unknown TTL: every write is assumed 1-hour, the dearest a token can be."""
    raw, out = _body("ok " * 12_000), _body("ok " * 12_000, injected="i" * 4_000)
    est = _estimate(raw, out, provider_tokens=10_000, cache_write=9_000)
    tier = _rates().base
    assert est.added_tokens < 9_000
    expected = -(Decimal(est.added_tokens) * tier.cache_write_1h)
    assert est.micro_usd == int(expected.to_integral_value(rounding="ROUND_FLOOR"))


def test_reported_five_minute_writes_are_priced_as_such() -> None:
    raw, out = _body("ok " * 12_000), _body("ok " * 12_000, injected="i" * 4_000)
    unknown = _estimate(raw, out, provider_tokens=10_000, cache_write=9_000)
    ns.SEEN.clear()
    known = _estimate(raw, out, provider_tokens=10_000, cache_write=9_000, cache_write_1h=0)
    tier = _rates().base
    expected = -(Decimal(known.added_tokens) * tier.cache_write_5m)
    assert known.micro_usd == int(expected.to_integral_value(rounding="ROUND_FLOOR"))
    assert unknown.micro_usd < known.micro_usd  # the old assumption was pessimistic


def test_reported_one_hour_writes_are_charged_first() -> None:
    raw, out = _body("ok " * 12_000), _body("ok " * 12_000, injected="i" * 4_000)
    est = _estimate(raw, out, provider_tokens=10_000, cache_write=9_000, cache_write_1h=100)
    tier = _rates().base
    expected = -(
        Decimal(min(est.added_tokens, 100)) * tier.cache_write_1h
        + Decimal(max(0, est.added_tokens - 100)) * tier.cache_write_5m
    )
    assert est.micro_usd == int(expected.to_integral_value(rounding="ROUND_FLOOR"))


def test_usage_parser_reads_the_ttl_split() -> None:
    from entroly.usage_ledger import parse_provider_usage, parse_stream_usage

    payload = {"usage": {"input_tokens": 5, "cache_creation_input_tokens": 120,
                         "cache_creation": {"ephemeral_5m_input_tokens": 100,
                                            "ephemeral_1h_input_tokens": 20}}}
    assert parse_provider_usage("anthropic", payload).cache_write_1h_tokens == 20
    assert parse_provider_usage(
        "anthropic", {"usage": {"cache_creation_input_tokens": 120}}
    ).cache_write_1h_tokens is None
    stream = "data: " + json.dumps({"type": "message_start", "message": payload}) + "\n\n"
    assert parse_stream_usage("anthropic", stream).cache_write_1h_tokens == 20


# ── the governor ────────────────────────────────────────────────────────


def _policy(**kw) -> ns.GovernorPolicy:
    return ns.GovernorPolicy(**kw)


def test_governor_needs_enough_priced_requests_before_holding() -> None:
    gov, policy = ns.NetSavingGovernor(), _policy()
    assert gov.record("c", -15_000, policy=policy).action == "observe"
    assert gov.record("c", -15_000, policy=policy).action == "observe"  # only 2 priced
    assert gov.record("c", 0, policy=policy).action == "hold"


def test_one_expensive_rewrite_turn_never_trips_the_governor() -> None:
    """A history rewrite is charged in full on its own turn, by design."""
    gov, policy = ns.NetSavingGovernor(), _policy()
    gov.record("c", -500_000, policy=policy)
    for _ in range(20):
        standing = gov.record("c", 0, policy=policy)
    assert standing.action == "observe"
    assert standing.losing_requests == 1
    assert gov.record("c", -1, policy=policy).action == "hold"  # a second loss does


def test_governor_holds_only_below_the_floor() -> None:
    gov, policy = ns.NetSavingGovernor(), _policy()
    for _ in range(3):
        standing = gov.record("c", -6_000, policy=policy)
    assert standing.cumulative_micro_usd == -18_000
    assert standing.action == "observe"  # -$0.018 is inside the $0.02 floor
    assert gov.record("c", -3_000, policy=policy).action == "hold"


def test_governor_hold_latches_for_the_conversation() -> None:
    gov, policy = ns.NetSavingGovernor(), _policy(min_requests=1)
    assert gov.record("c", -15_000, policy=policy).action == "observe"
    assert gov.record("c", -15_000, policy=policy).action == "hold"
    assert gov.record("c", +100_000, policy=policy).action == "hold"
    assert gov.standing("c", policy=policy).action == "hold"


def test_unpriced_requests_never_count() -> None:
    gov, policy = ns.NetSavingGovernor(), _policy(min_requests=1)
    for _ in range(10):
        assert gov.record("c", None, policy=policy) is None
    assert gov.standing("c", policy=policy).priced_requests == 0
    assert gov.standing("c", policy=policy).action == "observe"


def test_conversations_are_independent_and_bounded() -> None:
    gov, policy = ns.NetSavingGovernor(capacity=2), _policy(min_requests=1)
    gov.record("a", -30_000, policy=policy)
    gov.record("a", -30_000, policy=policy)
    assert gov.standing("a", policy=policy).action == "hold"
    assert gov.standing("b", policy=policy).action == "observe"
    gov.record("b", 1, policy=policy)
    gov.record("c", 1, policy=policy)
    assert gov.standing("a", policy=policy).priced_requests == 0  # evicted, fails open


def test_governor_keeps_no_conversation_content() -> None:
    gov, policy = ns.NetSavingGovernor(), _policy()
    gov.record("secret conversation anchor", -1, policy=policy)
    assert all("secret" not in key for key in gov._entries)


def test_disabled_governor_reports_nothing() -> None:
    gov, policy = ns.NetSavingGovernor(), _policy(enabled=False)
    assert gov.record("c", -1_000_000, policy=policy) is None
    assert gov.standing("c", policy=policy) is None


def test_governor_policy_reads_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_NET_SAVING_GOVERNOR", "off")
    monkeypatch.setenv("ENTROLY_NET_SAVING_FLOOR_USD", "0.5")
    monkeypatch.setenv("ENTROLY_NET_SAVING_MIN_REQUESTS", "7")
    assert ns.GovernorPolicy.from_env() == ns.GovernorPolicy(False, 500_000, 7)
    for bad in ("nan", "-1", "lots"):
        monkeypatch.setenv("ENTROLY_NET_SAVING_FLOOR_USD", bad)
        assert ns.GovernorPolicy.from_env().floor_micro_usd == 20_000
    monkeypatch.setenv("ENTROLY_NET_SAVING_MIN_REQUESTS", "0")
    assert ns.GovernorPolicy.from_env().min_requests == 1
