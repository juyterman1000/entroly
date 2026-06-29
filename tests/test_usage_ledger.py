from __future__ import annotations

from entroly.usage_ledger import (
    UsageLedger,
    UsagePricing,
    parse_provider_usage,
    price_usage,
)


def test_openai_usage_separates_cached_from_uncached_tokens() -> None:
    usage = parse_provider_usage(
        "openai",
        {
            "usage": {
                "prompt_tokens": 1_000,
                "completion_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 600},
            }
        },
    )
    pricing = UsagePricing.from_values(
        input_per_million=10,
        cache_read_per_million=1,
        output_per_million=20,
        source="test-catalog",
    )

    assert usage.uncached_input_tokens == 400
    assert usage.cache_read_tokens == 600
    assert price_usage(usage, pricing) == (6_600, 5_400)


def test_anthropic_usage_keeps_cache_write_category() -> None:
    usage = parse_provider_usage(
        "anthropic",
        {
            "usage": {
                "input_tokens": 50,
                "cache_read_input_tokens": 900,
                "cache_creation_input_tokens": 100,
                "output_tokens": 25,
            }
        },
    )

    assert usage.total_input_tokens == 1_050
    assert usage.cache_write_tokens == 100


def test_ledger_is_durable_and_idempotent(tmp_path) -> None:
    path = tmp_path / "usage.sqlite3"
    pricing = UsagePricing.from_values(
        input_per_million=10,
        cache_read_per_million=1,
        output_per_million=20,
    )
    payload = {
        "usage": {
            "input_tokens": 500,
            "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 300},
        }
    }

    with UsageLedger(path) as ledger:
        event = ledger.record_provider_payload(
            request_id="request-1",
            provider="openai",
            model="model",
            payload=payload,
            pricing=pricing,
            team="platform",
            project="entroly",
        )
        assert event.usage.cache_read_tokens == 300
        assert not ledger.record(event)
        assert ledger.summary(team="platform")["requests"] == 1

    with UsageLedger(path) as reopened:
        summary = reopened.summary(project="entroly")
        assert summary["requests"] == 1
        assert summary["cache_read_tokens"] == 300
        assert summary["cache_hit_token_ratio"] == 0.6
