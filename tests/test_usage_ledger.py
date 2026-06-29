from __future__ import annotations

from entroly.usage_ledger import (
    UsageLedger,
    UsagePricing,
    UsagePricingCatalog,
    parse_provider_usage,
    parse_stream_usage,
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


def test_gemini_usage_accepts_camel_case_usage_metadata() -> None:
    usage = parse_provider_usage(
        "gemini",
        {
            "usageMetadata": {
                "promptTokenCount": 1_000,
                "cachedContentTokenCount": 700,
                "candidatesTokenCount": 80,
            }
        },
    )

    assert usage.uncached_input_tokens == 300
    assert usage.cache_read_tokens == 700
    assert usage.output_tokens == 80


def test_stream_usage_reads_openai_terminal_frame() -> None:
    transcript = (
        'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\n'
        'data: {"id":"x","choices":[],"usage":{"prompt_tokens":1000,'
        '"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":800}}}\n\n'
        "data: [DONE]\n\n"
    )

    usage = parse_stream_usage("openai", transcript)

    assert usage is not None
    assert usage.uncached_input_tokens == 200
    assert usage.cache_read_tokens == 800
    assert usage.output_tokens == 50


def test_stream_usage_merges_anthropic_start_and_delta() -> None:
    transcript = (
        'data: {"type":"message_start","message":{"usage":{'
        '"input_tokens":100,"cache_read_input_tokens":900,'
        '"cache_creation_input_tokens":0,"output_tokens":1}}}\n\n'
        'data: {"type":"message_delta","usage":{"output_tokens":75}}\n\n'
    )

    usage = parse_stream_usage("anthropic", transcript)

    assert usage is not None
    assert usage.uncached_input_tokens == 100
    assert usage.cache_read_tokens == 900
    assert usage.output_tokens == 75


def test_pricing_catalog_resolves_exact_then_provider_default() -> None:
    catalog = UsagePricingCatalog.from_mapping(
        {
            "source": "contract-2026-q2",
            "models": {
                "openai:gpt-test": {
                    "input_per_million": "10",
                    "output_per_million": "20",
                    "cache_read_per_million": "1",
                },
                "anthropic:*": {
                    "input_per_million": "8",
                    "output_per_million": "16",
                    "cache_read_per_million": "0.8",
                    "cache_write_per_million": "10",
                },
            },
        }
    )

    exact = catalog.resolve("openai", "gpt-test")
    fallback = catalog.resolve("anthropic", "claude-future")

    assert exact is not None
    assert exact.input_per_million == 10
    assert exact.source == "contract-2026-q2:openai:gpt-test"
    assert fallback is not None
    assert fallback.cache_write_rate == 10
    assert catalog.resolve("gemini", "missing") is None


def test_duplicate_request_id_rejects_conflicting_usage(tmp_path) -> None:
    import pytest

    pricing = UsagePricing.from_values(
        input_per_million=10,
        cache_read_per_million=1,
        output_per_million=20,
        source="catalog-v1",
    )
    with UsageLedger(tmp_path / "usage.sqlite3") as ledger:
        first = ledger.record_provider_payload(
            request_id="same-request",
            provider="openai",
            model="model",
            payload={"usage": {"prompt_tokens": 100, "completion_tokens": 10}},
            pricing=pricing,
        )
        repeated = ledger.record_provider_payload(
            request_id="same-request",
            provider="openai",
            model="model",
            payload={"usage": {"prompt_tokens": 100, "completion_tokens": 10}},
            pricing=pricing,
        )

        assert repeated == first
        with pytest.raises(ValueError, match="different provider usage"):
            ledger.record_provider_payload(
                request_id="same-request",
                provider="openai",
                model="model",
                payload={
                    "usage": {
                        "prompt_tokens": 999,
                        "completion_tokens": 10,
                    }
                },
                pricing=pricing,
            )


def test_unpriced_usage_preserves_tokens_and_surfaces_reconciliation_gap() -> None:
    from entroly.usage_ledger import TokenUsage

    with UsageLedger() as ledger:
        event = ledger.record_usage(
            request_id="unpriced-request",
            provider="openai",
            model="future-model",
            usage=TokenUsage(
                uncached_input_tokens=100,
                cache_read_tokens=900,
                output_tokens=50,
            ),
            pricing=None,
        )
        summary = ledger.summary()

    assert event.pricing_source == "unpriced:openai:future-model"
    assert event.cost_micro_usd == 0
    assert summary["requests"] == 1
    assert summary["cache_read_tokens"] == 900
    assert summary["unpriced_requests"] == 1
