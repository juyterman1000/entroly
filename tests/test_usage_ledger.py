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

        with pytest.raises(ValueError, match="different provider usage"):
            ledger.record_provider_payload(
                request_id="same-request",
                provider="openai",
                model="model",
                payload={
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                    }
                },
                pricing=pricing,
                team="different-team",
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


def test_explicit_zero_cache_write_rate_is_not_replaced() -> None:
    pricing = UsagePricing.from_values(
        input_per_million=10,
        output_per_million=20,
        cache_read_per_million=1,
        cache_write_per_million=0,
    )

    assert pricing.cache_write_rate == 0


def _seed_ledger(ledger, n=5):
    """Insert n events with predictable timestamps and dimensions."""
    from entroly.usage_ledger import TokenUsage

    pricing = UsagePricing.from_values(
        input_per_million=10,
        cache_read_per_million=1,
        output_per_million=20,
    )
    base_time = 1_700_000_000.0
    for i in range(n):
        ledger.record_usage(
            request_id=f"req-{i}",
            provider="anthropic" if i % 2 == 0 else "openai",
            model="opus" if i % 2 == 0 else "gpt-4o",
            usage=TokenUsage(
                uncached_input_tokens=100 * (i + 1),
                cache_read_tokens=50 * (i + 1),
                output_tokens=20 * (i + 1),
            ),
            pricing=pricing,
            occurred_at=base_time + i * 3600,
            team="platform",
            project="test",
        )
    return base_time


def test_query_returns_events_in_reverse_chronological_order() -> None:
    with UsageLedger() as ledger:
        base = _seed_ledger(ledger)
        events = ledger.query(limit=10)

    assert len(events) == 5
    assert events[0].request_id == "req-4"
    assert events[-1].request_id == "req-0"
    for i in range(len(events) - 1):
        assert events[i].occurred_at >= events[i + 1].occurred_at


def test_query_filters_by_time_range() -> None:
    with UsageLedger() as ledger:
        base = _seed_ledger(ledger)
        events = ledger.query(
            since=base + 3600, until=base + 3 * 3600, limit=10
        )

    assert len(events) == 3
    request_ids = {e.request_id for e in events}
    assert request_ids == {"req-1", "req-2", "req-3"}


def test_query_filters_by_provider() -> None:
    with UsageLedger() as ledger:
        _seed_ledger(ledger)
        events = ledger.query(provider="anthropic", limit=10)

    assert all(e.provider == "anthropic" for e in events)
    assert len(events) == 3


def test_query_respects_limit_and_offset() -> None:
    with UsageLedger() as ledger:
        _seed_ledger(ledger)
        page1 = ledger.query(limit=2, offset=0)
        page2 = ledger.query(limit=2, offset=2)

    assert len(page1) == 2
    assert len(page2) == 2
    assert page1[0].request_id != page2[0].request_id


def test_count_matches_query_length() -> None:
    with UsageLedger() as ledger:
        _seed_ledger(ledger)
        total = ledger.count()
        filtered = ledger.count(provider="openai")

    assert total == 5
    assert filtered == 2


def test_export_csv_contains_header_and_all_events() -> None:
    import csv as _csv
    import io

    with UsageLedger() as ledger:
        _seed_ledger(ledger, n=3)
        output = ledger.export_csv()

    reader = _csv.reader(io.StringIO(output))
    rows = list(reader)
    assert rows[0][0] == "request_id"
    assert len(rows) == 4
    assert rows[1][2] in ("anthropic", "openai")


def test_export_csv_respects_filters() -> None:
    import csv as _csv
    import io

    with UsageLedger() as ledger:
        _seed_ledger(ledger, n=5)
        output = ledger.export_csv(provider="anthropic")

    reader = _csv.reader(io.StringIO(output))
    rows = list(reader)
    assert len(rows) == 4
    for row in rows[1:]:
        assert row[2] == "anthropic"
