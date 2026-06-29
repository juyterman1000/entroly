from __future__ import annotations

import json

import httpx
import pytest

from entroly.proxy import PromptCompilerProxy
from entroly.proxy_config import ProxyConfig
from entroly.usage_ledger import (
    TokenUsage,
    UsageLedger,
    UsagePricing,
    UsagePricingCatalog,
)


def _catalog(*, include_recommended: bool = True) -> UsagePricingCatalog:
    models: dict[str, dict[str, float]] = {
        "openai:gpt-current": {
            "input_per_million": 10.0,
            "output_per_million": 2.0,
            "cache_read_per_million": 0.1,
            "cache_write_per_million": 10.0,
        }
    }
    if include_recommended:
        models["openai:gpt-cheap"] = {
            "input_per_million": 1.0,
            "output_per_million": 1.0,
            "cache_read_per_million": 0.05,
            "cache_write_per_million": 1.0,
        }
    return UsagePricingCatalog.from_mapping(
        {"source": "test-catalog-v1", "models": models}
    )


def _proxy() -> PromptCompilerProxy:
    proxy = PromptCompilerProxy(object(), ProxyConfig())
    proxy._witness_enabled = False
    proxy._witness_analyzer = None
    proxy._enable_passive_feedback = False
    proxy._usage_ledger = UsageLedger()
    proxy._pricing_catalog = _catalog()
    return proxy


@pytest.mark.asyncio
async def test_live_json_forward_records_usage_and_cache_observation() -> None:
    async def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "response-1",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
            },
        )

    proxy = _proxy()
    proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    body = {
        "model": "gpt-current",
        "messages": [
            {"role": "system", "content": "stable policy"},
            {"role": "user", "content": "answer this"},
        ],
    }
    try:
        response = await proxy._forward_response(
            "https://provider.example/v1/chat/completions",
            {},
            body,
            provider="openai",
            request_id="request-json",
        )
        event = proxy._usage_ledger.get("request-json")
        assert response.status_code == 200
        assert event is not None
        assert event.usage == TokenUsage(
            uncached_input_tokens=20,
            cache_read_tokens=80,
            output_tokens=7,
        )
        assert event.cost_micro_usd > 0
        assert proxy._cache_router.stats()["observed_hits"] == 1
    finally:
        await proxy._client.aclose()
        proxy._usage_ledger.close()


@pytest.mark.asyncio
async def test_live_sse_forward_records_terminal_usage_without_rewriting_bytes() -> None:
    transcript = (
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        b'data: {"usage":{"prompt_tokens":120,"completion_tokens":9,'
        b'"prompt_tokens_details":{"cached_tokens":100}}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=transcript,
        )

    proxy = _proxy()
    proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    body = {
        "model": "gpt-current",
        "stream": True,
        "messages": [
            {"role": "system", "content": "stable policy"},
            {"role": "user", "content": "stream this"},
        ],
    }
    try:
        response = await proxy._stream_response(
            "https://provider.example/v1/chat/completions",
            {},
            body,
            provider="openai",
            request_id="request-stream",
        )
        forwarded = b"".join(
            [chunk async for chunk in response.body_iterator]
        )
        event = proxy._usage_ledger.get("request-stream")
        assert forwarded == transcript
        assert event is not None
        assert event.usage.cache_read_tokens == 100
        assert event.usage.output_tokens == 9
    finally:
        await proxy._client.aclose()
        proxy._usage_ledger.close()


def test_cache_economics_keep_a_large_warm_prefix_on_current_model() -> None:
    proxy = _proxy()
    body = {
        "model": "gpt-current",
        "messages": [
            {"role": "system", "content": "policy " * 60_000},
            {"role": "user", "content": "continue"},
        ],
    }
    conversation_id = proxy._routing_conversation_id(body, "openai")
    proxy._cache_router.observe(
        conversation_id,
        model="gpt-current",
        provider="openai",
        prefix_hash=proxy._cache_prefix_hash(body, "openai"),
        cached_prefix_tokens=100_000,
        cache_hit=True,
    )

    allowed, reason = proxy._cache_economics_allow_ravs(
        body=body,
        provider="openai",
        current_model="gpt-current",
        recommended_model="gpt-cheap",
        user_message="continue",
        risk="standard",
    )

    assert not allowed
    assert reason == "stay:warm_cache_economics"
    assert proxy._cache_route_stays == 1
    proxy._usage_ledger.close()


def test_cache_economics_switch_when_cold_model_is_materially_cheaper() -> None:
    proxy = _proxy()
    body = {
        "model": "gpt-current",
        "messages": [
            {"role": "system", "content": "policy " * 60_000},
            {"role": "user", "content": "continue"},
        ],
    }

    allowed, reason = proxy._cache_economics_allow_ravs(
        body=body,
        provider="openai",
        current_model="gpt-current",
        recommended_model="gpt-cheap",
        user_message="continue",
        risk="standard",
    )

    assert allowed
    assert reason == "switch:projected_savings"
    assert proxy._cache_route_switches == 1
    proxy._usage_ledger.close()


def test_cache_economics_fail_closed_when_catalog_is_incomplete() -> None:
    proxy = _proxy()
    proxy._pricing_catalog = _catalog(include_recommended=False)
    body = {
        "model": "gpt-current",
        "messages": [{"role": "user", "content": "continue"}],
    }

    allowed, reason = proxy._cache_economics_allow_ravs(
        body=body,
        provider="openai",
        current_model="gpt-current",
        recommended_model="gpt-cheap",
        user_message="continue",
        risk="standard",
    )

    assert not allowed
    assert reason == "stay:missing_pricing"
    assert proxy._cache_route_blocked_unpriced == 1
    proxy._usage_ledger.close()


def test_usage_ledger_survives_reopen(tmp_path) -> None:
    path = tmp_path / "usage.sqlite3"
    pricing = UsagePricing.from_values(
        input_per_million=3,
        output_per_million=6,
        cache_read_per_million=0.3,
        source="durability-test",
    )
    with UsageLedger(path) as ledger:
        ledger.record_usage(
            request_id="durable-request",
            provider="openai",
            model="gpt-current",
            usage=TokenUsage(
                uncached_input_tokens=10,
                cache_read_tokens=90,
                output_tokens=5,
            ),
            pricing=pricing,
        )

    with UsageLedger(path) as reopened:
        event = reopened.get("durable-request")
        assert event is not None
        assert event.pricing_source == "durability-test"
        assert event.usage.cache_read_tokens == 90
        assert json.loads(json.dumps(reopened.summary()))["requests"] == 1
