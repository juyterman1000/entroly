from __future__ import annotations

from entroly.proxy import PromptCompilerProxy
from entroly.proxy_config import ProxyConfig
from entroly.usage_ledger import TokenUsage


def _body(tool_content: str, *, append: bool = False) -> dict:
    messages = [
        {"role": "system", "content": "stable policy"},
        {"role": "user", "content": "run tests"},
        {"role": "tool", "tool_call_id": "call-1", "content": tool_content},
    ]
    if append:
        messages.extend(
            [
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "continue"},
            ]
        )
    return {"model": "test-model", "messages": messages}


def test_proxy_guard_uses_provider_observation_to_preserve_warm_prefix(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ENTROLY_SESSION_RESCUE", "0")
    proxy = PromptCompilerProxy(
        object(),
        ProxyConfig(enable_conversation_compression=False),
    )
    conversation_id = "conversation"
    original_tool = ("original factual output\n" * 300)
    rewritten_tool = ("rewritten factual output\n" * 300)
    first = _body(original_tool)
    proxy._prefix_continuity.observe(
        conversation_id,
        provider="openai",
        raw_body=first,
        outbound_body=first,
    )
    proxy._cache_router.observe(
        conversation_id,
        model="test-model",
        provider="openai",
        prefix_hash="observed-provider-prefix",
        cached_prefix_tokens=1_000,
        cache_hit=True,
    )
    baseline = _body(original_tool, append=True)
    candidate = _body(rewritten_tool, append=True)

    selected, headers = proxy._guard_optional_prefix_mutation(
        conversation_id=conversation_id,
        provider="openai",
        baseline_body=baseline,
        candidate_body=candidate,
    )

    assert selected == baseline
    assert headers["X-Entroly-Prefix-Guard"] == "preserve_warm_prefix"
    assert int(headers["X-Entroly-Prefix-Tokens-At-Risk"]) > 100


def test_proxy_continuity_headers_expose_only_aggregate_status(monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_SESSION_RESCUE", "0")
    proxy = PromptCompilerProxy(
        object(),
        ProxyConfig(enable_conversation_compression=False),
    )
    secret = "PRIVATE_PROMPT_MUST_NOT_REACH_HEADERS"
    body = _body(secret * 100)

    headers = proxy._observe_prefix_continuity(
        conversation_id="conversation",
        provider="openai",
        raw_body=body,
        outbound_body=body,
    )

    assert headers["X-Entroly-Prefix-Continuity"] == "first_observation"
    assert headers["X-Entroly-Prefix-Interference-Tokens"] == "0"
    assert secret not in repr(headers)


def test_proxy_tracks_content_blind_live_provider_usage_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_USAGE_LEDGER", raising=False)
    monkeypatch.setenv("ENTROLY_SESSION_RESCUE", "0")
    proxy = PromptCompilerProxy(object(), ProxyConfig())

    proxy._record_provider_usage(
        body=_body("tool output"),
        provider="openai",
        request_id="request-local-only",
        usage=TokenUsage(
            uncached_input_tokens=200,
            cache_read_tokens=800,
            cache_write_tokens=25,
            output_tokens=50,
        ),
        streaming=False,
    )

    assert proxy._usage_ledger is None
    assert proxy._live_usage_requests == 1
    assert proxy._live_uncached_input_tokens == 200
    assert proxy._live_cache_read_tokens == 800
    assert proxy._live_cache_write_tokens == 25
    assert proxy._live_output_tokens == 50
