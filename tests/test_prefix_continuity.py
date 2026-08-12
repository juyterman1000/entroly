from __future__ import annotations

from entroly.prefix_continuity import PrefixContinuityGuard
from entroly.proxy_transform import inject_context_live_zone


def _body(messages: list[dict]) -> dict:
    return {
        "model": "test-model",
        "messages": [{"role": "system", "content": "stable policy"}, *messages],
    }


def _large_tool(label: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": (label + " factual output\n") * 300,
    }


def test_append_only_history_preserves_prefix() -> None:
    guard = PrefixContinuityGuard(block_bytes=64, material_loss_tokens=16)
    first = _body([{"role": "user", "content": "first question"}])
    guard.observe("conversation", provider="openai", raw_body=first, outbound_body=first)

    second = _body(
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ]
    )
    observation = guard.observe(
        "conversation",
        provider="openai",
        raw_body=second,
        outbound_body=second,
    )

    assert observation.status == "prefix_preserved"
    assert observation.estimated_optimizer_interference_tokens == 0


def test_live_zone_suffix_does_not_materially_clobber_append_only_prefix() -> None:
    guard = PrefixContinuityGuard(block_bytes=64, material_loss_tokens=32)
    raw_first = _body([{"role": "user", "content": "inspect authentication"}])
    outbound_first, injected = inject_context_live_zone(
        raw_first, "relevant source evidence", "openai"
    )
    assert injected
    guard.observe(
        "conversation",
        provider="openai",
        raw_body=raw_first,
        outbound_body=outbound_first,
    )

    raw_second = _body(
        [
            {"role": "user", "content": "inspect authentication"},
            {"role": "assistant", "content": "analysis"},
            {"role": "user", "content": "add a regression test"},
        ]
    )
    outbound_second, injected = inject_context_live_zone(
        raw_second, "new relevant evidence", "openai"
    )
    assert injected
    observation = guard.observe(
        "conversation",
        provider="openai",
        raw_body=raw_second,
        outbound_body=outbound_second,
    )

    assert observation.status == "prefix_preserved"
    assert observation.estimated_optimizer_interference_tokens == 0


def test_rewriting_old_history_is_attributed_to_optimizer_interference() -> None:
    guard = PrefixContinuityGuard(block_bytes=64, material_loss_tokens=16)
    first = _body(
        [
            {"role": "user", "content": "run tests"},
            _large_tool("original"),
            {"role": "assistant", "content": "tests passed"},
        ]
    )
    guard.observe("conversation", provider="openai", raw_body=first, outbound_body=first)

    raw_second = _body(
        [
            {"role": "user", "content": "run tests"},
            _large_tool("original"),
            {"role": "assistant", "content": "tests passed"},
            {"role": "user", "content": "continue"},
        ]
    )
    rewritten = _body(
        [
            {"role": "user", "content": "run tests"},
            _large_tool("compressed"),
            {"role": "assistant", "content": "tests passed"},
            {"role": "user", "content": "continue"},
        ]
    )
    observation = guard.observe(
        "conversation",
        provider="openai",
        raw_body=raw_second,
        outbound_body=rewritten,
    )

    assert observation.status == "prefix_degraded"
    assert observation.estimated_optimizer_interference_tokens > 100


def test_stable_compression_is_not_mislabeled_as_interference() -> None:
    guard = PrefixContinuityGuard(block_bytes=64, material_loss_tokens=16)
    raw_first = _body(
        [{"role": "user", "content": "run tests"}, _large_tool("original")]
    )
    optimized_first = _body(
        [
            {"role": "user", "content": "run tests"},
            {"role": "tool", "tool_call_id": "call-1", "content": "31 passed"},
        ]
    )
    guard.observe(
        "conversation",
        provider="openai",
        raw_body=raw_first,
        outbound_body=optimized_first,
    )
    raw_second = _body(
        [
            {"role": "user", "content": "run tests"},
            _large_tool("original"),
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "continue"},
        ]
    )
    optimized_second = _body(
        [
            {"role": "user", "content": "run tests"},
            {"role": "tool", "tool_call_id": "call-1", "content": "31 passed"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "continue"},
        ]
    )

    observation = guard.observe(
        "conversation",
        provider="openai",
        raw_body=raw_second,
        outbound_body=optimized_second,
    )

    assert observation.status == "prefix_preserved"
    assert observation.estimated_optimizer_interference_tokens == 0


def test_warm_cache_guard_preserves_safer_optional_baseline() -> None:
    guard = PrefixContinuityGuard(block_bytes=64, material_loss_tokens=16)
    first = _body(
        [
            {"role": "user", "content": "run tests"},
            _large_tool("original"),
        ]
    )
    guard.observe("conversation", provider="openai", raw_body=first, outbound_body=first)
    baseline = _body(
        [
            {"role": "user", "content": "run tests"},
            _large_tool("original"),
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "continue"},
        ]
    )
    candidate = _body(
        [
            {"role": "user", "content": "run tests"},
            _large_tool("rewritten"),
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "continue"},
        ]
    )

    selected, decision = guard.choose(
        "conversation",
        provider="openai",
        baseline_body=baseline,
        candidate_body=candidate,
        cache_warm=True,
    )

    assert decision.preserved_baseline
    assert decision.estimated_tokens_at_risk > 100
    assert selected == baseline
    assert guard.stats()["guard_interventions"] == 1


def test_cold_cache_does_not_suppress_optional_candidate() -> None:
    guard = PrefixContinuityGuard(block_bytes=64, material_loss_tokens=16)
    first = _body([{"role": "user", "content": "q"}, _large_tool("original")])
    guard.observe("conversation", provider="openai", raw_body=first, outbound_body=first)
    candidate = _body([{"role": "user", "content": "q"}, _large_tool("rewritten")])

    selected, decision = guard.choose(
        "conversation",
        provider="openai",
        baseline_body=first,
        candidate_body=candidate,
        cache_warm=False,
    )

    assert not decision.preserved_baseline
    assert selected == candidate


def test_tracker_retains_no_prompt_content_or_raw_identifier() -> None:
    guard = PrefixContinuityGuard()
    secret = "PRIVATE_PROMPT_SENTINEL_DO_NOT_RETAIN"
    conversation_id = "raw-conversation-identifier"
    body = _body([{"role": "user", "content": secret}])

    guard.observe(
        conversation_id,
        provider="openai",
        raw_body=body,
        outbound_body=body,
    )

    retained = repr(guard._states)
    assert secret not in retained
    assert conversation_id not in retained
    assert guard.stats()["content_retained"] is False
