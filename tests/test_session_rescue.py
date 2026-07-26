from __future__ import annotations

from pathlib import Path

import pytest

from entroly.cache_routing import CacheAwareRouter
from entroly.compression_retrieval_store import CompressionRetrievalStore
from entroly.session_rescue import (
    SessionRescueController,
    SessionRescuePolicy,
    estimate_message_tokens,
)


def _controller(
    tmp_path: Path,
    **policy_overrides: float | int,
) -> tuple[SessionRescueController, CompressionRetrievalStore]:
    store = CompressionRetrievalStore(tmp_path / "session-rescue.json")
    policy = SessionRescuePolicy(**policy_overrides)
    return SessionRescueController(recovery_store=store, policy=policy), store


def _noisy_tool_message(name: str, lines: int = 500) -> dict[str, str]:
    content = "\n".join(
        f"2026-07-24T08:00:{index:02d}Z INFO worker={index % 5} completed"
        for index in range(lines)
    )
    content += "\n2026-07-24T08:09:59Z FATAL payment-service E_CONNRESET"
    return {"role": "tool", "name": name, "content": content}


def test_policy_rejects_unordered_watermarks() -> None:
    with pytest.raises(ValueError, match="not ordered"):
        SessionRescuePolicy(
            loop_min_watermark=0.4,
            target_watermark=0.75,
            soft_watermark=0.70,
        )


def test_soft_pressure_defers_when_provider_cache_is_warm(tmp_path: Path) -> None:
    controller, _ = _controller(
        tmp_path,
        soft_watermark=0.20,
        hard_watermark=0.90,
        target_watermark=0.10,
        failure_watermark=0.98,
        loop_min_watermark=0.05,
    )
    messages = [{"role": "user", "content": "keep exact prefix " * 200}]

    result = controller.rescue(
        "conv",
        messages,
        context_window=4_000,
        cache_warm=True,
    )

    assert result.action == "cache-deferred"
    assert result.cache_deferred is True
    assert result.messages == messages


def test_hard_pressure_overrides_cache_and_persists_recovery(
    tmp_path: Path,
) -> None:
    controller, store = _controller(
        tmp_path,
        soft_watermark=0.20,
        hard_watermark=0.30,
        target_watermark=0.10,
        failure_watermark=0.95,
        loop_min_watermark=0.05,
        tail_messages=2,
    )
    tool = _noisy_tool_message("pytest")
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        tool,
        {"role": "assistant", "content": "I will inspect the failure."},
        {"role": "user", "content": "continue"},
    ]

    result = controller.rescue(
        "conv",
        messages,
        context_window=4_000,
        cache_warm=True,
        query="payment-service E_CONNRESET",
    )

    assert result.action == "emergency-rescue"
    assert result.tokens_saved > 0
    assert result.recovery_receipts
    assert "payment-service E_CONNRESET" in result.messages[1]["content"]
    assert "[entroly-recovery:" in result.messages[1]["content"]
    stored = store.get_receipt(result.recovery_receipts[0])
    assert stored is not None and stored.spans
    assert stored.spans[0].content == tool["content"]
    assert (
        f"[entroly-recovery:{stored.receipt_id}:{stored.spans[0].span_id}]"
        in result.messages[1]["content"]
    )


def test_frozen_history_remains_byte_identical_when_turns_append(
    tmp_path: Path,
) -> None:
    controller, _ = _controller(
        tmp_path,
        soft_watermark=0.20,
        hard_watermark=0.30,
        target_watermark=0.10,
        failure_watermark=0.95,
        loop_min_watermark=0.05,
        tail_messages=2,
    )
    base = [
        {"role": "system", "content": "system"},
        _noisy_tool_message("cargo-test"),
        {"role": "assistant", "content": "checking"},
        {"role": "user", "content": "continue"},
    ]
    first = controller.rescue("conv", base, context_window=4_000)
    grown = [
        *base,
        {"role": "assistant", "content": "next step"},
        {"role": "user", "content": "continue again"},
    ]
    second = controller.rescue("conv", grown, context_window=4_000)

    assert first.messages[1] == second.messages[1]
    assert second.stable_prefix_messages >= 4


def test_restart_with_different_query_keeps_rescued_prefix_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import entroly.session_rescue as rescue_module

    class _Receipt:
        def as_dict(self) -> dict[str, object]:
            return {
                "original_tokens": 500,
                "compressed_tokens": 5,
                "omitted_spans": [],
            }

    class _Result:
        changed = True
        receipt = _Receipt()

        def __init__(self, query: str) -> None:
            self.compressed = f"structural-summary query={query}"

        def with_receipt_header(self) -> str:
            return self.compressed

    def query_sensitive_compressor(
        _text: str, *, query: str, budget_tokens: int, min_savings: float
    ) -> _Result:
        assert budget_tokens > 0
        assert min_savings > 0
        return _Result(query)

    monkeypatch.setattr(
        rescue_module, "compress_evidence_locked", query_sensitive_compressor
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    policy = {
        "soft_watermark": 0.20,
        "hard_watermark": 0.30,
        "target_watermark": 0.10,
        "failure_watermark": 0.95,
        "loop_min_watermark": 0.05,
        "tail_messages": 2,
    }
    first, _ = _controller(first_dir, **policy)
    restarted, _ = _controller(second_dir, **policy)
    messages = [
        {"role": "system", "content": "stable policy"},
        {"role": "tool", "content": "old tool output " * 500},
        {"role": "assistant", "content": "checking"},
        {"role": "user", "content": "continue"},
    ]

    before_restart = first.rescue(
        "conv", messages, context_window=4_000, query="payment failure"
    )
    after_restart = restarted.rescue(
        "conv", messages, context_window=4_000, query="unrelated auth question"
    )

    assert before_restart.messages[1] == after_restart.messages[1]
    assert before_restart.recovery_receipts == after_restart.recovery_receipts
    assert "query=" in before_restart.messages[1]["content"]
    assert "payment failure" not in before_restart.messages[1]["content"]
    assert "unrelated auth question" not in after_restart.messages[1]["content"]


def test_loop_signal_triggers_before_hard_watermark(tmp_path: Path) -> None:
    controller, _ = _controller(
        tmp_path,
        soft_watermark=0.70,
        hard_watermark=0.90,
        target_watermark=0.20,
        failure_watermark=0.98,
        loop_min_watermark=0.10,
        tail_messages=2,
    )
    messages = [
        _noisy_tool_message("retry"),
        {"role": "assistant", "content": "retrying"},
        {"role": "user", "content": "continue"},
    ]

    result = controller.rescue(
        "loop",
        messages,
        context_window=20_000,
        loop_detected=True,
    )

    assert result.action == "loop-rescue"
    assert result.tokens_saved > 0


def test_uncompressible_over_limit_request_is_blocked_with_clear_state(
    tmp_path: Path,
) -> None:
    controller, _ = _controller(
        tmp_path,
        soft_watermark=0.20,
        hard_watermark=0.30,
        target_watermark=0.10,
        failure_watermark=0.50,
        loop_min_watermark=0.05,
        tail_messages=2,
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "essential user evidence " * 1_000},
        {"role": "user", "content": "latest request " * 100},
    ]

    result = controller.rescue("conv", messages, context_window=1_000)

    assert result.action == "blocked"
    assert result.blocked is True
    assert result.messages == messages
    assert "safe context watermark" in result.error


def test_pressure_without_safe_candidate_is_reported_without_fake_savings(
    tmp_path: Path,
) -> None:
    controller, _ = _controller(
        tmp_path,
        soft_watermark=0.20,
        hard_watermark=0.90,
        target_watermark=0.10,
        failure_watermark=0.98,
        loop_min_watermark=0.05,
        tail_messages=2,
    )
    messages = [{"role": "user", "content": "essential evidence " * 100}]

    result = controller.rescue("conv", messages, context_window=2_000)

    assert result.action == "pressure-observed"
    assert result.tokens_saved == 0
    assert result.messages == messages
    assert controller.stats()["rescues"] == 0


def test_anthropic_tool_result_shape_is_preserved(tmp_path: Path) -> None:
    controller, _ = _controller(
        tmp_path,
        soft_watermark=0.20,
        hard_watermark=0.30,
        target_watermark=0.10,
        failure_watermark=0.95,
        loop_min_watermark=0.05,
        tail_messages=2,
    )
    message = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": _noisy_tool_message("pytest")["content"],
            }
        ],
    }
    messages = [
        message,
        {"role": "assistant", "content": "checking"},
        {"role": "user", "content": "continue"},
    ]

    result = controller.rescue("anthropic", messages, context_window=4_000)

    block = result.messages[0]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_1"
    assert "[entroly-recovery:" in block["content"]


def test_gemini_function_response_preserves_ids_and_thought_signature(
    tmp_path: Path,
) -> None:
    controller, _ = _controller(
        tmp_path,
        soft_watermark=0.20,
        hard_watermark=0.30,
        target_watermark=0.10,
        failure_watermark=0.95,
        loop_min_watermark=0.05,
        tail_messages=2,
    )
    noisy = _noisy_tool_message("gemini")["content"]
    message = {
        "role": "user",
        "content": [
            {
                "thoughtSignature": "opaque-signature",
                "functionResponse": {
                    "id": "call-1",
                    "name": "run_tests",
                    "response": {"output": noisy},
                },
            }
        ],
    }
    messages = [
        message,
        {"role": "model", "content": [{"text": "checking"}]},
        {"role": "user", "content": [{"text": "continue"}]},
    ]

    result = controller.rescue("gemini", messages, context_window=4_000)

    block = result.messages[0]["content"][0]
    assert block["thoughtSignature"] == "opaque-signature"
    assert block["functionResponse"]["id"] == "call-1"
    assert block["functionResponse"]["name"] == "run_tests"
    assert "[entroly-recovery:" in (
        block["functionResponse"]["response"]["output"]
    )


def test_cache_router_lease_snapshot_is_detached_and_expires() -> None:
    router = CacheAwareRouter()
    observed = router.observe(
        "conv",
        model="gpt-5.6",
        provider="openai",
        prefix_hash="abc",
        cached_prefix_tokens=2_000,
        cache_hit=True,
        observed_at=10.0,
        ttl_seconds=5.0,
    )

    snapshot = router.lease_snapshot("conv", now=11.0)
    assert snapshot is not None
    snapshot.cached_prefix_tokens = 0
    assert observed.cached_prefix_tokens == 2_000
    assert router.lease_snapshot("conv", now=16.0) is None


def test_estimator_counts_structured_tool_content() -> None:
    plain = estimate_message_tokens([{"role": "tool", "content": "hello"}])
    structured = estimate_message_tokens(
        [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "hello " * 100}
                ],
            }
        ]
    )
    assert structured > plain
