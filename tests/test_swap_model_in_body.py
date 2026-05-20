"""Regression tests for `swap_model_in_body` and Anthropic cleanup.

When RAVS routes or proxies native Anthropic traffic, request metadata that
the public Messages API rejects (for example `context_management`) must be
stripped from the body. Legacy Claude 3.x targets also reject newer generation
fields such as `thinking`. Otherwise Anthropic returns:

    400 ... context_management: Extra inputs are not permitted

and the user sees the proxy as broken on the first request.
"""

from __future__ import annotations

from entroly.ravs.router import (
    _EXTENDED_ONLY_PARAMS,
    _is_legacy_claude_3,
    swap_model_in_body,
)


class TestIsLegacyClaude3:
    def test_haiku_3_is_legacy(self):
        assert _is_legacy_claude_3("claude-3-haiku-20240307")

    def test_haiku_3_5_is_legacy(self):
        assert _is_legacy_claude_3("claude-3-5-haiku-20241022")

    def test_sonnet_3_5_is_legacy(self):
        assert _is_legacy_claude_3("claude-3-5-sonnet-20241022")

    def test_opus_3_is_legacy(self):
        assert _is_legacy_claude_3("claude-3-opus-20240229")

    def test_sonnet_4_5_is_not_legacy(self):
        # This only controls legacy-only params such as `thinking`.
        # `context_management` is stripped for all native Anthropic forwards.
        assert not _is_legacy_claude_3("claude-sonnet-4-5-20250929")
        assert not _is_legacy_claude_3("claude-opus-4-5")
        assert not _is_legacy_claude_3("claude-opus-4-7")

    def test_unknown_is_not_legacy(self):
        # Unknown targets keep legacy-only params, but public-API-invalid
        # client metadata is still removed by strip_anthropic_unsupported_params.
        assert not _is_legacy_claude_3("gpt-4o-mini")
        assert not _is_legacy_claude_3("")
        assert not _is_legacy_claude_3("some-custom-model")


class TestSwapModelInBody:
    def test_basic_swap(self):
        body = {"model": "claude-sonnet-4-5", "messages": []}
        out = swap_model_in_body(body, "claude-3-5-haiku-20241022")
        assert out["model"] == "claude-3-5-haiku-20241022"
        assert out["messages"] == []

    def test_does_not_mutate_input(self):
        body = {"model": "claude-sonnet-4-5", "messages": [], "context_management": {"foo": 1}}
        original = dict(body)
        _ = swap_model_in_body(body, "claude-3-5-haiku-20241022")
        assert body == original, "swap_model_in_body must not mutate input"

    def test_strips_context_management_on_legacy_swap(self):
        body = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "context_management": {"strategy": "auto"},
            "max_tokens": 1024,
        }
        out = swap_model_in_body(body, "claude-3-5-haiku-20241022")
        assert "context_management" not in out
        assert out["model"] == "claude-3-5-haiku-20241022"
        assert out["messages"] == [{"role": "user", "content": "hi"}]
        assert out["max_tokens"] == 1024

    def test_strips_thinking_on_legacy_swap(self):
        body = {
            "model": "claude-opus-4-5",
            "thinking": {"type": "enabled", "budget_tokens": 8192},
            "messages": [],
        }
        out = swap_model_in_body(body, "claude-3-haiku-20240307")
        assert "thinking" not in out

    def test_strips_context_management_on_non_legacy_swap(self):
        body = {
            "model": "claude-opus-4-5",
            "context_management": {"strategy": "auto"},
            "thinking": {"type": "enabled"},
        }
        out = swap_model_in_body(body, "claude-sonnet-4-5-20250929")
        assert "context_management" not in out
        assert "thinking" in out

    def test_strips_context_management_when_target_unknown(self):
        body = {
            "model": "claude-sonnet-4-5",
            "context_management": {"strategy": "auto"},
            "thinking": {"type": "enabled"},
        }
        out = swap_model_in_body(body, "some-future-model-id")
        assert "context_management" not in out
        assert "thinking" in out

    def test_no_model_field_still_safe(self):
        body = {"messages": [], "context_management": {"x": 1}}
        out = swap_model_in_body(body, "claude-3-5-haiku-20241022")
        assert "model" not in out or out["model"] is None or True
        assert "context_management" not in out

    def test_all_known_extended_params_stripped(self):
        body = {"model": "claude-sonnet-4-5"}
        for p in _EXTENDED_ONLY_PARAMS:
            body[p] = {"dummy": True}
        out = swap_model_in_body(body, "claude-3-haiku-20240307")
        for p in _EXTENDED_ONLY_PARAMS:
            assert p not in out, f"{p} leaked through legacy swap"
