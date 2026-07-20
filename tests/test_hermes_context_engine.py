"""Hermetic tests for the Entroly ContextEngine plugin.

These tests verify the Hermes ContextEngine adapter without importing
hermes-agent.  The engine uses a stub base class when Hermes is absent,
so all contract tests run in an isolated ``pip install entroly`` environment.
"""

from __future__ import annotations

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_engine(**kwargs):
    """Construct an EntrolyContextEngine with test defaults."""
    from entroly.integrations.hermes_context_engine import EntrolyContextEngine

    defaults = {
        "context_length": 128_000,
        "threshold_percent": 0.75,
        "protect_first_n": 3,
        "protect_last_n": 4,
        "profile": "balanced",
    }
    defaults.update(kwargs)
    return EntrolyContextEngine(**defaults)


def _make_messages(n: int = 10, content_len: int = 500) -> list[dict]:
    """Generate a synthetic OpenAI-format conversation."""
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"Message {i}: " + "x" * content_len})
    return msgs


def _make_hermes_messages(n: int = 10, content_len: int = 500) -> list[dict]:
    """Generate messages with a Hermes tool-calling system prompt."""
    tool_system = {
        "role": "system",
        "content": (
            "You are a function calling AI model. You are provided with "
            "function signatures within <tools></tools> XML tags."
        ),
    }
    msgs = [tool_system]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"Message {i}: " + "x" * content_len})
    return msgs


# ── Identity ─────────────────────────────────────────────────────────────

class TestIdentity:
    def test_name_is_entroly(self):
        engine = _make_engine()
        assert engine.name == "entroly"

    def test_init_defaults(self):
        engine = _make_engine()
        assert engine.context_length == 128_000
        assert engine.threshold_tokens == 96_000
        assert engine.compression_count == 0


# ── Token state tracking ─────────────────────────────────────────────────

class TestUpdateFromResponse:
    def test_legacy_keys(self):
        engine = _make_engine()
        engine.update_from_response({
            "prompt_tokens": 5000,
            "completion_tokens": 1200,
            "total_tokens": 6200,
        })
        assert engine.last_prompt_tokens == 5000
        assert engine.last_completion_tokens == 1200
        assert engine.last_total_tokens == 6200

    def test_canonical_keys(self):
        engine = _make_engine()
        engine.update_from_response({
            "input_tokens": 7000,
            "output_tokens": 800,
        })
        assert engine.last_prompt_tokens == 7000
        assert engine.last_completion_tokens == 800
        assert engine.last_total_tokens == 7800

    def test_empty_usage(self):
        engine = _make_engine()
        engine.update_from_response({})
        assert engine.last_prompt_tokens == 0
        assert engine.last_completion_tokens == 0


# ── Compression trigger ──────────────────────────────────────────────────

class TestShouldCompress:
    def test_below_threshold_returns_false(self):
        engine = _make_engine(context_length=100_000, threshold_percent=0.75)
        assert engine.should_compress(prompt_tokens=50_000) is False

    def test_above_threshold_returns_true(self):
        engine = _make_engine(context_length=100_000, threshold_percent=0.75)
        assert engine.should_compress(prompt_tokens=80_000) is True

    def test_at_threshold_returns_false(self):
        engine = _make_engine(context_length=100_000, threshold_percent=0.75)
        # Exactly at threshold: 75000 is not > 75000
        assert engine.should_compress(prompt_tokens=75_000) is False

    def test_uses_last_prompt_tokens_when_none(self):
        engine = _make_engine(context_length=100_000, threshold_percent=0.75)
        engine.last_prompt_tokens = 90_000
        assert engine.should_compress() is True


# ── Core compression ─────────────────────────────────────────────────────

class TestCompress:
    def test_empty_messages_returned_unchanged(self):
        engine = _make_engine()
        assert engine.compress([]) == []

    def test_compresses_large_conversation(self):
        engine = _make_engine(context_length=2000, profile="max")
        msgs = _make_messages(n=20, content_len=400)
        result = engine.compress(msgs, focus_topic="bug fix in auth module")

        # Must return a valid message list.
        assert isinstance(result, list)
        assert len(result) > 0
        # Every message must have role and content.
        for msg in result:
            assert "role" in msg
            assert "content" in msg

    def test_preserves_hermes_tool_system_prompt(self):
        engine = _make_engine(context_length=2000, profile="max")
        msgs = _make_hermes_messages(n=20, content_len=400)
        result = engine.compress(msgs)

        # The <tools> system prompt must survive verbatim.
        system_msgs = [m for m in result if m.get("role") == "system"]
        assert len(system_msgs) >= 1
        tool_system = [m for m in system_msgs if "<tools>" in m.get("content", "")]
        assert len(tool_system) == 1
        assert "function calling AI model" in tool_system[0]["content"]

    def test_compression_increments_count(self):
        engine = _make_engine(context_length=2000, profile="max")
        msgs = _make_messages(n=10, content_len=400)
        engine.compress(msgs)
        assert engine.compression_count >= 1

    def test_fail_open_on_error(self):
        """If Entroly SDK raises, messages are returned unchanged."""
        engine = _make_engine()
        msgs = _make_messages(n=3)
        # Patch _compress_with_entroly to raise
        original = engine._compress_with_entroly
        engine._compress_with_entroly = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("simulated entroly failure")
        )
        result = engine.compress(msgs)
        assert result == msgs
        engine._compress_with_entroly = original


# ── Lifecycle hooks ──────────────────────────────────────────────────────

class TestLifecycle:
    def test_session_start_resets_state(self):
        engine = _make_engine()
        engine.compression_count = 5
        engine._total_tokens_saved = 10000
        engine.on_session_start(session_id="test-123")
        assert engine._session_id == "test-123"
        assert engine.compression_count == 0
        assert engine._total_tokens_saved == 0

    def test_session_end_clears_session(self):
        engine = _make_engine()
        engine.on_session_start(session_id="test-456")
        engine.on_session_end(session_id="test-456")
        assert engine._session_id is None

    def test_session_reset_clears_token_state(self):
        engine = _make_engine()
        engine.last_prompt_tokens = 5000
        engine.compression_count = 3
        engine.on_session_reset()
        assert engine.last_prompt_tokens == 0
        assert engine.compression_count == 0


# ── Integration: focus_topic as evidence query ───────────────────────────

class TestFocusTopicIntegration:
    def test_focus_topic_does_not_crash(self):
        """focus_topic is passed through — verify no errors."""
        engine = _make_engine(context_length=2000, profile="balanced")
        msgs = _make_messages(n=10, content_len=300)
        result = engine.compress(msgs, focus_topic="fix authentication bug in login.py")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_none_focus_topic(self):
        engine = _make_engine(context_length=2000, profile="max")
        msgs = _make_messages(n=10, content_len=300)
        result = engine.compress(msgs, focus_topic=None)
        assert isinstance(result, list)
