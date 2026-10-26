"""Proxy savings must be measured on the same content on both sides.

The proxy booked the whole volume of Anthropic ``tool_result`` blocks as
"input tokens reduced": the inbound count read block content through
``_content_to_text`` while the outbound count read only ``text`` keys, so a
tool result counted on the way in and as nothing on the way out. One Claude
Code style request through ``entroly proxy`` then showed tens of thousands of
tokens "reduced" in ``entroly value`` although the forwarded body was larger
than the one the client sent.
"""

from __future__ import annotations

import asyncio
import json

from entroly.proxy import (
    PromptCompilerProxy,
    _content_to_text,
    _estimate_message_tokens,
)
from entroly.proxy_config import ProxyConfig
from entroly.tokens import count_tokens


class _OneFragmentEngine:
    """Engine stand-in: the object under test is the proxy's accounting."""

    def __init__(self):
        self._turn_counter = 0

    def advance_turn(self):
        return None

    def optimize_context(self, token_budget, query):
        return {
            "selected_fragments": [
                {
                    "id": "util-1",
                    "source": "src/util.py",
                    "content": "def make_str(value):\n    return str(value)\n",
                    "preview": "def make_str(value):",
                    "token_count": 12,
                    "entropy_score": 1.0,
                    "variant": "full",
                }
            ],
            "query_analysis": {},
        }


def _body_tokens(body: dict) -> int:
    """Neutral measure: message content plus system, read the same way."""
    return _estimate_message_tokens(body.get("messages") or []) + count_tokens(
        _content_to_text(body.get("system"))
    )


def _claude_code_request() -> dict:
    listing = "".join(
        f"tests/test_case_{i}.py::test_value_{i} PASSED [{i % 100:3d}%] node={i * 7919 % 10007}\n"
        for i in range(160)
    )
    return {
        "model": "claude-sonnet-4-6-20260501",
        "max_tokens": 256,
        "system": [{"type": "text", "text": "You are a careful release agent.", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Run the tests and fix make_str."}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "run_tests", "input": {"path": "tests"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": listing}]},
                {"type": "text", "text": "Now fix make_str so the regression test passes."},
            ]},
        ],
    }


def test_proxy_books_no_savings_for_tool_results_it_forwarded(tmp_path, monkeypatch):
    from httpx import ASGITransport, AsyncClient
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    import entroly.value_tracker as value_tracker

    tracker = value_tracker.ValueTracker(data_dir=tmp_path)
    monkeypatch.setattr(value_tracker, "_tracker", tracker)
    sent = _claude_code_request()

    async def run():
        cfg = ProxyConfig()
        cfg.enable_adaptive_budget = False
        cfg.enable_dynamic_budget = False
        cfg.enable_hierarchical_compression = False
        cfg.enable_passive_feedback = False
        cfg.enable_context_scaffold = False
        proxy = PromptCompilerProxy(_OneFragmentEngine(), cfg)
        proxy._confidence_threshold = 0.0
        app = Starlette(routes=[Route("/v1/messages", proxy.handle_proxy, methods=["POST"])])
        captured = {}

        async def capture(_url, _headers, body, *_args, **_kwargs):
            captured["body"] = json.loads(json.dumps(body))
            return JSONResponse({"id": "msg-test", "type": "message", "model": sent["model"],
                                 "content": [{"type": "text", "text": "done"}]})

        proxy._forward_response = capture
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/messages", json=sent, headers={
                "x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"})
        return response, captured, proxy

    response, captured, proxy = asyncio.run(run())
    assert response.status_code == 200
    forwarded = captured["body"]
    # The tool result reached the provider; it is not a saving.
    assert "test_value_159" in json.dumps(forwarded)

    measured = max(0, _body_tokens(sent) - _body_tokens(forwarded))
    booked = tracker._data["lifetime"]["provider_tokens_saved"]
    assert booked <= measured, (
        f"entroly value would show {booked:,} input tokens reduced, but the body "
        f"forwarded upstream is only {measured:,} tokens smaller than the one the "
        f"client sent ({_body_tokens(sent):,} -> {_body_tokens(forwarded):,})"
    )
    in_memory = max(0, proxy._total_original_tokens - proxy._total_optimized_tokens)
    assert in_memory <= measured


def test_outbound_estimate_reads_blocks_like_the_inbound_one():
    body = _claude_code_request()
    assert PromptCompilerProxy._estimate_body_tokens(body) == _body_tokens(body)
