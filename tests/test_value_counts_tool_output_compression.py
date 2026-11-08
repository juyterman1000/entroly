"""`entroly value` must count tool output the proxy compressed.

Dogfooding 1.0.88 through `entroly go`: an agent turn with a 3,000-line pytest
log and a 1,500-item JSON listing went upstream as 1,822 tokens instead of
64,646 (response headers said so), yet `entroly value` reported "Input tokens
reduced: 0" and the two requests were missing from `requests_observed`. The
proxy booked value only when it injected retrieved context, and measured it
against a body tool-output compression had already shrunk.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from entroly.proxy import PromptCompilerProxy, _first_request_banner
from entroly.proxy_config import ProxyConfig


class _NoContextEngine:
    """Selects nothing, so only tool-output compression can change the body."""

    def __init__(self):
        self._turn_counter = 0

    def advance_turn(self):
        return None

    def optimize_context(self, token_budget, query):
        return {"selected_fragments": [], "query_analysis": {}}


def _agent_turn() -> dict:
    log = "\n".join(
        f"tests/test_client.py::test_case_{i} PASSED [{i % 100:3d}%]" for i in range(3000)
    ) + "\nFAILED tests/test_redirects.py::test_cross_origin_auth - AssertionError\n"
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "system": "You are a coding agent.",
        "tools": [{"name": "bash", "description": "run a command",
                   "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}}}],
        "messages": [
            {"role": "user", "content": "Run the tests and tell me why the redirect test fails."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "pytest -q"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": log}]},
        ],
    }


def _send(tmp_path, monkeypatch, sent: dict):
    from httpx import ASGITransport, AsyncClient
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    import entroly.value_tracker as value_tracker

    monkeypatch.setenv("HOME", str(tmp_path))
    tracker = value_tracker.ValueTracker(data_dir=tmp_path / "value")
    monkeypatch.setattr(value_tracker, "_tracker", tracker)

    async def run():
        cfg = ProxyConfig()
        cfg.enable_adaptive_budget = False
        cfg.enable_dynamic_budget = False
        cfg.enable_passive_feedback = False
        cfg.enable_context_scaffold = False
        proxy = PromptCompilerProxy(_NoContextEngine(), cfg)
        app = Starlette(routes=[Route("/v1/messages", proxy.handle_proxy, methods=["POST"])])
        captured = {}

        async def capture(_url, _headers, body, *_args, **kwargs):
            captured["body"] = json.loads(json.dumps(body))
            captured["headers"] = dict(kwargs.get("extra_headers") or {})
            return JSONResponse({"id": "msg-test", "type": "message", "model": sent["model"],
                                 "content": [{"type": "text", "text": "ok"}]})

        proxy._forward_response = capture
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/v1/messages", json=sent, headers={
                "x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"})
        return proxy, captured

    proxy, captured = asyncio.run(run())
    return tracker, proxy, captured


def test_compressed_tool_output_reaches_the_value_receipt(tmp_path, monkeypatch) -> None:
    sent = _agent_turn()
    tracker, proxy, captured = _send(tmp_path, monkeypatch, sent)

    before = PromptCompilerProxy._estimate_body_tokens(sent)
    after = PromptCompilerProxy._estimate_body_tokens(captured["body"])
    assert after < before // 2, "tool output must have been compressed for this test to mean anything"

    provider = tracker.get_value_receipt()["provider_path"]
    assert provider["requests_observed"] == 1
    assert provider["requests_optimized"] == 1
    assert provider["input_tokens_reduced"] == before - after
    assert provider["input_tokens_added"] == 0

    assert proxy._total_original_tokens == before
    assert proxy._total_optimized_tokens == after
    assert proxy._requests_optimized == 1


def test_an_unchanged_request_is_observed_not_optimized(tmp_path, monkeypatch) -> None:
    sent = {"model": "claude-sonnet-4-5", "max_tokens": 64,
            "messages": [{"role": "user", "content": "?"}]}
    tracker, proxy, _captured = _send(tmp_path, monkeypatch, sent)

    provider = tracker.get_value_receipt()["provider_path"]
    assert provider["requests_observed"] == 1
    assert provider["requests_optimized"] == 0
    assert provider["input_tokens_reduced"] == 0
    assert proxy._requests_optimized == 0


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"contents": [{"role": "user", "parts": [{"text": "alpha beta gamma"}]}]}, True),
        ({"input": [{"role": "user", "content": [{"type": "input_text", "text": "alpha beta"}]}],
          "instructions": "be brief"}, True),
        ({"input": "alpha beta gamma"}, True),
    ],
)
def test_gemini_and_responses_bodies_are_measured(body, expected) -> None:
    assert (PromptCompilerProxy._estimate_body_tokens(body) > 0) is expected


def test_the_banner_never_calls_a_larger_request_optimized() -> None:
    grew = _first_request_banner(27_620, 27_765, [], 2963.9)
    assert "optimized" not in grew
    assert "+145" in grew

    shrank = _first_request_banner(64_646, 1_822, [], None)
    assert "First request optimized" in shrank
    assert "97% smaller" in shrank
