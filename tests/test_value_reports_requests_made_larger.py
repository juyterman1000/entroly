"""`entroly value` must say when the proxy made a request larger.

Through `entroly proxy` with the official SDKs, a "Say hello" request of 15
tokens was forwarded as 674: the proxy appended about 660 tokens of retrieved
repository context. The traffic receipt showed the growth, but the value
receipt the README tells users to check reported every such request as
"optimized" and counted none of the added tokens: the saving is clamped at
zero, and nothing records the other direction. Twenty-six requests that each
grew read as "requests_optimized: 26, input_tokens_reduced: 5".
"""

from __future__ import annotations

import asyncio
import json

from entroly.proxy import PromptCompilerProxy
from entroly.proxy_config import ProxyConfig


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
                    "content": "def make_str(value):\n    return str(value)\n" * 20,
                    "preview": "def make_str(value):",
                    "token_count": 240,
                    "entropy_score": 1.0,
                    "variant": "full",
                }
            ],
            "query_analysis": {},
        }


def _send(tracker_dir, monkeypatch, sent: dict):
    from httpx import ASGITransport, AsyncClient
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    import entroly.value_tracker as value_tracker

    tracker = value_tracker.ValueTracker(data_dir=tracker_dir)
    monkeypatch.setattr(value_tracker, "_tracker", tracker)

    async def run():
        cfg = ProxyConfig()
        # These tests are about what injection does, so they turn it on.
        cfg.context_injection = "always"
        cfg.enable_adaptive_budget = False
        cfg.enable_dynamic_budget = False
        cfg.enable_hierarchical_compression = False
        cfg.enable_passive_feedback = False
        cfg.enable_context_scaffold = False
        proxy = PromptCompilerProxy(_OneFragmentEngine(), cfg)
        proxy._confidence_threshold = 0.0
        app = Starlette(routes=[Route("/v1/messages", proxy.handle_proxy, methods=["POST"])])
        captured = {}

        async def capture(_url, _headers, body, *_args, **kwargs):
            captured["body"] = json.loads(json.dumps(body))
            captured["headers"] = dict(kwargs.get("extra_headers") or {})
            return JSONResponse({"id": "msg-test", "type": "message", "model": sent["model"],
                                 "content": [{"type": "text", "text": "hello"}]})

        proxy._forward_response = capture
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/v1/messages", json=sent, headers={
                "x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"})
        return captured

    captured = asyncio.run(run())
    return tracker, captured


def test_a_request_the_proxy_enlarged_is_reported_as_enlarged(tmp_path, monkeypatch) -> None:
    sent = {"model": "claude-sonnet-4-5", "max_tokens": 64,
            "messages": [{"role": "user", "content": "Say hello"}]}

    tracker, captured = _send(tmp_path, monkeypatch, sent)
    forwarded = captured["body"]

    grown = (PromptCompilerProxy._estimate_body_tokens(forwarded)
             - PromptCompilerProxy._estimate_body_tokens(sent))
    assert grown > 0, "the stand-in engine's context must have been injected"
    provider = tracker.get_value_receipt()["provider_path"]
    assert provider["requests_observed"] == 1
    assert provider["requests_optimized"] == 0, "a larger request is not an optimized one"
    assert provider["requests_enlarged"] == 1
    assert provider["input_tokens_added"] == grown
    assert provider["net_input_tokens_reduced"] == provider["input_tokens_reduced"] - grown


def test_old_value_files_load_with_the_new_counters_at_zero(tmp_path) -> None:
    import entroly.value_tracker as value_tracker

    tracker = value_tracker.ValueTracker(data_dir=tmp_path)
    tracker.record(tokens_saved=120, model="claude-sonnet-4-5", source="proxy")

    provider = tracker.get_value_receipt()["provider_path"]
    assert provider["requests_optimized"] == 1
    assert provider["input_tokens_added"] == 0
    assert provider["requests_enlarged"] == 0
    assert provider["net_input_tokens_reduced"] == 120


def test_the_outcome_header_does_not_call_an_enlarged_request_optimized(
    tmp_path, monkeypatch
) -> None:
    """docs/limitations.md points users at X-Entroly-Outcome for live traffic."""
    sent = {"model": "claude-sonnet-4-5", "max_tokens": 64,
            "messages": [{"role": "user", "content": "Say hello"}]}

    _tracker, captured = _send(tmp_path, monkeypatch, sent)

    outcome = {k.lower(): v for k, v in captured["headers"].items()}.get("x-entroly-outcome")
    assert outcome == "context_added", outcome
