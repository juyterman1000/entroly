"""The proxy must add repository context only when the query can justify it,
and its headers must describe the request they are attached to.

DF2-6: "?" and "What is this?" reached the provider carrying ~37k tokens of
repository context (147,924 / 148,113 bytes) while the headers reported 74 / 20
compressed tokens and nothing saved. Those queries contain no searchable term,
so nothing in them can rank context, and the engine fills the budget anyway.

DF2-7: on streaming responses X-Entroly-Tokens-Saved-Pct was computed from the
proxy's lifetime counters, so a request that saved nothing could report 65%.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Route

from entroly.config import EntrolyConfig
from entroly.proxy import PromptCompilerProxy
from entroly.proxy_config import ProxyConfig
from entroly.server import EntrolyEngine

FRAGMENT_MARKER = "warehouse_partition_ledger"


def _engine() -> EntrolyEngine:
    engine = EntrolyEngine(EntrolyConfig(use_persistent_index=False))
    for i in range(12):
        body = "\n".join(
            f"def reconcile_shelf_{i}_{n}(rows):\n"
            f"    # {FRAGMENT_MARKER} {i}\n"
            f"    return [row for row in rows if row.bin == {n}]\n"
            for n in range(8)
        )
        engine.ingest_fragment(body, f"pkg/shelf_{i}.py", len(body) // 4)
    return engine


def _proxy() -> PromptCompilerProxy:
    cfg = ProxyConfig()
    cfg.enable_adaptive_budget = False
    cfg.enable_dynamic_budget = False
    cfg.enable_hierarchical_compression = False
    cfg.enable_passive_feedback = False
    cfg.enable_context_scaffold = False
    proxy = PromptCompilerProxy(_engine(), cfg)
    proxy._confidence_threshold = 0.0
    return proxy


def _send(proxy: PromptCompilerProxy, question: str, *, stream: bool):
    forwarded: dict = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        forwarded["body"] = json.loads(request.content)
        if stream:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b'data: {"id":"c1","object":"chat.completion.chunk","choices":'
                    b'[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
                ),
            )
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
        })

    async def run():
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        app = Starlette(routes=[
            Route("/v1/chat/completions", proxy.handle_proxy, methods=["POST"]),
        ])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "stream": stream,
                      "messages": [{"role": "user", "content": question}]},
                headers={"authorization": "Bearer sk-test"},
            )
            await response.aread()
        await proxy._client.aclose()
        return response

    response = asyncio.run(run())
    return response, forwarded.get("body") or {}


@pytest.mark.parametrize("question", ["?", "What is this?"])
def test_a_query_without_terms_gets_no_repository_context(question):
    response, forwarded = _send(_proxy(), question, stream=False)

    assert response.status_code == 200
    assert FRAGMENT_MARKER not in json.dumps(forwarded), (
        f"{question!r} carried repository context upstream"
    )
    assert response.headers.get("x-entroly-context-tokens-added") == "0"


def test_a_query_with_terms_still_gets_context_and_reports_its_size():
    response, forwarded = _send(
        _proxy(), "How does reconcile_shelf_3_5 filter rows by bin?", stream=False
    )

    assert response.status_code == 200
    assert FRAGMENT_MARKER in json.dumps(forwarded)
    assert int(response.headers.get("x-entroly-context-tokens-added", "0")) > 0


def test_streaming_saved_pct_describes_this_request_not_the_process():
    proxy = _proxy()
    # Earlier traffic through the same process saved 90%.
    proxy._total_original_tokens = 10_000
    proxy._total_optimized_tokens = 1_000

    response, forwarded = _send(
        proxy, "How does reconcile_shelf_3_5 filter rows by bin?", stream=True
    )

    assert response.status_code == 200
    assert FRAGMENT_MARKER in json.dumps(forwarded)  # context was added: nothing saved
    assert response.headers.get("x-entroly-tokens-saved-pct") == "0", (
        response.headers.get("x-entroly-tokens-saved-pct")
    )
