from __future__ import annotations

import asyncio
import json

from entroly.integrations.asgi import EntrolyASGIMiddleware
from entroly.integrations.litellm import EntrolyLiteLLMCallback
from entroly.integrations.request_adapter import compress_request_payload


def _payload() -> dict:
    return {
        "model": "gpt-4o",
        "temperature": 0.2,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "messages": [
            {"role": "system", "content": "policy evidence " * 900},
            {"role": "assistant", "content": "analysis evidence " * 900, "tool_calls": [{"id": "call-1"}]},
            {"role": "user", "content": "What is the answer?"},
        ],
    }


def test_request_adapter_preserves_controls_and_tool_contracts() -> None:
    source = _payload()
    result = compress_request_payload(source, budget=150, preserve_last_n=1)
    assert result.changed
    assert result.payload["temperature"] == source["temperature"]
    assert result.payload["tools"] == source["tools"]
    assert result.payload["messages"][1]["tool_calls"] == [{"id": "call-1"}]
    assert result.payload["messages"][-1] == source["messages"][-1]


def test_litellm_pre_call_hook_returns_modified_copy() -> None:
    source = _payload()
    callback = EntrolyLiteLLMCallback(budget=150, preserve_last_n=1)
    output = asyncio.run(callback.async_pre_call_hook(None, None, source, "completion"))
    assert output is not source
    assert output["tools"] == source["tools"]
    assert callback.last_result.changed


def test_litellm_non_text_call_is_identity() -> None:
    source = {"input": [1, 2, 3]}
    callback = EntrolyLiteLLMCallback()
    output = asyncio.run(callback.async_pre_call_hook(None, None, source, "embeddings"))
    assert output is source


def test_asgi_middleware_replays_compressed_valid_json() -> None:
    captured: dict = {}

    async def app(scope, receive, send):
        captured["scope"] = scope
        captured["body"] = (await receive())["body"]

    raw = json.dumps(_payload()).encode()
    queue = [{"type": "http.request", "body": raw, "more_body": False}]

    async def receive():
        return queue.pop(0)

    middleware = EntrolyASGIMiddleware(app, budget=150, preserve_last_n=1)
    scope = {
        "type": "http", "method": "POST",
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())],
    }
    asyncio.run(middleware(scope, receive, lambda message: None))
    output = json.loads(captured["body"])
    assert output["tools"] == _payload()["tools"]
    assert output["messages"][-1] == _payload()["messages"][-1]
    assert int(dict(captured["scope"]["headers"])[b"content-length"]) == len(captured["body"])
