"""The proxy must hand the client the provider's own response metadata.

Measured with the official Anthropic and OpenAI SDKs against a local
upstream:

* Every response lost the provider's headers. ``request-id`` (Anthropic) and
  ``x-request-id`` (OpenAI) never reached the client, so the SDKs'
  ``_request_id`` was None, and the rate-limit headers Claude Code and SDK
  users read (``anthropic-ratelimit-*``, ``x-ratelimit-*``) were dropped on
  buffered and streamed responses alike.
* On the buffered path a 429 and any 5xx were replaced with the proxy's own
  JSON (``{"error": "rate_limited", ...}``, ``{"error": "upstream_error",
  "detail": "Upstream returned 529 after 2 retries"}``). The provider's
  ``error.type`` -- ``overloaded_error``, ``rate_limit_error`` -- and its
  request id were gone, so a client could not tell an overload from any other
  failure, and the streaming path already passed the provider body through.
"""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from entroly.proxy import create_proxy_app
from entroly.proxy_config import ProxyConfig

PROVIDER_HEADERS = {
    "request-id": "req_upstream_1",
    "x-request-id": "req_upstream_1",
    "anthropic-ratelimit-requests-remaining": "41",
    "anthropic-ratelimit-unified-status": "allowed",
    "x-ratelimit-remaining-requests": "99",
    "x-should-retry": "true",
}


def _upstream(status: int, body: dict | str, *, sse: bool = False, extra: dict | None = None):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("content-length", 0)))
            payload = (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(status)
            self.send_header("content-type", "text/event-stream" if sse else "application/json")
            self.send_header("content-length", str(len(payload)))
            for key, value in {**PROVIDER_HEADERS, **(extra or {})}.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):  # noqa: A003
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture()
def no_backoff(monkeypatch):
    """The 5xx retries sleep 1s and 2s; the behaviour under test is the result."""
    import entroly.proxy as proxy_module

    async def _instant(_seconds):
        return None

    monkeypatch.setattr(proxy_module.asyncio, "sleep", _instant)


def _post(server, path: str, payload: dict) -> httpx.Response:
    port = server.server_address[1]
    config = ProxyConfig(
        anthropic_base_url=f"http://127.0.0.1:{port}",
        openai_base_url=f"http://127.0.0.1:{port}",
    )
    app = create_proxy_app(object(), config, start_dashboard=False, start_autotune=False)
    app.state.proxy._bypass = True

    async def _run() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client:
            return await client.post(path, json=payload)

    try:
        return asyncio.run(_run())
    finally:
        server.shutdown()


ANTHROPIC = {"model": "claude-sonnet-4-5", "max_tokens": 16,
             "messages": [{"role": "user", "content": "hi"}]}
OPENAI = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
MESSAGE = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
           "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
           "stop_sequence": None, "usage": {"input_tokens": 3, "output_tokens": 1}}


def _assert_provider_headers(response: httpx.Response) -> None:
    for name, value in PROVIDER_HEADERS.items():
        assert response.headers.get(name) == value, f"{name} lost: {dict(response.headers)}"


def test_a_buffered_success_keeps_the_provider_headers() -> None:
    response = _post(_upstream(200, MESSAGE), "/v1/messages", ANTHROPIC)

    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "hello"
    _assert_provider_headers(response)


def test_a_streamed_success_keeps_the_provider_headers() -> None:
    events = (
        'event: message_start\ndata: {"type":"message_start","message":' + json.dumps(MESSAGE) + "}\n\n"
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )
    response = _post(_upstream(200, events, sse=True), "/v1/messages", {**ANTHROPIC, "stream": True})

    assert response.status_code == 200
    assert b"message_stop" in response.content
    _assert_provider_headers(response)


def test_an_openai_success_keeps_the_provider_headers() -> None:
    completion = {"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-4o",
                  "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"},
                               "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}

    response = _post(_upstream(200, completion), "/v1/chat/completions", OPENAI)

    assert response.status_code == 200
    _assert_provider_headers(response)


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(429, "rate_limit_error"), (500, "api_error"), (529, "overloaded_error")],
)
def test_a_buffered_error_reaches_the_client_as_the_provider_sent_it(
    no_backoff, status: int, error_type: str
) -> None:
    provider_error = {"type": "error", "error": {"type": error_type, "message": "upstream said so"},
                      "request_id": "req_upstream_1"}

    response = _post(_upstream(status, provider_error, extra={"retry-after": "42"}),
                     "/v1/messages", ANTHROPIC)

    assert response.status_code == status
    assert response.json() == provider_error
    _assert_provider_headers(response)
    assert response.headers.get("retry-after") == "42"
    assert response.headers.get("x-entroly-source") == "upstream"


def test_a_streamed_error_keeps_the_provider_headers() -> None:
    provider_error = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}

    response = _post(_upstream(429, provider_error, extra={"retry-after": "42"}),
                     "/v1/messages", {**ANTHROPIC, "stream": True})

    assert response.status_code == 429
    assert response.json() == provider_error
    _assert_provider_headers(response)


def test_framing_headers_are_not_copied_onto_a_rewritten_body() -> None:
    """content-length and content-encoding describe the upstream bytes, not ours."""
    response = _post(_upstream(200, MESSAGE, extra={"content-encoding": "identity"}),
                     "/v1/messages", ANTHROPIC)

    assert response.headers.get("content-encoding") is None
    assert int(response.headers["content-length"]) == len(response.content)
