"""A streaming request must report the upstream's status, like a buffered one.

`_stream_response` used to open the upstream connection inside the generator
passed to `StreamingResponse`. A StreamingResponse fixes its status when it is
constructed, and the generator does not run until those headers are already on
the wire, so the status was always 200. Only 429 and 5xx were noticed at all,
and they were reported by yielding a JSON blob into a stream that had already
claimed success.

That is the worst shape for the failure. Coding agents stream by default, and a
provider error body is not SSE-framed, so the client's parser yields zero
events and raises nothing: a wrong API key surfaced as an empty assistant turn
rather than an authentication error, a 429 arrived with no Retry-After so the
client's backoff never engaged, and a `400 context too long` -- the failure
this proxy exists to prevent -- was reported as success.

These tests pin the default configuration. The buffered WITNESS path already
propagated the status correctly, but `witness_mode` defaults to off, so the
path almost every user runs was the broken one.
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


def _upstream(status: int, body: dict, extra_headers: dict[str, str] | None = None):
    """A local upstream that answers every POST with one fixed response."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("content-length", 0)))
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):  # noqa: A003
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _proxy_app(port: int):
    config = ProxyConfig(
        anthropic_base_url=f"http://127.0.0.1:{port}",
        openai_base_url=f"http://127.0.0.1:{port}",
    )
    app = create_proxy_app(object(), config, start_dashboard=False, start_autotune=False)
    # Forward only: this is about status handling, not selection.
    app.state.proxy._bypass = True
    return app


def _post(app, *, stream: bool) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://proxy"
        ) as client:
            return await client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 16,
                    "stream": stream,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

    return asyncio.run(_run())


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 503])
def test_streaming_reports_the_upstream_status(status: int) -> None:
    server = _upstream(status, {"error": {"type": "e", "message": "upstream said no"}})
    try:
        response = _post(_proxy_app(server.server_address[1]), stream=True)
        assert response.status_code == status
    finally:
        server.shutdown()


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500])
def test_streaming_and_buffered_agree_on_status(status: int) -> None:
    """Whether the caller asked for a stream must not change the status."""
    server = _upstream(status, {"error": {"type": "e", "message": "upstream said no"}})
    try:
        app = _proxy_app(server.server_address[1])
        streamed = _post(app, stream=True)
        buffered = _post(app, stream=False)
        assert streamed.status_code == buffered.status_code == status
    finally:
        server.shutdown()


def test_retry_after_survives_the_streaming_path() -> None:
    """Without this header the client cannot honour the provider's backoff."""
    server = _upstream(
        429,
        {"error": {"type": "rate_limit_error", "message": "slow down"}},
        extra_headers={"retry-after": "42"},
    )
    try:
        response = _post(_proxy_app(server.server_address[1]), stream=True)
        assert response.status_code == 429
        assert response.headers.get("retry-after") == "42"
    finally:
        server.shutdown()


def test_successful_stream_still_streams() -> None:
    """The fix must not turn the success path into a buffered response."""
    server = _upstream(200, {"ok": True})
    try:
        response = _post(_proxy_app(server.server_address[1]), stream=True)
        assert response.status_code == 200
        assert response.content
    finally:
        server.shutdown()
