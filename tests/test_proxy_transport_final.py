from __future__ import annotations

import asyncio
import gzip
import json
from types import SimpleNamespace

import httpx
import pytest

import entroly.proxy as proxy_module
import entroly.proxy_transport_safe as safe_transport
from entroly.proxy_transport_final import (
    BoundedAsyncClient,
    _safe_build_headers,
    _safe_target_url,
)


def test_final_transport_class_is_active_in_startup_and_proxy() -> None:
    assert safe_transport.BoundedAsyncClient is BoundedAsyncClient
    assert proxy_module.PromptCompilerProxy._build_headers is _safe_build_headers

    async def run():
        dummy = SimpleNamespace(_client=None)
        await safe_transport._safe_startup(dummy)
        try:
            assert isinstance(dummy._client, BoundedAsyncClient)
        finally:
            await dummy._client.aclose()

    asyncio.run(run())


def test_decoded_gzip_body_has_coherent_headers_and_json() -> None:
    original = json.dumps({"ok": True, "value": "decoded"}).encode("utf-8")
    compressed = gzip.compress(original)

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
                "content-length": str(len(compressed)),
            },
            content=compressed,
        )

    async def run() -> httpx.Response:
        async with BoundedAsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=False,
            trust_env=False,
            max_response_bytes=1024,
        ) as client:
            return await client.get("https://api.openai.com/v1/models")

    response = asyncio.run(run())

    assert response.json() == {"ok": True, "value": "decoded"}
    assert "content-encoding" not in response.headers
    assert int(response.headers["content-length"]) == len(original)


def test_decoded_compression_bomb_is_bounded_after_decompression() -> None:
    original = b"x" * 100_000
    compressed = gzip.compress(original)

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            content=compressed,
        )

    async def run() -> httpx.Response:
        async with BoundedAsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=False,
            trust_env=False,
            max_response_bytes=4096,
        ) as client:
            return await client.get("https://api.openai.com/v1/models")

    response = asyncio.run(run())

    assert response.status_code == 424
    assert response.json()["error"] == "upstream_response_too_large"


def test_streaming_redirect_becomes_upstream_failure_without_location() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            308,
            headers={"location": "https://attacker.invalid/stream"},
            content=b"redirect",
        )

    async def run() -> tuple[int, dict, httpx.Headers]:
        async with BoundedAsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": "provider-secret"},
                json={"stream": True},
            ) as response:
                payload = json.loads((await response.aread()).decode("utf-8"))
                return response.status_code, payload, response.headers

    status, payload, headers = asyncio.run(run())

    assert status == 502
    assert payload["error"] == "upstream_redirect_blocked"
    assert "location" not in headers
    assert "attacker.invalid" not in json.dumps(payload)


def test_query_fragment_marker_is_rejected() -> None:
    with pytest.raises(ValueError, match="fragment marker"):
        _safe_target_url(
            "https://api.openai.com",
            "/v1/chat/completions",
            "api-version=1#https://attacker.invalid",
        )


def test_invalid_header_names_and_newlines_are_removed() -> None:
    original = {
        "Authorization": "Bearer provider-key",
        "Bad Header": "invalid-name",
        "X-Request-Test": "safe\r\nInjected: value",
        "Content-Type": "application/json",
    }

    forwarded = _safe_build_headers(SimpleNamespace(), original, "openai")
    lowered = {name.casefold(): value for name, value in forwarded.items()}

    assert lowered["authorization"] == "Bearer provider-key"
    assert lowered["content-type"] == "application/json"
    assert "bad header" not in lowered
    assert "x-request-test" not in lowered
