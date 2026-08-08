from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

import pytest

from entroly.integrations.gateway import (
    CompressionGatewayClient,
    GatewayCompression,
    GatewayReceipt,
    wrap_anthropic,
    wrap_openai,
)
from entroly.integrations.litellm import EntrolyLiteLLMHook


class FakeResponse:
    def __init__(self, body, headers=None):
        self._body = json.dumps(body).encode()
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self._body


def test_gateway_client_returns_recovery_receipt_and_rejects_remote():
    seen = {}

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["headers"] = dict(request.header_items())
        return FakeResponse(
            {"messages": [{"role": "user", "content": "compressed"}]},
            {
                "X-Entroly-Receipt-Count": "1",
                "X-Entroly-Recovery": "stored",
                "X-Entroly-Compression": "changed",
            },
        )

    client = CompressionGatewayClient(
        opener=opener,
        budget_tokens=700,
        access_token="x" * 40,
        sidecar_token="sidecar-secret",
    )
    result = client.compress_payload({"messages": []})

    assert result.receipt.count == 1
    assert result.receipt.recovery == "stored"
    assert "budget_tokens=700" in seen["url"]
    assert seen["headers"]["X-entroly-access-token"] == "x" * 40
    assert seen["headers"]["X-entroly-sidecar-token"] == "sidecar-secret"
    with pytest.raises(ValueError, match="positive"):
        client.compress_payload({"messages": []}, budget_tokens=0)
    with pytest.raises(ValueError, match="allow_remote"):
        CompressionGatewayClient("https://example.com")


def test_gateway_client_exposes_exact_image_recovery():
    seen = {}

    def opener(request, timeout):
        seen["url"] = request.full_url
        return FakeResponse(
            {
                "receipt_id": "img:0123456789abcdef01234567",
                "original_base64": "ZXhhY3Q=",
            }
        )

    result = CompressionGatewayClient(opener=opener).retrieve_image(
        "img:0123456789abcdef01234567"
    )

    assert "/retrieve-image?" in seen["url"]
    assert result["original_base64"] == "ZXhhY3Q="


class FakeGateway:
    def __init__(self):
        self.calls = []

    def compress_payload(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        changed = dict(payload)
        changed["messages"] = [{"role": "user", "content": "compressed"}]
        return GatewayCompression(
            changed,
            GatewayReceipt(1, "stored", "changed", {"x-entroly-recovery": "stored"}),
        )

    async def async_compress_payload(self, payload, **kwargs):
        return self.compress_payload(payload, **kwargs)


def test_openai_and_anthropic_wrappers_forward_compressed_payload():
    gateway = FakeGateway()
    openai_seen = {}
    anthropic_seen = {}
    openai = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: openai_seen.update(kwargs) or kwargs
            )
        ),
        responses=None,
    )
    anthropic = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: anthropic_seen.update(kwargs) or kwargs
        )
    )

    wrap_openai(openai, gateway=gateway).chat.completions.create(
        model="gpt", messages=[{"role": "user", "content": "raw"}]
    )
    wrap_anthropic(anthropic, gateway=gateway).messages.create(
        model="claude", messages=[{"role": "user", "content": "raw"}]
    )

    assert openai_seen["messages"][0]["content"] == "compressed"
    assert anthropic_seen["messages"][0]["content"] == "compressed"


def test_litellm_hook_matches_proxy_contract_and_exposes_receipt_headers():
    hook = EntrolyLiteLLMHook(gateway=FakeGateway())
    changed = asyncio.run(
        hook.async_pre_call_hook(None, None, {"messages": []}, "completion")
    )
    headers = asyncio.run(
        hook.async_post_call_response_headers_hook(changed, None, None)
    )

    assert changed["messages"][0]["content"] == "compressed"
    assert headers == {"x-entroly-recovery": "stored"}
    unchanged = asyncio.run(
        hook.async_pre_call_hook(None, None, {"input": "embed"}, "embeddings")
    )
    assert unchanged == {"input": "embed"}


def test_litellm_receipts_stay_bound_to_concurrent_payloads():
    class ReceiptGateway(FakeGateway):
        async def async_compress_payload(self, payload, **kwargs):
            marker = str(payload["marker"])
            changed = dict(payload)
            changed["compressed"] = True
            return GatewayCompression(
                changed,
                GatewayReceipt(1, "stored", "changed", {"x-request-marker": marker}),
            )

    hook = EntrolyLiteLLMHook(gateway=ReceiptGateway())

    async def run():
        first, second = await asyncio.gather(
            hook.async_pre_call_hook(None, None, {"marker": "first"}, "completion"),
            hook.async_pre_call_hook(None, None, {"marker": "second"}, "completion"),
        )
        second_headers = await hook.async_post_call_response_headers_hook(
            second, None, None
        )
        first_headers = await hook.async_post_call_response_headers_hook(
            first, None, None
        )
        return first_headers, second_headers

    first_headers, second_headers = asyncio.run(run())

    assert first_headers == {"x-request-marker": "first"}
    assert second_headers == {"x-request-marker": "second"}
