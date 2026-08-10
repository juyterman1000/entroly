from __future__ import annotations

import base64
import io
import struct

import pytest

from entroly.provider_adapters import optimize_provider_images
from entroly.proxy_config import ProxyConfig


def _large_png_header() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + (b"\x00" * 8) + struct.pack(">II", 2600, 1800) + b"payload"


def _encoded() -> str:
    return base64.b64encode(_large_png_header()).decode("ascii")


@pytest.mark.parametrize(
    ("provider", "body"),
    [
        ("openai", {"model": "gpt-4o", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encoded()}"}}]}]}),
        ("anthropic", {"model": "claude", "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _encoded()}}]}]}),
        ("gemini", {"contents": [{"parts": [{"inline_data": {"mime_type": "image/png", "data": _encoded()}}]}]}),
    ],
)
def test_default_is_byte_for_byte_preservation(provider: str, body: dict) -> None:
    result = optimize_provider_images(body, provider=provider)
    assert result.body == body
    assert result.examined == 0
    assert result.reasons == ("disabled",)


def test_opt_in_is_bounded_and_fail_open_without_valid_decoder() -> None:
    body = {"messages": [{"content": [{"image_url": {"url": f"data:image/png;base64,{_encoded()}"}}]}]}
    result = optimize_provider_images(body, provider="openai", enabled=True, max_images=1)
    assert result.examined == 1
    assert result.optimized == 0
    assert result.preserved == 1
    assert result.body == body


def test_external_image_url_is_never_fetched_or_changed() -> None:
    body = {"messages": [{"content": [{"image_url": {"url": "https://example.invalid/a.png"}}]}]}
    result = optimize_provider_images(body, provider="openai", enabled=True)
    assert result.body == body
    assert result.examined == 0


def test_opt_in_rewrites_supported_embedded_image_when_it_reduces_tokens() -> None:
    image_module = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image_module.new("RGB", (1700, 1400), color=(12, 80, 140)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    body = {"contents": [{"parts": [{"inline_data": {"mime_type": "image/png", "data": encoded}}]}]}
    result = optimize_provider_images(body, provider="gemini", enabled=True)
    assert result.examined == 1
    assert result.optimized == 1
    assert result.estimated_tokens_after < result.estimated_tokens_before
    assert result.body != body


def test_proxy_image_optimization_requires_environment_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_IMAGE_OPTIMIZATION", raising=False)
    assert ProxyConfig.from_env().enable_image_optimization is False
    monkeypatch.setenv("ENTROLY_IMAGE_OPTIMIZATION", "1")
    assert ProxyConfig.from_env().enable_image_optimization is True
