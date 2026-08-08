from __future__ import annotations

import base64
import asyncio
import io
import json
from types import SimpleNamespace

import pytest

from entroly.proxy_image_optimization import (
    ImageRecoveryStore,
    ImageTransformError,
    optimize_inline_images,
)
from entroly.proxy import _image_retrieve


def _large_png() -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGB", (2048, 2048), (23, 67, 101))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize(
    ("provider", "body", "expected_changed"),
    [
        (
            "openai",
            lambda encoded: {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}]}],
            },
            False,
        ),
        (
            "anthropic",
            lambda encoded: {
                "model": "claude",
                "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}}]}],
            },
            False,
        ),
        (
            "gemini",
            lambda encoded: {
                "model": "gemini",
                "contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": encoded}}]}],
            },
            True,
        ),
    ],
)
def test_provider_images_are_optimized_only_after_exact_original_is_stored(
    tmp_path, provider, body, expected_changed
):
    original = _large_png()
    encoded = base64.b64encode(original).decode("ascii")
    store = ImageRecoveryStore(tmp_path / "images")

    result = optimize_inline_images(
        body(encoded),
        provider=provider,
        model="",
        store=store,
        enabled=True,
        min_quality_ratio=0.5,
    )

    assert result.changed is expected_changed
    if expected_changed:
        assert len(result.receipts) == 1
        receipt = result.receipts[0]
        assert receipt.optimized_bytes < receipt.source_bytes
        assert store.retrieve(receipt.receipt_id) == original
        assert result.headers()["X-Entroly-Image-Receipt-Count"] == "1"
    else:
        # These provider estimators already account for upstream resizing, so
        # a smaller transport image would not honestly reduce estimated tokens.
        assert result.receipts == ()
        assert list(store.receipts.iterdir()) == []


def test_disabled_path_preserves_body_and_writes_nothing(tmp_path):
    original = _large_png()
    encoded = base64.b64encode(original).decode("ascii")
    body = {"type": "image_url", "image_url": f"data:image/png;base64,{encoded}"}
    store = ImageRecoveryStore(tmp_path / "images")

    result = optimize_inline_images(
        body, provider="openai", model="gpt-4o", store=store, enabled=False
    )

    assert result.body == body
    assert result.receipts == ()
    assert list(store.receipts.iterdir()) == []


def test_retrieve_rejects_untrusted_receipt_id(tmp_path):
    store = ImageRecoveryStore(tmp_path / "images")
    with pytest.raises(ImageTransformError, match="invalid image receipt"):
        store.retrieve("../../secret")


def test_authenticated_handler_returns_exact_original_without_local_path(tmp_path):
    original = _large_png()
    optimized = b"optimized-image-placeholder"
    store = ImageRecoveryStore(tmp_path / "images")
    receipt = store.store(
        original,
        optimized,
        provider="gemini",
        media_type="image/png",
        before_tokens=100,
        after_tokens=50,
        estimation_method="test",
    )
    request = SimpleNamespace(
        query_params={"receipt_id": receipt.receipt_id},
        app=SimpleNamespace(
            state=SimpleNamespace(
                proxy=SimpleNamespace(_image_recovery_store=store)
            )
        ),
    )

    response = asyncio.run(_image_retrieve(request))
    payload = json.loads(bytes(response.body))

    assert response.status_code == 200
    assert base64.b64decode(payload["original_base64"]) == original
    assert payload["source_sha256"] == receipt.source_sha256
    assert "original_object" not in payload
    assert response.headers["cache-control"] == "no-store"
