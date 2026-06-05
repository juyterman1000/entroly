from __future__ import annotations

import struct

from entroly.image_optimizer import (
    estimate_image_tokens_from_dimensions,
    image_dimensions,
    optimize_image_bytes,
    plan_image_optimization,
)


def _png(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"


def test_png_dimensions_are_read_without_decoding_pixels():
    assert image_dimensions(_png(640, 480)) == (640, 480)


def test_openai_high_detail_tile_estimate_matches_documented_shape():
    estimate = estimate_image_tokens_from_dimensions(
        1024,
        1024,
        provider="openai",
        model="gpt-4o",
        detail="high",
    )

    assert estimate.estimated_tokens == 765
    assert estimate.method == "openai_high_detail_tile_estimate"
    assert estimate.exact_provider_measurement is False


def test_gemini_small_image_fixed_estimate():
    estimate = estimate_image_tokens_from_dimensions(
        384,
        384,
        provider="gemini",
        model="gemini-2.0-flash",
    )

    assert estimate.estimated_tokens == 258


def test_image_optimizer_is_explicit_opt_in_and_quality_gated():
    data = _png(4096, 4096)

    gated = plan_image_optimization(data, provider="anthropic")
    assert gated.action == "preserve"
    assert gated.reason == "quality_gate"

    out, decision = optimize_image_bytes(data, provider="anthropic", enabled=False)
    assert out == data
    assert decision.action == "preserve"
    assert decision.reason == "disabled_explicit_opt_in_required"
