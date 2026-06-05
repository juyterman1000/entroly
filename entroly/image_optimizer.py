"""Provider-aware image token estimator and compliance-gated optimizer.

The default path is read-only. Entroly should not silently alter image inputs:
vision quality is task-dependent, and provider image-token accounting changes
over time. Call ``optimize_image_bytes(..., enabled=True)`` to opt in.
"""

from __future__ import annotations

import base64
import io
import math
import struct
from dataclasses import dataclass
from typing import Literal

ImageProvider = Literal["openai", "anthropic", "gemini", "unknown"]
ImageDetail = Literal["low", "high"]


@dataclass(frozen=True)
class ImageTokenEstimate:
    provider: ImageProvider
    width: int
    height: int
    detail: ImageDetail
    estimated_tokens: int
    method: str
    exact_provider_measurement: bool = False


@dataclass(frozen=True)
class ImageOptimizationDecision:
    action: Literal["preserve", "optimize"]
    reason: str
    before: ImageTokenEstimate
    after: ImageTokenEstimate | None
    target_width: int
    target_height: int


def image_dimensions(data: bytes) -> tuple[int, int]:
    """Return image dimensions for PNG/JPEG/WebP without decoding pixels."""
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    if data.startswith(b"\xff\xd8"):
        return _jpeg_dimensions(data)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _webp_dimensions(data)
    raise ValueError("unsupported image format for dimension probe")


def decode_base64_image(value: str) -> bytes:
    """Decode raw base64 or data-URL image content."""
    if "," in value and value[:64].lower().startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value, validate=True)


def estimate_image_tokens(
    data: bytes,
    *,
    provider: ImageProvider = "unknown",
    model: str = "",
    detail: ImageDetail = "high",
) -> ImageTokenEstimate:
    width, height = image_dimensions(data)
    return estimate_image_tokens_from_dimensions(
        width,
        height,
        provider=provider,
        model=model,
        detail=detail,
    )


def estimate_image_tokens_from_dimensions(
    width: int,
    height: int,
    *,
    provider: ImageProvider = "unknown",
    model: str = "",
    detail: ImageDetail = "high",
) -> ImageTokenEstimate:
    provider = provider if provider in {"openai", "anthropic", "gemini"} else "unknown"
    if provider == "openai":
        tokens, method = _openai_tokens(width, height, model=model, detail=detail)
    elif provider == "anthropic":
        tokens, method = _anthropic_tokens(width, height)
    elif provider == "gemini":
        tokens, method = _gemini_tokens(width, height)
    else:
        tokens = max(1, math.ceil(width * height / 750))
        method = "unknown_provider_pixels_over_750_estimate"
    return ImageTokenEstimate(
        provider=provider,
        width=width,
        height=height,
        detail=detail,
        estimated_tokens=tokens,
        method=method,
        exact_provider_measurement=False,
    )


def plan_image_optimization(
    data: bytes,
    *,
    provider: ImageProvider = "unknown",
    model: str = "",
    detail: ImageDetail = "high",
    min_quality_ratio: float = 0.72,
) -> ImageOptimizationDecision:
    before = estimate_image_tokens(data, provider=provider, model=model, detail=detail)
    target_width, target_height = _target_dimensions(
        before.width,
        before.height,
        provider=provider,
        detail=detail,
    )
    if (target_width, target_height) == (before.width, before.height):
        return ImageOptimizationDecision("preserve", "already_provider_aligned", before, None, target_width, target_height)

    quality_ratio = (target_width * target_height) / max(before.width * before.height, 1)
    if quality_ratio < min_quality_ratio:
        return ImageOptimizationDecision("preserve", "quality_gate", before, None, target_width, target_height)

    after = estimate_image_tokens_from_dimensions(
        target_width,
        target_height,
        provider=provider,
        model=model,
        detail=detail,
    )
    if after.estimated_tokens >= before.estimated_tokens:
        return ImageOptimizationDecision("preserve", "no_token_savings", before, after, target_width, target_height)
    return ImageOptimizationDecision("optimize", "provider_token_savings", before, after, target_width, target_height)


def optimize_image_bytes(
    data: bytes,
    *,
    provider: ImageProvider = "unknown",
    model: str = "",
    detail: ImageDetail = "high",
    enabled: bool = False,
    min_quality_ratio: float = 0.72,
) -> tuple[bytes, ImageOptimizationDecision]:
    """Return optimized image bytes only when explicitly enabled.

    Requires Pillow at runtime. If Pillow is unavailable, returns the original
    image and a preserve decision. This avoids adding a hard image dependency
    to Entroly's default install.
    """
    decision = plan_image_optimization(
        data,
        provider=provider,
        model=model,
        detail=detail,
        min_quality_ratio=min_quality_ratio,
    )
    if not enabled:
        return data, ImageOptimizationDecision(
            "preserve",
            "disabled_explicit_opt_in_required",
            decision.before,
            decision.after,
            decision.target_width,
            decision.target_height,
        )
    if decision.action != "optimize":
        return data, decision

    try:
        from PIL import Image  # type: ignore
    except Exception:
        return data, ImageOptimizationDecision(
            "preserve",
            "pillow_unavailable",
            decision.before,
            decision.after,
            decision.target_width,
            decision.target_height,
        )

    with Image.open(io.BytesIO(data)) as img:
        resized = img.resize((decision.target_width, decision.target_height))
        out = io.BytesIO()
        fmt = img.format or "PNG"
        if fmt.upper() in {"JPEG", "JPG"}:
            resized.save(out, format="JPEG", quality=88, optimize=True)
        else:
            resized.save(out, format=fmt)
        optimized = out.getvalue()
    return optimized if len(optimized) < len(data) else data, decision


def _openai_tokens(width: int, height: int, *, model: str, detail: ImageDetail) -> tuple[int, str]:
    model_l = model.lower()
    if "gpt-5" in model_l:
        base, tile = 70, 140
    elif "4o-mini" in model_l:
        base, tile = 2833, 5667
    elif "o1" in model_l or "o3" in model_l:
        base, tile = 75, 150
    elif "computer-use" in model_l:
        base, tile = 65, 129
    else:
        base, tile = 85, 170
    if detail == "low":
        return base, "openai_low_detail_fixed"
    scaled_w, scaled_h = _fit_box(width, height, 2048)
    scale = 768 / max(1, min(scaled_w, scaled_h))
    scaled_w = max(1, math.ceil(scaled_w * scale))
    scaled_h = max(1, math.ceil(scaled_h * scale))
    tiles = math.ceil(scaled_w / 512) * math.ceil(scaled_h / 512)
    return base + tile * tiles, "openai_high_detail_tile_estimate"


def _anthropic_tokens(width: int, height: int) -> tuple[int, str]:
    scaled_w, scaled_h = _fit_box(width, height, 1568)
    return max(1, math.ceil((scaled_w * scaled_h) / 750)), "anthropic_pixels_over_750_estimate"


def _gemini_tokens(width: int, height: int) -> tuple[int, str]:
    if width <= 384 and height <= 384:
        return 258, "gemini_small_image_fixed"
    tiles = math.ceil(width / 768) * math.ceil(height / 768)
    return max(258, tiles * 258), "gemini_768_tile_estimate"


def _target_dimensions(width: int, height: int, *, provider: ImageProvider, detail: ImageDetail) -> tuple[int, int]:
    if provider == "openai" and detail == "high":
        return _fit_box(width, height, 2048)
    if provider == "anthropic":
        return _fit_box(width, height, 1568)
    if provider == "gemini" and (width > 768 or height > 768):
        return _fit_box(width, height, 1536)
    return width, height


def _fit_box(width: int, height: int, max_edge: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= max_edge:
        return width, height
    scale = max_edge / longest
    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        i += 2
        if marker in {0xD8, 0xD9}:
            continue
        if i + 2 > len(data):
            break
        length = struct.unpack(">H", data[i:i + 2])[0]
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if i + 7 <= len(data):
                height, width = struct.unpack(">HH", data[i + 3:i + 7])
                return int(width), int(height)
        i += length
    raise ValueError("invalid jpeg dimensions")


def _webp_dimensions(data: bytes) -> tuple[int, int]:
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk == b"VP8 " and len(data) >= 30:
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    raise ValueError("unsupported webp dimensions")
