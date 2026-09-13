"""
Image/multimodal compression — reduces token cost of screenshots, diagrams,
and image-heavy context by 40-90%.

Three strategies:
  1. Resolution reduction: resize to fit token budget (vision models charge
     per tile, so smaller images = fewer tokens)
  2. Format conversion: PNG → WebP/JPEG for photographs, keep PNG for diagrams
  3. Text extraction: OCR fallback that converts image to text representation
     when the image is primarily text (terminal screenshots, error dialogs)

All processing is local-only — no remote API calls.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_QUALITY = 10


@dataclass
class CompressionResult:
    original_size: int       # bytes
    compressed_size: int     # bytes
    reduction_pct: float     # 0-100
    format: str              # output format (webp, jpeg, png, text)
    width: int
    height: int
    strategy: str            # resize, convert, ocr, combined
    data: bytes | None = None
    text: str | None = None  # OCR text if extracted
    data_uri: str | None = None


def _has_pillow() -> bool:
    try:
        from PIL import Image
        return True
    except ImportError:
        return False


def compress_image(
    source: str | Path | bytes,
    *,
    max_dimension: int = 1024,
    quality: int = 75,
    target_format: str | None = None,
    extract_text: bool = False,
    max_bytes: int | None = None,
) -> CompressionResult:
    """Compress an image for LLM context injection."""
    if not _has_pillow():
        raise RuntimeError(
            "Image compression requires Pillow. Install with: "
            "pip install 'entroly[full]'  # or:  pip install Pillow"
        )

    from PIL import Image

    original_bytes: bytes | None = None
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            original_bytes = path.read_bytes()
            img = Image.open(path)
        else:
            original_bytes = bytes(source)
            img = Image.open(io.BytesIO(original_bytes))
        img.load()
    except Exception as exc:
        logger.warning("Image compression failed to open image: %s", exc)
        fallback_size = len(original_bytes) if original_bytes is not None else 0
        return CompressionResult(
            original_size=fallback_size,
            compressed_size=fallback_size,
            reduction_pct=0.0,
            format="unknown",
            width=0,
            height=0,
            strategy="passthrough",
            data=original_bytes,
        )

    original_size = len(original_bytes)
    orig_w, orig_h = img.size
    strategy_parts: list[str] = []

    if max(orig_w, orig_h) > max_dimension:
        ratio = max_dimension / max(orig_w, orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        strategy_parts.append("resize")
    else:
        new_w, new_h = orig_w, orig_h

    if target_format is None:
        if img.mode == "RGBA" or _is_diagram(img):
            target_format = "png"
        else:
            target_format = "webp"
    strategy_parts.append("convert")

    compressed_data = _encode(img, target_format, quality)

    if max_bytes and len(compressed_data) > max_bytes and target_format not in ("png",):
        for q in range(max(quality - 10, MIN_QUALITY), MIN_QUALITY - 1, -10):
            compressed_data = _encode(img, target_format, q)
            if len(compressed_data) <= max_bytes:
                break

    compressed_size = len(compressed_data)

    extracted_text = None
    if extract_text:
        extracted_text = _extract_text(img)
        if extracted_text and len(extracted_text) > 20:
            strategy_parts.append("ocr")

    mime = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "webp": "image/webp",
    }.get(target_format, "image/png")
    data_uri = f"data:{mime};base64,{base64.b64encode(compressed_data).decode()}"

    reduction = ((original_size - compressed_size) / original_size * 100) if original_size > 0 else 0.0

    return CompressionResult(
        original_size=original_size,
        compressed_size=compressed_size,
        reduction_pct=round(reduction, 1),
        format=target_format,
        width=new_w,
        height=new_h,
        strategy="+".join(strategy_parts) if strategy_parts else "none",
        data=compressed_data,
        text=extracted_text,
        data_uri=data_uri,
    )


def _encode(img, fmt: str, quality: int) -> bytes:
    buf = io.BytesIO()
    if fmt in ("jpeg", "jpg"):
        out = img.convert("RGB") if img.mode == "RGBA" else img
        out.save(buf, format="JPEG", quality=quality, optimize=True)
    elif fmt == "webp":
        img.save(buf, format="WebP", quality=quality, method=4)
    else:
        img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _is_diagram(img) -> bool:
    """Heuristic: low colour diversity implies diagram/screenshot, not photo."""
    if img.mode == "RGBA":
        return True

    from PIL import Image
    small = img.resize((32, 32), Image.NEAREST)
    pixels = list(small.getdata())
    if not pixels:
        return False

    unique = len(set(pixels))
    return unique < 200


def _extract_text(img) -> str | None:
    """Extract text from image using available OCR."""
    try:
        import pytesseract
        text = pytesseract.image_to_string(img)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    try:
        import easyocr
        import numpy as np
        reader = easyocr.Reader(["en"], verbose=False)
        results = reader.readtext(np.array(img))
        if results:
            return "\n".join(r[1] for r in results)
    except Exception:
        pass

    return None


def estimate_vision_tokens(width: int, height: int, detail: str = "auto") -> int:
    """Estimate vision API token cost based on tile-based pricing."""
    if detail == "low":
        return 85

    if max(width, height) > 2048:
        ratio = 2048 / max(width, height)
        width = int(width * ratio)
        height = int(height * ratio)

    if min(width, height) > 768:
        ratio = 768 / min(width, height)
        width = int(width * ratio)
        height = int(height * ratio)

    tiles_w = (width + 511) // 512
    tiles_h = (height + 511) // 512
    return 85 + 170 * tiles_w * tiles_h
