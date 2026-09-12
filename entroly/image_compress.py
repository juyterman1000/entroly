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
    """
    Compress an image for LLM context.

    Args:
        source: File path or raw bytes
        max_dimension: Maximum width or height in pixels
        quality: JPEG/WebP quality (1-100)
        target_format: Force output format (webp, jpeg, png)
        extract_text: If True, attempt OCR text extraction
        max_bytes: Target maximum file size in bytes

    Returns:
        CompressionResult with compressed data and metadata
    """
    if not _has_pillow():
        raise RuntimeError(
            "Image compression requires Pillow. Install with: "
            "pip install 'entroly[full]'  # or:  pip install Pillow"
        )

    from PIL import Image

    # Load image
    if isinstance(source, (str, Path)):
        path = Path(source)
        original_bytes = path.read_bytes()
        img = Image.open(path)
    else:
        original_bytes = source
        img = Image.open(io.BytesIO(source))

    original_size = len(original_bytes)
    orig_w, orig_h = img.size
    strategy_parts = []

    # Strategy 1: Resize if larger than max_dimension
    if max(orig_w, orig_h) > max_dimension:
        ratio = max_dimension / max(orig_w, orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        strategy_parts.append("resize")
    else:
        new_w, new_h = orig_w, orig_h

    # Strategy 2: Format selection
    if target_format is None:
        if img.mode == "RGBA" or _is_diagram(img):
            target_format = "png"
        else:
            target_format = "webp"
    strategy_parts.append("convert")

    # Compress
    buf = io.BytesIO()

    if target_format in ("jpeg", "jpg"):
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    elif target_format == "webp":
        img.save(buf, format="WebP", quality=quality, method=4)
    else:
        img.save(buf, format="PNG", optimize=True)

    compressed_data = buf.getvalue()

    # If still too large, reduce quality iteratively
    if max_bytes and len(compressed_data) > max_bytes:
        for q in range(quality - 10, 10, -10):
            buf = io.BytesIO()
            if target_format in ("jpeg", "jpg"):
                img.save(buf, format="JPEG", quality=q, optimize=True)
            elif target_format == "webp":
                img.save(buf, format="WebP", quality=q, method=4)
            else:
                break
            compressed_data = buf.getvalue()
            if len(compressed_data) <= max_bytes:
                break

    compressed_size = len(compressed_data)

    # Strategy 3: OCR text extraction (optional)
    extracted_text = None
    if extract_text:
        extracted_text = _extract_text(img)
        if extracted_text and len(extracted_text) > 20:
            strategy_parts.append("ocr")

    # Build data URI for embedding
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


def _is_diagram(img) -> bool:
    """Heuristic: detect if image is a diagram/screenshot vs photograph."""
    from PIL import Image

    if img.mode == "RGBA":
        return True

    # Sample pixels and check color diversity
    small = img.resize((32, 32), Image.NEAREST)
    pixels = list(small.getdata())
    if not pixels:
        return False

    if isinstance(pixels[0], tuple):
        unique = len(set(pixels))
    else:
        unique = len(set(pixels))

    # Diagrams tend to have few unique colors (<100 in 32x32 = 1024 pixels)
    return unique < 200


def _extract_text(img) -> str | None:
    """Extract text from image using available OCR."""
    # Try pytesseract if available
    try:
        import pytesseract
        text = pytesseract.image_to_string(img)
        if text and text.strip():
            return text.strip()
    except (ImportError, Exception):
        pass

    # Try easyocr if available
    try:
        import easyocr
        import numpy as np
        reader = easyocr.Reader(["en"], verbose=False)
        results = reader.readtext(np.array(img))
        if results:
            return "\n".join(r[1] for r in results)
    except (ImportError, Exception):
        pass

    return None


def estimate_vision_tokens(width: int, height: int, detail: str = "auto") -> int:
    """
    Estimate vision API token cost for an image.

    Based on OpenAI's vision pricing model:
    - low detail: 85 tokens fixed
    - high detail: 85 base + 170 per 512x512 tile
    """
    if detail == "low":
        return 85

    # Scale to fit within 2048x2048
    if max(width, height) > 2048:
        ratio = 2048 / max(width, height)
        width = int(width * ratio)
        height = int(height * ratio)

    # Scale shortest side to 768
    if min(width, height) > 768:
        ratio = 768 / min(width, height)
        width = int(width * ratio)
        height = int(height * ratio)

    # Count 512x512 tiles
    tiles_w = (width + 511) // 512
    tiles_h = (height + 511) // 512
    total_tiles = tiles_w * tiles_h

    return 85 + 170 * total_tiles
