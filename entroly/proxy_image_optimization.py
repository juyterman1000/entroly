"""Opt-in, recoverable optimization of inline provider images."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .image_optimizer import decode_base64_image, optimize_image_bytes


SCHEMA_VERSION = "entroly.image-transform-receipt.v1"
_DATA_URL = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$", re.DOTALL)


class ImageTransformError(RuntimeError):
    """An opted-in image request could not be transformed recoverably."""


@dataclass(frozen=True)
class ImageTransformReceipt:
    receipt_id: str
    provider: str
    media_type: str
    source_sha256: str
    optimized_sha256: str
    source_bytes: int
    optimized_bytes: int
    before_tokens: int
    after_tokens: int
    estimation_method: str
    original_object: str


@dataclass(frozen=True)
class ImageRequestResult:
    body: dict[str, Any]
    receipts: tuple[ImageTransformReceipt, ...]

    @property
    def changed(self) -> bool:
        return bool(self.receipts)

    def headers(self) -> dict[str, str]:
        return {
            "X-Entroly-Image-Optimization": "changed" if self.changed else "preserved",
            "X-Entroly-Image-Receipt-Count": str(len(self.receipts)),
            "X-Entroly-Image-Receipts": ",".join(
                receipt.receipt_id for receipt in self.receipts[:8]
            ),
        }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class ImageRecoveryStore:
    """Content-addressed local originals plus digest-verifiable receipts."""

    def __init__(self, root: Path, *, max_image_bytes: int = 20 * 1024 * 1024) -> None:
        self.root = Path(root).expanduser().resolve()
        self.max_image_bytes = max(1, int(max_image_bytes))
        self.objects = self.root / "objects"
        self.receipts = self.root / "receipts"
        self._lock = threading.Lock()
        self.objects.mkdir(parents=True, exist_ok=True)
        self.receipts.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        original: bytes,
        optimized: bytes,
        *,
        provider: str,
        media_type: str,
        before_tokens: int,
        after_tokens: int,
        estimation_method: str,
    ) -> ImageTransformReceipt:
        if len(original) > self.max_image_bytes:
            raise ImageTransformError(
                f"inline image exceeds recovery limit ({len(original)} > {self.max_image_bytes})"
            )
        source_sha = _sha256(original)
        optimized_sha = _sha256(optimized)
        identity = hashlib.sha256(
            f"{source_sha}:{optimized_sha}:{provider}:{media_type}".encode("utf-8")
        ).hexdigest()[:24]
        receipt_id = f"img:{identity}"
        object_path = self.objects / source_sha[:2] / source_sha
        receipt_path = self.receipts / f"{identity}.json"
        receipt = ImageTransformReceipt(
            receipt_id=receipt_id,
            provider=provider,
            media_type=media_type,
            source_sha256=source_sha,
            optimized_sha256=optimized_sha,
            source_bytes=len(original),
            optimized_bytes=len(optimized),
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            estimation_method=estimation_method,
            original_object=str(object_path),
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            **asdict(receipt),
            "created_at_unix": int(time.time()),
        }
        with self._lock:
            if not object_path.exists():
                _atomic_write(object_path, original)
            elif _sha256(object_path.read_bytes()) != source_sha:
                raise ImageTransformError("content-addressed image object failed verification")
            _atomic_write(
                receipt_path,
                (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
        return receipt

    def recover(self, receipt_id: str) -> tuple[bytes, ImageTransformReceipt]:
        """Return the verified original and its typed receipt."""
        if not re.fullmatch(r"img:[0-9a-f]{24}", receipt_id):
            raise ImageTransformError("invalid image receipt id")
        receipt_path = self.receipts / f"{receipt_id[4:]}.json"
        try:
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unsupported image receipt schema")
            object_path = Path(payload["original_object"]).resolve()
            object_path.relative_to(self.objects.resolve())
            data = object_path.read_bytes()
            receipt = ImageTransformReceipt(
                **{field: payload[field] for field in ImageTransformReceipt.__dataclass_fields__}
            )
        except (FileNotFoundError, OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ImageTransformError(f"image receipt is unavailable: {receipt_id}") from exc
        if _sha256(data) != payload.get("source_sha256"):
            raise ImageTransformError("recovered image digest does not match its receipt")
        return data, receipt

    def retrieve(self, receipt_id: str) -> bytes:
        return self.recover(receipt_id)[0]


def _candidate(node: dict[str, Any]) -> tuple[str, str, str] | None:
    block_type = str(node.get("type", "")).casefold()
    if block_type in {"image_url", "input_image"}:
        image = node.get("image_url", node.get("url"))
        if isinstance(image, Mapping):
            image = image.get("url")
        if isinstance(image, str) and _DATA_URL.match(image):
            return "data_url", "image_url" if "image_url" in node else "url", image
    if block_type == "image" and isinstance(node.get("source"), Mapping):
        source = node["source"]
        if str(source.get("type", "")).casefold() == "base64" and isinstance(source.get("data"), str):
            return "anthropic", "source", str(source["data"])
    for key in ("inline_data", "inlineData"):
        inline = node.get(key)
        if isinstance(inline, Mapping) and isinstance(inline.get("data"), str):
            return "gemini", key, str(inline["data"])
    return None


def _replace_candidate(
    node: dict[str, Any],
    candidate: tuple[str, str, str],
    encoded: str,
    media_type: str,
) -> None:
    kind, key, original = candidate
    if kind == "data_url":
        match = _DATA_URL.match(original)
        assert match is not None
        replacement = f"data:{media_type};base64,{encoded}"
        if isinstance(node.get(key), Mapping):
            node[key] = dict(node[key])
            node[key]["url"] = replacement
        else:
            node[key] = replacement
    elif kind == "anthropic":
        node["source"] = dict(node["source"])
        node["source"]["data"] = encoded
        node["source"]["media_type"] = media_type
    else:
        node[key] = dict(node[key])
        node[key]["data"] = encoded
        mime_key = "mimeType" if "mimeType" in node[key] else "mime_type"
        node[key][mime_key] = media_type


def _candidate_media_type(node: Mapping[str, Any], candidate: tuple[str, str, str]) -> str:
    kind, key, value = candidate
    if kind == "data_url":
        match = _DATA_URL.match(value)
        return match.group(1).casefold() if match else "image/png"
    if kind == "anthropic":
        return str(node["source"].get("media_type", "image/png")).casefold()
    inline = node[key]
    return str(inline.get("mime_type", inline.get("mimeType", "image/png"))).casefold()


def optimize_inline_images(
    body: Mapping[str, Any],
    *,
    provider: str,
    model: str,
    store: ImageRecoveryStore,
    enabled: bool = False,
    min_quality_ratio: float = 0.72,
) -> ImageRequestResult:
    """Optimize supported base64 image blocks only after recoverability is proven."""
    if not enabled:
        return ImageRequestResult(copy.deepcopy(dict(body)), ())
    transformed = copy.deepcopy(dict(body))
    receipts: list[ImageTransformReceipt] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        candidate = _candidate(value)
        if candidate is not None:
            media_type = _candidate_media_type(value, candidate)
            try:
                original = decode_base64_image(candidate[2])
            except (ValueError, TypeError) as exc:
                raise ImageTransformError("invalid inline base64 image") from exc
            optimized, decision = optimize_image_bytes(
                original,
                provider=provider if provider in {"openai", "anthropic", "gemini"} else "unknown",
                model=model,
                enabled=True,
                min_quality_ratio=min_quality_ratio,
            )
            if optimized != original and decision.after is not None:
                receipt = store.store(
                    original,
                    optimized,
                    provider=provider,
                    media_type=media_type,
                    before_tokens=decision.before.estimated_tokens,
                    after_tokens=decision.after.estimated_tokens,
                    estimation_method=decision.after.method,
                )
                _replace_candidate(
                    value,
                    candidate,
                    base64.b64encode(optimized).decode("ascii"),
                    media_type,
                )
                receipts.append(receipt)
            return
        for child in value.values():
            visit(child)

    visit(transformed)
    return ImageRequestResult(transformed, tuple(receipts))


__all__ = [
    "ImageRecoveryStore",
    "ImageRequestResult",
    "ImageTransformError",
    "ImageTransformReceipt",
    "optimize_inline_images",
]
