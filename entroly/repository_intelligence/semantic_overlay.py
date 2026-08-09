"""Verified source binding for external LSP/compiler semantic relationships."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from .models import RepositoryIndex, normalize_relative

SEMANTIC_OVERLAY_SCHEMA_VERSION = "entroly.verified-semantic-overlay.v1"
_ALLOWED_KINDS = frozenset({
    "definition",
    "declaration",
    "implementation",
    "reference",
    "type-definition",
    "override",
})


def _strict_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.replace("\\", "/")
    if not raw or "\x00" in raw or raw.startswith(("/", "//")):
        return None
    if len(raw) >= 2 and raw[1] == ":":
        return None
    if any(part == ".." for part in raw.split("/")):
        return None
    return normalize_relative(raw) or None


def _utf16_to_byte(text: str, character: int) -> int | None:
    if character < 0:
        return None
    units = 0
    byte_offset = 0
    for value in text:
        if units == character:
            return byte_offset
        width = 2 if ord(value) > 0xFFFF else 1
        if units + width > character:
            return None
        units += width
        byte_offset += len(value.encode("utf-8", errors="surrogateescape"))
    return byte_offset if units == character else None


def _finish(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt["semantic_overlay_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def build_verified_semantic_overlay(
    root: Path,
    index: RepositoryIndex,
    relationships: Iterable[Mapping[str, object]],
    *,
    index_digest: str,
    provider: str,
    max_relationships: int = 100_000,
) -> dict[str, object]:
    """Verify LSP-style UTF-16 ranges without trusting compiler output blindly."""
    root = root.expanduser().resolve(strict=True)
    limit = max(1, min(int(max_relationships), 1_000_000))
    source_cache: dict[str, tuple[bytes | None, str, list[tuple[int, int, str]]]] = {}
    omissions: Counter[str] = Counter()
    edges: list[dict[str, object]] = []

    def source(path: str) -> tuple[bytes | None, str, list[tuple[int, int, str]]]:
        cached = source_cache.get(path)
        if cached is not None:
            return cached
        record = index.files.get(path)
        if record is None:
            result = (None, "unknown-path", [])
        else:
            try:
                candidate = (root / path).resolve(strict=True)
                candidate.relative_to(root)
                raw = candidate.read_bytes()
            except (OSError, RuntimeError, ValueError):
                result = (None, "unsafe-or-unreadable", [])
            else:
                if hashlib.sha256(raw).hexdigest() != record.sha256:
                    result = (None, "stale-index", [])
                else:
                    lines: list[tuple[int, int, str]] = []
                    offset = 0
                    for raw_line in raw.splitlines(keepends=True):
                        clean = raw_line.rstrip(b"\r\n")
                        lines.append((offset, offset + len(clean), clean.decode(
                            "utf-8", errors="surrogateescape"
                        )))
                        offset += len(raw_line)
                    if not lines and not raw:
                        lines.append((0, 0, ""))
                    result = (raw, "verified", lines)
        source_cache[path] = result
        return result

    def location(value: object) -> tuple[dict[str, object] | None, str]:
        if not isinstance(value, Mapping):
            return None, "invalid-location"
        path = _strict_path(value.get("path"))
        try:
            line = int(value.get("line", -1))
            start_character = int(value.get("start_character", -1))
            end_character = int(value.get("end_character", -1))
        except (TypeError, ValueError):
            return None, "invalid-location"
        if path is None or line < 0 or end_character <= start_character:
            return None, "invalid-location"
        raw, status, lines = source(path)
        if raw is None:
            return None, status
        if line >= len(lines):
            return None, "line-out-of-range"
        line_start, _line_end, text = lines[line]
        local_start = _utf16_to_byte(text, start_character)
        local_end = _utf16_to_byte(text, end_character)
        if local_start is None or local_end is None or local_end <= local_start:
            return None, "character-out-of-range"
        start = line_start + local_start
        end = line_start + local_end
        evidence = raw[start:end]
        enclosing = sorted(
            (
                symbol for symbol in index.symbols_for_path(path)
                if symbol.line_start <= line + 1 <= symbol.line_end
            ),
            key=lambda symbol: (symbol.line_end - symbol.line_start, symbol.symbol_id),
        )
        return {
            "path": path,
            "line": line,
            "start_character": start_character,
            "end_character": end_character,
            "start_byte": start,
            "end_byte": end,
            "evidence_sha256": hashlib.sha256(evidence).hexdigest(),
            "source_sha256": index.files[path].sha256,
            "symbol_id": enclosing[0].symbol_id if enclosing else None,
            "trust": "verified-source-range",
        }, "verified"

    for position, relationship in enumerate(relationships):
        if position >= limit:
            omissions["relationship-limit"] += 1
            break
        if not isinstance(relationship, Mapping):
            omissions["invalid-relationship"] += 1
            continue
        kind = str(relationship.get("kind", "")).strip().lower().replace("_", "-")
        if kind not in _ALLOWED_KINDS:
            omissions["invalid-kind"] += 1
            continue
        source_location, source_status = location(relationship.get("source"))
        target_location, target_status = location(relationship.get("target"))
        if source_location is None or target_location is None:
            omissions[
                f"source-{source_status}" if source_location is None else f"target-{target_status}"
            ] += 1
            continue
        edge_id = hashlib.sha256(json.dumps(
            [kind, source_location, target_location],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        edges.append({
            "edge_id": edge_id,
            "kind": kind,
            "source": source_location,
            "target": target_location,
            "confidence": "externally-reported-source-verified",
        })
    edges.sort(key=lambda item: str(item["edge_id"]))
    clean_provider = str(provider).strip()[:128] or "unspecified"
    payload: dict[str, object] = {
        "schema_version": SEMANTIC_OVERLAY_SCHEMA_VERSION,
        "index_digest": index_digest,
        "provider": clean_provider,
        "provider_trust": "untrusted-external-semantic-provider",
        "position_encoding": "utf-16",
        "relationships": edges,
        "receipt": {
            "freshness": "verified-against-indexed-source-sha256",
            "accepted_relationship_count": len(edges),
            "omissions_by_reason": dict(sorted(omissions.items())),
            "remote_calls": 0,
            "commitment_scope": "payload-excluding-generation-command-and-semantic-overlay-sha256",
        },
    }
    return _finish(payload)


def verify_semantic_overlay_commitment(payload: dict[str, object]) -> bool:
    """Verify an external semantic-overlay receipt without workspace access."""
    try:
        candidate = copy.deepcopy(payload)
        candidate.pop("generation", None)
        candidate.pop("command", None)
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("semantic_overlay_sha256"))
        canonical = json.dumps(
            candidate,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "SEMANTIC_OVERLAY_SCHEMA_VERSION",
    "build_verified_semantic_overlay",
    "verify_semantic_overlay_commitment",
]
