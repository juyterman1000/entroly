"""Verified source overlay for externally observed runtime events."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from .models import RepositoryIndex, normalize_relative

RUNTIME_OVERLAY_SCHEMA_VERSION = "entroly.verified-runtime-overlay.v1"
_ALLOWED_EVENTS = frozenset({"call", "return", "line", "exception", "covered"})


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


def _line_offsets(raw: bytes) -> list[int]:
    offsets = [0]
    for line in raw.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    if offsets[-1] < len(raw):
        offsets.append(len(raw))
    return offsets


def _finish(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt["runtime_overlay_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def build_verified_runtime_overlay(
    root: Path,
    index: RepositoryIndex,
    events: Iterable[Mapping[str, object]],
    *,
    index_digest: str,
    producer: str = "external-trace",
    max_events: int = 100_000,
) -> dict[str, object]:
    """Bind bounded external events to fresh repository source evidence.

    Event values are deliberately excluded. Only path, line, event kind, and
    count cross the boundary, limiting accidental secret capture from traces.
    """
    root = root.expanduser().resolve(strict=True)
    event_limit = max(1, min(int(max_events), 1_000_000))
    source_cache: dict[str, tuple[bytes | None, str, list[int]]] = {}
    aggregated: Counter[tuple[str, int, str, str | None, int, int, str]] = Counter()
    omissions: Counter[str] = Counter()
    accepted_input_count = 0

    def source(path: str) -> tuple[bytes | None, str, list[int]]:
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
                result = (
                    (raw, "verified", _line_offsets(raw))
                    if hashlib.sha256(raw).hexdigest() == record.sha256
                    else (None, "stale-index", [])
                )
        source_cache[path] = result
        return result

    for position, raw_event in enumerate(events):
        if position >= event_limit:
            omissions["event-limit"] += 1
            break
        if not isinstance(raw_event, Mapping):
            omissions["invalid-event"] += 1
            continue
        path = _strict_path(raw_event.get("path"))
        event = str(raw_event.get("event", "line")).strip().lower()
        try:
            line = int(raw_event.get("line", 0))
            count = int(raw_event.get("count", 1))
        except (TypeError, ValueError):
            omissions["invalid-event"] += 1
            continue
        if path is None or event not in _ALLOWED_EVENTS or line <= 0 or not 1 <= count <= 1_000_000_000:
            omissions["invalid-event"] += 1
            continue
        content, status, offsets = source(path)
        if content is None:
            omissions[status] += 1
            continue
        record = index.files[path]
        if line > record.line_count or line >= len(offsets):
            omissions["line-out-of-range"] += 1
            continue
        start = offsets[line - 1]
        end = offsets[line]
        evidence = content[start:end]
        enclosing = sorted(
            (
                symbol for symbol in index.symbols_for_path(path)
                if symbol.line_start <= line <= symbol.line_end
            ),
            key=lambda symbol: (
                symbol.line_end - symbol.line_start,
                symbol.symbol_id,
            ),
        )
        symbol_id = enclosing[0].symbol_id if enclosing else None
        evidence_sha256 = hashlib.sha256(evidence).hexdigest()
        aggregated[(path, line, event, symbol_id, start, end, evidence_sha256)] += count
        accepted_input_count += 1

    observations: list[dict[str, object]] = []
    hot_symbols: Counter[str] = Counter()
    for key, count in sorted(aggregated.items()):
        path, line, event, symbol_id, start, end, evidence_sha256 = key
        observation_id = hashlib.sha256(
            f"{path}\0{line}\0{event}\0{symbol_id or ''}\0{evidence_sha256}".encode("utf-8")
        ).hexdigest()
        observations.append({
            "observation_id": observation_id,
            "path": path,
            "line": line,
            "event": event,
            "count": count,
            "symbol_id": symbol_id,
            "start_byte": start,
            "end_byte": end,
            "evidence_sha256": evidence_sha256,
            "source_sha256": index.files[path].sha256,
            "trust": "untrusted-runtime-event-verified-source-location",
        })
        if symbol_id:
            hot_symbols[symbol_id] += count

    clean_producer = str(producer).strip()[:128] or "external-trace"
    payload: dict[str, object] = {
        "schema_version": RUNTIME_OVERLAY_SCHEMA_VERSION,
        "index_digest": index_digest,
        "producer": clean_producer,
        "observations": observations,
        "hot_symbols": [
            {"symbol_id": symbol_id, "count": count}
            for symbol_id, count in sorted(
                hot_symbols.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "receipt": {
            "freshness": "verified-against-indexed-source-sha256",
            "input_event_count": accepted_input_count + sum(omissions.values()),
            "accepted_event_count": accepted_input_count,
            "aggregated_observation_count": len(observations),
            "omissions_by_reason": dict(sorted(omissions.items())),
            "event_values_collected": False,
            "remote_calls": 0,
            "commitment_scope": "payload-excluding-generation-command-and-runtime-overlay-sha256",
        },
    }
    return _finish(payload)


def verify_runtime_overlay_commitment(payload: dict[str, object]) -> bool:
    """Verify a runtime-overlay receipt without workspace access."""
    try:
        candidate = copy.deepcopy(payload)
        candidate.pop("generation", None)
        candidate.pop("command", None)
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("runtime_overlay_sha256"))
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
    "RUNTIME_OVERLAY_SCHEMA_VERSION",
    "build_verified_runtime_overlay",
    "verify_runtime_overlay_commitment",
]
