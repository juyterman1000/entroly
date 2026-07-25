"""Capture an ordered QCCR selection from a runtime-bound frozen corpus.

Usage:
    python capture_selection.py <frozen_corpus.json> <budget> <query...>

The command refuses to run when the installed Entroly package or native module
does not exactly match the runtime identity stored by ``freeze_corpus.py``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

CORPUS_SCHEMA = "entroly.research.frozen-corpus.v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def selection_digest(order: list[dict]) -> str:
    """Digest every identity-bearing field of the ordered selection contract."""
    blob = json.dumps(
        [
            (
                item["rank"],
                item["source"],
                item["content_sha"],
                item["content_len"],
                item["source_fragment_ids"],
            )
            for item in order
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def canonical(selected: list[dict]) -> dict:
    order: list[dict] = []
    for rank, fragment in enumerate(selected):
        content = fragment.get("content") or ""
        source = fragment.get("source") or ""
        raw_origin_ids = fragment.get("source_fragment_ids") or []
        if not isinstance(content, str) or not isinstance(source, str):
            raise ValueError("selected source and content must be strings")
        if (
            not isinstance(raw_origin_ids, (list, tuple))
            or any(not isinstance(item, str) or not item for item in raw_origin_ids)
        ):
            raise ValueError("selected source_fragment_ids must be a list of strings")
        order.append(
            {
                "rank": rank,
                "source": source,
                "content_sha": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "content_len": len(content.encode("utf-8")),
                "source_fragment_ids": list(raw_origin_ids),
            }
        )
    return {
        "digest": selection_digest(order),
        "n": len(order),
        "order": order,
    }


def _validate_artifact(artifact: object) -> tuple[list[dict], dict]:
    if not isinstance(artifact, dict) or artifact.get("schema_version") != CORPUS_SCHEMA:
        raise ValueError(f"frozen corpus must use {CORPUS_SCHEMA}")
    fragments = artifact.get("fragments")
    metadata = artifact.get("metadata")
    if not isinstance(fragments, list) or not isinstance(metadata, dict):
        raise ValueError("frozen corpus fragments and metadata must be present")
    required_strings = (
        "source_commit",
        "source_tree",
        "entroly_version",
        "native_version",
        "native_module_sha256",
        "fragments_sha256",
    )
    if any(not isinstance(metadata.get(key), str) or not metadata[key] for key in required_strings):
        raise ValueError("frozen corpus runtime/source identity is incomplete")
    if not _COMMIT_RE.fullmatch(metadata["source_commit"]):
        raise ValueError("frozen corpus source_commit is invalid")
    if not _COMMIT_RE.fullmatch(metadata["source_tree"]):
        raise ValueError("frozen corpus source_tree is invalid")
    if not _SHA256_RE.fullmatch(metadata["native_module_sha256"]):
        raise ValueError("frozen corpus native_module_sha256 is invalid")
    if not _SHA256_RE.fullmatch(metadata["fragments_sha256"]):
        raise ValueError("frozen corpus fragments_sha256 is invalid")
    fragment_count = metadata.get("fragment_count")
    if (
        not isinstance(fragment_count, int)
        or isinstance(fragment_count, bool)
        or fragment_count != len(fragments)
    ):
        raise ValueError("frozen corpus fragment_count mismatch")
    canonical_fragments = json.dumps(
        fragments,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(canonical_fragments).hexdigest() != metadata["fragments_sha256"]:
        raise ValueError("frozen corpus fragment digest mismatch")
    return fragments, metadata


def _verify_runtime(metadata: dict) -> None:
    from entroly import __version__ as entroly_version
    from entroly.native_status import (
        QCCR_SYMBOLS,
        native_status,
        native_status_message,
    )

    status = native_status(QCCR_SYMBOLS)
    if not status.ok or not status.path or not status.version:
        raise RuntimeError(
            native_status_message(status, feature="reproducibility capture")
        )
    actual = {
        "entroly_version": entroly_version,
        "native_version": status.version,
        "native_module_sha256": _sha256_file(status.path),
    }
    mismatches = [
        f"{key}: expected {metadata[key]!r}, got {value!r}"
        for key, value in actual.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "runtime does not match frozen corpus identity: " + "; ".join(mismatches)
        )


def main() -> int:
    if len(sys.argv) < 4:
        raise ValueError("usage: capture_selection.py <corpus> <positive-budget> <query>")
    frozen_path = sys.argv[1]
    budget = int(sys.argv[2])
    query = " ".join(sys.argv[3:])
    if budget <= 0 or not query.strip():
        raise ValueError("budget must be positive and query must be non-empty")
    with open(frozen_path, encoding="utf-8") as source:
        artifact = json.load(source)
    fragments, metadata = _validate_artifact(artifact)
    _verify_runtime(metadata)

    from entroly import qccr

    selected = qccr.select(fragments, budget, query)
    print(json.dumps(canonical(selected), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
