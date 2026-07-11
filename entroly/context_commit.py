"""Portable, content-addressed commits for the context an agent received."""

from __future__ import annotations

import copy
import importlib
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any

from .context_receipts import ingest_documents, recover_omitted, select_from_index
from .context_receipts.models import stable_hash, text_fingerprint
from .context_receipts.recover import build_recovery_bundle

CONTEXT_COMMIT_SCHEMA = "entroly.context-commit.v1"


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "source"


def _rust_receipts_available() -> bool:
    try:
        core = importlib.import_module("entroly_core")
    except Exception:
        return False
    return all(
        hasattr(core, symbol)
        for symbol in ("context_receipts_ingest", "context_receipts_select")
    )


def _engine_identity(*, prefer_rust: bool) -> dict[str, str]:
    use_rust = prefer_rust and _rust_receipts_available()
    return {
        "implementation": "rust" if use_rust else "python",
        "entroly_version": _package_version("entroly"),
        "engine_version": _package_version("entroly-core") if use_rust else "python",
    }


def _selected_context_digest(receipt: Mapping[str, Any]) -> str:
    selected = receipt.get("selected_context", [])
    canonical = []
    if isinstance(selected, list):
        for item in selected:
            if isinstance(item, Mapping):
                canonical.append(
                    {
                        "chunk_id": str(item.get("chunk_id", "")),
                        "fingerprint": str(item.get("fingerprint", "")),
                        "source_path": str(item.get("source_path", "")),
                        "text": str(item.get("text", "")),
                    }
                )
    return stable_hash(canonical)


def _commit_payload(commit: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(commit))
    payload.pop("commit_id", None)
    return payload


@dataclass(frozen=True)
class ContextCommitVerification:
    valid: bool
    checks: dict[str, bool]
    errors: tuple[str, ...]
    selected_chunks: int
    omitted_chunks: int
    recovered_omitted_chunks: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_context_commit(
    documents: Iterable[tuple[str, str]],
    *,
    query: str,
    token_budget: int,
    chunk_tokens: int = 360,
    overlap_tokens: int = 32,
    parent_commit_id: str | None = None,
    prefer_rust: bool = True,
) -> dict[str, Any]:
    """Create a self-contained commit for selected and omitted context."""

    docs = [(str(path), str(text)) for path, text in documents]
    index = ingest_documents(
        docs,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        prefer_rust=prefer_rust,
    )
    receipt = select_from_index(
        index,
        query=query,
        token_budget=token_budget,
        prefer_rust=prefer_rust,
    )
    recovery = build_recovery_bundle(index)
    payload: dict[str, Any] = {
        "schema_version": CONTEXT_COMMIT_SCHEMA,
        "parent_commit_id": parent_commit_id,
        "engine": _engine_identity(prefer_rust=prefer_rust),
        "receipt": receipt,
        "selected_context_digest": _selected_context_digest(receipt),
        "recovery_bundle_digest": stable_hash(recovery),
        "recovery_bundle": recovery,
    }
    return {"commit_id": "ctx_" + stable_hash(payload)[:24], **payload}


def replay_context(commit: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return an exact, ordered copy of the context selected for the model."""

    receipt = commit.get("receipt", {})
    if not isinstance(receipt, Mapping):
        return []
    selected = receipt.get("selected_context", [])
    if not isinstance(selected, list):
        return []
    return [copy.deepcopy(dict(item)) for item in selected if isinstance(item, Mapping)]


def verify_context_commit(commit: Mapping[str, Any] | object) -> ContextCommitVerification:
    """Fail closed unless the commit, receipt, replay, and recovery all verify."""

    if not isinstance(commit, Mapping):
        return ContextCommitVerification(
            False, {}, ("commit is not a mapping",), 0, 0, 0
        )

    data = dict(commit)
    receipt = data.get("receipt", {})
    recovery = data.get("recovery_bundle", {})
    receipt_map = receipt if isinstance(receipt, Mapping) else {}
    recovery_map = recovery if isinstance(recovery, Mapping) else {}
    selected = replay_context(data)
    omitted_raw = receipt_map.get("omitted_context", [])
    omitted = omitted_raw if isinstance(omitted_raw, list) else []

    expected_commit_id = "ctx_" + stable_hash(_commit_payload(data))[:24]
    receipt_payload = {
        key: value
        for key, value in receipt_map.items()
        if key not in {"receipt_id", "reproducibility_hash"}
    }
    expected_receipt_hash = stable_hash(receipt_payload)
    expected_receipt_id = "cr_" + expected_receipt_hash[:12]

    chunks_raw = recovery_map.get("chunks", {})
    chunks = chunks_raw if isinstance(chunks_raw, Mapping) else {}
    chunk_integrity = True
    for entry in chunks.values():
        if not isinstance(entry, Mapping):
            chunk_integrity = False
            break
        text = str(entry.get("text", ""))
        if text_fingerprint(text) != str(entry.get("content_sha", "")):
            chunk_integrity = False
            break

    selected_integrity = True
    for item in selected:
        chunk_id = str(item.get("chunk_id", ""))
        entry = chunks.get(chunk_id)
        if not isinstance(entry, Mapping):
            selected_integrity = False
            break
        if (
            str(entry.get("text", "")) != str(item.get("text", ""))
            or str(entry.get("fingerprint", ""))
            != str(item.get("fingerprint", ""))
        ):
            selected_integrity = False
            break

    recovered = recover_omitted(dict(receipt_map), bundle=dict(recovery_map))
    recovered_count = sum(bool(item.get("verified")) for item in recovered)
    omitted_integrity = recovered_count == len(omitted)

    checks = {
        "schema": data.get("schema_version") == CONTEXT_COMMIT_SCHEMA,
        "commit_id": data.get("commit_id") == expected_commit_id,
        "receipt_hash": receipt_map.get("reproducibility_hash")
        == expected_receipt_hash,
        "receipt_id": receipt_map.get("receipt_id") == expected_receipt_id,
        "selected_context_digest": data.get("selected_context_digest")
        == _selected_context_digest(receipt_map),
        "recovery_bundle_digest": data.get("recovery_bundle_digest")
        == stable_hash(recovery_map),
        "recovery_chunk_integrity": chunk_integrity,
        "selected_context_integrity": selected_integrity,
        "omitted_context_recovery": omitted_integrity,
        "parent_lineage": data.get("parent_commit_id") is None
        or str(data.get("parent_commit_id", "")).startswith("ctx_"),
    }
    errors = tuple(name for name, passed in checks.items() if not passed)
    return ContextCommitVerification(
        valid=not errors,
        checks=checks,
        errors=errors,
        selected_chunks=len(selected),
        omitted_chunks=len(omitted),
        recovered_omitted_chunks=recovered_count,
    )


__all__ = [
    "CONTEXT_COMMIT_SCHEMA",
    "ContextCommitVerification",
    "create_context_commit",
    "replay_context",
    "verify_context_commit",
]
