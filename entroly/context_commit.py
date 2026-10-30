"""Portable, self-consistent commits for the context an agent received.

A commit's own hashes do not authenticate its issuer. A trusted caller can pin
``context_commit_anchor`` separately before handing the commit to another actor.
"""

from __future__ import annotations

import copy
import hmac
import importlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any, TYPE_CHECKING, Sequence

from .context_receipts import ingest_documents, recover_omitted, select_from_index
from .context_receipts.models import byte_digest, stable_hash, text_fingerprint
from .context_receipts.recover import build_recovery_bundle

CONTEXT_COMMIT_SCHEMA = "entroly.context-commit.v1"

if TYPE_CHECKING:
    from .context_assurance import AssuranceScope, HardObligation


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


def context_commit_anchor(commit: Mapping[str, Any]) -> str:
    """Full SHA-256 anchor to pin in trusted state when the commit is issued.

    Computing this from an untrusted commit at recovery time supplies no trust.
    The caller must retain the issued value independently of portable evidence.
    """
    if not isinstance(commit, Mapping):
        raise TypeError("context commit must be a mapping")
    # The payload includes CONTEXT_COMMIT_SCHEMA. Use the same full digest that
    # verification already computes for the commit_id's 24-hex prefix.
    return "sha256:" + stable_hash(_commit_payload(commit))


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
    assurance_scope: AssuranceScope | None = None,
    mandatory_ids: Sequence[str] = (),
    obligations: Sequence[HardObligation] = (),
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
    if assurance_scope is not None or mandatory_ids or obligations:
        from .context_assurance import audit_receipt, bind_assurance_scope

        certificate = audit_receipt(index, receipt, mandatory_ids=mandatory_ids,
                                    obligations=obligations)
        if assurance_scope is not None:
            certificate = bind_assurance_scope(certificate, assurance_scope)
        receipt["risk_summary"]["context_assurance"] = certificate
        receipt_hash = stable_hash({k: v for k, v in receipt.items()
                                    if k not in {"receipt_id", "reproducibility_hash"}})
        receipt.update(receipt_id="cr_" + receipt_hash[:12], reproducibility_hash=receipt_hash)
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
    if assurance_scope is not None:
        payload["assurance_scope"] = asdict(assurance_scope)
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


def verify_context_commit(
    commit: Mapping[str, Any] | object, *, trusted_anchor: str | None = None
) -> ContextCommitVerification:
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

    payload_digest = stable_hash(_commit_payload(data))
    expected_commit_id = "ctx_" + payload_digest[:24]
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
        if entry.get("fragment_sha256") and byte_digest(text) != entry["fragment_sha256"]:
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
    if trusted_anchor is not None:
        checks["trusted_anchor"] = isinstance(trusted_anchor, str) and hmac.compare_digest(
            "sha256:" + payload_digest, trusted_anchor
        )
    errors = tuple(name for name, passed in checks.items() if not passed)
    return ContextCommitVerification(
        valid=not errors,
        checks=checks,
        errors=errors,
        selected_chunks=len(selected),
        omitted_chunks=len(omitted),
        recovered_omitted_chunks=recovered_count,
    )


def recover_committed_chunks(
    commit: Mapping[str, Any], chunk_id: str | None = None, *,
    trusted_anchor: str | None = None,
) -> list[dict[str, Any]]:
    """Recover exact unselected snapshot bytes against an issued full anchor.

    The anchor must come from independent trusted issuance state, not from the
    supplied commit at recovery time. It binds this snapshot, not source-file
    freshness, semantic correctness or principal authorization. ``None`` returns
    every unselected chunk and may materialize substantial local evidence.
    """
    if not isinstance(trusted_anchor, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", trusted_anchor
    ):
        raise ValueError("context recovery requires an independently trusted anchor")
    verification = verify_context_commit(commit, trusted_anchor=trusted_anchor)
    if not verification.checks.get("trusted_anchor"):
        raise ValueError("trusted context commit anchor mismatch")
    if not verification.valid:
        raise ValueError("context commit integrity verification failed")
    if chunk_id is not None and (not isinstance(chunk_id, str) or not chunk_id):
        raise ValueError("chunk_id must be a nonempty string")
    receipt = commit["receipt"]
    selected = {item["chunk_id"] for item in receipt["selected_context"]}
    chunks = commit["recovery_bundle"]["chunks"]
    if chunk_id is not None:
        if chunk_id in selected:
            raise ValueError("requested evidence was not omitted")
        if chunk_id not in chunks:
            raise ValueError("requested evidence is not in the committed snapshot")
        ids = [chunk_id]
    else:
        ids = sorted(set(chunks) - selected)
    # These descriptors are authenticated by the independently pinned whole
    # commit, not invented as a replacement receipt. Nothing is persisted.
    descriptors = [
        {**chunks[cid], "chunk_id": cid, "omission_reason": "not_selected"}
        for cid in ids
    ]
    expanded = {**receipt, "omitted_context": descriptors}
    recovered = recover_omitted(expanded, bundle=commit["recovery_bundle"])
    if len(recovered) != len(ids) or any(
        item.get("verified") is not True
        or item.get("verification_level") != "exact_utf8_bytes"
        for item in recovered
    ):
        raise ValueError("exact committed evidence recovery failed")
    return recovered


__all__ = [
    "CONTEXT_COMMIT_SCHEMA",
    "ContextCommitVerification",
    "context_commit_anchor",
    "create_context_commit",
    "recover_committed_chunks",
    "replay_context",
    "verify_context_commit",
]
