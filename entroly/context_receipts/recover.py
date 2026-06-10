"""Recoverable Context Receipts — lossless, verifiable recovery of omitted context.

A Context Receipt already *explains* what was omitted and *why*. This module makes
it **recoverable**: given a receipt, return the full original text of any omitted
chunk — and prove it is byte-exact, not a re-derivation that may have drifted.

How it stays honest:
  • At receipt time we capture a project-local **recovery bundle** next to the
    receipt (``.entroly/receipts/<id>.recovery.json``) holding each chunk's full
    text plus a plain ``content_sha`` (``text_fingerprint`` of the text) and the
    chunk's recorded composite ``fingerprint``. Nothing leaves the machine.
  • On recover we verify two independent things: (a) the bundle's recorded
    ``fingerprint`` matches the fingerprint the *receipt* recorded for that chunk
    (this is the chunk the receipt omitted — not a look-alike), and (b)
    ``text_fingerprint(text) == content_sha`` (the stored text was not corrupted).
    Both pass ⇒ ``verified``: the returned text is provably exactly what was dropped.

This is strictly stronger than recover-only compression: receipts explain the
omission *and* hand back the exact content, with a cryptographic guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from . import store as _store
from .models import ContextReceipt, text_fingerprint

RECOVERY_SCHEMA = "context-receipt.recovery.v1"
RECOVERY_SUFFIX = ".recovery.json"


@dataclass
class RecoveredChunk:
    """One recovered omitted chunk. ``verified`` is the cryptographic guarantee."""

    chunk_id: str
    source_path: str
    section_heading: str | None
    token_count: int
    omission_reason: str
    text: str | None          # None when the chunk could not be recovered
    fingerprint: str          # the chunk fingerprint recorded by the receipt
    verified: bool            # text is provably the exact omitted content
    status: str               # "recovered" | "recovered_unverified" | "unavailable"
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── recovery bundle (the project-local store of recoverable content) ─────────
def build_recovery_bundle(index: dict[str, Any]) -> dict[str, Any]:
    """Capture full chunk text + integrity hashes from a ContextIndex dict.

    This is the *only* place full omitted text is retained; keep it project-local.
    """
    chunks: dict[str, Any] = {}
    for c in index.get("chunks", []):
        text = c.get("text", "") or ""
        chunks[str(c["chunk_id"])] = {
            "text": text,
            "fingerprint": c.get("fingerprint", ""),
            "content_sha": text_fingerprint(text),
            "source_path": c.get("source_path", ""),
            "section_heading": c.get("section_heading"),
            "token_count": int(c.get("token_count", 0) or 0),
        }
    return {"schema": RECOVERY_SCHEMA, "chunk_count": len(chunks), "chunks": chunks}


def recovery_path(receipt_id: str, store_dir: str | Path | None = None) -> Path:
    base = Path(store_dir) if store_dir is not None else _store.DEFAULT_STORE
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{receipt_id}{RECOVERY_SUFFIX}"


def save_recovery_bundle(
    receipt_id: str, bundle: dict[str, Any], store_dir: str | Path | None = None
) -> Path:
    return _store.write_json(recovery_path(receipt_id, store_dir), bundle)


def load_recovery_bundle(
    receipt_id: str, store_dir: str | Path | None = None
) -> dict[str, Any] | None:
    path = recovery_path(receipt_id, store_dir)
    if not path.exists():
        return None
    return _store.read_json(path)


# ── recovery ─────────────────────────────────────────────────────────────────
def _resolve_source(
    receipt_id: str,
    index: dict[str, Any] | None,
    bundle: dict[str, Any] | None,
    store_dir: str | Path | None,
) -> dict[str, dict[str, Any]]:
    """Resolve chunk_id -> recovery entry, in priority order: explicit bundle,
    explicit index (converted), then the persisted bundle for this receipt."""
    if bundle is not None:
        return dict(bundle.get("chunks", {}))
    if index is not None:
        return dict(build_recovery_bundle(index).get("chunks", {}))
    loaded = load_recovery_bundle(receipt_id, store_dir) if receipt_id else None
    return dict(loaded.get("chunks", {})) if loaded else {}


def recover_omitted(
    receipt: dict[str, Any] | ContextReceipt,
    chunk_id: str | None = None,
    *,
    index: dict[str, Any] | None = None,
    bundle: dict[str, Any] | None = None,
    store_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Recover the full, fingerprint-verified text of omitted chunk(s).

    Pass ``chunk_id`` to recover one chunk, or omit it to recover every omitted
    chunk. Full text is sourced (in priority order) from an explicit ``bundle``,
    an explicit ``index`` (a ContextIndex dict), or the persisted recovery bundle
    for this receipt. Returns a list of ``RecoveredChunk`` dicts; ``verified`` is
    True only when the returned text is provably the exact omitted content.
    """
    rec = receipt if isinstance(receipt, dict) else receipt.to_dict()
    receipt_id = str(rec.get("receipt_id", ""))
    omitted = list(rec.get("omitted_context", []))
    if chunk_id is not None:
        omitted = [o for o in omitted if o.get("chunk_id") == chunk_id]

    source = _resolve_source(receipt_id, index, bundle, store_dir)

    results: list[RecoveredChunk] = []
    for item in omitted:
        cid = str(item.get("chunk_id", ""))
        recorded_fp = item.get("fingerprint", "")
        common = dict(
            chunk_id=cid,
            source_path=item.get("source_path", ""),
            section_heading=item.get("section_heading"),
            token_count=int(item.get("token_count", 0) or 0),
            omission_reason=item.get("omission_reason", ""),
            fingerprint=recorded_fp,
        )
        entry = source.get(cid)
        if entry is None:
            results.append(RecoveredChunk(
                **common, text=None, verified=False, status="unavailable",
                note="No recovery data for this chunk. Create the receipt via "
                     "run_recoverable_pipeline(), or pass index=/bundle=.",
            ))
            continue

        text = entry.get("text", "") or ""
        # (a) is this the chunk the receipt omitted?  (b) is the stored text intact?
        chunk_matches = (not recorded_fp) or entry.get("fingerprint", "") == recorded_fp
        text_intact = text_fingerprint(text) == entry.get("content_sha", "")
        verified = bool(chunk_matches and text_intact)
        results.append(RecoveredChunk(
            **common, text=text, verified=verified,
            status="recovered" if verified else "recovered_unverified",
            note=None if verified else
                 ("Chunk fingerprint does not match the receipt." if not chunk_matches
                  else "Stored text failed its integrity hash."),
        ))
    return [r.to_dict() for r in results]
