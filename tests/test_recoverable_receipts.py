"""Recoverable Context Receipts — receipts that explain AND recover.

Proves the breakthrough guarantee: any omitted chunk can be recovered byte-exact
and *verified* against its recorded fingerprint, from the project-local store
alone — and that corruption or missing data is detected, never silently trusted.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from entroly import context_receipts as cr

# Enough distinct tokens that chunking yields many chunks and a small budget omits most.
_TEXT = " ".join(f"token{i}" for i in range(800))
_DOCS = [("doc.txt", _TEXT)]
_KW = dict(query="token5 token6", token_budget=40, chunk_tokens=40, overlap_tokens=8)


def _index_and_receipt():
    index = cr.ingest_documents(_DOCS, chunk_tokens=_KW["chunk_tokens"], overlap_tokens=_KW["overlap_tokens"])
    receipt = cr.select_from_index(index, query=_KW["query"], token_budget=_KW["token_budget"])
    return index, receipt


def test_pipeline_omits_chunks():
    _, receipt = _index_and_receipt()
    assert receipt["omitted_context"], "test needs a budget that forces omission"


def test_recover_is_lossless_and_verified_from_index():
    index, receipt = _index_and_receipt()
    original = {c["chunk_id"]: c["text"] for c in index["chunks"]}
    recovered = cr.recover_omitted(receipt, index=index)
    assert recovered
    for r in recovered:
        assert r["verified"] is True, r
        assert r["status"] == "recovered"
        assert r["text"] == original[r["chunk_id"]]  # byte-exact


def test_recover_from_store_alone(tmp_path):
    out = cr.run_recoverable_pipeline(_DOCS, store_dir=tmp_path, **_KW)
    assert out["recovery_path"] and Path(out["recovery_path"]).exists()
    receipt = out["receipt"]
    # No index/bundle passed — recovery must come from the persisted bundle.
    recovered = cr.recover_omitted(receipt, store_dir=tmp_path)
    assert recovered
    assert all(r["verified"] for r in recovered)
    assert all(r["text"] for r in recovered)


def test_recover_specific_chunk(tmp_path):
    out = cr.run_recoverable_pipeline(_DOCS, store_dir=tmp_path, **_KW)
    receipt = out["receipt"]
    target = receipt["omitted_context"][0]["chunk_id"]
    recovered = cr.recover_omitted(receipt, target, store_dir=tmp_path)
    assert len(recovered) == 1
    assert recovered[0]["chunk_id"] == target
    assert recovered[0]["verified"]


def test_corruption_is_detected(tmp_path):
    out = cr.run_recoverable_pipeline(_DOCS, store_dir=tmp_path, **_KW)
    receipt = out["receipt"]
    target = receipt["omitted_context"][0]["chunk_id"]
    # Tamper with the stored text for an omitted chunk.
    rp = Path(out["recovery_path"])
    bundle = json.loads(rp.read_text(encoding="utf-8"))
    bundle["chunks"][target]["text"] += " <<TAMPERED>>"
    rp.write_text(json.dumps(bundle), encoding="utf-8")

    recovered = cr.recover_omitted(receipt, target, store_dir=tmp_path)
    assert len(recovered) == 1
    assert recovered[0]["verified"] is False
    assert recovered[0]["status"] == "recovered_unverified"


def test_unavailable_without_recovery_data(tmp_path):
    # A receipt with no persisted bundle and no index/bundle passed: must report
    # unavailable rather than fabricate content.
    _, receipt = _index_and_receipt()
    recovered = cr.recover_omitted(receipt, store_dir=tmp_path)  # empty store
    assert recovered
    assert all(r["status"] == "unavailable" and r["text"] is None for r in recovered)


def test_recover_no_omissions_returns_empty():
    # A receipt that omitted nothing has nothing to recover.
    receipt = {"receipt_id": "r1", "omitted_context": []}
    assert cr.recover_omitted(receipt) == []


def test_sdk_recoverable_roundtrip(tmp_path, monkeypatch):
    # Full public surface: create_context_receipt(recoverable=True) persists the
    # bundle to the project-local store; recover_receipt_omission reads it back.
    from entroly import sdk

    monkeypatch.chdir(tmp_path)  # .entroly/receipts/ lands under tmp
    receipt = sdk.create_context_receipt(
        dict(_DOCS),
        query=_KW["query"],
        budget=_KW["token_budget"],
        chunk_tokens=_KW["chunk_tokens"],
        overlap_tokens=_KW["overlap_tokens"],
        recoverable=True,
    )
    assert receipt["omitted_context"]
    recovered = sdk.recover_receipt_omission(receipt)
    assert recovered
    assert all(r["verified"] for r in recovered)
    assert all(r["text"] for r in recovered)
