from __future__ import annotations

import copy
import json

import pytest

from entroly.compression_retrieval_store_resilient import CompressionRetrievalStore


def _put(store: CompressionRetrievalStore, text: str):
    return store.put(
        original_text=text,
        compressed_text=text.splitlines()[0],
        receipt={
            "original_tokens": 20,
            "compressed_tokens": 5,
            "omitted_spans": [{"start_line": 2, "end_line": 2, "reason": "budget"}],
        },
        metadata={"provider": "anthropic"},
    )


def test_one_corrupt_record_does_not_disable_healthy_recovery(tmp_path):
    path = tmp_path / "recovery.json"
    store = CompressionRetrievalStore(path, scope_id="test")
    healthy = _put(store, "keep\nexact evidence\n")

    raw = json.loads(path.read_text(encoding="utf-8"))
    poisoned = copy.deepcopy(raw["items"][0])
    poisoned["receipt_id"] = "corrupt-receipt"
    # Its span still points at the healthy receipt: exact recovery must reject it.
    raw["items"].append(poisoned)
    path.write_text(json.dumps(raw), encoding="utf-8")

    reloaded = CompressionRetrievalStore(path, scope_id="test")

    assert reloaded.get_receipt(healthy.receipt_id) is not None
    assert reloaded.get_receipt("corrupt-receipt") is None
    summary = reloaded.quarantine_summary()
    assert summary["quarantined_records"] == 1
    assert any("span_receipt_mismatch" in reason for reason in summary["by_reason"])


def test_future_write_preserves_forensic_quarantine(tmp_path):
    path = tmp_path / "recovery.json"
    store = CompressionRetrievalStore(path, scope_id="test")
    _put(store, "first\nomitted\n")
    raw = json.loads(path.read_text(encoding="utf-8"))
    poisoned = copy.deepcopy(raw["items"][0])
    poisoned["receipt_id"] = "poisoned"
    raw["items"].append(poisoned)
    path.write_text(json.dumps(raw), encoding="utf-8")

    reloaded = CompressionRetrievalStore(path, scope_id="test")
    _put(reloaded, "second\nother evidence\n")

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert len(persisted["quarantined_items"]) == 1
    assert persisted["quarantined_items"][0]["item"]["receipt_id"] == "poisoned"


def test_malformed_store_root_still_fails_closed(tmp_path):
    path = tmp_path / "recovery.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="root"):
        CompressionRetrievalStore(path, scope_id="test")
