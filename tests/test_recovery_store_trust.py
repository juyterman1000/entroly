from __future__ import annotations

import os
from pathlib import Path

import pytest

import entroly.compression_proxy as compression_proxy
from entroly import compression_retrieval_store as legacy_module
from entroly.compression_mcp import _safe_store_path
from entroly.compression_retrieval_store_safe import (
    _LegacyCompressionRetrievalStore,
)
from entroly.compression_retrieval_store_secure import CompressionRetrievalStore


def _receipt() -> dict:
    return {
        "original_tokens": 100,
        "compressed_tokens": 10,
        "omitted_spans": [{"start_line": 1, "end_line": 1}],
    }


def _put(store: CompressionRetrievalStore, text: str = "private recovery evidence"):
    return store.put(
        original_text=text,
        compressed_text="[omitted]",
        receipt=_receipt(),
        metadata={
            "query": "user asked about a private production incident",
            "session_id": "session-secret-123",
            "component": "trust-test",
        },
    )


def test_historical_and_proxy_imports_are_patched_to_secure_store() -> None:
    assert legacy_module.CompressionRetrievalStore is CompressionRetrievalStore
    assert compression_proxy.CompressionRetrievalStore is CompressionRetrievalStore


def test_identical_content_has_scope_distinct_receipt_ids_and_access(tmp_path: Path) -> None:
    path = tmp_path / "shared.json"
    alpha = CompressionRetrievalStore(path, scope_id="workspace-alpha")
    alpha_record = _put(alpha, "same exact source")

    beta = CompressionRetrievalStore(path, scope_id="workspace-beta")
    beta_record = _put(beta, "same exact source")

    assert alpha_record.receipt_id != beta_record.receipt_id
    assert alpha.get_receipt(beta_record.receipt_id) is None
    assert beta.get_receipt(alpha_record.receipt_id) is None
    assert alpha.search("exact source") == [
        alpha.get_span(alpha_record.receipt_id, alpha_record.spans[0].span_id)
    ]
    assert beta.search("exact source") == [
        beta.get_span(beta_record.receipt_id, beta_record.spans[0].span_id)
    ]


def test_scope_survives_restart_and_denies_other_workspace(tmp_path: Path) -> None:
    path = tmp_path / "restart.json"
    written = _put(CompressionRetrievalStore(path, scope_id="alpha"))

    restored = CompressionRetrievalStore(path, scope_id="alpha")
    denied = CompressionRetrievalStore(path, scope_id="beta")

    assert restored.get_span(written.receipt_id, written.spans[0].span_id) is not None
    assert denied.get_span(written.receipt_id, written.spans[0].span_id) is None
    assert denied.list_receipts() == []
    assert denied.savings_summary()["gross_tokens_saved"] == 0


def test_returned_objects_are_defensive_snapshots(tmp_path: Path) -> None:
    store = CompressionRetrievalStore(tmp_path / "snapshots.json", scope_id="alpha")
    returned = _put(store)
    original_receipt_id = returned.receipt_id
    original_span_id = returned.spans[0].span_id

    returned.metadata["component"] = "tampered"
    returned.spans[0].content = "tampered content"
    returned.spans[0].retrieval_ids.append("forged-debit")

    observed = store.get_receipt(original_receipt_id)
    span = store.get_span(original_receipt_id, original_span_id)
    assert observed is not None
    assert span is not None
    assert observed.metadata["component"] == "trust-test"
    assert span.content == "private recovery evidence"
    assert "forged-debit" not in span.retrieval_ids

    listed = store.list_receipts()
    listed[0]["metadata"]["component"] = "also-tampered"
    assert store.list_receipts()[0]["metadata"]["component"] == "trust-test"


def test_prompt_and_session_metadata_are_hashed_not_disclosed(tmp_path: Path) -> None:
    store = CompressionRetrievalStore(tmp_path / "metadata.json", scope_id="alpha")
    _put(store)

    metadata = store.list_receipts()[0]["metadata"]
    serialized = repr(metadata)
    assert "private production incident" not in serialized
    assert "session-secret-123" not in serialized
    assert metadata["component"] == "trust-test"
    assert len(str(metadata["query_sha256"])) == 64
    assert len(str(metadata["session_id_sha256"])) == 64
    assert not any(str(key).startswith("_") for key in metadata)


def test_parent_directory_fsync_failure_does_not_roll_back_committed_file(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "durable.json"

    def unsupported_directory_fsync(_self) -> None:
        raise OSError("directory fsync unsupported")

    monkeypatch.setattr(
        _LegacyCompressionRetrievalStore,
        "_sync_parent_directory",
        unsupported_directory_fsync,
    )
    store = CompressionRetrievalStore(path, scope_id="alpha")
    written = _put(store)

    assert path.exists()
    assert store.get_receipt(written.receipt_id) is not None
    restarted = CompressionRetrievalStore(path, scope_id="alpha")
    assert restarted.get_receipt(written.receipt_id) is not None


def test_retrieval_history_saturation_keeps_recovery_and_zeros_claimed_savings(
    tmp_path: Path,
) -> None:
    store = CompressionRetrievalStore(
        tmp_path / "bounded-history.json",
        scope_id="alpha",
        max_retrieval_ids=32,
    )
    written = _put(store, "bounded accounting evidence")
    span_id = written.spans[0].span_id

    for index in range(40):
        span = store.retrieve_span(
            written.receipt_id,
            span_id,
            retrieval_id=f"request-{index}",
        )
        assert span is not None
        assert span.content == "bounded accounting evidence"

    observed = store.get_span(written.receipt_id, span_id)
    receipt = store.list_receipts()[0]
    assert observed is not None
    assert len(observed.retrieval_ids) <= 32
    assert receipt["metadata"]["retrieval_accounting_exhausted"] is True
    assert receipt["metadata"]["retrieval_accounting_confidence"] == "conservative_zero"
    assert receipt["net_realized_saved_tokens"] == 0


def test_legacy_unscoped_records_require_explicit_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    legacy = _LegacyCompressionRetrievalStore(path)
    written = legacy.put(
        original_text="legacy evidence",
        compressed_text="[omitted]",
        receipt=_receipt(),
    )

    strict = CompressionRetrievalStore(path, scope_id="alpha")
    compatible = CompressionRetrievalStore(
        path,
        scope_id="alpha",
        allow_legacy_unscoped=True,
    )

    assert strict.get_receipt(written.receipt_id) is None
    assert compatible.get_receipt(written.receipt_id) is not None


class _FakeLedger:
    def __init__(self) -> None:
        self.events = []
        self.adjustments = []

    def record(self, event) -> None:
        self.events.append(event)

    def adjust(self, adjustment) -> None:
        self.adjustments.append(adjustment)


def test_ledger_replay_is_filtered_to_active_scope(tmp_path: Path) -> None:
    path = tmp_path / "ledger-scope.json"
    alpha = CompressionRetrievalStore(path, scope_id="alpha")
    beta = CompressionRetrievalStore(path, scope_id="beta")
    alpha_record = _put(alpha, "alpha evidence")
    beta_record = _put(beta, "beta evidence")
    ledger = _FakeLedger()

    CompressionRetrievalStore(path, scope_id="beta", optimization_ledger=ledger)

    event_ids = {event.event_id for event in ledger.events}
    assert f"compression:{beta_record.receipt_id}" in event_ids
    assert f"compression:{alpha_record.receipt_id}" not in event_ids


def test_first_run_default_path_can_have_missing_parent(tmp_path: Path) -> None:
    candidate = tmp_path / "new" / "nested" / "compression-store.json"

    resolved = _safe_store_path("", str(candidate))

    assert resolved == candidate
    assert not candidate.parent.exists()


def test_mcp_store_override_is_disabled_by_default(monkeypatch, tmp_path: Path) -> None:
    configured = tmp_path / "configured" / "store.json"
    monkeypatch.delenv("ENTROLY_ALLOW_STORE_PATH_OVERRIDE", raising=False)

    with pytest.raises(ValueError, match="disabled"):
        _safe_store_path(str(tmp_path / "other.json"), str(configured))


def test_enabled_override_is_contained_to_operator_root(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    nested = root / "nested"
    nested.mkdir(parents=True)
    configured = root / "store.json"
    monkeypatch.setenv("ENTROLY_ALLOW_STORE_PATH_OVERRIDE", "1")
    monkeypatch.setenv("ENTROLY_STORE_OVERRIDE_ROOT", str(root))

    inside = _safe_store_path(str(nested / "inside.json"), str(configured))
    assert inside == nested / "inside.json"

    with pytest.raises(ValueError, match="escapes"):
        _safe_store_path(str(tmp_path / "outside.json"), str(configured))


def test_enabled_override_rejects_missing_parent(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    configured = root / "store.json"
    monkeypatch.setenv("ENTROLY_ALLOW_STORE_PATH_OVERRIDE", "1")
    monkeypatch.setenv("ENTROLY_STORE_OVERRIDE_ROOT", str(root))

    with pytest.raises(ValueError, match="escapes"):
        _safe_store_path(str(root / "missing" / "inside.json"), str(configured))


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_override_symlink_cannot_escape_allowed_root(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is not permitted")
    monkeypatch.setenv("ENTROLY_ALLOW_STORE_PATH_OVERRIDE", "1")
    monkeypatch.setenv("ENTROLY_STORE_OVERRIDE_ROOT", str(root))

    with pytest.raises(ValueError, match="escapes"):
        _safe_store_path(str(link / "store.json"), str(root / "configured.json"))
