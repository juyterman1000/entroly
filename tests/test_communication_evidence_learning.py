from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import entroly.communication as communication
from entroly.communication import (
    CommunicationActionProposal,
    CommunicationReceiptLedger,
    CommunicationStateConflict,
    CommunicationStateError,
    CommunicationStore,
    CommunicationTasteOptimizer,
    SelectedTasteEvidence,
    build_action_receipt,
    event_from_adapter,
)
from entroly.daemon import EntrolyDaemon
from entroly.openclaw_bridge import handle_request
from entroly.receipt_merkle import SignedTreeHead, verify_inclusion


def _birthday_event(message_id: str = "birthday-1"):
    return event_from_adapter(
        {
            "direction": "inbound",
            "channel": "whatsapp",
            "account_id": "personal",
            "conversation_id": "dm-a",
            "conversation_kind": "direct",
            "sender_id": "person-a",
            "message_id": message_id,
            "timestamp": 1_700_000_000,
            "content": "Happy birthday!",
            "source": "test",
        }
    )


def _birthday_proposal(event):
    return CommunicationActionProposal.build(
        action_type="reply",
        channel="whatsapp",
        account_id="personal",
        conversation_id="dm-a",
        conversation_kind="direct",
        source_event_ids=[event.event_id],
        payload="Thank you so much!",
        category="birthday_wish",
        risk_class="low",
    )


def test_evidence_verification_rejects_payload_outside_proposal_commitment() -> None:
    event = _birthday_event()
    proposal = _birthday_proposal(event)

    verdict = communication.action_evidence_verification(
        proposal,
        [event],
        payload="A different response",
    )

    assert verdict == {"status": "unavailable", "reason": "payload_commitment_mismatch"}


def test_signed_merkle_receipt_survives_restart_without_raw_message_text(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "communication.sqlite3"
    receipt_dir = tmp_path / "receipts"
    event = _birthday_event()
    proposal = _birthday_proposal(event)

    with CommunicationStore(store_path, retention_days=0) as store:
        store.record_event(event, observed_at=100)
        receipt = build_action_receipt(
            proposal,
            decision="allow",
            reasons=("policy:bounded_allow",),
            source_events=[event],
            evidence_verification={
                "status": "not_applicable",
                "reason": "routine_nonfactual_acknowledgement",
            },
        )
        first = CommunicationReceiptLedger(receipt_dir).record(store, receipt)
        rows = store.receipt_rows()

    assert first["verified"] is True
    assert first["tree_size"] == 1
    assert len(rows) == 1
    serialized = json.dumps(rows[0]["receipt"], sort_keys=True)
    assert "Happy birthday" not in serialized
    assert "Thank you so much" not in serialized
    assert event.content_sha256 in serialized
    assert event.commitment_sha256 in serialized

    # New process-equivalent objects reuse the same durable key and receipt row.
    with CommunicationStore(store_path, retention_days=0) as reopened:
        second_ledger = CommunicationReceiptLedger(receipt_dir)
        proof = second_ledger.prove(reopened, receipt["receipt_id"])
        assert proof is not None
        assert proof["operator_public_key"] == first["operator_public_key"]
        stored = reopened.receipt_rows()[0]["receipt"]

    head = SignedTreeHead(
        tree_size=int(proof["tree_size"]),
        root_hash=str(proof["root_hash"]),
        signature=str(proof["operator_signature"]),
        public_key=str(proof["operator_public_key"]),
        timestamp=float(proof["signed_at"]),
    )
    assert head.verify(public_key=second_ledger.public_key)
    assert verify_inclusion(
        int(proof["index"]),
        int(proof["tree_size"]),
        bytes.fromhex(str(proof["leaf_hex"])),
        [bytes.fromhex(item) for item in proof["audit_path"]],
        bytes.fromhex(str(proof["root_hash"])),
    )
    assert stored["receipt_id"] == receipt["receipt_id"]


def test_duplicate_receipt_keeps_later_merkle_index_dense(tmp_path: Path) -> None:
    with CommunicationStore(tmp_path / "communication.sqlite3", retention_days=0) as store:
        ledger = CommunicationReceiptLedger(tmp_path / "receipts")
        for number in (1, 2):
            event = _birthday_event(f"birthday-{number}")
            store.record_event(event)
            receipt = build_action_receipt(
                _birthday_proposal(event),
                decision="allow",
                reasons=("policy:bounded_allow",),
                source_events=[event],
            )
            proof = ledger.record(store, receipt)
            if number == 1:
                assert ledger.record(store, receipt)["index"] == 0

        assert proof["index"] == 1
        assert proof["tree_size"] == 2
        assert proof["verified"] is True


def test_receipt_tampering_is_detected_before_rebuild(tmp_path: Path) -> None:
    store_path = tmp_path / "communication.sqlite3"
    receipt_dir = tmp_path / "receipts"
    event = _birthday_event()
    proposal = _birthday_proposal(event)

    with CommunicationStore(store_path, retention_days=0) as store:
        store.record_event(event, observed_at=100)
        receipt = build_action_receipt(
            proposal,
            decision="allow",
            reasons=("policy:bounded_allow",),
            source_events=[event],
        )
        CommunicationReceiptLedger(receipt_dir).record(store, receipt)

    conn = sqlite3.connect(store_path)
    try:
        conn.execute(
            """
            UPDATE communication_receipts
            SET receipt_json = ?
            WHERE receipt_id = ?
            """,
            ('{"tampered":true}', receipt["receipt_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    with CommunicationStore(store_path, retention_days=0) as store:
        with pytest.raises(CommunicationStateConflict, match="hash mismatch"):
            store.receipt_rows()


def test_failed_signing_cannot_leave_allow_receipt_row(tmp_path: Path) -> None:
    class BrokenKey:
        def public_hex(self) -> str:
            return "00" * 32

        def sign(self, _message: bytes) -> str:
            raise RuntimeError("signing unavailable")

    store_path = tmp_path / "communication.sqlite3"
    ledger = CommunicationReceiptLedger(tmp_path / "receipts")
    ledger._key = BrokenKey()  # type: ignore[assignment]
    event = _birthday_event()
    proposal = _birthday_proposal(event)

    with CommunicationStore(store_path, retention_days=0) as store:
        store.record_event(event, observed_at=100)
        receipt = build_action_receipt(
            proposal,
            decision="allow",
            reasons=("policy:bounded_allow",),
            source_events=[event],
        )
        with pytest.raises(RuntimeError, match="signing unavailable"):
            ledger.record(store, receipt)
        assert store.receipt_count() == 0


def test_bridge_allow_requires_signed_receipt(tmp_path: Path) -> None:
    store_path = tmp_path / "communication.sqlite3"
    ingested = handle_request(
        {
            "operation": "communication_ingest",
            "store_path": str(store_path),
            "retention_days": 0,
            "event": {
                "direction": "inbound",
                "channel": "whatsapp",
                "account_id": "personal",
                "conversation_id": "dm-a",
                "conversation_kind": "direct",
                "sender_id": "person-a",
                "message_id": "birthday-1",
                "content": "Happy birthday!",
            },
        }
    )
    result = handle_request(
        {
            "operation": "communication_assure",
            "store_path": str(store_path),
            "retention_days": 0,
            "channel": "whatsapp",
            "account_id": "personal",
            "conversation_id": "dm-a",
            "action_type": "reply",
            "source_event_ids": [ingested["event_id"]],
            "payload": "Thank you so much!",
            "policy_mode": "bounded",
            "auto_actions": ["reply"],
            "auto_categories": ["birthday_wish"],
        }
    )

    assert result["decision"] == "allow"
    assert result["execution_state"] == "assured"
    assert result["receipt"]["verified"] is True
    assert result["evidence_verification"]["status"] == "not_applicable"


def test_receipt_failure_downgrades_auto_action_to_human_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "communication.sqlite3"
    ingested = handle_request(
        {
            "operation": "communication_ingest",
            "store_path": str(store_path),
            "retention_days": 0,
            "event": {
                "direction": "inbound",
                "channel": "whatsapp",
                "account_id": "personal",
                "conversation_id": "dm-a",
                "conversation_kind": "direct",
                "sender_id": "person-a",
                "message_id": "birthday-1",
                "content": "Happy birthday!",
            },
        }
    )

    def fail_record(self, store, receipt):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(
        communication.CommunicationReceiptLedger,
        "record",
        fail_record,
    )
    result = handle_request(
        {
            "operation": "communication_assure",
            "store_path": str(store_path),
            "retention_days": 0,
            "channel": "whatsapp",
            "account_id": "personal",
            "conversation_id": "dm-a",
            "action_type": "reply",
            "source_event_ids": [ingested["event_id"]],
            "payload": "Thank you!",
            "policy_mode": "bounded",
            "auto_actions": ["reply"],
            "auto_categories": ["birthday_wish"],
        }
    )

    assert result["decision"] == "approval_required"
    assert result["execution_state"] == "awaiting_approval"
    assert "audit:signed_receipt_unavailable" in result["reasons"]
    assert result["receipt"] is None


def test_eicv_can_only_remove_automatic_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "communication.sqlite3"
    ingested = handle_request(
        {
            "operation": "communication_ingest",
            "store_path": str(store_path),
            "retention_days": 0,
            "event": {
                "direction": "inbound",
                "channel": "whatsapp",
                "account_id": "personal",
                "conversation_id": "dm-a",
                "conversation_kind": "direct",
                "sender_id": "person-a",
                "message_id": "request-1",
                "content": "Can you check the reservation and reply?",
            },
        }
    )

    monkeypatch.setattr(
        communication,
        "action_evidence_verification",
        lambda proposal, source_events, *, payload: {
            "status": "verified",
            "decision": "hallucinated",
            "phi": 0.1,
        },
    )
    result = handle_request(
        {
            "operation": "communication_assure",
            "store_path": str(store_path),
            "retention_days": 0,
            "channel": "whatsapp",
            "account_id": "personal",
            "conversation_id": "dm-a",
            "action_type": "reply",
            "source_event_ids": [ingested["event_id"]],
            "payload": "The reservation is confirmed.",
            "policy_mode": "bounded",
            "auto_actions": ["reply"],
            "auto_categories": ["actionable"],
        }
    )

    assert result["decision"] == "approval_required"
    assert "eicv:automatic_action_not_supported" in result["reasons"]


def test_prism_feedback_rejects_model_self_report_and_keeps_no_authority_surface(
    tmp_path: Path,
) -> None:
    optimizer = CommunicationTasteOptimizer(
        tmp_path / "prism.json",
        tmp_path / "feedback.jsonl",
    )
    selection = SelectedTasteEvidence(
        event_ids=("e1", "e2", "e3"),
        feature_mean={
            "w_recency": 0.9,
            "w_frequency": 0.8,
            "w_semantic": 0.7,
            "w_entropy": 0.6,
            "w_resonance": 0.8,
        },
        weights={
            "w_recency": 0.30,
            "w_frequency": 0.20,
            "w_semantic": 0.25,
            "w_entropy": 0.15,
            "w_resonance": 0.10,
        },
    )

    with pytest.raises(CommunicationStateError, match="user-grounded"):
        optimizer.prepare_feedback(
            scope_type="owner",
            scope_id="owner-global",
            reward=1.0,
            selection=selection,
            source="model_self_report",
        )

    prepared = optimizer.prepare_feedback(
        scope_type="owner",
        scope_id="owner-global",
        reward=1.0,
        selection=selection,
        source="explicit_approval",
    )
    optimizer.append_feedback(prepared, receipt_id="commrcpt_" + "a" * 40)
    processed = optimizer.process_pending()
    stats = optimizer.stats()

    assert processed["processed"] == 0
    assert processed["reason"] == "signed_receipt_verifier_required"
    assert stats["authority_surface"] == "none"
    assert stats["processed_feedback"] == 0


def test_shadow_autotune_does_not_promote_without_holdout(tmp_path: Path) -> None:
    optimizer = CommunicationTasteOptimizer(
        tmp_path / "prism.json",
        tmp_path / "feedback.jsonl",
    )
    events = [
        event_from_adapter(
            {
                "direction": "outbound",
                "channel": "whatsapp",
                "account_id": "personal",
                "conversation_id": "dm-a",
                "conversation_kind": "direct",
                "recipient_id": "person-a",
                "message_id": f"out-{index}",
                "timestamp": 1_700_000_000 + index,
                "content": "Thanks!",
                "delivery_state": "sent",
            }
        )
        for index in range(8)
    ]

    result = optimizer.shadow_autotune(
        events,
        scope_type="owner",
        scope_id="owner-global",
    )

    assert result["status"] == "insufficient_holdout"
    assert result["promoted"] is False


def test_daemon_communication_learning_cycle_is_separate_authority_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "communication.sqlite3"
    state_path = tmp_path / "prism.json"
    journal_path = tmp_path / "feedback.jsonl"
    monkeypatch.setenv("ENTROLY_COMMUNICATION_STORE", str(store_path))
    monkeypatch.setenv("ENTROLY_COMMUNICATION_PRISM_STATE", str(state_path))
    monkeypatch.setenv("ENTROLY_COMMUNICATION_PRISM_JOURNAL", str(journal_path))

    with CommunicationStore(store_path, retention_days=0) as store:
        for index in range(6):
            store.record_event(
                event_from_adapter(
                    {
                        "direction": "outbound",
                        "channel": "whatsapp",
                        "account_id": "personal",
                        "conversation_id": "dm-a",
                        "conversation_kind": "direct",
                        "recipient_id": "person-a",
                        "message_id": f"out-{index}",
                        "timestamp": 1_700_000_000 + index,
                        "content": "Thanks!",
                        "delivery_state": "sent",
                    }
                ),
                observed_at=1_800_000_000 + index,
            )

    daemon = EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    result = daemon._run_communication_learning_cycle()

    assert result["authority_surface"] == "none"
    assert result["autotune"]["promoted"] is False
