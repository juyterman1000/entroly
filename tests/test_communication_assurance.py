from __future__ import annotations

from pathlib import Path

import pytest

from entroly.communication import (
    CommunicationActionProposal,
    CommunicationPolicy,
    CommunicationStateConflict,
    CommunicationStateError,
    CommunicationStore,
    event_from_adapter,
)
from entroly.openclaw_bridge import handle_request


def _event(
    *,
    conversation: str = "chat-a",
    message_id: str = "m-1",
    content: str = "hello",
    sender: str = "person-a",
    kind: str = "unknown",
):
    return event_from_adapter(
        {
            "direction": "inbound",
            "channel": "whatsapp",
            "account_id": "personal",
            "conversation_id": conversation,
            "conversation_kind": kind,
            "sender_id": sender,
            "message_id": message_id,
            "timestamp": 1_700_000_000,
            "content": content,
            "source": "test",
        }
    )


def test_adapter_event_identity_is_stable_and_does_not_infer_group_kind() -> None:
    first = _event(content="Happy birthday")
    second = _event(content="Happy birthday")

    assert first.event_id == second.event_id
    assert first.identity_strength == "message_id"
    assert first.conversation_kind == "unknown"
    assert first.content_sha256 == second.content_sha256


def test_store_is_idempotent_but_rejects_identity_reuse_with_changed_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "communication.sqlite3"
    original = _event(content="first")
    changed = _event(content="changed")

    with CommunicationStore(path, retention_days=0) as store:
        assert store.record_event(original, observed_at=100) is True
        assert store.record_event(original, observed_at=101) is False
        with pytest.raises(CommunicationStateConflict):
            store.record_event(changed, observed_at=102)


def test_scoped_retrieval_cannot_cross_conversations(tmp_path: Path) -> None:
    path = tmp_path / "communication.sqlite3"
    a = _event(conversation="private-a", message_id="a")
    b = _event(conversation="private-b", message_id="b", sender="person-b")

    with CommunicationStore(path, retention_days=0) as store:
        store.record_event(a, observed_at=100)
        store.record_event(b, observed_at=101)
        rows = store.events_for_scope(
            channel="whatsapp",
            account_id="personal",
            conversation_id="private-a",
        )

    assert [item.event_id for item in rows] == [a.event_id]
    assert all(item.conversation_id == "private-a" for item in rows)


def test_store_refuses_unscoped_evidence() -> None:
    with pytest.raises(CommunicationStateError, match="conversation_id"):
        event_from_adapter(
            {
                "direction": "inbound",
                "channel": "whatsapp",
                "message_id": "m",
                "content": "private",
            }
        )


def test_retention_prunes_raw_evidence(tmp_path: Path) -> None:
    path = tmp_path / "communication.sqlite3"
    old = _event(message_id="old")
    new = _event(message_id="new")

    with CommunicationStore(path, retention_days=1) as store:
        store.record_event(old, observed_at=100)
        store.record_event(new, observed_at=100 + (2 * 24 * 60 * 60))
        assert store.get_event(old.event_id) is None
        assert store.get_event(new.event_id) is not None


def test_default_policy_never_authorizes_external_send() -> None:
    event = _event()
    proposal = CommunicationActionProposal.build(
        action_type="reply",
        channel="whatsapp",
        account_id="personal",
        conversation_id="chat-a",
        source_event_ids=[event.event_id],
        payload="Thanks",
        category="social_ack",
        risk_class="low",
    )

    decision, reasons = CommunicationPolicy().evaluate(
        proposal,
        source_events=[event],
    )

    assert decision == "approval_required"
    assert reasons == ("policy:observe",)


def test_bounded_policy_can_allow_only_explicit_low_risk_action() -> None:
    event = _event(kind="direct")
    proposal = CommunicationActionProposal.build(
        action_type="reply",
        channel="whatsapp",
        account_id="personal",
        conversation_id="chat-a",
        conversation_kind="direct",
        source_event_ids=[event.event_id],
        payload="Thank you",
        category="social_ack",
        risk_class="low",
    )
    policy = CommunicationPolicy(
        mode="bounded",
        auto_categories=("social_ack",),
        auto_actions=("reply",),
    )

    assert policy.evaluate(proposal, source_events=[event]) == (
        "allow",
        ("policy:bounded_allow",),
    )


def test_group_reply_fails_closed_when_group_kind_is_not_verified() -> None:
    event = _event(kind="unknown")
    proposal = CommunicationActionProposal.build(
        action_type="group_reply",
        channel="whatsapp",
        account_id="personal",
        conversation_id="chat-a",
        conversation_kind="unknown",
        source_event_ids=[event.event_id],
        payload="Thanks everyone",
        category="social_ack",
        risk_class="low",
    )
    policy = CommunicationPolicy(
        mode="bounded",
        auto_categories=("social_ack",),
        auto_actions=("group_reply",),
    )

    assert policy.evaluate(proposal, source_events=[event]) == (
        "ambiguous",
        ("scope:group_not_verified",),
    )


def test_cross_conversation_evidence_blocks_action() -> None:
    event = _event(conversation="chat-b")
    proposal = CommunicationActionProposal.build(
        action_type="reply",
        channel="whatsapp",
        account_id="personal",
        conversation_id="chat-a",
        source_event_ids=[event.event_id],
        payload="reply",
        category="social_ack",
        risk_class="low",
    )

    assert CommunicationPolicy(mode="bounded").evaluate(
        proposal,
        source_events=[event],
    ) == ("deny", ("scope:cross_conversation",))


def test_commitment_creation_always_requires_approval() -> None:
    event = _event(kind="direct")
    proposal = CommunicationActionProposal.build(
        action_type="reply",
        channel="whatsapp",
        account_id="personal",
        conversation_id="chat-a",
        conversation_kind="direct",
        source_event_ids=[event.event_id],
        payload="I will send it tomorrow",
        category="social_ack",
        risk_class="low",
        creates_commitment=True,
    )
    policy = CommunicationPolicy(
        mode="bounded",
        auto_categories=("social_ack",),
        auto_actions=("reply",),
    )

    assert policy.evaluate(proposal, source_events=[event]) == (
        "approval_required",
        ("action:creates_commitment",),
    )


def test_action_store_requires_local_same_scope_evidence(tmp_path: Path) -> None:
    path = tmp_path / "communication.sqlite3"
    event = _event(conversation="chat-a")
    foreign = CommunicationActionProposal.build(
        action_type="reply",
        channel="whatsapp",
        account_id="personal",
        conversation_id="chat-b",
        source_event_ids=[event.event_id],
        payload="reply",
        category="social_ack",
        risk_class="low",
    )

    with CommunicationStore(path, retention_days=0) as store:
        store.record_event(event, observed_at=100)
        with pytest.raises(CommunicationStateError, match="crosses"):
            store.record_action(
                foreign,
                decision="deny",
                reasons=("scope:cross_conversation",),
                now=101,
            )


def test_action_outcome_is_observed_before_becoming_handled(tmp_path: Path) -> None:
    path = tmp_path / "communication.sqlite3"
    event = _event(kind="direct")
    proposal = CommunicationActionProposal.build(
        action_type="reply",
        channel="whatsapp",
        account_id="personal",
        conversation_id="chat-a",
        conversation_kind="direct",
        source_event_ids=[event.event_id],
        payload="Thank you",
        category="social_ack",
        risk_class="low",
    )

    with CommunicationStore(path, retention_days=0) as store:
        store.record_event(event, observed_at=100)
        store.record_action(
            proposal,
            decision="approval_required",
            reasons=("policy:approve",),
            now=101,
        )
        assert store.action_is_handled(proposal.action_id) is False
        store.record_action_outcome(
            proposal.action_id,
            success=True,
            outbound_message_id="sent-1",
            now=102,
        )
        assert store.action_is_handled(proposal.action_id) is True


def test_openclaw_bridge_ingests_idempotently_and_reports_scalar_status(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "openclaw-communication.sqlite3"
    request = {
        "operation": "communication_ingest",
        "store_path": str(store_path),
        "retention_days": 0,
        "event": {
            "direction": "inbound",
            "channel": "whatsapp",
            "account_id": "personal",
            "conversation_id": "chat-a",
            "conversation_kind": "unknown",
            "sender_id": "person-a",
            "message_id": "m-1",
            "content": "hello",
            "source": "openclaw.message_received",
        },
    }

    first = handle_request(request)
    second = handle_request(request)
    status = handle_request(
        {
            "operation": "communication_status",
            "store_path": str(store_path),
            "retention_days": 0,
        }
    )

    assert first["ok"] is True
    assert first["local_only"] is True
    assert first["provider_call_performed"] is False
    assert first["inserted"] is True
    assert second["inserted"] is False
    assert status["stats"]["events"] == 1
    assert status["stats"]["inbound"] == 1
    assert "content" not in status["stats"]


def test_openclaw_bridge_refuses_relative_private_store_path() -> None:
    with pytest.raises(CommunicationStateError, match="absolute"):
        handle_request(
            {
                "operation": "communication_status",
                "store_path": "relative/private.sqlite3",
            }
        )
