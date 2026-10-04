from __future__ import annotations

from pathlib import Path

import pytest

from entroly.communication import (
    CommunicationActionProposal,
    CommunicationMemory,
    CommunicationPolicy,
    CommunicationStateConflict,
    CommunicationStateError,
    CommunicationStore,
    CommunicationTaste,
    assess_event,
    build_digest,
    build_group_episodes,
    event_from_adapter,
    infer_taste_from_outbound,
    resolve_taste,
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
    event = _event(kind="direct")
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


def test_birthday_burst_100_direct_messages_remain_isolated_and_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "birthday-direct.sqlite3"
    with CommunicationStore(path, retention_days=0) as store:
        events = [
            event_from_adapter(
                {
                    "direction": "inbound",
                    "channel": "whatsapp",
                    "account_id": "personal",
                    "conversation_id": f"dm-{index:03d}",
                    "conversation_kind": "direct",
                    "sender_id": f"person-{index:03d}",
                    "message_id": f"birthday-{index:03d}",
                    "timestamp": 1_700_000_000 + index,
                    "content": "Happy birthday!",
                    "source": "birthday-scenario",
                }
            )
            for index in range(100)
        ]
        assert all(store.record_event(event, observed_at=2_000 + index) for index, event in enumerate(events))
        assert store.stats()["events"] == 100
        assert store.stats()["conversations"] == 100

        for index, event in enumerate(events):
            scoped = store.events_for_scope(
                channel="whatsapp",
                account_id="personal",
                conversation_id=f"dm-{index:03d}",
            )
            assert [item.event_id for item in scoped] == [event.event_id]
            assert scoped[0].sender_id == f"person-{index:03d}"

        # Replaying the full burst must not create a second copy of any message.
        assert all(
            store.record_event(event, observed_at=5_000 + index) is False
            for index, event in enumerate(events)
        )
        assert store.stats()["events"] == 100


def test_birthday_burst_100_group_messages_share_only_verified_group_scope(
    tmp_path: Path,
) -> None:
    path = tmp_path / "birthday-group.sqlite3"
    group_id = "family-birthday-group"
    with CommunicationStore(path, retention_days=0) as store:
        events = [
            event_from_adapter(
                {
                    "direction": "inbound",
                    "channel": "whatsapp",
                    "account_id": "personal",
                    "conversation_id": group_id,
                    "conversation_kind": "group",
                    "sender_id": f"member-{index:03d}",
                    "message_id": f"group-birthday-{index:03d}",
                    "timestamp": 1_700_100_000 + index,
                    "content": f"Happy birthday from member {index}!",
                    "source": "birthday-scenario",
                }
            )
            for index in range(100)
        ]
        for index, event in enumerate(events):
            store.record_event(event, observed_at=10_000 + index)

        scoped = store.events_for_scope(
            channel="whatsapp",
            account_id="personal",
            conversation_id=group_id,
            limit=200,
        )

        assert len(scoped) == 100
        assert {item.sender_id for item in scoped} == {
            f"member-{index:03d}" for index in range(100)
        }
        assert all(item.conversation_kind == "group" for item in scoped)
        assert store.stats()["conversations"] == 1


def test_birthday_100_direct_reply_proposals_need_approval_by_default() -> None:
    policy = CommunicationPolicy()
    for index in range(100):
        event = event_from_adapter(
            {
                "direction": "inbound",
                "channel": "whatsapp",
                "account_id": "personal",
                "conversation_id": f"dm-{index:03d}",
                "conversation_kind": "direct",
                "sender_id": f"person-{index:03d}",
                "message_id": f"birthday-{index:03d}",
                "content": "Happy birthday!",
            }
        )
        proposal = CommunicationActionProposal.build(
            action_type="reply",
            channel="whatsapp",
            account_id="personal",
            conversation_id=f"dm-{index:03d}",
            conversation_kind="direct",
            source_event_ids=[event.event_id],
            payload="Thank you so much!",
            category="social_ack",
            risk_class="low",
        )
        assert policy.evaluate(proposal, source_events=[event]) == (
            "approval_required",
            ("policy:observe",),
        )


def test_birthday_group_reply_is_allowed_only_when_explicitly_delegated_and_verified() -> None:
    events = [
        event_from_adapter(
            {
                "direction": "inbound",
                "channel": "whatsapp",
                "account_id": "personal",
                "conversation_id": "family-birthday-group",
                "conversation_kind": "group",
                "sender_id": f"member-{index:03d}",
                "message_id": f"group-birthday-{index:03d}",
                "content": "Happy birthday!",
            }
        )
        for index in range(100)
    ]
    proposal = CommunicationActionProposal.build(
        action_type="group_reply",
        channel="whatsapp",
        account_id="personal",
        conversation_id="family-birthday-group",
        conversation_kind="group",
        source_event_ids=[event.event_id for event in events],
        payload="Thank you everyone for the wishes!",
        category="social_ack",
        risk_class="low",
    )
    bounded = CommunicationPolicy(
        mode="bounded",
        auto_categories=("social_ack",),
        auto_actions=("group_reply",),
    )

    assert bounded.evaluate(proposal, source_events=events) == (
        "allow",
        ("policy:bounded_allow",),
    )

    unknown_scope = CommunicationActionProposal.build(
        action_type="group_reply",
        channel="whatsapp",
        account_id="personal",
        conversation_id="family-birthday-group",
        conversation_kind="unknown",
        source_event_ids=[event.event_id for event in events],
        payload="Thank you everyone for the wishes!",
        category="social_ack",
        risk_class="low",
    )
    assert bounded.evaluate(unknown_scope, source_events=events) == (
        "ambiguous",
        ("scope:group_not_verified",),
    )


def test_birthday_burst_never_reuses_one_chat_evidence_for_another_reply() -> None:
    event_a = event_from_adapter(
        {
            "direction": "inbound",
            "channel": "whatsapp",
            "account_id": "personal",
            "conversation_id": "dm-a",
            "conversation_kind": "direct",
            "sender_id": "person-a",
            "message_id": "birthday-a",
            "content": "Happy birthday!",
        }
    )
    proposal_b = CommunicationActionProposal.build(
        action_type="reply",
        channel="whatsapp",
        account_id="personal",
        conversation_id="dm-b",
        conversation_kind="direct",
        source_event_ids=[event_a.event_id],
        payload="Thank you!",
        category="social_ack",
        risk_class="low",
    )

    assert CommunicationPolicy(
        mode="bounded",
        auto_categories=("social_ack",),
        auto_actions=("reply",),
    ).evaluate(proposal_b, source_events=[event_a]) == (
        "deny",
        ("scope:cross_conversation",),
    )


def test_triage_separates_simple_birthday_from_mixed_financial_request() -> None:
    routine = _event(
        message_id="birthday-simple",
        content="Happy birthday! 🎉",
        kind="direct",
    )
    mixed = _event(
        message_id="birthday-money",
        content="Happy birthday! Can you send me $500 today?",
        kind="direct",
    )

    routine_assessment = assess_event(routine)
    mixed_assessment = assess_event(mixed)

    assert routine_assessment.category == "birthday_wish"
    assert routine_assessment.attention == "routine_candidate"
    assert routine_assessment.risk_class == "low"
    assert routine_assessment.routine_social is True
    assert routine_assessment.direct_reply_candidate is True

    assert mixed_assessment.category == "financial"
    assert mixed_assessment.attention == "review"
    assert mixed_assessment.risk_class == "high"
    assert mixed_assessment.request is True
    assert mixed_assessment.routine_social is False
    assert mixed_assessment.direct_reply_candidate is False


def test_group_episode_coalesces_100_verified_wishes_but_not_unknown_scope() -> None:
    group_events = [
        event_from_adapter(
            {
                "direction": "inbound",
                "channel": "whatsapp",
                "account_id": "personal",
                "conversation_id": "family-group",
                "conversation_kind": "group",
                "sender_id": f"member-{index}",
                "message_id": f"group-wish-{index}",
                "timestamp": 1_700_000_000 + index,
                "content": "Happy birthday!",
            }
        )
        for index in range(100)
    ]
    unknown = event_from_adapter(
        {
            "direction": "inbound",
            "channel": "whatsapp",
            "account_id": "personal",
            "conversation_id": "unknown-chat",
            "conversation_kind": "unknown",
            "sender_id": "person-x",
            "message_id": "unknown-wish",
            "timestamp": 1_700_000_050,
            "content": "Happy birthday!",
        }
    )

    episodes = build_group_episodes(group_events + [unknown])

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.category == "birthday_wish"
    assert episode.conversation_id == "family-group"
    assert len(episode.source_event_ids) == 100
    assert len(episode.participant_ids) == 100
    assert unknown.event_id not in episode.source_event_ids


def test_digest_reduces_mass_social_noise_but_surfaces_exception() -> None:
    routine = [
        event_from_adapter(
            {
                "direction": "inbound",
                "channel": "whatsapp",
                "account_id": "personal",
                "conversation_id": f"dm-{index}",
                "conversation_kind": "direct",
                "sender_id": f"person-{index}",
                "message_id": f"wish-{index}",
                "timestamp": 1_700_000_000 + index,
                "content": "Happy birthday!",
            }
        )
        for index in range(100)
    ]
    exception = event_from_adapter(
        {
            "direction": "inbound",
            "channel": "whatsapp",
            "account_id": "personal",
            "conversation_id": "dm-important",
            "conversation_kind": "direct",
            "sender_id": "important-person",
            "message_id": "wish-question",
            "timestamp": 1_700_000_500,
            "content": "Happy birthday! Can you call me urgently?",
        }
    )

    digest = build_digest(routine + [exception])

    assert digest["total_events"] == 101
    assert digest["routine_candidate_count"] == 100
    assert digest["attention_count"] == 1
    assert digest["urgent_count"] == 1
    assert digest["attention_items"][0]["event_id"] == exception.event_id


def test_inferred_taste_requires_real_evidence() -> None:
    with pytest.raises(CommunicationStateError, match="evidence"):
        CommunicationTaste.build(
            scope_type="contact",
            scope_id="person-a",
            source="inferred",
            confidence=0.8,
            response_length="short",
        )


def test_explicit_narrow_taste_overrides_broader_inferred_taste() -> None:
    owner = CommunicationTaste.build(
        scope_type="owner",
        scope_id="owner",
        source="inferred",
        confidence=0.9,
        evidence_event_ids=("e1", "e2", "e3"),
        response_length="short",
        emoji_level="expressive",
    )
    contact = CommunicationTaste.build(
        scope_type="contact",
        scope_id="person-a",
        source="explicit",
        confidence=1.0,
        response_length="very_short",
        emoji_level="none",
    )

    resolved = resolve_taste(owner, contact)

    assert resolved["response_length"] == "very_short"
    assert resolved["emoji_level"] == "none"


def test_taste_inference_uses_only_observed_outbound_surface_traits() -> None:
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
                "content": text,
                "delivery_state": "sent",
            }
        )
        for index, text in enumerate(("Thanks 😊", "Sure 👍", "Done 😊"))
    ]

    taste = infer_taste_from_outbound(
        events,
        scope_type="contact",
        scope_id="person-a",
    )

    assert taste is not None
    assert taste.source == "inferred"
    assert taste.routine_action == "none"
    assert taste.response_length == "very_short"
    assert taste.emoji_level == "expressive"
    assert set(taste.evidence_event_ids) == {event.event_id for event in events}


def test_communication_memory_persists_partitioned_episodic_taste(
    tmp_path: Path,
) -> None:
    memory_path = tmp_path / "taste-memory.json"
    taste = CommunicationTaste.build(
        scope_type="contact",
        scope_id="person-a",
        source="inferred",
        confidence=0.9,
        evidence_event_ids=("e1", "e2", "e3"),
        response_length="short",
        emoji_level="light",
    )

    memory = CommunicationMemory(
        memory_path,
        enable_long_term=False,
        enable_native=False,
    )
    remembered = memory.remember_taste(taste)

    assert remembered["tier"] == "episodic"
    assert remembered["authority_expanded"] is False

    restored = CommunicationMemory(
        memory_path,
        enable_long_term=False,
        enable_native=False,
    )
    recalled = restored.recall_tastes(
        scope_type="contact",
        scope_id="person-a",
    )
    wrong_scope = restored.recall_tastes(
        scope_type="contact",
        scope_id="person-b",
    )

    assert recalled
    assert recalled[0].response_length == "short"
    assert recalled[0].emoji_level == "light"
    assert wrong_scope == []


def test_contact_taste_never_mirrors_to_global_hippocampus(tmp_path: Path) -> None:
    class FakeLongTerm:
        active = True

        def __init__(self) -> None:
            self.calls = []

        def remember_fragments(self, fragments, *, selected_ids):
            self.calls.append((fragments, selected_ids))
            return 1

    memory = CommunicationMemory(
        tmp_path / "taste-memory.json",
        enable_long_term=False,
        enable_native=False,
    )
    fake = FakeLongTerm()
    memory.fabric._long_term = fake
    taste = CommunicationTaste.build(
        scope_type="contact",
        scope_id="person-a",
        source="inferred",
        confidence=0.95,
        evidence_event_ids=("e1", "e2", "e3", "e4"),
        response_length="short",
    )

    result = memory.remember_taste(taste)

    assert result["long_term"]["reason"] == "scope_partition_required"
    assert fake.calls == []


def test_strong_owner_taste_can_use_optional_hippocampus_mirror(
    tmp_path: Path,
) -> None:
    class FakeLongTerm:
        active = True

        def __init__(self) -> None:
            self.calls = []

        def remember_fragments(self, fragments, *, selected_ids):
            self.calls.append((fragments, selected_ids))
            return 1

    memory = CommunicationMemory(
        tmp_path / "taste-memory.json",
        enable_long_term=False,
        enable_native=False,
    )
    fake = FakeLongTerm()
    memory.fabric._long_term = fake
    taste = CommunicationTaste.build(
        scope_type="owner",
        scope_id="owner-main",
        source="inferred",
        confidence=0.9,
        evidence_event_ids=("e1", "e2", "e3"),
        response_length="short",
    )

    result = memory.remember_taste(taste)

    assert result["long_term"]["remembered"] is True
    assert result["long_term"]["reason"] == "hippocampus_active"
    assert len(fake.calls) == 1


def test_explicit_taste_replaces_current_policy_state(tmp_path: Path) -> None:
    path = tmp_path / "communication.sqlite3"
    first = CommunicationTaste.build(
        scope_type="contact",
        scope_id="person-a",
        source="explicit",
        confidence=1.0,
        response_length="short",
        emoji_level="light",
    )
    replacement = CommunicationTaste.build(
        scope_type="contact",
        scope_id="person-a",
        source="explicit",
        confidence=1.0,
        response_length="very_short",
        emoji_level="none",
    )

    with CommunicationStore(path, retention_days=0) as store:
        store.set_explicit_taste(first, now=100)
        store.set_explicit_taste(replacement, now=101)
        current = store.get_explicit_taste(
            scope_type="contact",
            scope_id="person-a",
        )

    assert current is not None
    assert current.response_length == "very_short"
    assert current.emoji_level == "none"


def test_bridge_assure_claim_delivery_and_restart_prevent_duplicate_send(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "communication.sqlite3"
    inbound = {
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
            "timestamp": 1_700_000_000,
            "content": "Happy birthday!",
            "source": "test",
        },
    }
    ingested = handle_request(inbound)
    event_id = ingested["event_id"]
    assure_request = {
        "operation": "communication_assure",
        "store_path": str(store_path),
        "retention_days": 0,
        "channel": "whatsapp",
        "account_id": "personal",
        "conversation_id": "dm-a",
        "action_type": "reply",
        "source_event_ids": [event_id],
        "payload": "Thank you so much!",
        "policy_mode": "bounded",
        "auto_actions": ["reply"],
        "auto_categories": ["birthday_wish"],
    }

    assured = handle_request(assure_request)
    assert assured["decision"] == "allow"
    assert assured["execution_state"] == "assured"

    first_claim = handle_request(
        {
            "operation": "communication_begin_action",
            "store_path": str(store_path),
            "action_id": assured["action_id"],
        }
    )
    second_claim = handle_request(
        {
            "operation": "communication_begin_action",
            "store_path": str(store_path),
            "action_id": assured["action_id"],
        }
    )
    assert first_claim["claimed"] is True
    assert first_claim["execution_state"] == "dispatching"
    assert second_claim["claimed"] is False

    outbound = handle_request(
        {
            "operation": "communication_ingest",
            "store_path": str(store_path),
            "retention_days": 0,
            "event": {
                "direction": "outbound",
                "channel": "whatsapp",
                "account_id": "personal",
                "conversation_id": "dm-a",
                "conversation_kind": "direct",
                "recipient_id": "person-a",
                "message_id": "sent-1",
                "content": "Thank you so much!",
                "delivery_state": "sent",
                "source": "openclaw.message_sent",
            },
        }
    )
    assert outbound["correlated_action_id"] == assured["action_id"]

    # A fresh bridge/store instance sees the durable handled state.
    after_restart = handle_request(assure_request)
    assert after_restart["decision"] == "already_handled"
    assert after_restart["execution_state"] == "sent"


def test_unknown_chat_kind_blocks_bounded_text_reply(tmp_path: Path) -> None:
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
                "conversation_id": "opaque-chat",
                "conversation_kind": "unknown",
                "sender_id": "person-a",
                "message_id": "birthday-1",
                "content": "Happy birthday!",
            },
        }
    )

    assured = handle_request(
        {
            "operation": "communication_assure",
            "store_path": str(store_path),
            "retention_days": 0,
            "channel": "whatsapp",
            "account_id": "personal",
            "conversation_id": "opaque-chat",
            "action_type": "reply",
            "source_event_ids": [ingested["event_id"]],
            "payload": "Thank you!",
            "policy_mode": "bounded",
            "auto_actions": ["reply"],
            "auto_categories": ["birthday_wish"],
        }
    )

    assert assured["decision"] == "ambiguous"
    assert assured["reasons"] == ["scope:direct_not_verified"]


def test_global_digest_requires_trusted_owner_authority(tmp_path: Path) -> None:
    store_path = tmp_path / "communication.sqlite3"
    handle_request(
        {
            "operation": "communication_ingest",
            "store_path": str(store_path),
            "event": {
                "direction": "inbound",
                "channel": "whatsapp",
                "conversation_id": "dm-a",
                "message_id": "m-1",
                "content": "Can you call me?",
            },
        }
    )

    with pytest.raises(PermissionError, match="owner"):
        handle_request(
            {
                "operation": "communication_digest",
                "store_path": str(store_path),
                "channel": "whatsapp",
            }
        )

    result = handle_request(
        {
            "operation": "communication_digest",
            "store_path": str(store_path),
            "channel": "whatsapp",
            "owner_authorized": True,
        }
    )
    assert result["scope"] == "owner_global"
    assert result["digest"]["attention_count"] == 1


def test_explicit_taste_bridge_requires_owner_and_overrides_memory(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "communication.sqlite3"
    memory_path = tmp_path / "taste-memory.json"

    with pytest.raises(PermissionError, match="owner"):
        handle_request(
            {
                "operation": "communication_set_taste",
                "store_path": str(store_path),
                "scope_type": "owner",
                "scope_id": "main",
                "taste": {"response_length": "very_short"},
            }
        )

    handle_request(
        {
            "operation": "communication_set_taste",
            "store_path": str(store_path),
            "scope_type": "owner",
            "scope_id": "main",
            "owner_authorized": True,
            "taste": {
                "response_length": "very_short",
                "emoji_level": "none",
            },
        }
    )
    result = handle_request(
        {
            "operation": "communication_resolve_taste",
            "store_path": str(store_path),
            "memory_path": str(memory_path),
            "owner_authorized": True,
            "scopes": [{"scope_type": "owner", "scope_id": "main"}],
        }
    )

    assert result["resolved"]["response_length"] == "very_short"
    assert result["resolved"]["emoji_level"] == "none"
    assert result["authority_expanded"] is False
