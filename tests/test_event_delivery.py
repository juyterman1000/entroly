from __future__ import annotations

from pathlib import Path

import pytest

from entroly.integrations.event_delivery import (
    DeliveryReceipt,
    EventDeliveryStore,
    IdempotencyConflict,
    ReliableEventDispatcher,
)


class FakeClock:
    def __init__(self, value: float = 1_700_000_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_idempotency_key_deduplicates_same_event_and_rejects_drift(
    tmp_path: Path,
) -> None:
    store = EventDeliveryStore(tmp_path / "delivery.sqlite3")
    first, inserted_first = store.enqueue(
        channel="slack",
        destination_hash="a" * 64,
        payload={"text": "skill promoted"},
        idempotency_key="skills_promoted:7",
    )
    second, inserted_second = store.enqueue(
        channel="slack",
        destination_hash="a" * 64,
        payload={"text": "skill promoted"},
        idempotency_key="skills_promoted:7",
    )

    assert inserted_first is True
    assert inserted_second is False
    assert first.event_id == second.event_id

    with pytest.raises(IdempotencyConflict):
        store.enqueue(
            channel="slack",
            destination_hash="a" * 64,
            payload={"text": "different payload"},
            idempotency_key="skills_promoted:7",
        )


def test_failed_delivery_survives_restart_and_replays_once_due(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    db = tmp_path / "delivery.sqlite3"
    calls: list[str] = []

    failing = ReliableEventDispatcher(
        channel="discord",
        destination_identity="https://discord.com/api/webhooks/secret",
        sender=lambda event: {"ok": False, "error": "temporary outage"},
        db_path=db,
        base_delay_s=10.0,
        max_delay_s=60.0,
        clock=clock,
    )
    initial = failing.publish("hello", idempotency_key="event:1")

    assert initial["ok"] is False
    assert initial["queued"] is True
    assert initial["attempts"] == 1
    event_id = initial["event_id"]
    pending = failing.store.get(event_id)
    assert pending is not None

    def deliver(event):
        calls.append(event.payload["text"])
        return {"ok": True, "status": 204}

    succeeding = ReliableEventDispatcher(
        channel="discord",
        destination_identity="https://discord.com/api/webhooks/secret",
        sender=deliver,
        db_path=db,
        base_delay_s=10.0,
        max_delay_s=60.0,
        clock=clock,
    )

    assert succeeding.flush() == []
    clock.advance((pending.not_before - clock()) + 0.001)
    outcomes = succeeding.flush()

    assert len(outcomes) == 1
    assert outcomes[0].ok is True
    assert calls == ["hello"]
    delivered = succeeding.store.get(event_id)
    assert delivered is not None
    assert delivered.state == "delivered"
    assert delivered.attempts == 2
    receipt = DeliveryReceipt.from_json(delivered.receipt_json)
    assert receipt.verify() is True
    assert receipt.destination_hash != "https://discord.com/api/webhooks/secret"


def test_retry_exhaustion_moves_event_to_dead_letter(tmp_path: Path) -> None:
    clock = FakeClock()
    dispatcher = ReliableEventDispatcher(
        channel="telegram",
        destination_identity="token:chat",
        sender=lambda event: {"ok": False, "error": "provider unavailable"},
        db_path=tmp_path / "delivery.sqlite3",
        max_attempts=2,
        base_delay_s=0.0,
        max_delay_s=0.0,
        clock=clock,
    )

    first = dispatcher.publish("alert", idempotency_key="alert:1")
    assert first["state"] == "pending"
    outcomes = dispatcher.flush()

    assert len(outcomes) == 1
    assert outcomes[0].state == "dead"
    assert outcomes[0].attempts == 2
    assert dispatcher.stats() == {
        "pending": 0,
        "delivered": 0,
        "dead": 1,
        "total": 1,
    }
    assert dispatcher.store.retry_dead(first["event_id"]) is True
    assert dispatcher.stats()["pending"] == 1


def test_queue_never_persists_destination_or_common_credentials(
    tmp_path: Path,
) -> None:
    db = tmp_path / "delivery.sqlite3"
    webhook = "https://hooks.slack.com/services/T000/B000/VERYSECRET"
    telegram_token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
    dispatcher = ReliableEventDispatcher(
        channel="slack",
        destination_identity=webhook,
        sender=lambda event: {"ok": False, "error": "offline"},
        db_path=db,
    )

    result = dispatcher.publish(
        f"webhook={webhook} token={telegram_token}",
        idempotency_key="secret-test",
    )

    persisted = b"".join(
        path.read_bytes() for path in tmp_path.glob("delivery.sqlite3*")
    )
    assert webhook.encode() not in persisted
    assert telegram_token.encode() not in persisted
    queued = dispatcher.store.get(result["event_id"])
    assert queued is not None
    assert queued.state == "pending"
    assert "VERYSECRET" not in str(queued.payload)
    assert telegram_token not in str(queued.payload)
    assert "[REDACTED]" in str(queued.payload)


def test_receipt_detects_tampering(tmp_path: Path) -> None:
    dispatcher = ReliableEventDispatcher(
        channel="slack",
        destination_identity="destination",
        sender=lambda event: {"ok": True, "status": 200},
        db_path=tmp_path / "delivery.sqlite3",
    )
    result = dispatcher.publish("delivered", idempotency_key="receipt:1")
    event = dispatcher.store.get(result["event_id"])

    assert event is not None
    receipt = DeliveryReceipt.from_json(event.receipt_json)
    assert receipt.verify() is True
    tampered = DeliveryReceipt(
        schema=receipt.schema,
        event_id=receipt.event_id,
        channel=receipt.channel,
        destination_hash=receipt.destination_hash,
        payload_digest=receipt.payload_digest,
        attempts=receipt.attempts + 1,
        delivered_at=receipt.delivered_at,
        provider_status=receipt.provider_status,
        receipt_digest=receipt.receipt_digest,
    )
    assert tampered.verify() is False


def test_prune_removes_only_old_delivered_events(tmp_path: Path) -> None:
    clock = FakeClock()
    dispatcher = ReliableEventDispatcher(
        channel="slack",
        destination_identity="destination",
        sender=lambda event: {"ok": True},
        db_path=tmp_path / "delivery.sqlite3",
        clock=clock,
    )
    dispatcher.publish("old", idempotency_key="old")
    clock.advance(100)
    dispatcher.publish("new", idempotency_key="new")

    assert dispatcher.store.prune_delivered(older_than_s=50) == 1
    assert dispatcher.stats()["delivered"] == 1
