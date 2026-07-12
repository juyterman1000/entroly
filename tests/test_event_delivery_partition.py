from __future__ import annotations

from pathlib import Path

from entroly.integrations.event_delivery import ReliableEventDispatcher


def test_shared_database_never_cross_delivers_channels_or_destinations(
    tmp_path: Path,
) -> None:
    db = tmp_path / "delivery.sqlite3"
    calls: list[tuple[str, str]] = []

    slack_a = ReliableEventDispatcher(
        channel="slack",
        destination_identity="slack-a",
        sender=lambda event: calls.append(("slack-a", event.payload["text"]))
        or {"ok": True},
        db_path=db,
    )
    slack_b = ReliableEventDispatcher(
        channel="slack",
        destination_identity="slack-b",
        sender=lambda event: calls.append(("slack-b", event.payload["text"]))
        or {"ok": True},
        db_path=db,
    )
    discord = ReliableEventDispatcher(
        channel="discord",
        destination_identity="discord-a",
        sender=lambda event: calls.append(("discord", event.payload["text"]))
        or {"ok": True},
        db_path=db,
    )

    slack_a.publish("one", idempotency_key="one", immediate=False)
    slack_b.publish("two", idempotency_key="two", immediate=False)
    discord.publish("three", idempotency_key="three", immediate=False)

    first = slack_a.flush()
    assert [outcome.state for outcome in first] == ["delivered"]
    assert calls == [("slack-a", "one")]
    assert slack_a.stats() == {
        "pending": 2,
        "delivered": 1,
        "dead": 0,
        "total": 3,
    }

    second = discord.flush()
    assert [outcome.state for outcome in second] == ["delivered"]
    assert calls == [("slack-a", "one"), ("discord", "three")]

    third = slack_b.flush()
    assert [outcome.state for outcome in third] == ["delivered"]
    assert calls == [
        ("slack-a", "one"),
        ("discord", "three"),
        ("slack-b", "two"),
    ]
    assert slack_b.stats()["pending"] == 0
