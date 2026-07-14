from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

import pytest

from entroly.integrations.discord_gateway import DiscordGateway
from entroly.integrations.slack_gateway import SlackGateway
from entroly.integrations.telegram_gateway import TelegramGateway


@pytest.mark.parametrize("gateway_kind", ["slack", "discord", "telegram"])
def test_gateway_send_uses_durable_dispatcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gateway_kind: str,
) -> None:
    db = tmp_path / "delivery.sqlite3"
    calls: list[str] = []

    if gateway_kind == "slack":
        gateway: Any = SlackGateway(
            "https://hooks.slack.com/services/test",
            delivery_db_path=db,
        )
        monkeypatch.setattr(
            gateway,
            "_send_now",
            lambda text: calls.append(text) or {"ok": True, "status": 200},
        )
    elif gateway_kind == "discord":
        gateway = DiscordGateway(
            "https://discord.com/api/webhooks/test",
            delivery_db_path=db,
        )
        monkeypatch.setattr(
            gateway,
            "_send_now",
            lambda text: calls.append(text) or {"ok": True, "status": 204},
        )
    else:
        gateway = TelegramGateway(
            "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef",
            "42",
            delivery_db_path=db,
        )

        def telegram_call(method: str, **params: Any) -> dict[str, Any]:
            calls.append(str(params.get("text", "")))
            return {"ok": True, "status": 200}

        monkeypatch.setattr(gateway, "_call", telegram_call)

    first = gateway.send("event", event_key="event:1")
    second = gateway.send("event", event_key="event:1")

    assert first["ok"] is True
    assert first["inserted"] is True
    assert len(first["receipt"]) == 64
    assert second["ok"] is True
    assert second["inserted"] is False
    assert second["event_id"] == first["event_id"]
    assert calls == ["event"]
    assert gateway.delivery_stats() == {
        "pending": 0,
        "delivered": 1,
        "dead": 0,
        "total": 1,
    }


def test_daemon_counter_event_is_not_resent_after_gateway_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "delivery.sqlite3"
    calls: list[str] = []
    webhook = "https://hooks.slack.com/services/restart-test"

    first = SlackGateway(webhook, delivery_db_path=db)
    monkeypatch.setattr(
        first,
        "_send_now",
        lambda text: calls.append(text) or {"ok": True, "status": 200},
    )
    stats = {
        "skills_promoted": 3,
        "skills_pruned": 0,
        "structural_successes": 0,
        "dream_cycles": 0,
    }
    first._surface_delta({}, stats)

    restarted = SlackGateway(webhook, delivery_db_path=db)
    monkeypatch.setattr(
        restarted,
        "_send_now",
        lambda text: calls.append(text) or {"ok": True, "status": 200},
    )
    restarted._surface_delta({}, stats)

    assert len(calls) == 1
    assert "Total: 3" in calls[0]
    assert restarted.delivery_stats()["delivered"] == 1


def test_failed_gateway_send_returns_queued_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = DiscordGateway(
        "https://discord.com/api/webhooks/outage",
        delivery_db_path=tmp_path / "delivery.sqlite3",
    )
    monkeypatch.setattr(
        gateway,
        "_send_now",
        lambda text: {"ok": False, "error": "timeout"},
    )

    result = gateway.send("will retry", event_key="outage:1")

    assert result["ok"] is False
    assert result["queued"] is True
    assert result["attempts"] == 1
    assert result["next_attempt_at"] is not None
    assert gateway.delivery_stats()["pending"] == 1


@pytest.mark.parametrize("gateway_kind", ["slack", "discord", "telegram"])
def test_gateway_start_replays_due_events_before_announcing_online(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gateway_kind: str,
) -> None:
    db = tmp_path / "delivery.sqlite3"
    if gateway_kind == "slack":
        destination = "https://hooks.slack.com/services/replay-test"
        failed: Any = SlackGateway(destination, delivery_db_path=db)
        monkeypatch.setattr(failed, "_send_now", lambda text: {"ok": False, "error": "offline"})
    elif gateway_kind == "discord":
        destination = "https://discord.com/api/webhooks/replay-test"
        failed = DiscordGateway(destination, delivery_db_path=db)
        monkeypatch.setattr(failed, "_send_now", lambda text: {"ok": False, "error": "offline"})
    else:
        destination = ("123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef", "42")
        failed = TelegramGateway(*destination, delivery_db_path=db)
        monkeypatch.setattr(failed, "_call", lambda method, **params: {"ok": False, "error": "offline"})

    pending = failed.send("persisted-before-restart", event_key="restart:1")
    assert pending["queued"] is True
    with sqlite3.connect(db) as connection:
        connection.execute(
            "UPDATE delivery_events SET not_before = 0 WHERE event_id = ?",
            (pending["event_id"],),
        )

    calls: list[str] = []
    if gateway_kind == "slack":
        restarted: Any = SlackGateway(destination, poll_interval_s=60, delivery_db_path=db)
        monkeypatch.setattr(
            restarted,
            "_send_now",
            lambda text: calls.append(text) or {"ok": True, "status": 200},
        )
    elif gateway_kind == "discord":
        restarted = DiscordGateway(destination, poll_interval_s=60, delivery_db_path=db)
        monkeypatch.setattr(
            restarted,
            "_send_now",
            lambda text: calls.append(text) or {"ok": True, "status": 204},
        )
    else:
        restarted = TelegramGateway(*destination, poll_interval_s=60, delivery_db_path=db)

        def telegram_call(method: str, **params: Any) -> dict[str, Any]:
            if method == "sendMessage":
                calls.append(str(params.get("text", "")))
            return {"ok": True, "status": 200, "result": []}

        monkeypatch.setattr(restarted, "_call", telegram_call)

    restarted.start()
    restarted.stop()

    assert calls[0] == "persisted-before-restart"
    assert len(restarted.startup_replay()) == 1
    assert restarted.startup_replay()[0]["ok"] is True
    assert restarted.delivery_stats()["pending"] == 0
