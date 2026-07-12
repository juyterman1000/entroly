from __future__ import annotations

from pathlib import Path
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
        monkeypatch.setattr(
            gateway,
            "_call",
            lambda method, **params: calls.append(str(params.get("text", "")))
            or {"ok": True, "status": 200},
        )

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


def test_failed_gateway_send_returns_queued_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
