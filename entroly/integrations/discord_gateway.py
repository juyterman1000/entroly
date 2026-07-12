"""
Discord Gateway
===============

Surfaces Entroly operational events through an incoming webhook. Delivery is
persisted before network I/O, retried with bounded exponential backoff, and
replayed after restart. The webhook credential is never persisted.

Configuration (env):
    ENTROLY_DISCORD_WEBHOOK   Full webhook URL from channel settings
    ENTROLY_DISCORD_POLL_S    Seconds between daemon-stat polls (default 30)
    ENTROLY_DELIVERY_DB       Shared durable queue path
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .event_delivery import DeliveryEvent, ReliableEventDispatcher

logger = logging.getLogger("entroly.discord_gateway")


class DiscordGateway:
    def __init__(
        self,
        webhook_url: str,
        poll_interval_s: float = 30.0,
        username: str = "Entroly",
        delivery_db_path: str | Path | None = None,
        max_delivery_attempts: int = 8,
    ):
        self._url = webhook_url
        self._poll_s = poll_interval_s
        self._username = username

        self._daemon: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_stats: dict[str, Any] = {}
        db_path = delivery_db_path or os.environ.get(
            "ENTROLY_DELIVERY_DB", ".entroly/event-delivery.sqlite3"
        )
        self._delivery = ReliableEventDispatcher(
            channel="discord",
            destination_identity=webhook_url,
            sender=self._deliver_event,
            db_path=db_path,
            max_attempts=max_delivery_attempts,
        )

    def attach(self, daemon: Any) -> None:
        self._daemon = daemon

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="entroly-discord-gateway", daemon=True
        )
        self._thread.start()
        self.send("🧬 **Entroly gateway online** — watching the self-evolution loop.")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._delivery.flush()

    @staticmethod
    def _retry_after(headers: Any) -> float | None:
        try:
            value = headers.get("Retry-After")
            return float(value) if value is not None else None
        except (AttributeError, TypeError, ValueError):
            return None

    def _send_now(self, content: str) -> dict[str, Any]:
        payload = json.dumps(
            {"username": self._username, "content": content}
        ).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return {"ok": response.status < 300, "status": response.status}
        except urllib.error.HTTPError as exc:
            logger.debug("Discord send failed with HTTP %s", exc.code)
            return {
                "ok": False,
                "status": exc.code,
                "error": f"http_{exc.code}",
                "retry_after_s": self._retry_after(exc.headers),
            }
        except Exception as exc:
            logger.debug("Discord send failed: %s", type(exc).__name__)
            return {"ok": False, "error": type(exc).__name__}

    def _deliver_event(self, event: DeliveryEvent) -> dict[str, Any]:
        return self._send_now(str(event.payload.get("text", "")))

    def send(
        self,
        content: str,
        *,
        event_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist then deliver one event; repeated ``event_key`` values deduplicate."""
        return self._delivery.publish(
            content,
            idempotency_key=event_key,
            metadata=metadata,
        )

    def flush(self) -> list[Any]:
        return self._delivery.flush()

    def delivery_stats(self) -> dict[str, int]:
        return self._delivery.stats()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._delivery.flush()
                if self._daemon is not None:
                    stats = self._daemon.stats()
                    self._surface_delta(self._last_stats, stats)
                    self._last_stats = dict(stats)
            except Exception as exc:
                logger.debug("Gateway tick error: %s", type(exc).__name__)
            self._stop.wait(self._poll_s)

    def _surface_delta(
        self, prev: dict[str, Any], now: dict[str, Any]
    ) -> None:
        def diff(key: str) -> int:
            return int(now.get(key, 0)) - int(prev.get(key, 0))

        promoted = diff("skills_promoted")
        if promoted:
            total = int(now.get("skills_promoted", 0))
            self.send(
                f"✅ **Skill promoted** — +{promoted}. Total: {total}.",
                event_key=f"skills_promoted:{total}",
            )
        pruned = diff("skills_pruned")
        if pruned:
            total = int(now.get("skills_pruned", 0))
            self.send(
                f"🗑️ **Skill pruned** — +{pruned}.",
                event_key=f"skills_pruned:{total}",
            )
        structural = diff("structural_successes")
        if structural:
            total = int(now.get("structural_successes", 0))
            self.send(
                f"🧠 **Structural synthesis** — +{structural} ($0, deterministic).",
                event_key=f"structural_successes:{total}",
            )
        dreams = diff("dream_cycles")
        if dreams:
            total = int(now.get("dream_cycles", 0))
            self.send(
                f"💭 **Dream cycle complete** — +{dreams}.",
                event_key=f"dream_cycles:{total}",
            )


def _main() -> int:
    url = os.environ.get("ENTROLY_DISCORD_WEBHOOK")
    if not url:
        print("Set ENTROLY_DISCORD_WEBHOOK to run the gateway.", flush=True)
        return 2

    poll_s = float(os.environ.get("ENTROLY_DISCORD_POLL_S", "30"))
    gw = DiscordGateway(webhook_url=url, poll_interval_s=poll_s)

    try:
        from entroly.evolution_daemon import EvolutionDaemon
        from entroly.evolution_logger import EvolutionLogger
        from entroly.value_tracker import ValueTracker
        from entroly.vault import VaultConfig, VaultManager

        vm = VaultManager(VaultConfig(base_path=".entroly/vault"))
        vm.ensure_structure()
        daemon = EvolutionDaemon(
            vault=vm,
            evolution_logger=EvolutionLogger(vm),
            value_tracker=ValueTracker(),
        )
        daemon.start()
        gw.attach(daemon)
    except Exception as exc:
        logger.warning("Running without attached daemon: %s", exc)

    gw.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        gw.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
