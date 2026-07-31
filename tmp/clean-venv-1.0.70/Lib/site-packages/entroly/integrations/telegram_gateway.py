"""
Telegram Gateway
================

Makes Entroly's self-evolution daemon visible in Telegram. Outbound messages are
persisted before network I/O, retried with bounded exponential backoff, and
replayed after restart. Bot tokens and chat identifiers are never persisted.

Configuration (env):
    ENTROLY_TG_TOKEN      Bot token from @BotFather
    ENTROLY_TG_CHAT_ID    Target chat ID (user, group, or channel)
    ENTROLY_TG_POLL_S     Seconds between daemon-stat polls (default 30)
    ENTROLY_DELIVERY_DB   Shared durable queue path
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .event_delivery import DeliveryEvent, ReliableEventDispatcher

logger = logging.getLogger("entroly.telegram_gateway")

API_BASE = "https://api.telegram.org/bot{token}/{method}"
_MAX_RESPONSE_BYTES = 1_048_576


class TelegramGateway:
    def __init__(
        self,
        token: str,
        chat_id: str | int,
        vault_path: str | Path = ".entroly/vault",
        poll_interval_s: float = 30.0,
        delivery_db_path: str | Path | None = None,
        max_delivery_attempts: int = 8,
    ):
        self._token = token
        self._chat_id = str(chat_id)
        self._vault = Path(vault_path)
        self._poll_s = poll_interval_s

        self._daemon: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._update_offset = 0
        self._last_stats: dict[str, Any] = {}
        self._startup_replay: tuple[dict[str, Any], ...] = ()
        db_path = delivery_db_path or os.environ.get(
            "ENTROLY_DELIVERY_DB", ".entroly/event-delivery.sqlite3"
        )
        self._delivery = ReliableEventDispatcher(
            channel="telegram",
            destination_identity=f"{token}:{self._chat_id}",
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
        replayed = self._delivery.flush()
        self._startup_replay = tuple(outcome.to_dict() for outcome in replayed)
        failed = [outcome for outcome in replayed if not outcome.ok]
        if failed:
            logger.warning(
                "Telegram gateway startup replay left %d event(s) queued or dead",
                len(failed),
            )
        self._thread = threading.Thread(
            target=self._loop, name="entroly-tg-gateway", daemon=True
        )
        self._thread.start()
        self.send("🧬 *Entroly gateway online* — watching the self-evolution loop.")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._delivery.flush()

    @staticmethod
    def _retry_after(payload: Any) -> float | None:
        try:
            value = payload.get("parameters", {}).get("retry_after")
            return float(value) if value is not None else None
        except (AttributeError, TypeError, ValueError):
            return None

    def _call(self, method: str, **params: Any) -> dict[str, Any]:
        url = API_BASE.format(token=self._token, method=method)
        data = urllib.parse.urlencode(
            {key: value for key, value in params.items() if value is not None}
        ).encode("utf-8")
        try:
            with urllib.request.urlopen(url, data=data, timeout=15) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    return {"ok": False, "error": "response_too_large"}
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    return {"ok": False, "error": "invalid_response_shape"}
                retry_after = self._retry_after(payload)
                if retry_after is not None:
                    payload["retry_after_s"] = retry_after
                payload.setdefault("status", response.status)
                return payload
        except urllib.error.HTTPError as exc:
            payload: dict[str, Any] = {}
            try:
                raw = exc.read(_MAX_RESPONSE_BYTES + 1)
                if len(raw) <= _MAX_RESPONSE_BYTES:
                    decoded = json.loads(raw.decode("utf-8"))
                    if isinstance(decoded, dict):
                        payload = decoded
            except Exception:
                payload = {}
            logger.debug("Telegram call failed with HTTP %s", exc.code)
            return {
                "ok": False,
                "status": exc.code,
                "error": f"http_{exc.code}",
                "retry_after_s": self._retry_after(payload),
            }
        except Exception as exc:
            logger.debug("Telegram call failed: %s", type(exc).__name__)
            return {"ok": False, "error": type(exc).__name__}

    def _deliver_event(self, event: DeliveryEvent) -> dict[str, Any]:
        return self._call(
            "sendMessage",
            chat_id=self._chat_id,
            text=str(event.payload.get("text", "")),
            parse_mode="Markdown",
            disable_web_page_preview="true",
        )

    def send(
        self,
        text: str,
        *,
        event_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist then deliver one event; repeated ``event_key`` values deduplicate."""
        return self._delivery.publish(
            text,
            idempotency_key=event_key,
            metadata=metadata,
        )

    def flush(self) -> list[Any]:
        return self._delivery.flush()

    def delivery_stats(self) -> dict[str, int]:
        return self._delivery.stats()

    def startup_replay(self) -> tuple[dict[str, Any], ...]:
        """Return immutable outcomes from the synchronous restart replay."""
        return self._startup_replay

    def _get_updates(self) -> list[dict[str, Any]]:
        response = self._call("getUpdates", offset=self._update_offset, timeout=0)
        if not response.get("ok"):
            return []
        updates = response.get("result", [])
        if not isinstance(updates, list):
            return []
        if updates:
            last = updates[-1]
            if isinstance(last, dict) and "update_id" in last:
                self._update_offset = int(last["update_id"]) + 1
        return [item for item in updates if isinstance(item, dict)]

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._delivery.flush()
                self._tick()
            except Exception as exc:
                logger.debug("Gateway tick error: %s", type(exc).__name__)
            self._stop.wait(self._poll_s)

    def _tick(self) -> None:
        if self._daemon is not None:
            stats = self._daemon.stats()
            self._surface_delta(self._last_stats, stats)
            self._last_stats = dict(stats)

        for update in self._get_updates():
            message = update.get("message") or update.get("channel_post") or {}
            if not isinstance(message, dict):
                continue
            text = str(message.get("text") or "").strip()
            if not text.startswith("/"):
                continue
            command = text.split()[0].lower().split("@")[0]
            self._handle_command(command)

    def _surface_delta(
        self, prev: dict[str, Any], now: dict[str, Any]
    ) -> None:
        def diff(key: str) -> int:
            return int(now.get(key, 0)) - int(prev.get(key, 0))

        promoted = diff("skills_promoted")
        if promoted:
            total = int(now.get("skills_promoted", 0))
            self.send(
                f"✅ *Skill promoted* — +{promoted}. Total promoted: {total}.",
                event_key=f"skills_promoted:{total}",
            )
        pruned = diff("skills_pruned")
        if pruned:
            total = int(now.get("skills_pruned", 0))
            self.send(
                f"🗑️ *Skill pruned* — +{pruned}.",
                event_key=f"skills_pruned:{total}",
            )
        structural = diff("structural_successes")
        if structural:
            total = int(now.get("structural_successes", 0))
            self.send(
                f"🧠 *Structural synthesis* — +{structural} ($0, deterministic).",
                event_key=f"structural_successes:{total}",
            )
        dreams = diff("dream_cycles")
        if dreams:
            total = int(now.get("dream_cycles", 0))
            self.send(
                f"💭 *Dream cycle complete* — +{dreams}.",
                event_key=f"dream_cycles:{total}",
            )

    def _handle_command(self, command: str) -> None:
        if command == "/status":
            self._cmd_status()
        elif command == "/skills":
            self._cmd_skills()
        elif command == "/gaps":
            self._cmd_gaps()
        elif command == "/dream":
            self._cmd_dream()
        elif command in {"/help", "/start"}:
            self.send(
                "Commands:\n"
                "/status — daemon stats + budget\n"
                "/skills — promoted skills\n"
                "/gaps — pending coverage gaps\n"
                "/dream — last dreaming-loop result"
            )

    def _cmd_status(self) -> None:
        if self._daemon is None:
            self.send("_No daemon attached._")
            return
        stats = self._daemon.stats()
        budget = stats.get("budget", {})
        self.send(
            "*Entroly daemon status*\n"
            f"running: `{stats.get('running')}`\n"
            f"structural successes: `{stats.get('structural_successes', 0)}`\n"
            f"skills promoted: `{stats.get('skills_promoted', 0)}`\n"
            f"skills pruned: `{stats.get('skills_pruned', 0)}`\n"
            f"dream cycles: `{stats.get('dream_cycles', 0)}`\n"
            f"evolution budget: `${budget.get('available_usd', 0):.4f}` "
            f"(can_evolve: `{budget.get('can_evolve')}`)"
        )

    def _cmd_skills(self) -> None:
        registry = self._vault / "evolution" / "registry.md"
        if not registry.exists():
            self.send("_No registry yet._")
            return
        self.send(
            "*Skill registry*\n```\n"
            + registry.read_text(encoding="utf-8")
            + "\n```"
        )

    def _cmd_gaps(self) -> None:
        gaps_dir = self._vault / "evolution"
        if not gaps_dir.exists():
            self.send("_No gaps._")
            return
        gaps = sorted(gaps_dir.glob("gap_*.md"))
        if not gaps:
            self.send("_No pending gaps._ 🎯")
            return
        lines = [f"• `{gap.stem}`" for gap in gaps[:20]]
        self.send("*Pending gaps*\n" + "\n".join(lines))

    def _cmd_dream(self) -> None:
        if self._daemon is None:
            self.send("_No daemon attached._")
            return
        dream = self._daemon.stats().get("dreaming")
        if not dream:
            self.send("_Dreaming loop not configured._")
            return
        self.send("*Dreaming loop*\n```\n" + json.dumps(dream, indent=2) + "\n```")


def _main() -> int:
    token = os.environ.get("ENTROLY_TG_TOKEN")
    chat_id = os.environ.get("ENTROLY_TG_CHAT_ID")
    if not token or not chat_id:
        print(
            "Set ENTROLY_TG_TOKEN and ENTROLY_TG_CHAT_ID to run the gateway.",
            flush=True,
        )
        return 2

    poll_s = float(os.environ.get("ENTROLY_TG_POLL_S", "30"))
    gateway = TelegramGateway(token=token, chat_id=chat_id, poll_interval_s=poll_s)

    try:
        from entroly.evolution_daemon import EvolutionDaemon
        from entroly.evolution_logger import EvolutionLogger
        from entroly.value_tracker import ValueTracker
        from entroly.vault import VaultConfig, VaultManager

        vault = VaultManager(VaultConfig(base_path=".entroly/vault"))
        vault.ensure_structure()
        daemon = EvolutionDaemon(
            vault=vault,
            evolution_logger=EvolutionLogger(vault),
            value_tracker=ValueTracker(),
        )
        daemon.start()
        gateway.attach(daemon)
    except Exception as exc:
        logger.warning("Running gateway without attached daemon: %s", exc)

    gateway.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        gateway.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
