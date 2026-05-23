"""
Telegram Gateway
================

Makes the self-evolution daemon *visible*: surfaces gap detection,
structural-synthesis events, skill promotions, and dreaming-loop wins
to a Telegram chat so the autonomy is observable in real time.

Zero hard dependencies — uses stdlib urllib for the Bot API. No
`python-telegram-bot` required.

Configuration (env):
    ENTROLY_TG_TOKEN      Bot token from @BotFather
    ENTROLY_TG_CHAT_ID    Target chat ID (user, group, or channel)
    ENTROLY_TG_POLL_S     Seconds between daemon-stat polls (default 30)

Usage:
    # As a standalone process
    python -m entroly.integrations.telegram_gateway

    # Programmatic
    from entroly.integrations.telegram_gateway import TelegramGateway
    gw = TelegramGateway(token=..., chat_id=...)
    gw.attach(daemon)           # wire to an EvolutionDaemon
    gw.start()                  # background thread

Commands (in chat):
    /status     current daemon stats + budget
    /skills     list promoted skills
    /gaps       pending coverage gaps
    /dream      last dreaming-loop result
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger("entroly.telegram_gateway")

API_BASE = "https://api.telegram.org/bot{token}/{method}"


class TelegramGateway:
    def __init__(
        self,
        token: str,
        chat_id: str | int,
        vault_path: str | Path = ".entroly/vault",
        poll_interval_s: float = 30.0,
    ):
        self._token = token
        self._chat_id = str(chat_id)
        self._vault = Path(vault_path)
        self._poll_s = poll_interval_s

        self._daemon: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._update_offset = 0

        # Snapshot of daemon stats for diff detection
        self._last_stats: dict[str, Any] = {}

    # ── Lifecycle ──────────────────────────────────────────────

    def attach(self, daemon: Any) -> None:
        """Wire a running EvolutionDaemon so we can surface its events."""
        self._daemon = daemon

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="entroly-tg-gateway", daemon=True
        )
        self._thread.start()
        self.send(
            "🧬 *Entroly gateway online* — watching the self-evolution loop."
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    # ── Telegram Bot API ───────────────────────────────────────

    def _call(self, method: str, **params: Any) -> dict[str, Any]:
        url = API_BASE.format(token=self._token, method=method)
        data = urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None}
        ).encode("utf-8")
        try:
            with urllib.request.urlopen(url, data=data, timeout=15) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            logger.debug("Telegram call failed: %s", e)
            return {"ok": False, "error": str(e)}

    def send(self, text: str) -> dict[str, Any]:
        return self._call(
            "sendMessage",
            chat_id=self._chat_id,
            text=text,
            parse_mode="Markdown",
            disable_web_page_preview="true",
        )

    def _get_updates(self) -> list[dict[str, Any]]:
        res = self._call("getUpdates", offset=self._update_offset, timeout=0)
        if not res.get("ok"):
            return []
        updates = res.get("result", [])
        if updates:
            self._update_offset = updates[-1]["update_id"] + 1
        return updates

    # ── Main loop ──────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                logger.debug("Gateway tick error: %s", e)
            self._stop.wait(self._poll_s)

    def _tick(self) -> None:
        # 1. Surface daemon event deltas
        if self._daemon is not None:
            stats = self._daemon.stats()
            self._surface_delta(self._last_stats, stats)
            self._last_stats = dict(stats)

        # 2. Handle inbound commands
        for update in self._get_updates():
            msg = update.get("message") or update.get("channel_post") or {}
            text = (msg.get("text") or "").strip()
            if not text.startswith("/"):
                continue
            cmd = text.split()[0].lower().split("@")[0]
            self._handle_command(cmd)

    # ── Event surfacing ────────────────────────────────────────

    def _surface_delta(
        self, prev: dict[str, Any], now: dict[str, Any]
    ) -> None:
        def diff(key: str) -> int:
            return int(now.get(key, 0)) - int(prev.get(key, 0))

        promoted = diff("skills_promoted")
        pruned = diff("skills_pruned")
        structural = diff("structural_successes")
        dreams = diff("dream_cycles")

        if promoted:
            self.send(
                f"✅ *Skill promoted* — +{promoted}. "
                f"Total promoted: {now.get('skills_promoted', 0)}."
            )
        if pruned:
            self.send(f"🗑️ *Skill pruned* — +{pruned}.")
        if structural:
            self.send(
                f"🧠 *Structural synthesis* — +{structural} ($0, deterministic)."
            )
        if dreams:
            self.send(f"💭 *Dream cycle complete* — +{dreams}.")

    # ── Commands ───────────────────────────────────────────────

    def _handle_command(self, cmd: str) -> None:
        if cmd == "/status":
            self._cmd_status()
        elif cmd == "/skills":
            self._cmd_skills()
        elif cmd == "/gaps":
            self._cmd_gaps()
        elif cmd == "/dream":
            self._cmd_dream()
        elif cmd == "/help" or cmd == "/start":
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
        s = self._daemon.stats()
        budget = s.get("budget", {})
        self.send(
            "*Entroly daemon status*\n"
            f"running: `{s.get('running')}`\n"
            f"structural successes: `{s.get('structural_successes', 0)}`\n"
            f"skills promoted: `{s.get('skills_promoted', 0)}`\n"
            f"skills pruned: `{s.get('skills_pruned', 0)}`\n"
            f"dream cycles: `{s.get('dream_cycles', 0)}`\n"
            f"evolution budget: `${budget.get('available_usd', 0):.4f}` "
            f"(can_evolve: `{budget.get('can_evolve')}`)"
        )

    def _cmd_skills(self) -> None:
        reg = self._vault / "evolution" / "registry.md"
        if not reg.exists():
            self.send("_No registry yet._")
            return
        self.send("*Skill registry*\n```\n" + reg.read_text(encoding="utf-8") + "\n```")

    def _cmd_gaps(self) -> None:
        gaps_dir = self._vault / "evolution"
        if not gaps_dir.exists():
            self.send("_No gaps._")
            return
        gaps = sorted(gaps_dir.glob("gap_*.md"))
        if not gaps:
            self.send("_No pending gaps._ 🎯")
            return
        lines = [f"• `{g.stem}`" for g in gaps[:20]]
        self.send("*Pending gaps*\n" + "\n".join(lines))

    def _cmd_dream(self) -> None:
        if self._daemon is None:
            self.send("_No daemon attached._")
            return
        stats = self._daemon.stats()
        dream = stats.get("dreaming")
        if not dream:
            self.send("_Dreaming loop not configured._")
            return
        self.send("*Dreaming loop*\n```\n" + json.dumps(dream, indent=2) + "\n```")


# ── Standalone entrypoint ──────────────────────────────────────


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
    gw = TelegramGateway(token=token, chat_id=chat_id, poll_interval_s=poll_s)

    # Best-effort: boot a real daemon if dependencies are available.
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
    except Exception as e:
        logger.warning("Running gateway without attached daemon: %s", e)

    gw.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        gw.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
