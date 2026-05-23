"""
Slack Gateway
=============

Surfaces the self-evolution daemon's events to a Slack channel via
an Incoming Webhook. Zero dependencies — stdlib urllib only.

Configuration (env):
    ENTROLY_SLACK_WEBHOOK   Incoming webhook URL from Slack app settings
    ENTROLY_SLACK_POLL_S    Seconds between daemon-stat polls (default 30)

Usage:
    # Standalone
    python -m entroly.integrations.slack_gateway

    # Programmatic
    from entroly.integrations.slack_gateway import SlackGateway
    gw = SlackGateway(webhook_url=...)
    gw.attach(daemon)
    gw.start()
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Any

logger = logging.getLogger("entroly.slack_gateway")


class SlackGateway:
    def __init__(
        self,
        webhook_url: str,
        poll_interval_s: float = 30.0,
    ):
        self._url = webhook_url
        self._poll_s = poll_interval_s

        self._daemon: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_stats: dict[str, Any] = {}

    def attach(self, daemon: Any) -> None:
        self._daemon = daemon

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="entroly-slack-gateway", daemon=True
        )
        self._thread.start()
        self.send(":dna: *Entroly gateway online* — watching the self-evolution loop.")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def send(self, text: str) -> dict[str, Any]:
        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return {"ok": r.status < 300, "status": r.status}
        except Exception as e:
            logger.debug("Slack send failed: %s", e)
            return {"ok": False, "error": str(e)}

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self._daemon is not None:
                    stats = self._daemon.stats()
                    self._surface_delta(self._last_stats, stats)
                    self._last_stats = dict(stats)
            except Exception as e:
                logger.debug("Gateway tick error: %s", e)
            self._stop.wait(self._poll_s)

    def _surface_delta(
        self, prev: dict[str, Any], now: dict[str, Any]
    ) -> None:
        def diff(key: str) -> int:
            return int(now.get(key, 0)) - int(prev.get(key, 0))

        if diff("skills_promoted"):
            self.send(
                f":white_check_mark: *Skill promoted* — +{diff('skills_promoted')}. "
                f"Total: {now.get('skills_promoted', 0)}."
            )
        if diff("skills_pruned"):
            self.send(f":wastebasket: *Skill pruned* — +{diff('skills_pruned')}.")
        if diff("structural_successes"):
            self.send(
                f":brain: *Structural synthesis* — +{diff('structural_successes')} "
                f"($0, deterministic)."
            )
        if diff("dream_cycles"):
            self.send(f":thought_balloon: *Dream cycle complete* — +{diff('dream_cycles')}.")


def _main() -> int:
    url = os.environ.get("ENTROLY_SLACK_WEBHOOK")
    if not url:
        print("Set ENTROLY_SLACK_WEBHOOK to run the gateway.", flush=True)
        return 2

    poll_s = float(os.environ.get("ENTROLY_SLACK_POLL_S", "30"))
    gw = SlackGateway(webhook_url=url, poll_interval_s=poll_s)

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
        logger.warning("Running without attached daemon: %s", e)

    gw.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        gw.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
