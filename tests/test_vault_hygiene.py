"""Tests for the belief-vs-belief hygiene scan (entroly/vault_hygiene.py)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from entroly.vault import BeliefArtifact, VaultConfig, VaultManager
from entroly.vault_hygiene import VaultHygiene
from entroly.vault_time import BeliefLedger


@pytest.fixture()
def vault(tmp_path):
    return VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))


def _write(vault, entity, body, **kw):
    vault.write_belief(BeliefArtifact(entity=entity, title=entity, body=body,
                                      sources=["test"], **kw))


def test_detects_contradicting_beliefs(vault):
    _write(vault, "payments", "The payment service uses Stripe for all transactions.")
    _write(vault, "payments-v2", "The payment service does not use Stripe for transactions.")
    _write(vault, "database", "The primary database runs on postgres 15.")

    report = VaultHygiene(vault._base).scan()
    assert report["healthy"] is False
    assert len(report["contradictions"]) == 1
    pair = set(report["contradictions"][0]["entities"])
    assert pair == {"payments", "payments-v2"}
    # The unrelated belief must NOT be flagged (contradiction, not tension).
    assert all("database" not in c["entities"] for c in report["contradictions"])


def test_detects_near_duplicates(vault):
    _write(vault, "cache-a", "Caching uses LRU eviction with a 512MB cap.")
    _write(vault, "cache-b", "Caching uses LRU eviction with a 512MB cap today.")

    report = VaultHygiene(vault._base).scan()
    assert len(report["duplicates"]) == 1
    assert report["duplicates"][0]["suggestion"] == "merge"


def test_flags_stale_beliefs(vault):
    old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    _write(vault, "ancient", "The build uses webpack 4.", last_checked=old)
    _write(vault, "fresh", "The build uses vite.")

    report = VaultHygiene(vault._base).scan()
    stale_entities = {s["entity"] for s in report["stale"]}
    assert "ancient" in stale_entities
    assert "fresh" not in stale_entities
    assert report["stale"][0]["suggestion"] == "refresh_beliefs"


def test_detects_confidence_flapping_from_ledger(vault):
    ledger = BeliefLedger(vault._base)
    for conf in (0.9, 0.3, 0.8, 0.2, 0.7):
        ledger.record(BeliefArtifact(entity="flappy", title="flappy",
                                     body="Service X handles retries.",
                                     confidence=conf))
    report = VaultHygiene(vault._base).scan()
    assert len(report["confidence_flapping"]) == 1
    flap = report["confidence_flapping"][0]
    assert flap["entity"] == "flappy"
    assert flap["reversals"] >= 3
    assert flap["suggestion"] == "escalate_verification"


def test_clean_vault_reports_healthy(vault):
    _write(vault, "api", "REST endpoints live under /api/v2.")
    _write(vault, "auth", "Login uses OAuth device flow.")

    report = VaultHygiene(vault._base).scan()
    assert report["healthy"] is True
    assert report["contradictions"] == []
    assert report["duplicates"] == []
