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


def test_reports_belief_whose_source_is_gone(tmp_path):
    """Compilation only ever adds, so a belief outlives the file it came from.

    Nothing previously distinguished such a belief from a live one: it kept
    its confidence and ranked beside beliefs that still have evidence.
    """

    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "src" / "live.py").write_text("x = 1\n", encoding="utf-8")

    vault = VaultManager(VaultConfig(base_path=str(project / ".entroly" / "vault")))
    vault.write_belief(
        BeliefArtifact(entity="live", title="live", body="Describes a real module.",
                       sources=["src/live.py:1"])
    )
    vault.write_belief(
        BeliefArtifact(entity="removed", title="removed", body="Describes a deleted module.",
                       sources=["src/removed.py:1"])
    )

    report = VaultHygiene(vault._base).scan()
    entities = {row["entity"] for row in report["ungrounded"]}

    assert "removed" in entities
    assert "live" not in entities


def test_source_recorded_relative_to_the_compiled_dir_is_still_grounded(tmp_path):
    """A belief does not record which directory was compiled.

    `entroly compile scripts` writes `helper.py` while `entroly compile .`
    writes `scripts/helper.py`, for the same file. Resolving only against the
    project root marks the first form dead -- which is what made a naive check
    retract 275 of 715 real beliefs on the entroly repo.
    """

    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts" / "helper.py").write_text("y = 2\n", encoding="utf-8")

    vault = VaultManager(VaultConfig(base_path=str(project / ".entroly" / "vault")))
    vault.write_belief(
        BeliefArtifact(entity="helper", title="helper", body="Compiled from scripts/.",
                       sources=["helper.py:1"])
    )

    report = VaultHygiene(vault._base).scan()

    assert [row["entity"] for row in report["ungrounded"]] == []


def test_groundedness_is_reported_but_does_not_fail_the_health_verdict(tmp_path):
    """Groundedness depends on inferring a root the belief never recorded.

    A vault read from an unexpected working directory would otherwise report
    every belief dead, so the finding informs a caller instead of gating.
    """

    project = tmp_path / "project"
    project.mkdir()
    vault = VaultManager(VaultConfig(base_path=str(project / ".entroly" / "vault")))
    vault.write_belief(
        BeliefArtifact(entity="orphan", title="orphan", body="No file backs this.",
                       sources=["nowhere/at/all.py:1"])
    )

    report = VaultHygiene(vault._base).scan()

    assert report["ungrounded"]
    assert report["healthy"] is True
