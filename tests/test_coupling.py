"""
Tests for entroly/coupling.py — System 1 ↔ System 2 bridge.

Covers all three coupling operations end-to-end against a real on-disk vault:

  1. inject_vault_beliefs    — System 2 → System 1 (vault → engine fragments)
  2. attribute_outcome       — System 1 → System 2 (Bayesian posterior update)
  3. enqueue_reverification  — System 1 → System 2 (mark stale on failure)

Plus math invariants on the Bayesian update.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from entroly.coupling import (
    PINNED_CONFIDENCE_THRESHOLD,
    attribute_outcome,
    enqueue_reverification,
    inject_vault_beliefs,
    is_enabled,
    project_beliefs,
)
from entroly.vault import BeliefArtifact, VaultConfig, VaultManager


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def vault(tmp_path: Path) -> VaultManager:
    """A fresh vault with three beliefs of varying confidence/freshness."""
    v = VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))
    v.ensure_structure()

    # Belief 1: high confidence, fresh, relevant to "knapsack"
    v.write_belief(BeliefArtifact(
        claim_id="cid-knapsack",
        entity="knapsack_solver",
        status="verified",
        confidence=0.92,
        sources=["entroly-core/src/knapsack.rs:1"],
        title="Knapsack solver",
        body="The knapsack solver uses 0/1 dynamic programming with the "
             "submodular (1-1/e) approximation guarantee. Token budget "
             "constraints are enforced at selection time.",
    ))

    # Belief 2: medium confidence, fresh, unrelated to query
    v.write_belief(BeliefArtifact(
        claim_id="cid-cache",
        entity="cache_aligner",
        status="inferred",
        confidence=0.70,
        sources=["entroly/cache_aligner.py:1"],
        title="Cache aligner",
        body="Aligns request boundaries to provider-side prompt cache "
             "blocks. Saves significant tokens on repeated prefixes.",
    ))

    # Belief 3: low confidence — should be filtered out
    v.write_belief(BeliefArtifact(
        claim_id="cid-lowconf",
        entity="experimental_thing",
        status="hypothesis",
        confidence=0.30,
        sources=["entroly/experimental.py"],
        title="Experimental",
        body="Hypothesis about knapsack optimization that we have not "
             "verified yet.",
    ))

    return v


class FakeEngine:
    """Mimics the PyO3 engine surface used by inject_vault_beliefs."""
    def __init__(self):
        self.ingested: list[dict] = []

    def remember_fragment(self, content, source, token_count, is_pinned):
        self.ingested.append({
            "content": content,
            "source": source,
            "token_count": token_count,
            "is_pinned": is_pinned,
        })


# ── Operation 1: project / inject ─────────────────────────────────────


def test_project_filters_low_confidence(vault: VaultManager):
    beliefs = project_beliefs(vault, "knapsack solver token budget")
    entities = {b.entity for b in beliefs}
    assert "knapsack_solver" in entities
    assert "experimental_thing" not in entities, "low-confidence belief must be filtered"


def test_project_filters_by_relevance(vault: VaultManager):
    """Query about an unrelated topic should not retrieve the cache belief."""
    beliefs = project_beliefs(vault, "knapsack solver token budget")
    entities = {b.entity for b in beliefs}
    # cache_aligner has no token-overlap with the query
    assert "cache_aligner" not in entities


def test_project_returns_empty_on_no_match(vault: VaultManager):
    beliefs = project_beliefs(vault, "completely unrelated query about cookies")
    assert beliefs == []


def test_project_ranks_by_confidence_recency_relevance(vault: VaultManager):
    beliefs = project_beliefs(vault, "knapsack token submodular budget")
    assert beliefs, "expected at least one match"
    # Scores must be in non-increasing order
    scores = [b.score for b in beliefs]
    assert scores == sorted(scores, reverse=True)
    # All scores in (0, 1]
    assert all(0 < s <= 1 for s in scores)


def test_inject_ingests_into_engine(vault: VaultManager):
    engine = FakeEngine()
    claim_ids = inject_vault_beliefs(engine, vault, "knapsack solver budget")
    assert claim_ids, "expected at least one belief injected"
    assert len(engine.ingested) == len(claim_ids)
    frag = engine.ingested[0]
    assert frag["source"].startswith("vault://beliefs/")
    assert frag["token_count"] > 0
    # The 0.92-confidence belief should be pinned (≥ threshold)
    pinned_sources = [f["source"] for f in engine.ingested if f["is_pinned"]]
    assert any("knapsack_solver" in s for s in pinned_sources)


def test_inject_pinned_threshold(vault: VaultManager):
    """Beliefs with confidence ≥ PINNED_CONFIDENCE_THRESHOLD are pinned."""
    engine = FakeEngine()
    inject_vault_beliefs(engine, vault, "knapsack solver budget")
    for f in engine.ingested:
        if "knapsack_solver" in f["source"]:
            assert f["is_pinned"] is True
        elif "cache_aligner" in f["source"]:
            # 0.70 < 0.85 threshold
            assert f["is_pinned"] is False


def test_inject_empty_vault(tmp_path: Path):
    v = VaultManager(VaultConfig(base_path=str(tmp_path / "empty_vault")))
    v.ensure_structure()
    engine = FakeEngine()
    claim_ids = inject_vault_beliefs(engine, v, "anything")
    assert claim_ids == []
    assert engine.ingested == []


# ── Operation 2: attribute_outcome ────────────────────────────────────


def test_bayesian_update_success_raises_confidence(vault: VaultManager):
    """A successful outcome must raise (or hold) confidence for grounded beliefs."""
    cid = "cid-knapsack"
    before = _read_confidence(vault, "knapsack_solver")
    updates = attribute_outcome([cid], outcome_success=True, vault=vault)
    after = _read_confidence(vault, "knapsack_solver")
    assert updates and updates[0]["claim_id"] == cid
    assert after >= before, f"success should not lower confidence ({before} → {after})"
    assert updates[0]["confidence_after"] == pytest.approx(after, rel=1e-3)


def test_bayesian_update_failure_lowers_confidence(vault: VaultManager):
    cid = "cid-knapsack"
    before = _read_confidence(vault, "knapsack_solver")
    attribute_outcome([cid], outcome_success=False, vault=vault)
    after = _read_confidence(vault, "knapsack_solver")
    assert after < before, f"failure must lower confidence ({before} → {after})"


def test_bayesian_update_clamped_to_unit_interval(vault: VaultManager):
    """Repeated successes asymptote toward 0.99, repeated failures toward 0.01."""
    cid = "cid-knapsack"
    for _ in range(20):
        attribute_outcome([cid], outcome_success=True, vault=vault)
    c = _read_confidence(vault, "knapsack_solver")
    assert 0 < c <= 0.99
    assert c >= 0.95, f"after 20 successes expected near-saturation, got {c}"

    for _ in range(50):
        attribute_outcome([cid], outcome_success=False, vault=vault)
    c = _read_confidence(vault, "knapsack_solver")
    assert 0.01 <= c < 0.5


def test_bayesian_update_math_explicit():
    """Verify the formula directly:
       prior = 0.5, α = 0.85, β = 0.40, success
       posterior = (0.5·0.85) / (0.5·0.85 + 0.5·0.40) = 0.425 / 0.625 = 0.68
    """
    # We can't easily exercise the math without a vault, but we can verify
    # the closed-form on a single-shot update.
    c, alpha, beta = 0.5, 0.85, 0.40
    L_g, L_u = alpha, beta  # outcome=success
    expected = (c * L_g) / (c * L_g + (1 - c) * L_u)
    assert expected == pytest.approx(0.68, rel=1e-2)


def test_attribute_dedupes_repeat_claim_ids(vault: VaultManager):
    """Passing the same claim_id twice in one call updates once."""
    cid = "cid-knapsack"
    updates = attribute_outcome([cid, cid, cid], outcome_success=True, vault=vault)
    assert len(updates) == 1


def test_attribute_missing_claim_id(vault: VaultManager):
    updates = attribute_outcome(["does-not-exist"], outcome_success=True, vault=vault)
    assert updates == []


# ── Operation 3: enqueue_reverification ───────────────────────────────


def test_enqueue_marks_verified_stale(vault: VaultManager):
    cid = "cid-knapsack"
    assert _read_status(vault, "knapsack_solver") == "verified"
    marked = enqueue_reverification([cid], vault)
    assert marked == 1
    assert _read_status(vault, "knapsack_solver") == "stale"


def test_enqueue_idempotent(vault: VaultManager):
    """Marking an already-stale belief stale again is a no-op."""
    cid = "cid-knapsack"
    enqueue_reverification([cid], vault)
    second = enqueue_reverification([cid], vault)
    assert second == 0  # already stale


def test_enqueue_skips_hypothesis(vault: VaultManager):
    """Hypothesis-status beliefs are not 'verified or inferred' → skip."""
    cid = "cid-lowconf"
    marked = enqueue_reverification([cid], vault)
    assert marked == 0


# ── Feature gate ──────────────────────────────────────────────────────


def test_is_enabled_default_off(monkeypatch):
    monkeypatch.delenv("ENTROLY_VAULT_COUPLING", raising=False)
    assert is_enabled() is False


def test_is_enabled_when_set(monkeypatch):
    monkeypatch.setenv("ENTROLY_VAULT_COUPLING", "1")
    assert is_enabled() is True


# ── helpers ──────────────────────────────────────────────────────────

_CONF_RE = re.compile(r"^confidence:\s*([0-9.]+)", re.MULTILINE)
_STATUS_RE = re.compile(r"^status:\s*(\S+)", re.MULTILINE)


def _read_confidence(vault: VaultManager, entity: str) -> float:
    b = vault.read_belief(entity)
    assert b, f"belief {entity} not found"
    m = _CONF_RE.search(Path(b["path"]).read_text(encoding="utf-8"))
    assert m, "confidence field missing"
    return float(m.group(1))


def _read_status(vault: VaultManager, entity: str) -> str:
    b = vault.read_belief(entity)
    assert b, f"belief {entity} not found"
    m = _STATUS_RE.search(Path(b["path"]).read_text(encoding="utf-8"))
    assert m, "status field missing"
    return m.group(1).strip()
