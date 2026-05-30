"""
Tests for belief-conditioned compression — H(X | beliefs).

Two layers:
  * Engine layer (native entroly_core, skipped if unavailable): the discount,
    the vault:// self-skip, IDEMPOTENCE on a persistent engine (the bug the
    feature must never regress), corpus-change restore, and the binding.
  * Coupling layer (pure Python): couple_beliefs injects beliefs AND drives the
    discount from a single projection, degrading gracefully on older engines.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from entroly.coupling import couple_beliefs
from entroly.vault import BeliefArtifact, VaultConfig, VaultManager

# ── Engine layer ──────────────────────────────────────────────────────
# Engine tests need the native core; the coupling tests below are pure Python
# and must still run in the no-engine (pure-Python fallback) CI job — so we
# skip per-test, not per-module.
try:
    import entroly_core
    _HAS_ENGINE = True
except ImportError:  # pragma: no cover - exercised in the pure-Python CI job
    entroly_core = None
    _HAS_ENGINE = False

requires_engine = pytest.mark.skipif(
    not _HAS_ENGINE, reason="native entroly_core engine not installed"
)

_BELIEF = "authentication validates a bearer token with hmac compare_digest against a secret"
_RATE_BELIEF = "the rate limiter uses a sliding window of request timestamps per key"


def _engine(frags):
    eng = entroly_core.EntrolyEngine()
    for src, txt in frags:
        eng.ingest(txt, src, max(1, len(txt) // 4), False)
    return eng


def _scores(eng):
    return {f["source"]: f["entropy_score"] for f in eng.export_fragments()}


@requires_engine
def test_restatement_discounted_novel_untouched():
    eng = _engine([
        ("ide://restate", _BELIEF + " today"),
        ("ide://novel", "websocket reconnect uses exponential backoff with a jitter cap"),
    ])
    before = _scores(eng)
    eng.set_belief_corpus([(_BELIEF, 0.95)])
    adjusted = eng.apply_belief_conditioning()
    after = _scores(eng)
    assert adjusted >= 1
    assert after["ide://restate"] < before["ide://restate"], "restatement must be discounted"
    assert after["ide://novel"] == pytest.approx(before["ide://novel"]), "novel must be untouched"


@requires_engine
def test_vault_sourced_fragment_self_skip():
    # A belief restated as a vault:// fragment must NOT discount itself.
    eng = _engine([("vault://beliefs/auth#abcd", _BELIEF)])
    before = _scores(eng)
    eng.set_belief_corpus([(_BELIEF, 0.95)])
    eng.apply_belief_conditioning()
    after = _scores(eng)
    assert after["vault://beliefs/auth#abcd"] == pytest.approx(before["vault://beliefs/auth#abcd"])


@requires_engine
def test_idempotent_no_compounding_on_persistent_engine():
    # THE regression guard: repeated passes (persistent proxy/MCP engine) must
    # not compound the discount toward zero.
    eng = _engine([("ide://r", _RATE_BELIEF + " counts")])
    eng.set_belief_corpus([(_RATE_BELIEF, 0.95)])
    eng.apply_belief_conditioning()
    s1 = _scores(eng)["ide://r"]
    eng.apply_belief_conditioning()
    s2 = _scores(eng)["ide://r"]
    eng.apply_belief_conditioning()
    s3 = _scores(eng)["ide://r"]
    assert s1 == pytest.approx(s2) == pytest.approx(s3), "discount must be idempotent"


@requires_engine
def test_corpus_change_restores_toward_baseline():
    eng = _engine([("ide://r", _RATE_BELIEF + " counts")])
    eng.set_belief_corpus([(_RATE_BELIEF, 0.95)])
    eng.apply_belief_conditioning()
    discounted = _scores(eng)["ide://r"]
    # Swap to an unrelated belief: the fragment is no longer "known".
    eng.set_belief_corpus([("totally unrelated belief about parsing json grammars", 0.9)])
    eng.apply_belief_conditioning()
    restored = _scores(eng)["ide://r"]
    assert restored > discounted, "removing the matching belief must restore value"


@requires_engine
def test_empty_corpus_is_noop():
    eng = _engine([("ide://x", "some representative source content for the test")])
    before = _scores(eng)
    assert eng.apply_belief_conditioning() == 0
    assert _scores(eng) == before


@requires_engine
def test_binding_strict_superset_and_discount():
    frag = "the rate limiter uses a sliding window of requests per key"
    base = entroly_core.py_information_score(frag, [])
    # Empty beliefs == information_score (strict superset).
    assert entroly_core.py_conditional_information_score(frag, [], []) == pytest.approx(base)
    # Known high-confidence belief discounts.
    known = entroly_core.py_conditional_information_score(frag, [], [(frag, 1.0)])
    assert known < base


# ── Coupling layer (pure Python) ──────────────────────────────────────


@pytest.fixture
def vault(tmp_path: Path) -> VaultManager:
    v = VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))
    v.ensure_structure()
    v.write_belief(BeliefArtifact(
        claim_id="cid-knapsack", entity="knapsack_solver", status="verified",
        confidence=0.92, sources=["entroly-core/src/knapsack.rs:1"],
        title="Knapsack solver",
        body="The knapsack solver uses 0/1 dynamic programming with a token "
             "budget constraint enforced at selection time.",
    ))
    return v


class _FakeEngine:
    """Records ingestion only (older engine without conditioning support)."""
    def __init__(self):
        self.ingested = []

    def remember_fragment(self, content, source, token_count, is_pinned):
        self.ingested.append((source, is_pinned))


class _FakeConditioningEngine(_FakeEngine):
    """Adds the belief-conditioning surface."""
    def __init__(self):
        super().__init__()
        self.corpus = None
        self.conditioning_passes = 0

    def set_belief_corpus(self, corpus):
        self.corpus = corpus

    def clear_belief_corpus(self):
        self.corpus = []

    def apply_belief_conditioning(self):
        self.conditioning_passes += 1
        return len(self.corpus or [])


def test_couple_beliefs_injects_and_conditions(vault: VaultManager):
    eng = _FakeConditioningEngine()
    claim_ids = couple_beliefs(eng, vault, "knapsack solver token budget")
    assert claim_ids, "beliefs should be injected"
    assert eng.corpus is not None, "belief corpus must be set"
    assert all(isinstance(t, str) and isinstance(c, float) for t, c in eng.corpus)
    assert eng.conditioning_passes == 1, "conditioning must run exactly once per request"


def test_couple_beliefs_no_match_no_conditioning(vault: VaultManager):
    eng = _FakeConditioningEngine()
    claim_ids = couple_beliefs(eng, vault, "completely unrelated query about cookies")
    assert claim_ids == []
    assert eng.conditioning_passes == 0


def test_couple_beliefs_graceful_on_old_engine(vault: VaultManager):
    # Engine without set_belief_corpus must still inject without crashing.
    eng = _FakeEngine()
    claim_ids = couple_beliefs(eng, vault, "knapsack solver token budget")
    assert claim_ids, "injection must still work on engines lacking conditioning"
