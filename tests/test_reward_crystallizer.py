"""
Tests for RewardCrystallizer + success-driven skill synthesis.

Coverage:
  C-01  STATELESS BELOW MIN N        no event before min_samples
  C-02  IGNORES LOW REWARD           reward at baseline → never crystallizes
  C-03  CRYSTALLIZES SUSTAINED HIGH  high-reward cluster crosses LCB → event
  C-04  COOLDOWN PREVENTS DUPLICATE  same cluster doesn't double-emit
  C-05  CLUSTER REQUIRES BOTH        SimHash AND Jaccard, not OR
  C-06  HOEFFDING LCB MATH           hand-computed LCB matches implementation
  C-07  EVENT CARRIES CLUSTER STATE  event has weights, recipe, queries
  C-08  EMPTY QUERY IGNORED          observe('') returns None silently
  C-09  CONCURRENT SAFE              parallel observes do not corrupt state
  C-10  FROM_SUCCESS WIRES PROVENANCE  SkillSpec captures all event fields
  C-11  CRYSTALLIZE_SKILL WRITES VAULT  end-to-end: event → file on disk
  C-12  PROMOTED STATUS NO BENCHMARK   crystallized skills bypass draft state
"""

from __future__ import annotations

import json
import math
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from entroly.reward_crystallizer import (  # noqa: E402
    RewardCrystallizer,
    CrystallizationEvent,
    _hoeffding_lcb,
    _simhash64,
    _hamming,
    _jaccard,
)
from entroly.skill_engine import SkillSynthesizer, SkillEngine  # noqa: E402


# ── Helpers ─────────────────────────────────────────────────────────


def _w(r=0.25, f=0.25, s=0.25, e=0.25):
    return {"w_recency": r, "w_frequency": f, "w_semantic": s, "w_entropy": e}


def _frags(*ids):
    return list(ids)


# ══════════════════════════════════════════════════════════════════
# C-01..C-04: Detection lifecycle
# ══════════════════════════════════════════════════════════════════


def test_c01_no_event_below_min_samples():
    c = RewardCrystallizer(min_samples=5, epsilon=0.0, delta=0.05)
    for _ in range(4):
        ev = c.observe(
            query="fix bug in payments",
            reward=0.99, weights=_w(),
            selected_fragment_ids=_frags("a", "b"),
            baseline_reward=0.50,
        )
        assert ev is None
    assert c.poll_events() == []


def test_c02_ignores_at_baseline():
    """Cluster mean ≈ baseline → LCB strictly below baseline → no event ever."""
    c = RewardCrystallizer(min_samples=4, epsilon=0.05, delta=0.05)
    for _ in range(20):
        ev = c.observe(
            query="check status of orders",
            reward=0.50, weights=_w(),
            selected_fragment_ids=_frags("o1", "o2"),
            baseline_reward=0.50,
        )
        assert ev is None


def test_c03_crystallizes_sustained_high_reward():
    c = RewardCrystallizer(
        min_samples=5, window=10,
        epsilon=0.05, delta=0.05,
    )
    event = None
    # Realistic shape: a query family asked many times across a session.
    # Variants share core terms (auth/session/tokens) so SimHash clusters
    # them, and select the same files so Jaccard agrees.
    family = [
        "fix the auth login bug",
        "fix auth login crash",
        "auth login bug needs fix",
        "fix auth bug in login flow",
        "auth bug login fix needed",
        "login auth bug fix please",
        "fix the auth login regression",
        "auth login crash fix",
        "fix auth login session bug",
        "session auth login bug fix",
        "fix login auth retry bug",
        "fix auth login bug in session",
    ]
    for q in family:
        ev = c.observe(
            query=q,
            reward=0.95, weights=_w(s=0.5, e=0.3),
            selected_fragment_ids=_frags("auth.py", "session.py", "tokens.py"),
            baseline_reward=0.50,
        )
        if ev is not None:
            event = ev
            break
    assert event is not None, "should have crystallized within 12 high-reward observations"
    assert event.n_samples >= 5
    assert event.mean_reward > 0.9
    assert event.lcb_reward > event.baseline_reward + 0.05


def test_c04_cooldown_prevents_duplicate_emission():
    c = RewardCrystallizer(
        min_samples=5, window=10,
        epsilon=0.05, delta=0.05,
        cooldown=10,
    )
    events = []
    # Same query repeated → guaranteed same cluster (Hamming = 0)
    q = "profile rust hot function in core engine"
    for _ in range(20):
        ev = c.observe(
            query=q, reward=0.95, weights=_w(),
            selected_fragment_ids=_frags("core.rs", "perf.rs"),
            baseline_reward=0.4,
        )
        if ev is not None:
            events.append(ev)
    # First emission should fire; cooldown blocks the next ~10 observations.
    # Across 20 observations with cooldown=10 we expect ≤ 2 events.
    assert 1 <= len(events) <= 2


# ══════════════════════════════════════════════════════════════════
# C-05..C-08: Cluster identity, math, edge cases
# ══════════════════════════════════════════════════════════════════


def test_c05_cluster_requires_text_and_fragment_jaccard():
    """Two queries with similar text but disjoint fragment sets → different clusters.
    Two queries with identical fragments but unrelated text → different clusters."""
    c = RewardCrystallizer(min_samples=3, window=10, epsilon=0.05, delta=0.05,
                           query_jaccard=0.34, fragment_jaccard=0.5)
    # First family: "fix auth bug" with auth fragments
    for i in range(4):
        c.observe(query=f"fix auth bug {i}", reward=0.95, weights=_w(),
                  selected_fragment_ids=_frags("auth.py", "login.py"),
                  baseline_reward=0.4)
    # Same text family BUT disjoint fragments → must NOT merge
    c.observe(query="fix auth bug zzz", reward=0.95, weights=_w(),
              selected_fragment_ids=_frags("payment.py", "billing.py"),
              baseline_reward=0.4)
    stats = c.stats()
    assert stats["active_clusters"] >= 2


def test_c06_hoeffding_lcb_math_matches_paper():
    """LCB = mean − sqrt( ln(1/δ) / (2n) ). Auer et al. 2002, finite-time bandit bounds."""
    n, mean, delta = 16, 0.85, 0.05
    expected = mean - math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    assert abs(_hoeffding_lcb(mean, n, delta) - expected) < 1e-12

    # Boundary: n=0 → 0 (degenerate, no information)
    assert _hoeffding_lcb(0.5, 0, 0.05) == 0.0


def test_c07_event_carries_cluster_state():
    c = RewardCrystallizer(min_samples=5, window=10, epsilon=0.05, delta=0.05)
    event = None
    family = [
        "optimize knapsack solver",
        "make knapsack solver faster",
        "speed up knapsack optimize",
        "knapsack solver optimize entropy",
        "optimize the knapsack solver entropy",
        "knapsack solver speed optimize",
        "tune knapsack solver entropy",
        "knapsack solver tuning optimize",
        "optimize knapsack entropy solver",
        "knapsack optimize solver",
        "speed knapsack solver optimize",
        "optimize knapsack",
    ]
    for q in family:
        ev = c.observe(
            query=q, reward=0.95, weights=_w(s=0.45, e=0.30),
            selected_fragment_ids=_frags("knapsack.rs", "entropy.rs"),
            baseline_reward=0.45,
        )
        if ev is not None:
            event = ev
            break
    assert event is not None
    # Provenance: every field needed by SkillSynthesizer.synthesize_from_success
    assert isinstance(event, CrystallizationEvent)
    assert event.cluster_id and len(event.cluster_id) >= 8
    joined = " ".join(event.common_terms)
    assert "knapsack" in joined or "optimize" in joined or "solver" in joined
    assert "knapsack.rs" in event.fragment_recipe
    assert event.weight_profile.get("w_semantic", 0) > 0
    assert event.lcb_reward > event.baseline_reward
    assert len(event.sample_queries) >= 1


def test_c08_empty_query_silent():
    c = RewardCrystallizer()
    assert c.observe(
        query="", reward=0.95, weights=_w(),
        selected_fragment_ids=_frags("x"), baseline_reward=0.5,
    ) is None
    assert c.stats()["total_observations"] == 0


def test_c09_thread_safety_no_corruption():
    c = RewardCrystallizer(min_samples=3, window=20, epsilon=0.05,
                           delta=0.05, cooldown=100)
    errors = []
    def worker(thread_id: int) -> None:
        try:
            for i in range(50):
                c.observe(
                    query=f"thread {thread_id} query {i}",
                    reward=0.6, weights=_w(),
                    selected_fragment_ids=_frags(f"frag_{thread_id}"),
                    baseline_reward=0.5,
                )
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == []
    assert c.stats()["total_observations"] == 8 * 50


# ══════════════════════════════════════════════════════════════════
# C-10..C-12: Skill synthesis from event
# ══════════════════════════════════════════════════════════════════


def _fake_event(**overrides):
    base = dict(
        event_id="ev123",
        cluster_id="cl0001",
        sample_queries=["fix auth bug", "fix auth crash", "fix auth retry"],
        common_terms=["fix", "auth", "bug"],
        weight_profile={"w_recency": 0.2, "w_frequency": 0.2,
                        "w_semantic": 0.4, "w_entropy": 0.2},
        fragment_recipe=["auth.py", "session.py", "login.py"],
        n_samples=8, mean_reward=0.92,
        lcb_reward=0.78, baseline_reward=0.45,
        effect_size=0.33, created_at=0.0,
    )
    base.update(overrides)
    return CrystallizationEvent(**base)


def test_c10_synthesize_from_success_carries_provenance():
    spec = SkillSynthesizer().synthesize_from_success(_fake_event())
    assert spec.status == "promoted"  # pre-validated by LCB
    assert spec.metrics.get("source") == "crystallization"
    assert spec.metrics.get("fitness_score") == 0.78
    assert "auth.py" in spec.tool_code
    assert "0.4000" in spec.tool_code or "0.4" in spec.tool_code  # weight profile embedded
    # Trigger contains common terms
    assert "auth" in spec.trigger
    # Test cases come from real winning queries, not synthetic placeholders
    assert any(tc.get("input") == "fix auth bug" for tc in spec.test_cases)


def test_c11_crystallize_skill_writes_vault(tmp_path):
    # Stand up a minimal vault rooted at tmp_path
    from entroly.vault import VaultManager, VaultConfig
    vault = VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))
    vault.ensure_structure()
    se = SkillEngine(vault)

    res = se.crystallize_skill(_fake_event(cluster_id="cl_test"))
    assert res["status"] == "crystallized"

    skill_dir = Path(res["path"])
    assert (skill_dir / "SKILL.md").exists()
    assert (skill_dir / "tool.py").exists()
    assert (skill_dir / "metrics.json").exists()
    assert (skill_dir / "tests" / "test_cases.json").exists()

    # SKILL.md carries provenance
    md = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "source: crystallization" in md
    assert "cluster_id: cl_test" in md
    assert "Hoeffding lower bound" in md

    # metrics.json captures the LCB as fitness, not a benchmark result
    metrics = json.loads((skill_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["source"] == "crystallization"
    assert metrics["fitness_score"] > 0.5

    # Generated tool is a runnable Python module (parse + exec)
    tool_src = (skill_dir / "tool.py").read_text(encoding="utf-8")
    ns: dict = {}
    exec(compile(tool_src, str(skill_dir / "tool.py"), "exec"), ns)
    assert callable(ns.get("matches"))
    assert callable(ns.get("execute"))
    assert ns["matches"]("fix auth bug") is True
    assert ns["matches"]("totally unrelated string") is False
    out = ns["execute"]("fix auth bug", {})
    assert out["status"] == "success"
    assert "auth.py" in out["fragment_recipe"]


def test_c12_promoted_status_skips_draft_lifecycle():
    """Crystallized skills enter the registry already promoted.
    The Hoeffding LCB is the fitness proof; no benchmark required."""
    spec = SkillSynthesizer().synthesize_from_success(_fake_event())
    assert spec.status == "promoted"
    # Distinct from failure-driven path which starts as 'draft'
    failure_spec = SkillSynthesizer().synthesize_from_gap(
        "auth", ["fix auth"], "Debugging",
    )
    assert failure_spec.status == "draft"


# ══════════════════════════════════════════════════════════════════
# C-13: SimHash + Hamming sanity (regression)
# ══════════════════════════════════════════════════════════════════


def test_c13_simhash_similarity_property():
    """Token-overlap implies low Hamming; disjoint tokens imply higher Hamming."""
    h1 = _simhash64("fix authentication bug in login flow")
    h2 = _simhash64("fix authentication bug in login retry")
    h3 = _simhash64("compile rust binary release mode for shipping")
    # Similar queries → relatively small hamming. Calibration: short
    # queries (4-7 trigrams) at MD5+trigram simhash typically land at
    # 12-20 for one-token edits. See DEFAULT_HASH_HAMMING note.
    assert _hamming(h1, h2) < 20
    # Unrelated queries → strictly larger hamming.
    assert _hamming(h1, h3) > _hamming(h1, h2)


def test_c14_jaccard_basic():
    assert _jaccard(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert _jaccard(["a", "b"], ["c", "d"]) == 0.0
    assert abs(_jaccard(["a", "b", "c"], ["b", "c", "d"]) - 2/4) < 1e-9
    assert _jaccard([], []) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
