"""Quick smoke test for the daemon learning loop integration."""
import tempfile
import time as _time

from entroly.daemon import EntrolyDaemon, EntrolyDaemonState
from entroly.autotune import DreamingLoop, FeedbackJournal, TaskProfileOptimizer
from entroly.online_learner import OnlinePrism

# 1. Verify DaemonState has learning fields
state = EntrolyDaemonState()
d = state.to_dict()
assert d["learning"]["local_enabled"] is True
assert d["learning"]["autotune_enabled"] is True
assert d["learning"]["dreaming_active"] is True
print("[PASS] DaemonState has learning + dreaming fields")

# 2. Verify OnlinePrism 5D works
prism = OnlinePrism(prior_weights={
    "w_recency": 0.30, "w_frequency": 0.25,
    "w_semantic": 0.25, "w_entropy": 0.20,
})
w = prism.weights()
assert abs(sum(w.values()) - 1.0) < 0.01
print(f"[PASS] OnlinePrism weights sum to {sum(w.values()):.4f}")

# 3. Test observe updates weights
new_w = prism.observe(0.8, {
    "w_recency": 0.5, "w_frequency": 0.2,
    "w_semantic": 0.2, "w_entropy": 0.1,
})
stats = prism.stats()
assert stats["n_observations"] == 1
assert stats["phase"] == "warmup"
print(f"[PASS] OnlinePrism after observe: n={stats['n_observations']}, phase={stats['phase']}")

# 4. After warmup, weights should have shifted
for _ in range(5):
    prism.observe(0.9, {
        "w_recency": 0.6, "w_frequency": 0.1,
        "w_semantic": 0.2, "w_entropy": 0.1,
    })
w2 = prism.weights()
assert w2["w_recency"] > w["w_recency"], "Recency should have increased"
stats2 = prism.stats()
assert stats2["phase"] == "learning"
print(f"[PASS] OnlinePrism converging: recency {w['w_recency']:.3f} -> {w2['w_recency']:.3f}")

# 5. Test FeedbackJournal
with tempfile.TemporaryDirectory() as td:
    journal = FeedbackJournal(td)
    journal.log(weights={"w_r": 0.3, "w_f": 0.25, "w_s": 0.25, "w_e": 0.2}, reward=0.7)
    journal.log(weights={"w_r": 0.3, "w_f": 0.25, "w_s": 0.25, "w_e": 0.2}, reward=0.8)
    journal.log(weights={"w_r": 0.3, "w_f": 0.25, "w_s": 0.25, "w_e": 0.2}, reward=0.6)
    assert journal.count() == 3
    s = journal.stats()
    assert s["episodes"] == 3
    assert s["successes"] == 3
    print(f"[PASS] FeedbackJournal: {s['episodes']} episodes, avg_reward={s['avg_reward']}")

    # 6. TaskProfileOptimizer
    tpo = TaskProfileOptimizer(journal)
    profiles = tpo.optimize_all()
    assert len(profiles) >= 1
    print(f"[PASS] TaskProfileOptimizer: {len(profiles)} task profiles")

    # 7. DreamingLoop
    dl = DreamingLoop(journal=journal, max_iterations=3)
    assert not dl.should_dream()
    pre = dl.stats()
    assert pre["last_dream_at"] is None
    assert pre["last_check_at"] is None
    r = dl.run_dream_cycle()
    assert r["status"] == "not_idle"
    mid = dl.stats()
    assert mid["last_dream_at"] is None, "Not-idle checks should not count as dreams"
    assert mid["last_check_at"] is not None
    assert mid["next_dream_in_s"] > 0
    dl._last_activity = _time.time() - 120  # fake idle
    assert dl.should_dream()
    ds = dl.stats()
    assert ds["will_dream"] is True
    print(f"[PASS] DreamingLoop idle detection: idle={ds['idle_seconds']:.0f}s, will_dream={ds['will_dream']}")

    # 8. Daemon outcome callback resets idle timer before journal logging
    daemon = EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    daemon._feedback_journal = journal
    daemon._dreaming_loop = dl
    daemon._log_learning_episode(
        weights={"w_r": 0.3, "w_f": 0.25, "w_s": 0.25, "w_e": 0.2},
        reward=0.9,
    )
    assert journal.count() == 4
    assert not dl.should_dream()
    print("[PASS] Daemon learning callback records activity before dreaming")

print()
print("All 8 learning loop integration tests passed!")
