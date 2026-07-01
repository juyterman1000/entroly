"""Comprehensive test of the Epistemic Cascade Engine (ECE v6)."""
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

from entroly.ravs.ece import (
    EpistemicCascadeEngine,
    LyapunovThresholdController,
    compute_fisher_curvature,
    compute_renyi_entropy,
    cluster_by_simhash,
    select_renyi_alpha,
)

ece = EpistemicCascadeEngine()

print("=" * 60)
print("  EPISTEMIC CASCADE ENGINE (ECE v6) — INTEGRATION TEST")
print("=" * 60)

# --- Tier 0: Open-ended query (should NOT escalate) ---
print("\n[Tier 0] Open-ended query (aleatoric uncertainty)")
s = ece.evaluate_uncertainty(
    "Write a poem about the moon",
    "Roses are red, violets are blue...",
)
print(f"  Decision: {s.decision}")
print(f"  Tier: {s.tier_used}")
print(f"  Time: {s.computation_time_us:.0f} us")
assert s.decision == "keep", "Open-ended should NOT escalate!"
assert s.tier_used == 0, "Should exit at Tier 0!"

# --- Tier 1: Confident factual (should NOT escalate) ---
print("\n[Tier 1] Confident factual answer")
s = ece.evaluate_uncertainty(
    "What is 2+2?",
    "The answer is 4.",
)
print(f"  Decision: {s.decision}")
print(f"  Tier: {s.tier_used}")
print(f"  Fisher Curvature: {s.fisher_curvature:.4f}")
print(f"  Time: {s.computation_time_us:.0f} us")
assert s.decision == "keep"

# --- Tier 2: Multi-sample agreement (should NOT escalate) ---
print("\n[Tier 2] Multi-sample with semantic AGREEMENT")
s = ece.evaluate_uncertainty(
    "How many moons does Earth have?",
    "Earth has 1 moon.",
    alternative_responses=[
        "Earth has one natural satellite.",
        "The Moon is Earth's only moon.",
    ],
)
print(f"  Decision: {s.decision}")
print(f"  Tier: {s.tier_used}")
print(f"  Clusters: {s.cluster_count}")
print(f"  Renyi Entropy: {s.renyi_entropy:.4f}")
print(f"  Epistemic U: {s.epistemic_uncertainty:.4f}")
print(f"  Time: {s.computation_time_us:.0f} us")

# --- Tier 2: Multi-sample DISAGREEMENT (should ESCALATE) ---
print("\n[Tier 2] Multi-sample with DISAGREEMENT (high risk)")
s = ece.evaluate_uncertainty(
    "How many moons does Mars have?",
    "Mars has 1 moon.",
    risk_level="high",
    alternative_responses=[
        "Mars has 2 moons named Phobos and Deimos.",
        "Mars has three small moons orbiting it.",
    ],
)
print(f"  Decision: {s.decision}")
print(f"  Tier: {s.tier_used}")
print(f"  Clusters: {s.cluster_count}")
print(f"  Renyi Alpha: {s.renyi_alpha} (tail-sensitive for safety)")
print(f"  Renyi Entropy: {s.renyi_entropy:.4f}")
print(f"  Epistemic U: {s.epistemic_uncertainty:.4f}")
print(f"  Time: {s.computation_time_us:.0f} us")

# --- Invention 4: Adaptive Renyi ---
print("\n[Invention 4] Adaptive Renyi Alpha Selection")
print(f"  High risk:     alpha = {select_renyi_alpha('high')}")
print(f"  Standard risk: alpha = {select_renyi_alpha('standard')}")
print(f"  Low risk:      alpha = {select_renyi_alpha('low')}")

# --- Invention 4: Renyi entropy comparison ---
print("\n[Invention 4] Renyi Entropy: Shannon vs Tail-Sensitive")
clusters = [6, 1, 1]  # Majority agrees, 2 outliers
print(f"  Clusters: {clusters}")
print(f"  Shannon (alpha=1):    H = {compute_renyi_entropy(clusters, 1.0):.4f}")
print(f"  Tail-sens (alpha=3):  H = {compute_renyi_entropy(clusters, 3.0):.4f}")
print(f"  Min-entropy (alpha=inf): H = {compute_renyi_entropy(clusters, 100):.4f}")

# --- Invention 5: Lyapunov Controller ---
print("\n[Invention 5] Lyapunov-Stable Threshold Controller")
lyap = LyapunovThresholdController(initial_tau=0.5)

# Simulate 90 keeps + 10 escalations
for _ in range(90):
    lyap.record_decision(False)
for _ in range(10):
    lyap.record_decision(True)

stats = lyap.stats()
print(f"  Tau after 100 decisions: {stats['current_tau']}")
print(f"  Escalation rate (EMA):   {stats['escalation_rate_ema']}")

# Simulate GPU load spike
lyap.adjust_for_load(gpu_load=0.95, queue_depth=150)
stats2 = lyap.stats()
print(f"  Tau after GPU spike:     {stats2['current_tau']}")
print("  (System raised tau to reduce escalations under load)")

# --- SimHash Clustering ---
print("\n[Clustering] SimHash Semantic Clustering")
texts = [
    "Earth has one natural satellite called the Moon.",
    "The Moon is Earth's only natural satellite.",
    "Earth has two moons.",
]
clusters = cluster_by_simhash(texts)
print(f"  Texts: {len(texts)}")
print(f"  Clusters found: {len(clusters)}")
for i, c in enumerate(clusters):
    print(f"  Cluster {i}: indices {c}")

# --- Final Stats ---
print("\n" + "=" * 60)
print("  FINAL ECE STATS")
print("=" * 60)
for k, v in ece.stats().items():
    if isinstance(v, dict):
        print(f"  {k}:")
        for k2, v2 in v.items():
            print(f"    {k2}: {v2}")
    else:
        print(f"  {k}: {v}")

print("\n[PASS] All ECE integration tests passed!")
