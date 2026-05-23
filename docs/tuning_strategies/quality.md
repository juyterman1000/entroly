---
name: quality
description: Maximum recall/precision — optimises for correct context selection at cost of some latency.
optimize_for: [recall, precision, f1]
time_budget_ms: 500
---

# Quality Strategy

Use this when correctness matters more than speed. The autotuner maximises F1
score — the harmonic mean of recall and precision — without a tight latency
constraint.

**Best for**: Security-sensitive code, debugging sessions, architecture reviews
where you need every relevant fragment and zero noise.

## What This Optimizes

- **Recall** (40%): Every relevant fragment is found
- **Precision** (40%): No irrelevant fragments included
- **F1** (20%): Balance of the above

Latency budget is 500ms (relaxed). Quality > speed.

## Weight Hints

| Parameter | Hint | Reason |
|-----------|------|--------|
| weight_recency | 0.25 | Moderate — don't over-weight recent |
| weight_frequency | 0.20 | Moderate |
| weight_semantic_sim | 0.35 | Semantic sim is highest quality signal |
| weight_entropy | 0.20 | Dense code is often important |

## Bounds Overrides

| Parameter | Min | Max | Step |
|-----------|-----|-----|------|
| decay.min_relevance_threshold | 0.01 | 0.08 | 0.01 |
| knapsack.exploration_rate | 0.05 | 0.25 | 0.02 |

Lower min_relevance keeps more candidates. Higher exploration ensures breadth.

## When to Switch

- Switch to `latency` for interactive coding sessions
- Quality mode is slower — 100-500ms per optimize call
