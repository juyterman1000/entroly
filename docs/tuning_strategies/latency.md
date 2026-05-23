---
name: latency
description: Ultra-low latency — guarantees optimize_context() completes in <50ms. Sacrifices recall for speed.
optimize_for: [latency]
time_budget_ms: 50
---

# Latency Strategy

Use this when you need a hard latency guarantee. The autotuner hard-kills any
config that exceeds 50ms and only keeps configs that maintain speed.

**Marketing claim this enables**: "Entroly guarantees optimal context in under 50ms."

## What This Optimizes

- **Latency** (primary): Sub-50ms hard guarantee — any violation = discard
- **Efficiency** (secondary): Token density within the speed constraint

## Weight Hints

| Parameter | Hint | Reason |
|-----------|------|--------|
| weight_recency | 0.40 | Recency is cheap to compute |
| weight_frequency | 0.30 | Counter lookup is O(1) |
| weight_semantic_sim | 0.20 | SimHash is fast but skip refine pass |
| weight_entropy | 0.10 | Entropy slightly slower on large frags |

## Bounds Overrides

| Parameter | Min | Max | Step |
|-----------|-----|-----|------|
| decay.half_life_turns | 5 | 20 | 3 |
| knapsack.exploration_rate | 0.0 | 0.05 | 0.01 |

Time budget is HARD at 50ms — no soft penalty. Timeouts automatically get score 0.0.

## When to Switch

- Switch to `balanced` if latency is fine but recall suffers
- This strategy intentionally accepts lower f1 for speed
