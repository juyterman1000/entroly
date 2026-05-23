---
name: balanced
description: Balanced default — equal weight to all signals. Good starting point for most codebases.
optimize_for: [recall, precision, context_efficiency]
time_budget_ms: 500
---

# Balanced Strategy

Use this when you don't have a specific bottleneck. The autotuner explores all
parameters with equal curiosity.

## What This Optimizes

- **Recall** (50%): Correct fragments are selected
- **Precision** (25%): No noise fragments sneak in
- **Efficiency** (25%): High information density per token used

## Weight Hints

These biases guide the autotuner's mutation budget:

| Parameter | Hint | Reason |
|-----------|------|--------|
| weight_recency | 0.30 | Recent info matters moderately |
| weight_frequency | 0.25 | Access patterns matter |
| weight_semantic_sim | 0.25 | Query relevance matters |
| weight_entropy | 0.20 | Information density matters |

## Bounds

All parameters use their full default search range. No special restrictions.

## When to Switch

- Switch to `latency` if optimize_context() latency exceeds 100ms
- Switch to `quality` if recall drops below 0.8
- Switch to `monorepo` if your codebase has >10,000 files
