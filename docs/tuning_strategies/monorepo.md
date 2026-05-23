---
name: monorepo
description: Large monorepo — prioritises structural dependency graph and prefetch. Good for >10k file codebases.
optimize_for: [recall, prefetch_accuracy]
time_budget_ms: 200
---

# Monorepo Strategy

Use this for large codebases with deep dependency trees (e.g., Google-style
monorepos, Turborepo workspaces, Cargo workspaces). The autotuner biases toward
structural signals (dep graph, import chains) over recency.

## What This Optimizes

- **Structural recall** (50%): Files related via imports/calls are surfaced
- **Prefetch accuracy** (30%): Co-access patterns learned fast
- **Efficiency** (20%): Token budget used for structurally important code

## Weight Hints

| Parameter | Hint | Reason |
|-----------|------|--------|
| weight_recency | 0.20 | In a monorepo, recency is less signal |
| weight_frequency | 0.35 | Cross-module access patterns are stable |
| weight_semantic_sim | 0.25 | Still needed for query matching |
| weight_entropy | 0.20 | Dense files (core libs) should rank high |

## Bounds Overrides

| Parameter | Min | Max | Step |
|-----------|-----|-----|------|
| prefetch.depth | 3 | 5 | 1 |
| prefetch.max_fragments | 15 | 30 | 5 |
| decay.half_life_turns | 20 | 50 | 5 |

## When to Switch

- Switch to `latency` if optimize is slow on large dep graphs
- Switch to `balanced` for single-service repos
