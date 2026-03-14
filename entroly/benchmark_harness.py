"""
Entroly Benchmark Harness — READ ONLY. Never modified by autotune.py.
=====================================================================

Fixed, immutable evaluation harness: defines what "better" means for Entroly's
autotuner. The autotuner reads results from this module but NEVER modifies it.

Metric: context_efficiency = cumulative_information / (cumulative_tokens / 1000)
  - Higher = better (more information density per token spent)
  - The single objective the autotuner maximizes

Usage (internal, called by autotune.py):
    from entroly.benchmark_harness import run_benchmark
    result = run_benchmark(engine, budget_seconds=10)
"""

from __future__ import annotations

import gc
import time
import math
from typing import Any, Dict, List

# ── Benchmark payload — fixed test corpus (never changes) ─────────────
# These fragments represent a realistic codebase context load.
# The harness ingests them and then optimizes — result is the efficiency score.
BENCHMARK_FRAGMENTS: List[Dict[str, Any]] = [
    {"content": "def authenticate(token: str) -> bool:\n    if not token:\n        raise ValueError('Empty token')\n    return hmac.compare_digest(token, SECRET)", "source": "auth/service.py"},
    {"content": "class PaymentProcessor:\n    def charge(self, amount: float, currency: str = 'USD') -> dict:\n        return self._gateway.process({'amount': amount, 'currency': currency})", "source": "payments/processor.py"},
    {"content": "SELECT u.id, u.email, o.total FROM users u JOIN orders o ON o.user_id = u.id WHERE u.active = 1 ORDER BY o.created_at DESC LIMIT 100", "source": "db/queries.sql"},
    {"content": "pub fn knapsack_optimize(fragments: &[Fragment], budget: u32) -> Vec<Fragment> {\n    // DP table: dp[i][w] = max value using first i items with weight ≤ w\n    let n = fragments.len();\n    let mut dp = vec![vec![0.0f64; (budget + 1) as usize]; n + 1];\n    // ... fill DP\n    dp[n][budget as usize]\n}", "source": "entroly-core/src/knapsack.rs"},
    {"content": "import React, { useState, useEffect } from 'react';\nexport const Dashboard = ({ userId }) => {\n  const [data, setData] = useState(null);\n  useEffect(() => { fetchDashboard(userId).then(setData); }, [userId]);\n  return <div>{data ? <DataView data={data} /> : <Spinner />}</div>;\n};", "source": "web/components/Dashboard.tsx"},
    {"content": "class RateLimiter:\n    def __init__(self, max_requests: int, window_seconds: int):\n        self.max_requests = max_requests\n        self.window = window_seconds\n        self._counts: dict[str, list[float]] = {}\n\n    def is_allowed(self, key: str) -> bool:\n        now = time.time()\n        hits = [t for t in self._counts.get(key, []) if now - t < self.window]\n        self._counts[key] = hits\n        if len(hits) >= self.max_requests:\n            return False\n        self._counts[key].append(now)\n        return True", "source": "middleware/rate_limiter.py"},
    {"content": "type UserProfile = {\n  id: string;\n  email: string;\n  role: 'admin' | 'editor' | 'viewer';\n  createdAt: Date;\n  preferences: Record<string, unknown>;\n};", "source": "types/user.ts"},
    {"content": "fn main() {\n    let args: Vec<String> = std::env::args().collect();\n    let config = Config::from_file(&args[1]).expect('Failed to load config');\n    let engine = Engine::new(config);\n    engine.run();\n}", "source": "src/main.rs"},
    {"content": "# Production deployment checklist\n## Pre-deploy\n- [ ] Run `cargo test` — 0 failures\n- [ ] Check memory profile under load\n- [ ] Verify rate limits configured\n## Post-deploy\n- [ ] Monitor error rate for 10 minutes\n- [ ] Check context efficiency metric", "source": "docs/DEPLOY.md"},
    {"content": "OPENAI_KEY=sk-proj-xxxx\nDATABASE_URL=postgres://user:pass@localhost/prod\nMAX_TOKENS=128000\nLOG_LEVEL=INFO", "source": ".env.example"},
]

BENCHMARK_QUERY = "authenticate user and process payment with rate limiting"
BENCHMARK_BUDGET = 8000  # tokens


def run_benchmark(engine: Any, budget_seconds: float = 10.0) -> Dict[str, Any]:
    """
    Run the fixed evaluation payload and return the context_efficiency score.

    READ ONLY — this function is the ground truth metric. autotune.py calls
    this but never modifies it. The engine and benchmark corpus are fixed.

    Returns:
        {
            "context_efficiency": float,   # primary metric (higher = better)
            "total_tokens_saved": int,
            "num_fragments_selected": int,
            "wall_seconds": float,
            "timed_out": bool,
        }
    """
    t_start = time.monotonic()
    timed_out = False

    # Reset engine state for a clean benchmark run (don't carry over prior state)
    # We ingest the fixed corpus and then optimize against the benchmark query.
    gc.disable()
    try:
        for frag in BENCHMARK_FRAGMENTS:
            if time.monotonic() - t_start > budget_seconds:
                timed_out = True
                break
            engine.ingest_fragment(
                content=frag["content"],
                source=frag["source"],
            )
    finally:
        gc.enable()
        gc.collect()

    if timed_out:
        return {
            "context_efficiency": 0.0,
            "total_tokens_saved": 0,
            "num_fragments_selected": 0,
            "wall_seconds": time.monotonic() - t_start,
            "timed_out": True,
        }

    # Optimize — this is what the autotuner is scoring
    gc.disable()
    try:
        result = engine.optimize_context(
            token_budget=BENCHMARK_BUDGET,
            query=BENCHMARK_QUERY,
        )
    finally:
        gc.enable()
        gc.collect()

    wall = time.monotonic() - t_start

    # Check time budget (fixed budget eval is the fair comparison metric)
    if wall > budget_seconds:
        timed_out = True

    # Primary metric: context_efficiency from engine stats
    stats = engine.stats()
    eff_block = stats.get("context_efficiency", {})
    ctx_eff = eff_block.get("context_efficiency", 0.0)

    # Penalize timeouts severely
    if timed_out:
        ctx_eff = max(0.0, ctx_eff - 0.5)

    return {
        "context_efficiency": ctx_eff,
        "total_tokens_saved": stats.get("savings", {}).get("total_tokens_saved", 0),
        "num_fragments_selected": len(result.get("selected", [])),
        "wall_seconds": round(wall, 3),
        "timed_out": timed_out,
    }
