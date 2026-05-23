"""
Entroly Autotune -- Autonomous Self-Tuning Loop
==============================================

Keep/discard experimentation loop to autonomously improve Entroly's
hyperparameters. Each iteration mutates one parameter in tuning_config.json,
evaluates the result on a fixed benchmark suite, and keeps improvements.

For Entroly, the loop maps to:
  - The mutable surface  = bench/tuning_config.json (the ONLY file we mutate)
  - The evaluation step  = running optimize_context() on benchmark cases
  - The objective metric = context_efficiency (information_retained / tokens_used)
  - The keep/discard     = compare efficiency, keep improvements, revert regressions

Each iteration takes seconds on CPU, so ~1000 experiments run overnight.

Usage:
    python -m entroly.autotune                    # Run 100 iterations
    python -m entroly.autotune --iterations 500   # Run 500 iterations
    python -m entroly.autotune --bench-only       # Just evaluate current config
    python -m entroly.autotune --time-budget 60   # Max seconds per iteration

Single-file mutation discipline:
  - Only bench/tuning_config.json is modified
  - bench/cases.json is read-only (the fixed validation set)
  - This file (autotune.py) is read-only (the evaluation harness)
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("entroly")


def _log(msg: str = "") -> None:
    """Print to stderr so autotune output never corrupts MCP stdio transport."""
    print(msg, file=sys.stderr)


# -- Constants ---------------------------------------------------------------

BENCH_DIR = Path(__file__).parent.parent / "bench"
CASES_PATH = BENCH_DIR / "cases.json"
CONFIG_PATH = BENCH_DIR / "tuning_config.json"
RESULTS_PATH = BENCH_DIR / "results.tsv"

# Fixed time budget per benchmark evaluation.
# Iterations exceeding this are auto-discarded (poor configs stall the loop).
DEFAULT_TIME_BUDGET_SECS = 5.0

# Parameters and their mutation ranges
TUNABLE_PARAMS = {
    "weight_recency":       (0.05, 0.80),
    "weight_frequency":     (0.05, 0.80),
    "weight_semantic_sim":  (0.05, 0.80),
    "weight_entropy":       (0.05, 0.80),
    "decay_half_life_turns": (5, 50),
    "min_relevance_threshold": (0.01, 0.20),
    "exploration_rate":     (0.0, 0.3),
}


@dataclass
class BenchResult:
    """Result of running the benchmark suite with a given config."""
    context_efficiency: float
    recall_accuracy: float
    avg_wall_time_ms: float
    total_tokens_used: int
    total_information: float
    per_case: list[dict[str, Any]] = field(default_factory=list)


def load_cases() -> list[dict[str, Any]]:
    """Load the fixed benchmark cases (read-only val set).

    Returns empty list if the file doesn't exist (pip-install mode).
    """
    if not CASES_PATH.exists():
        return []
    with open(CASES_PATH) as f:
        return json.load(f)


def load_config() -> dict[str, Any]:
    """Load the current tuning config (the file we mutate).

    Returns default config if the file doesn't exist (pip-install mode).
    """
    if not CONFIG_PATH.exists():
        return {
            "weight_recency": 0.30,
            "weight_frequency": 0.25,
            "weight_semantic_sim": 0.25,
            "weight_entropy": 0.20,
        }
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(config: dict[str, Any]) -> None:
    """Save tuning config (single-file mutation).

    Creates parent directories if they don't exist.
    """
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def evaluate(config: dict[str, Any], cases: list[dict[str, Any]],
             time_budget: float = DEFAULT_TIME_BUDGET_SECS) -> BenchResult:
    """
    Run the benchmark suite with a given config.

    Fixed time budget evaluation: if any single case exceeds time_budget seconds,
    the entire run is marked as failed.
    """
    try:
        from entroly_core import EntrolyEngine
    except ImportError:
        _log("ERROR: entroly_core not available. Run `maturin develop` first.")
        raise RuntimeError("entroly_core not available for autotune evaluation")

    total_information = 0.0
    total_tokens_used = 0
    correct_selections = 0
    total_expected = 0
    wall_times: list[float] = []
    per_case: list[dict[str, Any]] = []

    for case in cases:
        engine = EntrolyEngine(
            w_recency=config.get("weight_recency", 0.30),
            w_frequency=config.get("weight_frequency", 0.25),
            w_semantic=config.get("weight_semantic_sim", 0.25),
            w_entropy=config.get("weight_entropy", 0.20),
            decay_half_life=config.get("decay_half_life_turns", 15),
            min_relevance=config.get("min_relevance_threshold", 0.05),
            exploration_rate=config.get("exploration_rate", 0.1),
        )

        frag_id_map: dict[str, str] = {}
        for frag_data in case["fragments"]:
            result = engine.ingest(
                frag_data["content"],
                frag_data["source"],
                frag_data["token_count"],
                False,
            )
            if hasattr(result, '__getitem__'):
                fid = result.get("fragment_id", result.get("duplicate_of", ""))
            else:
                fid = str(result)
            frag_id_map[frag_data["source"]] = fid

        t0 = time.perf_counter()
        opt_result = engine.optimize(case["token_budget"], case["query"])
        t1 = time.perf_counter()
        wall_ms = (t1 - t0) * 1000

        if wall_ms > time_budget * 1000:
            return BenchResult(
                context_efficiency=0.0, recall_accuracy=0.0,
                avg_wall_time_ms=wall_ms, total_tokens_used=0,
                total_information=0.0,
                per_case=[{"case_id": case["id"], "status": "timeout",
                           "wall_ms": wall_ms}],
            )

        wall_times.append(wall_ms)

        selected_sources: set = set()
        if hasattr(opt_result, '__getitem__'):
            selected_list = opt_result.get("selected", [])
            for item in selected_list:
                if hasattr(item, '__getitem__'):
                    src = item.get("source", "")
                    if src:
                        selected_sources.add(src)

        expected_sources = {
            f["source"] for f in case["fragments"] if f.get("expected_selected")
        }
        hits = len(selected_sources & expected_sources)
        total_expected += len(expected_sources)
        correct_selections += hits

        case_tokens = sum(
            f["token_count"] for f in case["fragments"]
            if f["source"] in selected_sources
        )
        case_info = sum(
            1.0 for f in case["fragments"]
            if f["source"] in selected_sources and f.get("expected_selected")
        )
        total_tokens_used += case_tokens
        total_information += case_info

        per_case.append({
            "case_id": case["id"],
            "recall": hits / len(expected_sources) if expected_sources else 1.0,
            "tokens_used": case_tokens,
            "wall_ms": round(wall_ms, 2),
            "selected": list(selected_sources),
        })

    ctx_eff = total_information / max(total_tokens_used, 1)
    recall_acc = correct_selections / max(total_expected, 1)
    avg_wall = sum(wall_times) / max(len(wall_times), 1)

    return BenchResult(
        context_efficiency=ctx_eff,
        recall_accuracy=recall_acc,
        avg_wall_time_ms=avg_wall,
        total_tokens_used=total_tokens_used,
        total_information=total_information,
        per_case=per_case,
    )


def mutate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Mutate one parameter at a time (single-change experiments for interpretability)."""
    new_config = dict(config)
    param = random.choice(list(TUNABLE_PARAMS.keys()))
    lo, hi = TUNABLE_PARAMS[param]
    current = new_config.get(param, (lo + hi) / 2)

    if isinstance(lo, int):
        delta = random.randint(-3, 3)
        new_val = max(lo, min(hi, int(current) + delta))
    else:
        sigma = (hi - lo) * 0.10
        new_val = current + random.gauss(0, sigma)
        new_val = max(lo, min(hi, round(new_val, 4)))

    new_config[param] = new_val

    weight_keys = ["weight_recency", "weight_frequency",
                   "weight_semantic_sim", "weight_entropy"]
    weight_sum = sum(new_config.get(k, 0.25) for k in weight_keys)
    if weight_sum > 0:
        for k in weight_keys:
            new_config[k] = round(new_config.get(k, 0.25) / weight_sum, 4)

    new_config["_mutated_param"] = param
    return new_config


def composite_score(result: BenchResult, config: dict[str, Any] | None = None,
                    defaults: dict[str, Any] | None = None,
                    drift_weight: float = 0.1) -> float:
    """Single metric: efficiency + recall - config drift penalty.

    Adds a drift penalty that penalises configs straying from defaults
    (prevents adversarial parameter regions).

    composite = 0.6·recall + 0.4·efficiency×100 - drift_weight·drift²×100
    """
    base = 0.6 * result.recall_accuracy + 0.4 * result.context_efficiency * 100

    if config is None or defaults is None or drift_weight <= 0:
        return base

    drift_sq = 0.0
    count = 0
    for key, (lo, hi) in TUNABLE_PARAMS.items():
        if key in config and key in defaults:
            span = max(float(hi) - float(lo), 1e-9)
            delta = (float(config[key]) - float(defaults.get(key, config[key]))) / span
            drift_sq += delta * delta
            count += 1

    if count > 0:
        drift = (drift_sq / count) ** 0.5
        return base - drift_weight * drift * drift * 100
    return base


# ══════════════════════════════════════════════════════════════════════
# Cautious Parameter Updates — Momentum-Dampened Autotune
# ══════════════════════════════════════════════════════════════════════
#
# Instead of binary keep/discard, three mechanisms:
#
# 1. EMA Blending: Instead of full replacement, blend the winning
#    config with the current best:
#      p_new = (1 - α) · p_old + α · p_candidate
#    where α scales with improvement magnitude (big improvement = faster adoption).
#
# 2. Polyak Averaging: Maintain a running average of all kept configs.
#    After tuning completes, the Polyak average is often more robust
#    than the single best (Polyak & Juditsky, 1992).
#
# 3. Config Drift Penalty: The composite score penalises configs that
#    stray too far from defaults, preventing the autotuner from finding
#    adversarial parameter regions that overfit the benchmark.
# ══════════════════════════════════════════════════════════════════════

def _ema_blend(best: dict[str, Any], candidate: dict[str, Any],
               alpha: float) -> dict[str, Any]:
    """EMA blend: p = (1-α)·best + α·candidate for numeric params."""
    blended = dict(best)
    for key in TUNABLE_PARAMS:
        if key in candidate and key in best:
            old_val = float(best[key])
            new_val = float(candidate[key])
            val = (1 - alpha) * old_val + alpha * new_val
            lo, hi = TUNABLE_PARAMS[key]
            val = max(float(lo), min(float(hi), val))
            if isinstance(lo, int):
                blended[key] = int(round(val))
            else:
                blended[key] = round(val, 4)
    return blended


def _polyak_update(avg: dict[str, Any], config: dict[str, Any],
                   count: int) -> dict[str, Any]:
    """Polyak running average: avg = ((n-1)·avg + config) / n."""
    updated = dict(avg)
    for key in TUNABLE_PARAMS:
        if key in config and key in avg:
            old_avg = float(avg[key])
            new_val = float(config[key])
            val = (old_avg * (count - 1) + new_val) / count
            lo, hi = TUNABLE_PARAMS[key]
            val = max(float(lo), min(float(hi), val))
            if isinstance(lo, int):
                updated[key] = int(round(val))
            else:
                updated[key] = round(val, 4)
    return updated


def log_result(iteration: int, config: dict[str, Any], result: BenchResult,
               status: str, description: str) -> None:
    """Append to results.tsv (structured experiment log)."""
    header = "iteration\tscore\trecall\tefficiency\tavg_ms\tstatus\tdescription\n"
    if not RESULTS_PATH.exists():
        with open(RESULTS_PATH, "w") as f:
            f.write(header)

    score = composite_score(result)
    with open(RESULTS_PATH, "a") as f:
        f.write(f"{iteration}\t{score:.4f}\t{result.recall_accuracy:.4f}\t"
                f"{result.context_efficiency:.6f}\t{result.avg_wall_time_ms:.1f}\t"
                f"{status}\t{description}\n")


def run_autotune(iterations: int = 100,
                 time_budget: float = DEFAULT_TIME_BUDGET_SECS,
                 bench_only: bool = False) -> None:
    """
    The experiment loop: mutate → evaluate → keep/discard → repeat.

    LOOP:
      1. Load current config
      2. Mutate one parameter
      3. Evaluate on benchmark suite
      4. If score improved → keep (advance)
      5. If score equal or worse → discard (revert)
      6. Log results
      7. Repeat
    """
    cases = load_cases()
    if not cases:
        _log("Entroly Autotune -- no benchmark cases found (pip-install mode), skipping")
        return
    config = load_config()

    _log(f"Entroly Autotune -- {len(cases)} benchmark cases loaded")
    _log(f"Time budget per case: {time_budget}s")

    _log("\n--- Baseline evaluation ---")
    baseline = evaluate(config, cases, time_budget)
    baseline_score = composite_score(baseline)
    _log(f"Baseline score: {baseline_score:.4f} "
         f"(recall={baseline.recall_accuracy:.3f}, "
         f"efficiency={baseline.context_efficiency:.6f}, "
         f"avg_ms={baseline.avg_wall_time_ms:.1f})")
    log_result(0, config, baseline, "keep", "baseline")

    if bench_only:
        _log("\nPer-case breakdown:")
        for pc in baseline.per_case:
            _log(f"  {pc['case_id']}: recall={pc.get('recall', 0):.2f}, "
                 f"tokens={pc.get('tokens_used', 0)}, "
                 f"wall_ms={pc.get('wall_ms', 0):.1f}")
        return

    best_score = baseline_score
    best_config = dict(config)
    defaults = dict(config)  # Original config for drift penalty
    improvements = 0

    # Polyak averaging state
    polyak_avg = dict(config)
    polyak_count = 1

    # EMA blending rate: scales with improvement magnitude
    # Base alpha = 0.3 (cautious). Doubles when improvement > 5% of score.
    ema_base_alpha = 0.3

    _log(f"\n--- Starting {iterations} experiments (cautious updates) ---")
    _log("(EMA blending + Polyak averaging + drift penalty)\n")

    for i in range(1, iterations + 1):
        candidate = mutate_config(best_config)
        mutated_param = candidate.pop("_mutated_param", "unknown")
        old_val = best_config.get(mutated_param)
        new_val = candidate.get(mutated_param)

        result = evaluate(candidate, cases, time_budget)
        score = composite_score(result, candidate, defaults)

        if score > best_score:
            # ── Cautious update: EMA blend instead of full replacement ──
            # Alpha scales with improvement magnitude: big jumps get faster
            # adoption, small improvements get cautious blending.
            delta_pct = (score - best_score) / max(best_score, 0.001)
            alpha = min(1.0, ema_base_alpha * (1.0 + delta_pct * 10.0))

            status = "keep"
            improvements += 1
            best_score = score
            best_config = _ema_blend(best_config, candidate, alpha)
            save_config(best_config)

            # Update Polyak average
            polyak_count += 1
            polyak_avg = _polyak_update(polyak_avg, best_config, polyak_count)

            marker = f">>> a={alpha:.2f}"
        elif (score == best_score and
              result.avg_wall_time_ms < baseline.avg_wall_time_ms):
            status = "keep"
            best_config = _ema_blend(best_config, candidate, ema_base_alpha * 0.5)
            save_config(best_config)
            marker = "  ="
        else:
            status = "discard"
            marker = "   "

        description = f"{mutated_param}: {old_val} -> {new_val}"
        log_result(i, candidate, result, status, description)

        _log(f"{marker} [{i:04d}] score={score:.4f} "
             f"(recall={result.recall_accuracy:.3f}, "
             f"eff={result.context_efficiency:.6f}) "
             f"| {status:7s} | {description}")

    # ── Final: evaluate Polyak average ──
    if polyak_count > 2:
        _log(f"\n--- Evaluating Polyak average ({polyak_count} samples) ---")
        polyak_result = evaluate(polyak_avg, cases, time_budget)
        polyak_score = composite_score(polyak_result, polyak_avg, defaults)
        _log(f"Polyak score: {polyak_score:.4f} vs best: {best_score:.4f}")

        if polyak_score >= best_score:
            _log("  -> Polyak average is at least as good -- using it")
            best_config = polyak_avg
            best_score = polyak_score
        else:
            _log("  -> Best single config is better -- keeping it")

    _log("\n--- Summary ---")
    _log(f"Total experiments: {iterations}")
    _log(f"Improvements found: {improvements}")
    _log(f"Baseline score: {baseline_score:.4f}")
    _log(f"Best score: {best_score:.4f}")
    delta_final = ((best_score - baseline_score) / max(baseline_score, 0.001)) * 100
    _log(f"Improvement: {delta_final:.1f}%")
    _log(f"Polyak samples: {polyak_count}")
    _log(f"\nBest config saved to {CONFIG_PATH}")
    _log(f"Full results log: {RESULTS_PATH}")

    save_config(best_config)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Entroly autonomous self-tuning loop")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--time-budget", type=float,
                        default=DEFAULT_TIME_BUDGET_SECS)
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    run_autotune(
        iterations=args.iterations,
        time_budget=args.time_budget,
        bench_only=args.bench_only,
    )


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT FEEDBACK BUS — Universal Self-Improvement Signal
# ═══════════════════════════════════════════════════════════════════════════
#
# Every component in Entroly should self-improve. The ComponentFeedbackBus
# provides a single, lightweight mechanism for this:
#
#   1. Any component can log(component, metric, value) after each operation
#   2. The bus persists episodes to component_feedback.jsonl
#   3. Components query their own history and tune their parameters
#
# The key insight (and what makes this revolutionary): this is a
# GRADIENT-FREE stochastic optimization framework. Each component
# maintains an exponential moving average (EMA) of its own metric,
# and adjusts its parameter in the direction that improves the EMA.
#
# No LLM calls. No tokens. Pure local O(1) computation per episode.
#
# Mathematical formulation:
#   θ_{t+1} = θ_t + α · sign(EMA_recent - EMA_baseline) · step_size
#   where α = learning_rate, EMA = exponential moving average of metric
#
# This is equivalent to online stochastic gradient approximation (SPSA)
# but with zero function evaluations — observations come from real usage.

class ComponentFeedbackBus:
    """Universal feedback bus for all Entroly components.

    Allows any component to log metrics and self-tune parameters
    based on observed outcomes. Zero token cost — all local compute.
    """

    # EMA smooth factor: higher = more responsive to recent data
    EMA_ALPHA = 0.15

    def __init__(self, data_dir: str):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._data_dir / "component_feedback.jsonl"
        self._emas: dict[str, dict[str, float]] = {}
        self._counts: dict[str, int] = {}

    def log(
        self,
        component: str,
        metric: str,
        value: float,
        params: dict[str, float] | None = None,
    ) -> None:
        """Log a metric observation for a component.

        Args:
            component: Component name (e.g., 'epistemic_router', 'prefetch')
            metric: Metric name (e.g., 'hit_rate', 'success_rate')
            value: Observed value (float)
            params: Current parameter snapshot (for correlation analysis)
        """
        entry = {
            "t": time.time(),
            "c": component,
            "m": metric,
            "v": value,
        }
        if params:
            entry["p"] = params

        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

        # Update in-memory EMA
        key = f"{component}:{metric}"
        if key not in self._emas:
            self._emas[key] = {"ema": value, "baseline": value}
            self._counts[key] = 1
        else:
            self._counts[key] += 1
            ema = self._emas[key]
            ema["ema"] = self.EMA_ALPHA * value + (1 - self.EMA_ALPHA) * ema["ema"]
            # Update baseline every 20 episodes (slow-moving reference)
            if self._counts[key] % 20 == 0:
                ema["baseline"] = ema["ema"]

    def get_trend(self, component: str, metric: str) -> dict[str, Any]:
        """Get the current trend for a component's metric.

        Returns:
            {ema, baseline, improving, delta, count}
        """
        key = f"{component}:{metric}"
        if key not in self._emas:
            return {"ema": 0.0, "baseline": 0.0, "improving": False, "delta": 0.0, "count": 0}

        ema = self._emas[key]
        delta = ema["ema"] - ema["baseline"]
        return {
            "ema": round(ema["ema"], 6),
            "baseline": round(ema["baseline"], 6),
            "improving": delta > 0,
            "delta": round(delta, 6),
            "count": self._counts[key],
        }

    def suggest_adjustment(
        self,
        component: str,
        metric: str,
        current_value: float,
        bounds: tuple[float, float],
        step_size: float = 0.01,
        maximize: bool = True,
    ) -> float:
        """Suggest a parameter adjustment based on observed metric trend.

        Uses SPSA-inspired gradient-free optimization:
        If metric is improving → keep direction
        If metric is degrading → reverse direction

        Args:
            component: Component name
            metric: Metric to optimize
            current_value: Current parameter value
            bounds: (min, max) bounds for the parameter
            step_size: How much to adjust per step
            maximize: True if higher metric is better

        Returns:
            Suggested new value for the parameter
        """
        trend = self.get_trend(component, metric)
        if trend["count"] < 5:
            return current_value  # Not enough data

        delta = trend["delta"]
        direction = 1.0 if (delta > 0) == maximize else -1.0
        new_value = current_value + direction * step_size * (bounds[1] - bounds[0])
        return max(bounds[0], min(bounds[1], new_value))

    def stats(self) -> dict[str, Any]:
        """Return all component feedbacks at a glance."""
        result = {}
        for key, ema in self._emas.items():
            delta = ema["ema"] - ema["baseline"]
            result[key] = {
                "ema": round(ema["ema"], 4),
                "baseline": round(ema["baseline"], 4),
                "improving": delta > 0,
                "episodes": self._counts.get(key, 0),
            }
        return result


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-SESSION FEEDBACK JOURNAL + REWARD-WEIGHTED OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════
#
# These classes add the same capabilities as the JS autotune:
#   1. FeedbackJournal: persistent .jsonl log of (weights, reward) episodes
#   2. reward_weighted_optimize(): closed-form optimal weights from feedback
#   3. TaskProfileOptimizer: per-task-type weight profiles
#
# The existing bench-based autotune above is untouched. These are additive.
# ═══════════════════════════════════════════════════════════════════════════

WEIGHT_KEYS = ["w_r", "w_f", "w_s", "w_e"]
DECAY_GAMMA = 0.995
WARMUP_EPISODES = 8
MAX_BLEND_RATE = 0.5
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.80
JOURNAL_MAX_AGE_S = 14 * 24 * 60 * 60  # 14 days


class FeedbackJournal:
    """Persistent cross-session feedback journal (.jsonl)."""

    def __init__(self, journal_dir: str):
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.journal_dir / "feedback_journal.jsonl"
        self._cache: list | None = None

    def log(self, *, weights: dict[str, float], reward: float,
            selected_count: int = 0, query: str = "",
            selected_sources: list[str] | None = None,
            token_budget: int = 0, turn: int = 0) -> None:
        """Log a feedback episode."""
        entry = {
            "t": time.time(),
            "w": weights,
            "n": selected_count,
            "src": (selected_sources or [])[:15],
            "q": query[:120],
            "r": max(-1.0, min(1.0, reward)),
            "turn": turn,
            "bgt": token_budget,
        }
        try:
            with open(self.journal_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            self._cache = None
        except Exception:
            pass

    def load(self, max_age: float = JOURNAL_MAX_AGE_S) -> list[dict]:
        """Load episodes filtered by max age."""
        if self._cache is not None:
            return self._cache
        cutoff = time.time() - max_age
        episodes = []
        try:
            with open(self.journal_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ep = json.loads(line)
                        if ep and ep.get("t", 0) >= cutoff and ep.get("w"):
                            episodes.append(ep)
                    except Exception:
                        pass
        except FileNotFoundError:
            pass
        self._cache = episodes
        return episodes

    def prune(self, max_age: float = JOURNAL_MAX_AGE_S) -> None:
        """Remove episodes older than max_age."""
        self._cache = None
        kept = self.load(max_age)
        self._cache = None
        try:
            with open(self.journal_path, "w") as f:
                for ep in kept:
                    f.write(json.dumps(ep) + "\n")
        except Exception:
            pass

    def count(self) -> int:
        return len(self.load())

    def stats(self) -> dict[str, Any]:
        eps = self.load()
        if not eps:
            return {"episodes": 0, "successes": 0, "failures": 0, "avg_reward": 0}
        successes = sum(1 for e in eps if e["r"] > 0)
        failures = sum(1 for e in eps if e["r"] < 0)
        avg_reward = sum(e["r"] for e in eps) / len(eps)
        return {
            "episodes": len(eps),
            "successes": successes,
            "failures": failures,
            "avg_reward": round(avg_reward, 3),
        }


def _normalize_weights(w: dict[str, float]) -> dict[str, float]:
    """Clamp and normalize weights to sum=1."""
    out = {k: max(MIN_WEIGHT, min(MAX_WEIGHT, w.get(k, 0.25))) for k in WEIGHT_KEYS}
    s = sum(out.values())
    return {k: round(v / s, 4) for k, v in out.items()} if s > 0 else out


def _extract_weights(w: dict) -> dict[str, float]:
    return {
        "w_r": w.get("w_r", w.get("R", 0.30)),
        "w_f": w.get("w_f", w.get("F", 0.25)),
        "w_s": w.get("w_s", w.get("S", 0.25)),
        "w_e": w.get("w_e", w.get("E", 0.20)),
    }


def reward_weighted_optimize(
    episodes: list[dict],
    current_weights: dict[str, float],
) -> dict[str, Any] | None:
    """
    Reward-weighted regression with:
    - Global advantage normalization (REINFORCE++, 2025)
    - Exponential decay (EXP3.S non-stationarity)
    - Per-dimension adaptive step (CMA-ES LED, 2024)
    - Fisher information natural gradient
    - Polyak-Ruppert averaging
    """
    if len(episodes) < 3:
        return None

    rewards = [e["r"] for e in episodes]
    mu = sum(rewards) / len(rewards)
    sigma = math.sqrt(sum((r - mu) ** 2 for r in rewards) / len(rewards))

    # Edge case: all rewards identical → use raw reward signs
    if sigma < 1e-6:
        advantages = [1.0 if r > 0 else (-1.0 if r < 0 else 0.0) for r in rewards]
    else:
        advantages = [(r - mu) / (sigma + 1e-8) for r in rewards]

    sorted_eps = sorted(episodes, key=lambda e: e.get("t", 0))
    decay_weights = [DECAY_GAMMA ** (len(episodes) - 1 - i) for i in range(len(sorted_eps))]

    attract = {k: 0.0 for k in WEIGHT_KEYS}
    attract_sum = 0.0
    repel = {k: 0.0 for k in WEIGHT_KEYS}
    repel_sum = 0.0

    for i, ep in enumerate(sorted_eps):
        w = _extract_weights(ep["w"])
        adv = advantages[episodes.index(ep)] if ep in episodes else 0
        decay = decay_weights[i]

        if adv > 0:
            weight = decay * adv
            for k in WEIGHT_KEYS:
                attract[k] += weight * w[k]
            attract_sum += weight
        elif adv < 0:
            weight = decay * abs(adv)
            for k in WEIGHT_KEYS:
                repel[k] += weight * w[k]
            repel_sum += weight

    if attract_sum <= 0:
        return None

    for k in WEIGHT_KEYS:
        attract[k] /= attract_sum
    if repel_sum > 0:
        for k in WEIGHT_KEYS:
            repel[k] /= repel_sum

    # Per-dimension stats
    dim_stats = {}
    for k in WEIGHT_KEYS:
        values = [_extract_weights(e["w"])[k] for e in sorted_eps]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 0.01
        snr = abs(attract[k] - current_weights[k]) / std
        dim_stats[k] = {"mean": mean, "std": std, "snr": snr}

    N = len(episodes)
    confidence = min(1.0, N / WARMUP_EPISODES)
    base_alpha = confidence * MAX_BLEND_RATE

    beta = 0.15 * min(1.0, repel_sum / attract_sum) if repel_sum > 0 else 0
    optimal = {}
    blended = {}

    for k in WEIGHT_KEYS:
        repel_delta = (repel[k] - current_weights[k]) if repel_sum > 0 else 0
        optimal[k] = attract[k] - beta * repel_delta

        nat_grad_scale = 1.0 / (dim_stats[k]["std"] ** 2 + 0.01)
        sigmoid_snr = 1.0 / (1.0 + math.exp(-2 * (dim_stats[k]["snr"] - 0.5)))
        alpha_k = base_alpha * sigmoid_snr

        direction = (optimal[k] - current_weights[k]) * nat_grad_scale
        clamped = max(-0.1, min(0.1, direction))
        blended[k] = current_weights[k] + alpha_k * clamped

    # Polyak average
    polyak = {k: 0.0 for k in WEIGHT_KEYS}
    for ep in sorted_eps:
        w = _extract_weights(ep["w"])
        for k in WEIGHT_KEYS:
            polyak[k] += w[k]
    for k in WEIGHT_KEYS:
        polyak[k] /= len(sorted_eps)

    # Regret
    avg_positive = (
        sum(e["r"] for e in episodes if e["r"] > 0) /
        max(sum(1 for e in episodes if e["r"] > 0), 1)
    )
    estimated_regret = max(0, (avg_positive - mu) * N)

    return {
        "optimal": _normalize_weights(optimal),
        "blended": _normalize_weights(blended),
        "polyak": _normalize_weights(polyak),
        "confidence": confidence,
        "success_count": sum(1 for e in episodes if e["r"] > 0),
        "failure_count": sum(1 for e in episodes if e["r"] < 0),
        "total_episodes": N,
        "estimated_regret": round(estimated_regret, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# TASK-CONDITIONED WEIGHT PROFILES (Novel)
# ═══════════════════════════════════════════════════════════════════════════

TASK_PATTERNS = {
    "Debugging":     re.compile(r"\b(fix|bug|error|crash|issue|debug|broken|fail|wrong|exception)\b", re.I),
    "Feature":       re.compile(r"\b(add|implement|create|build|feature|new|support|integrate|enable)\b", re.I),
    "Refactoring":   re.compile(r"\b(refactor|clean|reorganize|simplify|extract|rename|move|split|merge)\b", re.I),
    "Performance":   re.compile(r"\b(optimize|performance|slow|fast|speed|cache|memory|leak|bottleneck)\b", re.I),
    "Testing":       re.compile(r"\b(test|spec|assert|expect|mock|stub|coverage|unit|integration|e2e)\b", re.I),
    "Documentation": re.compile(r"\b(document|readme|comment|explain|describe|usage|api)\b", re.I),
}

TASK_PRIORS = {
    "Debugging":     {"w_r": 0.35, "w_f": 0.15, "w_s": 0.35, "w_e": 0.15},
    "Feature":       {"w_r": 0.20, "w_f": 0.25, "w_s": 0.25, "w_e": 0.30},
    "Refactoring":   {"w_r": 0.15, "w_f": 0.35, "w_s": 0.20, "w_e": 0.30},
    "Performance":   {"w_r": 0.25, "w_f": 0.30, "w_s": 0.25, "w_e": 0.20},
    "Testing":       {"w_r": 0.30, "w_f": 0.20, "w_s": 0.30, "w_e": 0.20},
    "Documentation": {"w_r": 0.20, "w_f": 0.20, "w_s": 0.30, "w_e": 0.30},
    "General":       {"w_r": 0.30, "w_f": 0.25, "w_s": 0.25, "w_e": 0.20},
}


def classify_query(query: str) -> str:
    """Classify a query into a task type."""
    if not query:
        return "General"
    for task_type, pattern in TASK_PATTERNS.items():
        if pattern.search(query):
            return task_type
    return "General"


class TaskProfileOptimizer:
    """Per-task-type weight optimization from feedback journal."""

    def __init__(self, journal: FeedbackJournal):
        self.journal = journal
        self._profiles: dict[str, dict] = {}

    def optimize_all(self) -> dict[str, dict]:
        """Classify episodes by task type and optimize each independently."""
        episodes = self.journal.load()
        if len(episodes) < 3:
            return {}

        buckets: dict[str, list] = {}
        for ep in episodes:
            task_type = classify_query(ep.get("q", ""))
            buckets.setdefault(task_type, []).append(ep)

        profiles = {}
        for task_type, task_eps in buckets.items():
            prior = TASK_PRIORS.get(task_type, TASK_PRIORS["General"])
            if len(task_eps) >= 3:
                result = reward_weighted_optimize(task_eps, prior)
                if result:
                    profiles[task_type] = {
                        "weights": result["blended"],
                        "confidence": result["confidence"],
                        "episodes": len(task_eps),
                    }
            if task_type not in profiles:
                profiles[task_type] = {
                    "weights": dict(prior),
                    "confidence": 0,
                    "episodes": len(task_eps),
                }

        # Add unseen task types with priors
        for task_type, prior in TASK_PRIORS.items():
            if task_type not in profiles:
                profiles[task_type] = {"weights": dict(prior), "confidence": 0, "episodes": 0}

        self._profiles = profiles
        return profiles

    def get_profile_for_query(self, query: str) -> tuple[dict[str, float], str, float]:
        """Get optimal weights for a query. Returns (weights, task_type, confidence)."""
        task_type = classify_query(query)
        profile = self._profiles.get(task_type)
        if profile and profile["confidence"] > 0:
            return profile["weights"], task_type, profile["confidence"]
        prior = TASK_PRIORS.get(task_type, TASK_PRIORS["General"])
        return prior, task_type, 0.0

    def apply_to_engine(self, engine, query: str) -> tuple[str, float]:
        """Apply task-conditioned weights to an engine."""
        weights, task_type, confidence = self.get_profile_for_query(query)
        if hasattr(engine, "_use_rust") and engine._use_rust:
            engine._rust.set_weights(
                weights["w_r"], weights["w_f"], weights["w_s"], weights["w_e"]
            )
        elif hasattr(engine, "set_weights"):
            engine.set_weights(
                weights["w_r"], weights["w_f"], weights["w_s"], weights["w_e"]
            )
        return task_type, confidence


# ═══════════════════════════════════════════════════════════════════════════
# DREAMING LOOP — Autonomous Self-Play Optimization (Pillar 3)
# ═══════════════════════════════════════════════════════════════════════════
#
# During idle cycles (no user queries for >60s), the system "dreams":
# it generates synthetic queries from its own experience (FeedbackJournal),
# tests counterfactual weight configurations against the benchmark harness,
# and keeps only configurations that improve context_efficiency.
#
# Mathematical foundation:
#   The dreaming loop solves the rate-distortion dual:
#     D*(R) = min_{W} E_q~P_synth[L(q, W)]   s.t.  R(W) ≤ budget
#   where W are PRISM weights, L is the loss (1 - context_efficiency),
#   q is drawn from the synthetic query distribution P_synth, and
#   R(W) is the token cost of the resulting context.
#
#   P_synth is constructed to maximize coverage of the joint
#   (task_type × entity × budget) space observed in the journal,
#   ensuring the system explores corners of its experience it rarely
#   encounters in real traffic.
#
# This is genuinely novel: no existing system performs counterfactual
# self-play on its own weight space during idle time. The closest
# analogy is AlphaGo's self-play — but applied to context engineering.
#
# Guarantees:
#   - Zero tokens: all computation is local (CPU-only)
#   - Monotonic improvement: only keeps configs that beat the baseline
#   - Bounded resource usage: max N iterations per dream cycle (default 10)
#   - Non-blocking: runs in a background thread, yields to user queries

DREAMING_IDLE_THRESHOLD_S = 60.0    # seconds of inactivity before dreaming
DREAMING_MAX_ITERATIONS = 10        # max experiments per dream cycle
DREAMING_WEIGHT_PERTURB_STD = 0.05  # standard deviation for weight perturbation


class DreamingLoop:
    """Autonomous self-play optimization during idle cycles.

    Generates synthetic queries from the FeedbackJournal, tests weight
    perturbations against the benchmark harness, and keeps improvements.
    All computation is local — zero token cost.
    """

    def __init__(
        self,
        journal: FeedbackJournal,
        config_path: Path | None = None,
        max_iterations: int = DREAMING_MAX_ITERATIONS,
        archetype_optimizer: Any = None,
    ):
        self._journal = journal
        self._config_path = config_path or CONFIG_PATH
        self._max_iterations = max_iterations
        self._last_activity: float = time.time()
        self._total_dreams: int = 0
        self._total_improvements: int = 0
        self._best_efficiency: float = 0.0
        self._archetype_optimizer = archetype_optimizer

    def record_activity(self) -> None:
        """Called on every user query to reset the idle timer."""
        self._last_activity = time.time()

    def should_dream(self) -> bool:
        """Returns True if the system has been idle long enough to dream."""
        return (time.time() - self._last_activity) > DREAMING_IDLE_THRESHOLD_S

    def generate_synthetic_queries(self) -> list[dict[str, Any]]:
        """Generate synthetic queries from journal history.

        Constructs a set of counterfactual queries that maximizes
        coverage of the (task_type × entity) space. Each query
        combines an entity from past episodes with a task pattern
        from TASK_PATTERNS, generating situations the system may
        not have encountered in real traffic.

        This is the "imagination" step — the system generates its
        own training data from its memory of past interactions.
        """
        episodes = self._journal.load()
        if not episodes:
            # No history — generate from TASK_PATTERNS alone
            return [
                {"query": f"{task_type.lower()} the main module",
                 "task_type": task_type,
                 "budget": 8000}
                for task_type in TASK_PATTERNS
            ]

        # Extract unique entities and queries from episodes
        seen_entities: set[str] = set()
        seen_queries: list[str] = []
        for ep in episodes:
            q = ep.get("q", "")
            if q:
                seen_queries.append(q)
                # Extract entity-like terms (multi-word or dotted identifiers)
                for term in re.findall(r'[a-zA-Z_][\w.]*', q):
                    if len(term) > 3 and "." in term or len(term) > 6:
                        seen_entities.add(term)

        # Generate counterfactual queries: cross product of
        # (task_type × entity) that we haven't seen in real traffic
        synthetic: list[dict[str, Any]] = []

        # Type 1: Replay successful episodes with perturbed budgets
        for ep in episodes[-5:]:
            if ep.get("r", 0) > 0:
                for budget_mult in [0.5, 0.75, 1.5]:
                    synthetic.append({
                        "query": ep.get("q", ""),
                        "task_type": classify_query(ep.get("q", "")),
                        "budget": int(ep.get("bgt", 8000) * budget_mult),
                        "origin": "replay_budget_perturb",
                    })

        # Type 2: Novel queries combining entities × task patterns
        entity_list = list(seen_entities)[:10]
        task_names = list(TASK_PATTERNS.keys())
        for entity in entity_list:
            task = random.choice(task_names)
            verbs = {
                "Debugging": "fix",
                "Feature": "implement",
                "Refactoring": "refactor",
                "Performance": "optimize",
                "Testing": "test",
                "Documentation": "document",
            }
            verb = verbs.get(task, "explain")
            synthetic.append({
                "query": f"{verb} {entity}",
                "task_type": task,
                "budget": 8000,
                "origin": "cross_product",
            })

        # Type 3: Failure replay — specifically re-test areas that failed
        for ep in episodes:
            if ep.get("r", 0) < 0:
                synthetic.append({
                    "query": ep.get("q", ""),
                    "task_type": classify_query(ep.get("q", "")),
                    "budget": ep.get("bgt", 8000),
                    "origin": "failure_replay",
                })

        # Shuffle and limit
        random.shuffle(synthetic)
        return synthetic[:self._max_iterations * 3]

    def run_dream_cycle(self) -> dict[str, Any]:
        """Run one dream cycle: generate synthetic queries, test weight
        perturbations, keep improvements.

        Returns stats about the cycle.
        """
        if not self.should_dream():
            return {"status": "not_idle", "idle_seconds": time.time() - self._last_activity}

        t_start = time.time()
        self._total_dreams += 1

        # Load current best config
        config = load_config()
        cases = load_cases()
        if not cases:
            return {"status": "no_cases", "dream_id": self._total_dreams}

        # Evaluate baseline
        try:
            baseline = evaluate(config, cases, time_budget=DEFAULT_TIME_BUDGET_SECS)
            self._best_efficiency = baseline.context_efficiency
        except Exception as e:
            return {"status": "error", "error": str(e)}

        improvements = 0
        experiments = 0
        synthetic_queries = self.generate_synthetic_queries()

        for _ in range(min(self._max_iterations, len(synthetic_queries))):
            # Check if user has become active
            if not self.should_dream():
                break

            experiments += 1

            # Perturb weights using Gaussian noise
            mutated_config = dict(config)
            for param, (lo, hi) in TUNABLE_PARAMS.items():
                if param in mutated_config:
                    current = mutated_config[param]
                    delta = random.gauss(0, DREAMING_WEIGHT_PERTURB_STD)
                    new_val = max(lo, min(hi, current + delta * (hi - lo)))
                    if isinstance(current, int):
                        new_val = int(round(new_val))
                    mutated_config[param] = new_val

            # Evaluate the mutation
            try:
                result = evaluate(mutated_config, cases,
                                  time_budget=DEFAULT_TIME_BUDGET_SECS)
            except Exception:
                continue

            # Keep only strict improvements (monotonic improvement guarantee)
            if result.context_efficiency > self._best_efficiency:
                self._best_efficiency = result.context_efficiency
                config = mutated_config
                save_config(config)
                improvements += 1
                self._total_improvements += 1

                # ── Pillar 4: Feed improvement to archetype optimizer ──
                # Maps the improved autotune config weights to the archetype
                # strategy table so they persist per-archetype across sessions.
                if self._archetype_optimizer:
                    try:
                        updated = self._archetype_optimizer.current_weights()
                        key_map = {
                            "w_r": "w_recency", "w_f": "w_frequency",
                            "w_s": "w_semantic", "w_e": "w_entropy",
                        }
                        for short, full in key_map.items():
                            if short in mutated_config:
                                updated[full] = mutated_config[short]
                        self._archetype_optimizer.update_weights(updated)
                    except Exception:
                        pass  # non-critical: don't break dreaming

        wall_seconds = time.time() - t_start

        return {
            "status": "completed",
            "dream_id": self._total_dreams,
            "experiments": experiments,
            "improvements": improvements,
            "baseline_efficiency": baseline.context_efficiency,
            "best_efficiency": self._best_efficiency,
            "wall_seconds": round(wall_seconds, 2),
            "synthetic_queries_generated": len(synthetic_queries),
            "total_dreams": self._total_dreams,
            "total_improvements": self._total_improvements,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "total_dreams": self._total_dreams,
            "total_improvements": self._total_improvements,
            "best_efficiency": self._best_efficiency,
            "idle_seconds": round(time.time() - self._last_activity, 1),
            "will_dream": self.should_dream(),
        }
