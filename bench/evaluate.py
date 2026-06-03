"""
Fixed Time Budget Evaluation Harness
====================================

Deterministic, reproducible quality check that runs against a fixed
benchmark suite. Results are the single number the autotuner optimizes.

Measures:
  - recall:            fraction of expected fragments correctly selected
  - precision:         fraction of selected fragments that were expected
  - context_efficiency: sum(entropy * tokens) / total_tokens for selected set
  - latency_ms:        wall-clock time for optimize_context() call

Any config that exceeds MAX_LATENCY_MS is auto-discarded. This prevents
the autotuner from finding "optimal" configs that are too slow for
real-time MCP use.

Usage:
    python -m bench.evaluate                         # run with default config
    python -m bench.evaluate --config tuning_config.json  # run with specific config
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Fixed evaluation constants (DO NOT CHANGE — this is the fixed metric)
MAX_LATENCY_MS = 500  # any optimize() call exceeding this fails the case

# Canonical default tuning config. Mirrors the EntrolyEngine constructor
# defaults in entroly-core/src/lib.rs (w_recency=0.30, w_frequency=0.25,
# w_semantic=0.25, w_entropy=0.20, decay_half_life=15, min_relevance=0.05,
# hamming_threshold=3, exploration_rate=0.1) plus the IOS defaults used by
# create_engine_from_config. Used as the autotune baseline when no
# tuning_config.json exists yet, so `entroly autotune` works on a fresh repo
# instead of crashing with FileNotFoundError.
DEFAULT_TUNING_CONFIG: dict = {
    "weights": {
        "recency": 0.30,
        "frequency": 0.25,
        "semantic_sim": 0.25,
        "entropy": 0.20,
    },
    "decay": {
        "half_life_turns": 15,
        "min_relevance_threshold": 0.05,
    },
    "knapsack": {
        "exploration_rate": 0.1,
    },
    "dedup": {
        "hamming_threshold": 3,
    },
    "sliding_window": {
        "long_window_fraction": 0.30,
    },
    "prism": {
        "learning_rate": 0.01,
        "beta": 0.90,
    },
    "egtc": {
        "alpha": 1.0,
        "gamma": 1.0,
        "epsilon": 0.5,
        "fisher_scale": 0.5,
        "trajectory_c_min": 0.6,
        "trajectory_lambda": 0.05,
    },
    "ios": {
        "skeleton_info_factor": 0.70,
        "reference_info_factor": 0.15,
        "diversity_floor": 0.10,
    },
    "ecdb": {
        "min_budget": 500,
        "max_fraction": 0.30,
        "sigmoid_steepness": 3.0,
        "sigmoid_base": 0.5,
        "sigmoid_range": 1.5,
        "codebase_divisor": 200,
        "codebase_cap": 2.0,
    },
}


def load_cases(path: Path | None = None) -> list[dict]:
    if path is None:
        path = Path(__file__).parent / "cases.json"
    with open(path) as f:
        return json.load(f)


def validate_tuning_config(config: dict) -> list[str]:
    """Validate tuning_config.json schema and value ranges.

    Returns a list of error strings. Empty list means valid.
    """
    errors: list[str] = []

    # Required top-level sections
    required_sections = ["weights", "decay", "knapsack"]
    for sec in required_sections:
        if sec not in config:
            errors.append(f"missing required section: '{sec}'")

    # Weights: must be numeric, positive, sum to ~1.0
    w = config.get("weights", {})
    weight_keys = ["recency", "frequency", "semantic_sim", "entropy"]
    for k in weight_keys:
        if k not in w:
            errors.append(f"weights.{k} missing")
        elif not isinstance(w[k], (int, float)):
            errors.append(f"weights.{k} must be numeric, got {type(w[k]).__name__}")
        elif w[k] < 0:
            errors.append(f"weights.{k} must be >= 0, got {w[k]}")

    if all(k in w and isinstance(w[k], (int, float)) for k in weight_keys):
        total = sum(w[k] for k in weight_keys)
        if abs(total - 1.0) > 0.01:
            errors.append(f"weights must sum to ~1.0, got {total:.4f}")

    # Range-checked numeric fields: (section, key, min, max)
    range_checks: list[tuple[str, str, float, float]] = [
        ("decay", "half_life_turns", 1, 1000),
        ("decay", "min_relevance_threshold", 0.0, 1.0),
        ("knapsack", "exploration_rate", 0.0, 1.0),
        ("sliding_window", "long_window_fraction", 0.0, 1.0),
        ("prism", "learning_rate", 0.0, 1.0),
        ("prism", "beta", 0.0, 1.0),
        ("egtc", "alpha", 0.0, 10.0),
        ("egtc", "gamma", 0.0, 10.0),
        ("egtc", "epsilon", 0.0, 10.0),
        ("egtc", "fisher_scale", 0.0, 5.0),
        ("ecdb", "min_budget", 1, 100000),
        ("ecdb", "max_fraction", 0.0, 1.0),
        ("ios", "skeleton_info_factor", 0.0, 1.0),
        ("ios", "reference_info_factor", 0.0, 1.0),
        ("ios", "diversity_floor", 0.0, 1.0),
    ]
    for section, key, lo, hi in range_checks:
        sec = config.get(section, {})
        if key in sec:
            val = sec[key]
            if not isinstance(val, (int, float)):
                errors.append(f"{section}.{key} must be numeric, got {type(val).__name__}")
            elif val < lo or val > hi:
                errors.append(f"{section}.{key}={val} out of range [{lo}, {hi}]")

    return errors


def load_tuning_config(path: Path | None = None) -> dict:
    """Load the tuning config, falling back to canonical defaults.

    The config lives alongside the harness at ``bench/tuning_config.json``.
    If it is missing or unparseable we return a deep copy of
    ``DEFAULT_TUNING_CONFIG`` (with a warning) rather than crashing — this is
    what lets ``entroly autotune`` run on a fresh checkout that has never been
    tuned before.
    """
    import copy
    import logging

    log = logging.getLogger("entroly")

    if path is None:
        path = Path(__file__).parent / "tuning_config.json"

    try:
        with open(path) as f:
            config = json.load(f)
    except FileNotFoundError:
        log.warning(
            "tuning_config not found at %s — using built-in defaults", path
        )
        return copy.deepcopy(DEFAULT_TUNING_CONFIG)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning(
            "tuning_config at %s is unreadable (%s) — using built-in defaults",
            path, exc,
        )
        return copy.deepcopy(DEFAULT_TUNING_CONFIG)

    errors = validate_tuning_config(config)
    if errors:
        for err in errors:
            log.warning(f"tuning_config validation: {err}")

    return config


def create_engine_from_config(config: dict):
    """Create an EntrolyEngine using the tuning config weights."""
    try:
        from entroly_core import EntrolyEngine
    except ImportError:
        raise ImportError(
            "entroly_core not available. Run `maturin develop` in entroly-core/."
        )

    w = config["weights"]
    d = config["decay"]
    k = config["knapsack"]
    sw = config.get("sliding_window", {})
    at = config.get("autotuner", {})
    # IOS parameters from tuning config
    ios = config.get("ios", {})
    return EntrolyEngine(
        w_recency=w["recency"],
        w_frequency=w["frequency"],
        w_semantic=w["semantic_sim"],
        w_entropy=w["entropy"],
        decay_half_life=d["half_life_turns"],
        min_relevance=d["min_relevance_threshold"],
        hamming_threshold=config["dedup"]["hamming_threshold"],
        exploration_rate=k["exploration_rate"],
        ios_skeleton_info_factor=ios.get("skeleton_info_factor", 0.70),
        ios_reference_info_factor=ios.get("reference_info_factor", 0.15),
        ios_diversity_floor=ios.get("diversity_floor", 0.10),
    )


def run_case(engine_factory, case: dict) -> dict[str, Any]:
    """Run a single benchmark case. Returns metrics dict."""
    engine = engine_factory()

    # Ingest all fragments
    source_to_id: dict[str, str] = {}
    for frag in case["fragments"]:
        result = engine.ingest(
            frag["content"],
            frag["source"],
            frag.get("token_count", 0),
            False,
        )
        result = dict(result)
        if result.get("status") == "ingested":
            source_to_id[frag["source"]] = result["fragment_id"]

    # Run optimize with time budget enforcement
    t0 = time.perf_counter()
    result = dict(engine.optimize(case["token_budget"], case.get("query", "")))
    latency_ms = (time.perf_counter() - t0) * 1000

    # Extract selected sources
    selected = result.get("selected", [])
    selected_sources = set()
    for item in selected:
        item = dict(item)
        selected_sources.add(item.get("source", ""))

    # Compute recall and precision against expected
    expected_selected = {
        f["source"] for f in case["fragments"] if f.get("expected_selected", False)
    }
    expected_not_selected = {
        f["source"] for f in case["fragments"] if not f.get("expected_selected", True)
    }

    true_positives = expected_selected & selected_sources
    false_negatives = expected_selected - selected_sources
    false_positives = expected_not_selected & selected_sources

    recall = len(true_positives) / max(len(expected_selected), 1)
    precision = len(true_positives) / max(len(selected_sources), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # Context efficiency from the optimize result, with fallback computation
    context_efficiency = result.get("context_efficiency", 0.0)
    if context_efficiency == 0.0 and selected:
        # Fallback: compute from per-fragment entropy × tokens / total_tokens
        total_tok = sum(dict(item).get("token_count", 0) for item in selected)
        if total_tok > 0:
            weighted = sum(
                dict(item).get("entropy_score", 0.0) * dict(item).get("token_count", 0)
                for item in selected
            )
            context_efficiency = weighted / total_tok

    return {
        "case_id": case["id"],
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "context_efficiency": context_efficiency,
        "latency_ms": round(latency_ms, 2),
        "latency_ok": latency_ms <= MAX_LATENCY_MS,
        "selected_sources": sorted(selected_sources),
        "expected_selected": sorted(expected_selected),
        "true_positives": sorted(true_positives),
        "false_negatives": sorted(false_negatives),
        "false_positives": sorted(false_positives),
        "tokens_used": result.get("total_tokens", 0),
        "method": result.get("method", "unknown"),
    }


def evaluate(config: dict | None = None, cases_path: Path | None = None) -> dict:
    """Run the full benchmark suite. Returns aggregate metrics."""
    if config is None:
        config = load_tuning_config()
    cases = load_cases(cases_path)

    factory = lambda: create_engine_from_config(config)

    results = []
    for case in cases:
        r = run_case(factory, case)
        results.append(r)

    # Aggregate metrics (primary objective for autotuner)
    avg_recall = sum(r["recall"] for r in results) / max(len(results), 1)
    avg_precision = sum(r["precision"] for r in results) / max(len(results), 1)
    avg_f1 = sum(r["f1"] for r in results) / max(len(results), 1)
    avg_efficiency = sum(r["context_efficiency"] for r in results) / max(len(results), 1)
    avg_latency = sum(r["latency_ms"] for r in results) / max(len(results), 1)
    all_latency_ok = all(r["latency_ok"] for r in results)

    composite = (
        0.50 * avg_recall
        + 0.25 * avg_precision
        + 0.25 * avg_efficiency
    )

    return {
        "composite_score": round(composite, 4),
        "avg_recall": round(avg_recall, 4),
        "avg_precision": round(avg_precision, 4),
        "avg_f1": round(avg_f1, 4),
        "avg_context_efficiency": round(avg_efficiency, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "all_latency_ok": all_latency_ok,
        "total_cases": len(results),
        "cases": results,
        "config": config,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Entroly benchmark evaluation")
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to tuning_config.json"
    )
    parser.add_argument(
        "--cases", type=Path, default=None, help="Path to cases.json"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output raw JSON"
    )
    args = parser.parse_args()

    result = evaluate(
        config=load_tuning_config(args.config) if args.config else None,
        cases_path=args.cases,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Composite Score: {result['composite_score']:.4f}")
        print(f"  Recall:     {result['avg_recall']:.4f}")
        print(f"  Precision:  {result['avg_precision']:.4f}")
        print(f"  F1 Score:   {result['avg_f1']:.4f}")
        print(f"  Efficiency: {result['avg_context_efficiency']:.4f}")
        print(f"  Latency:    {result['avg_latency_ms']:.1f} ms (ok={result['all_latency_ok']})")
        print(f"  Cases:      {result['total_cases']}")
        print()
        for c in result["cases"]:
            status = "PASS" if c["recall"] >= 0.5 and c["latency_ok"] else "FAIL"
            print(
                f"  [{status}] {c['case_id']}: "
                f"R={c['recall']:.2f} P={c['precision']:.2f} F1={c['f1']:.2f} "
                f"latency={c['latency_ms']:.1f}ms"
            )


if __name__ == "__main__":
    main()
