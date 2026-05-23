"""
Entroly Autonomous Self-Tuning Loop
====================================

Keep/discard experimentation loop that optimizes hyperparameters overnight by
trying small parameter mutations, evaluating the result, and keeping improvements.

In Entroly:  autotune mutates tuning_config.json → benchmark eval → composite_score → keep/discard.

Single-file mutation discipline: this script ONLY modifies tuning_config.json.
The benchmark harness (evaluate.py) and Rust core are read-only.

Each iteration:
  1. Load current best config from tuning_config.json
  2. Mutate one parameter (or use PRISM spectral directions)
  3. Run benchmark suite (evaluate.py)
  4. If composite_score improves: keep (write to tuning_config.json)
  5. If composite_score regresses: discard (restore previous config)
  6. Log the result and repeat

Runs entirely on CPU within 32GB RAM. Each iteration takes seconds.

Usage:
    python -m bench.autotune                    # 50 iterations (default)
    python -m bench.autotune --iterations 200   # run overnight
    python -m bench.autotune --strategy spectral  # use PRISM directions
"""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evaluate import evaluate, load_tuning_config


# ── Tunable parameter definitions ──────────────────────────────────────

@dataclass
class TunableParam:
    """A single tunable parameter with its JSON path and bounds."""
    path: list[str]       # e.g., ["weights", "recency"]
    min_val: float
    max_val: float
    step_size: float      # mutation step size (fraction of range)
    is_integer: bool = False

    def get(self, config: dict) -> float:
        obj = config
        for key in self.path[:-1]:
            obj = obj[key]
        return obj[self.path[-1]]

    def set(self, config: dict, value: float) -> None:
        obj = config
        for key in self.path[:-1]:
            obj = obj[key]
        if self.is_integer:
            obj[self.path[-1]] = int(round(value))
        else:
            obj[self.path[-1]] = round(value, 6)


# All tunable parameters and their bounds
TUNABLE_PARAMS = [
    TunableParam(["weights", "recency"],           0.05, 0.60, 0.05),
    TunableParam(["weights", "frequency"],          0.05, 0.60, 0.05),
    TunableParam(["weights", "semantic_sim"],       0.05, 0.60, 0.05),
    TunableParam(["weights", "entropy"],            0.05, 0.60, 0.05),
    TunableParam(["decay", "half_life_turns"],      5,    50,   5, is_integer=True),
    TunableParam(["decay", "min_relevance_threshold"], 0.01, 0.20, 0.02),
    TunableParam(["knapsack", "exploration_rate"],  0.0,  0.30, 0.02),
    TunableParam(["sliding_window", "long_window_fraction"], 0.10, 0.50, 0.05),
    TunableParam(["prism", "learning_rate"],        0.001, 0.05, 0.005),
    TunableParam(["prism", "beta"],                 0.80, 0.99, 0.02),
    # EGTC v2 temperature calibration coefficients
    TunableParam(["egtc", "alpha"],                 0.5,  3.0,  0.1),
    TunableParam(["egtc", "gamma"],                 0.3,  2.5,  0.1),
    TunableParam(["egtc", "epsilon"],               0.1,  1.5,  0.1),
    TunableParam(["egtc", "fisher_scale"],          0.3,  0.8,  0.05),
    TunableParam(["egtc", "trajectory_c_min"],      0.3,  0.9,  0.05),
    TunableParam(["egtc", "trajectory_lambda"],     0.02, 0.15, 0.01),
    # IOS: Multi-Resolution Knapsack + Submodular Diversity Selection
    TunableParam(["ios", "skeleton_info_factor"],    0.30, 0.90, 0.05),
    TunableParam(["ios", "reference_info_factor"],   0.05, 0.40, 0.03),
    TunableParam(["ios", "diversity_floor"],          0.0, 0.30, 0.02),
    # ECDB: Entropy-Calibrated Dynamic Budget
    TunableParam(["ecdb", "min_budget"],             200, 1000, 50, is_integer=True),
    TunableParam(["ecdb", "max_fraction"],          0.10, 0.50, 0.03),
    TunableParam(["ecdb", "sigmoid_steepness"],      1.0, 6.0,  0.3),
    TunableParam(["ecdb", "sigmoid_base"],           0.2, 1.0,  0.1),
    TunableParam(["ecdb", "sigmoid_range"],          0.5, 3.0,  0.2),
    TunableParam(["ecdb", "codebase_divisor"],       50,  500,  25),
    TunableParam(["ecdb", "codebase_cap"],           1.0, 4.0,  0.2),
]


@dataclass
class Experiment:
    """Record of a single autotuning experiment."""
    iteration: int
    param_name: str
    old_value: float
    new_value: float
    old_score: float
    new_score: float
    kept: bool
    duration_ms: float


def normalize_weights(config: dict) -> None:
    """Ensure the 4 scoring weights sum to 1.0."""
    w = config["weights"]
    total = w["recency"] + w["frequency"] + w["semantic_sim"] + w["entropy"]
    if total > 0:
        w["recency"]      = round(w["recency"] / total, 6)
        w["frequency"]    = round(w["frequency"] / total, 6)
        w["semantic_sim"] = round(w["semantic_sim"] / total, 6)
        w["entropy"]      = round(w["entropy"] / total, 6)


def mutate_random(config: dict, rng: random.Random) -> tuple[dict, str, float, float]:
    """Mutate one random parameter. Returns (new_config, param_name, old_val, new_val)."""
    config = copy.deepcopy(config)
    param = rng.choice(TUNABLE_PARAMS)
    old_val = param.get(config)
    delta = rng.uniform(-param.step_size, param.step_size) * (param.max_val - param.min_val)
    new_val = max(param.min_val, min(param.max_val, old_val + delta))
    param.set(config, new_val)

    # Re-normalize weights if we touched one
    if param.path[0] == "weights":
        normalize_weights(config)

    name = ".".join(param.path)
    return config, name, old_val, param.get(config)


def save_config(config: dict, path: Path) -> None:
    """Atomically write config: tmpfile + fsync + rename. Crash-safe."""
    import os as _os
    import tempfile as _tmp
    data = json.dumps(config, indent=2) + "\n"
    fd, tmp_path = _tmp.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        _os.write(fd, data.encode())
        _os.fsync(fd)
        _os.close(fd)
        _os.replace(tmp_path, str(path))
    except BaseException:
        try:
            _os.close(fd)
        except OSError:
            pass
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass
        raise


def snapshot_config(config_path: Path) -> Path | None:
    """Save a timestamped snapshot of the current config before tuning.

    Stored at <config_dir>/tuning_config.<timestamp>.bak.json.
    Enables `autotune --rollback` to restore the last known-good config.
    """
    if not config_path.exists():
        return None
    ts = int(time.time())
    backup = config_path.with_suffix(f".{ts}.bak.json")
    import shutil
    shutil.copy2(config_path, backup)
    return backup


def rollback_config(config_path: Path) -> dict[str, Any]:
    """Restore the most recent backup of tuning_config.json.

    Returns dict with status, restored_from path, and the restored config.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "tuning_config.json"

    # Find all backups: tuning_config.*.bak.json
    parent = config_path.parent
    stem = config_path.stem  # "tuning_config"
    backups = sorted(
        parent.glob(f"{stem}.*.bak.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not backups:
        return {"status": "no_backup_found"}

    latest = backups[0]
    import shutil
    shutil.copy2(latest, config_path)
    restored = json.loads(config_path.read_text())

    return {
        "status": "rolled_back",
        "restored_from": str(latest),
        "config": restored,
    }


def autotune(
    iterations: int = 50,
    config_path: Path | None = None,
    cases_path: Path | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the autonomous self-tuning loop."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "tuning_config.json"

    rng = random.Random(seed)

    # Snapshot current config before mutating (enables --rollback)
    backup = snapshot_config(config_path)
    if backup and verbose:
        print(f"  Config snapshot saved: {backup}")

    best_config = load_tuning_config(config_path)
    best_result = evaluate(best_config, cases_path)
    best_score = best_result["composite_score"]

    experiments: list[Experiment] = []
    improvements = 0
    start_time = time.time()

    if verbose:
        print(f"Autotune starting: baseline composite_score = {best_score:.4f}")
        print(f"  recall={best_result['avg_recall']:.4f} "
              f"precision={best_result['avg_precision']:.4f} "
              f"efficiency={best_result['avg_context_efficiency']:.4f}")
        print(f"  {iterations} iterations planned")
        print()

    for i in range(iterations):
        # Mutate
        candidate_config, param_name, old_val, new_val = mutate_random(best_config, rng)

        # Evaluate
        t0 = time.perf_counter()
        try:
            candidate_result = evaluate(candidate_config, cases_path)
            candidate_score = candidate_result["composite_score"]
        except Exception as e:
            if verbose:
                print(f"  [{i+1:3d}] ERROR evaluating {param_name}: {e}")
            continue
        duration_ms = (time.perf_counter() - t0) * 1000

        # Keep/discard decision
        kept = candidate_score > best_score and candidate_result["all_latency_ok"]
        if kept:
            best_config = candidate_config
            best_score = candidate_score
            best_result = candidate_result
            save_config(best_config, config_path)
            improvements += 1

        exp = Experiment(
            iteration=i + 1,
            param_name=param_name,
            old_value=old_val,
            new_value=new_val,
            old_score=best_score if not kept else candidate_score - (candidate_score - best_score),
            new_score=candidate_score,
            kept=kept,
            duration_ms=duration_ms,
        )
        experiments.append(exp)

        if verbose:
            status = "KEEP" if kept else "SKIP"
            delta = candidate_score - (best_score if not kept else best_score)
            print(
                f"  [{i+1:3d}/{iterations}] [{status}] "
                f"{param_name}: {old_val:.4f} -> {new_val:.4f}  "
                f"score={candidate_score:.4f} (delta={delta:+.4f})  "
                f"{duration_ms:.0f}ms"
            )

    elapsed = time.time() - start_time

    if verbose:
        print()
        print(f"Autotune complete: {improvements}/{iterations} improvements kept")
        print(f"  Final composite_score = {best_score:.4f}")
        print(f"  Total time: {elapsed:.1f}s ({elapsed/max(iterations,1)*1000:.0f}ms/iter)")

    return {
        "final_score": best_score,
        "final_result": best_result,
        "improvements": improvements,
        "iterations": iterations,
        "elapsed_seconds": round(elapsed, 1),
        "experiments": [
            {
                "iteration": e.iteration,
                "param": e.param_name,
                "old": e.old_value,
                "new": e.new_value,
                "score": e.new_score,
                "kept": e.kept,
                "ms": e.duration_ms,
            }
            for e in experiments
        ],
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Entroly autonomous self-tuning loop"
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=50,
        help="Number of tuning iterations (default: 50)"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to tuning_config.json (default: entroly/tuning_config.json)"
    )
    parser.add_argument(
        "--cases", type=Path, default=None,
        help="Path to benchmark cases.json"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--rollback", action="store_true",
        help="Restore the previous tuning_config.json (undo last autotune)"
    )
    args = parser.parse_args()

    if args.rollback:
        cfg_path = args.config or (Path(__file__).parent.parent / "tuning_config.json")
        result = rollback_config(cfg_path)
        if result["status"] == "no_backup_found":
            print("No backup found — nothing to roll back.")
            sys.exit(1)
        print(f"Rolled back to: {result['restored_from']}")
        sys.exit(0)

    result = autotune(
        iterations=args.iterations,
        config_path=args.config,
        cases_path=args.cases,
        seed=args.seed,
        verbose=not args.json,
    )

    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
