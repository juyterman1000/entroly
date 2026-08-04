#!/usr/bin/env python3
"""
End-to-End Coding Agent Benchmark Harness
=========================================

Intended to measure true end-to-end outcomes: verified tasks completed per
dollar, across compressed and uncompressed arms.

STATUS: the real execution path is NOT wired. This harness does not call a
model, does not route through the Entroly proxy, and does not run each task's
test oracle. Running it without --simulate raises NotImplementedError rather
than emitting invented numbers. Building the real path is the work gated by
benchmarks/AGENTIC_TASKS_PREREGISTRATION.md.

Modes:
  raw: No compression, full context.
  entroly: Full closed-loop Entroly (compression, routing, recovery, PRISM).
  entroly_compress_only: Entroly compression only (no recovery loop).

Usage:
  python benchmarks/coding_tasks.py --dry-run
  python benchmarks/coding_tasks.py --simulate   # fabricated, quarantined
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("coding_benchmark")

# Cost constants (example for gpt-4o)
COST_PER_1K_IN = 0.005
COST_PER_1K_OUT = 0.015

# benchmarks/results/ holds measured, provenance-gated artifacts that the README
# cites. Simulated output is quarantined so the two can never be confused.
_REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_RESULTS_DIR = _REPO_ROOT / "benchmarks" / "results"
SIMULATED_RESULTS_DIR = _REPO_ROOT / "benchmarks" / "simulated"

@dataclass
class CodingTask:
    id: str
    description: str
    target_repo_path: str  # Path relative to test workspace
    setup_command: str
    test_command: str
    expected_exit_code: int = 0
    # Files to provide in context (for raw mode)
    context_files: list[str] = None

# A minimal set of dummy tasks for local validation without cloning huge repos.
# In a real environment, these would map to SWE-bench Lite or similar.
TASKS = [
    CodingTask(
        id="python_syntax_fix",
        description="Fix the syntax error in server.py.",
        target_repo_path="test_repos/simple_py",
        setup_command="echo 'def hello() print(\"world\")' > server.py",
        test_command="python -m py_compile server.py",
        context_files=["server.py"],
    ),
    CodingTask(
        id="rust_test_fix",
        description="Make the failing test pass in lib.rs by returning 42 instead of 0.",
        target_repo_path="test_repos/simple_rs",
        setup_command="mkdir -p src && echo '#[test] fn test_val() { assert_eq!(get_val(), 42); } fn get_val() -> i32 { 0 }' > src/lib.rs",
        test_command="cargo test",
        context_files=["src/lib.rs"],
    ),
]

class BenchmarkHarness:
    def __init__(self, workspace_root: Path, model: str, simulate: bool = False):
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.simulate = simulate
        self.results: list[dict[str, Any]] = []

    def _setup_task(self, task: CodingTask) -> Path:
        """Create an isolated workspace for the task."""
        task_dir = self.workspace_root / f"{task.id}_{int(time.time())}"
        if task_dir.exists():
            shutil.rmtree(task_dir)
        task_dir.mkdir(parents=True)

        # Run setup
        try:
            subprocess.run(
                task.setup_command,
                shell=True,
                cwd=task_dir,
                check=True,
                capture_output=True,
                timeout=10,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Task {task.id} setup failed: {e.stderr.decode()}")
            raise

        return task_dir

    def _run_agent(self, task_dir: Path, task: CodingTask, mode: str) -> dict[str, Any]:
        """Execute one task in one arm.

        The real execution path is not wired yet. Rather than silently
        substituting invented numbers, this fails closed: a caller who has not
        explicitly opted into simulation gets an error naming the missing work,
        not a plausible-looking result.
        """
        if not self.simulate:
            raise NotImplementedError(
                "coding_tasks.py has no real execution path yet: it does not "
                "call a model, does not route through the Entroly proxy, and "
                "does not run the task's test oracle. Wiring those is the work "
                "gated by benchmarks/AGENTIC_TASKS_PREREGISTRATION.md.\n\n"
                "Pass --simulate to exercise the harness plumbing with invented "
                "numbers. Simulated output is quarantined outside "
                f"{REAL_RESULTS_DIR}/ and is not evidence of anything."
            )
        return self._simulate_agent(task_dir, task, mode)

    def _simulate_agent(self, task_dir: Path, task: CodingTask, mode: str) -> dict[str, Any]:
        """Invent metrics to exercise the harness plumbing.

        Every number below is fabricated. The arm ordering is hardcoded so that
        Entroly wins; it reflects an assumption, not a measurement. Nothing
        produced here may be cited, published, or compared against a real run.
        """
        logger.warning(
            "SIMULATED run of task %s (mode=%s): metrics below are invented, "
            "not measured", task.id, mode,
        )
        t0 = time.time()

        # Fabricated. Encodes the hoped-for outcome, proves nothing about it.
        input_tokens = 10000 if mode == "raw" else 3000
        output_tokens = 500
        repair_count = 0 if mode == "raw" else (1 if mode == "entroly" else 0)

        import random
        if mode == "raw":
            passed = random.random() > 0.3
        elif mode == "entroly_compress_only":
            passed = random.random() > 0.4  # assumed degradation from compression
        elif mode == "entroly":
            passed = random.random() > 0.1  # assumed recovery-loop repair
        else:
            passed = False

        latency = time.time() - t0 + (random.random() * 5)

        cost = (input_tokens / 1000 * COST_PER_1K_IN) + (output_tokens / 1000 * COST_PER_1K_OUT)

        return {
            "simulated": True,
            "passed": passed,
            "latency_s": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "repair_count": repair_count,
            "dollars": cost,
        }

    def _guard_output_dir(self, output_dir: Path) -> Path:
        """Keep fabricated output out of the directory that holds real evidence.

        benchmarks/results/ is cited by the README and gated by
        scripts/verify_readme_claims.py. A simulated artifact landing there
        would be indistinguishable at a glance from a measured one, so a
        simulated run is redirected rather than trusted to be labelled.
        """
        if not self.simulate:
            return output_dir
        try:
            resolved = output_dir.resolve()
            if resolved == REAL_RESULTS_DIR.resolve() or REAL_RESULTS_DIR.resolve() in resolved.parents:
                logger.warning(
                    "Refusing to write simulated output into %s; redirecting to %s",
                    output_dir, SIMULATED_RESULTS_DIR,
                )
                return SIMULATED_RESULTS_DIR
        except OSError:
            return SIMULATED_RESULTS_DIR
        return output_dir

    def run(self, modes: list[str], runs_per_task: int = 1, dry_run: bool = False):
        logger.info(f"Starting benchmark for {len(TASKS)} tasks in modes {modes} ({runs_per_task} runs/task)")

        for task in TASKS:
            if dry_run:
                logger.info(f"Dry run: {task.id} (no execution)")
                continue

            for mode in modes:
                for run_idx in range(runs_per_task):
                    task_dir = self._setup_task(task)

                    try:
                        metrics = self._run_agent(task_dir, task, mode)

                        self.results.append({
                            "task_id": task.id,
                            "mode": mode,
                            "run": run_idx,
                            "model": self.model,
                            **metrics
                        })
                    finally:
                        # Cleanup
                        shutil.rmtree(task_dir, ignore_errors=True)

        return self.results

    def generate_report(self, output_dir: Path):
        output_dir = self._guard_output_dir(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        prefix = "SIMULATED_NOT_EVIDENCE_" if self.simulate else ""
        out_file = output_dir / f"{prefix}coding_tasks_{timestamp}.json"

        # Aggregate
        summary = {}
        for r in self.results:
            mode = r["mode"]
            if mode not in summary:
                summary[mode] = {
                    "total_tasks": 0,
                    "passed_tasks": 0,
                    "total_dollars": 0.0,
                    "total_latency_s": 0.0,
                }

            summary[mode]["total_tasks"] += 1
            if r["passed"]:
                summary[mode]["passed_tasks"] += 1
            summary[mode]["total_dollars"] += r["dollars"]
            summary[mode]["total_latency_s"] += r["latency_s"]

        # Compute headline metric
        for mode, stats in summary.items():
            if stats["total_dollars"] > 0:
                stats["verified_tasks_per_dollar"] = stats["passed_tasks"] / stats["total_dollars"]
            else:
                stats["verified_tasks_per_dollar"] = 0.0

        report = {
            "metadata": {
                "timestamp": timestamp,
                "model": self.model,
                "simulated": self.simulate,
                "valid_as_evidence": not self.simulate,
            },
            "summary": summary,
            "raw_traces": self.results,
        }
        if self.simulate:
            report["limitations"] = [
                "Every number in this file is fabricated by "
                "BenchmarkHarness._simulate_agent. No model was called.",
                "No task test oracle was executed; 'passed' is a random draw.",
                "Token counts are hardcoded constants (10000 raw / 3000 "
                "compressed), not measured, so any implied saving is an "
                "assumption restated as a result.",
                "Arm ordering is hardcoded to favour Entroly and therefore "
                "cannot evidence that Entroly is favourable.",
                "This file must never be cited, published, or compared "
                "against a measured run.",
            ]

        out_file.write_text(json.dumps(report, indent=2))
        logger.info(f"Report written to {out_file}")

        # Print summary
        print("\n=== Benchmark Summary ===")
        if self.simulate:
            print("*** SIMULATED - ALL NUMBERS BELOW ARE FABRICATED ***")
            print("*** No model was called. No test oracle ran. Not evidence. ***\n")
        for mode, stats in summary.items():
            print(f"Mode: {mode}")
            print(f"  Pass Rate: {stats['passed_tasks']}/{stats['total_tasks']} ({(stats['passed_tasks']/max(1,stats['total_tasks']))*100:.1f}%)")
            print(f"  Cost: ${stats['total_dollars']:.4f}")
            print(f"  Headline (Tasks/$): {stats['verified_tasks_per_dollar']:.2f}")
            print()
        if self.simulate:
            print("*** SIMULATED - see 'limitations' in the JSON. Not evidence. ***")

def main():
    parser = argparse.ArgumentParser(description="End-to-End Coding Agent Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Validate tasks without running agents")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to benchmark")
    parser.add_argument("--runs", type=int, default=1, help="Runs per task per mode")
    parser.add_argument("--workspace", type=str, default="benchmarks/workspace", help="Temp workspace dir")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Exercise the harness plumbing with FABRICATED metrics. No model is "
            "called and no test oracle runs. Output is quarantined under "
            "benchmarks/simulated/ and is not evidence."
        ),
    )

    args = parser.parse_args()

    harness = BenchmarkHarness(Path(args.workspace), args.model, simulate=args.simulate)
    modes = ["raw", "entroly_compress_only", "entroly"]

    harness.run(modes=modes, runs_per_task=args.runs, dry_run=args.dry_run)

    if not args.dry_run:
        harness.generate_report(
            SIMULATED_RESULTS_DIR if args.simulate else REAL_RESULTS_DIR
        )

if __name__ == "__main__":
    main()
