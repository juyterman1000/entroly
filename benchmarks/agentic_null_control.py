#!/usr/bin/env python3
"""Run the agentic pilot with no repository context.

This is a construct-validity control, not a competitor arm. It answers the
question that must be settled before RAW-versus-compressed task success means
anything: can the model solve the task from the query and visible test alone?

A task that passes here is not evidence about context selection. Exclude it
from context-quality comparisons or redesign it so answer-critical information
exists only in the repository context. The control uses the same model,
decoding parameters, prompt template, answer extraction, and pytest oracle as
``agentic_tasks_run.py``; the only difference is an empty context field.

Usage:
    python benchmarks/agentic_null_control.py --model qwen2.5-coder:1.5b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agentic_tasks_run import (  # noqa: E402
    DEFAULT_BASE_URL,
    PROMPT,
    Task,
    build_dependency_tasks,
    build_tasks,
    call_model,
    extract_source,
    run_oracle,
)

NULL_ARM = "null_context"


def build_null_prompt(task: Task) -> str:
    """Build the paired prompt with no repository context whatsoever."""
    return PROMPT.format(
        query=task.query,
        context="",
        target=task.target_file,
        test=task.test_source,
    )


def run_null_arm(
    task: Task,
    *,
    model: str,
    base_url: str,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    """Call the real model and score its answer with the existing oracle."""
    generation = call_model(
        base_url=base_url,
        model=model,
        prompt=build_null_prompt(task),
        seed=seed,
        timeout=timeout,
    )
    passed, detail = run_oracle(task, extract_source(generation["text"]))
    return {
        "task_id": task.task_id,
        "arm": NULL_ARM,
        "passed": passed,
        "input_tokens": generation["input_tokens"],
        "output_tokens": generation["output_tokens"],
        "latency_s": generation["latency_s"],
        "context": {
            "arm": NULL_ARM,
            "estimated_context_tokens": 0,
            "included_sources": [],
            "omitted_sources": [fragment.source for fragment in task.fragments()],
            "notes": ["no repository context supplied"],
        },
        "oracle_tail": detail.strip()[-200:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--distractors", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--task-set", choices=("dependency", "self-contained"), default="dependency"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results"
        / "agentic_tasks_null_control.json",
    )
    args = parser.parse_args()

    # Must validate the same set the comparison runs, or it certifies nothing.
    tasks = (
        build_dependency_tasks(args.distractors)
        if args.task_set == "dependency"
        else build_tasks(args.distractors)
    )
    rows: list[dict[str, Any]] = []
    for task in tasks:
        print(f"  {task.task_id:22} {NULL_ARM:22} ...", end="", flush=True)
        row = run_null_arm(
            task,
            model=args.model,
            base_url=args.base_url,
            seed=args.seed,
            timeout=args.timeout,
        )
        rows.append(row)
        print(
            f" {'PASS' if row['passed'] else 'FAIL'}"
            f"  in={row['input_tokens']} out={row['output_tokens']}"
            f" {row['latency_s']}s"
        )

    passed = sum(row["passed"] for row in rows)
    null_solvable = sorted(row["task_id"] for row in rows if row["passed"])
    artifact = {
        "metadata": {
            "model": args.model,
            "seed": args.seed,
            "distractors": args.distractors,
            "simulated": False,
            "arm": NULL_ARM,
            "token_source": "provider prompt_eval_count / eval_count",
            "oracle": "pytest exit code",
        },
        "summary": {
            "tasks": len(rows),
            "passed_without_repository_context": passed,
            "null_success_rate": round(passed / len(rows), 4) if rows else None,
            "context_diagnostic_tasks": len(rows) - passed,
        },
        "construct_validity": {
            "all_tasks_require_context": passed == 0,
            "null_solvable_tasks": null_solvable,
            "interpretation": (
                "Tasks passing the null arm cannot support claims that raw or "
                "compressed repository context preserved answer-critical evidence."
            ),
        },
        "rows": rows,
        "limitations": [
            "This control measures prompt leakage and task self-sufficiency; it is not a product benchmark.",
            "A null failure does not prove raw context is sufficient; RAW must still pass the paired task.",
            "Synthetic tasks remain unsuitable for a leadership claim even after filtering null-solvable tasks.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print("\n=== construct-validity control ===")
    print(f"passed without context : {passed}/{len(rows)}")
    print(f"diagnostic tasks       : {len(rows) - passed}/{len(rows)}")
    print(f"artifact               : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
