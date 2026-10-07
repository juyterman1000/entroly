#!/usr/bin/env python3
"""Execute Context Arena Phase 2 adapter preparation with fail-closed artifacts.

This is the adapter/preflight layer for the frozen Context Arena protocol.
Model execution and test-oracle scoring are separate stages; this command never
turns an unavailable competitor into a comparative loss.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase2 = _load("context_arena_phase2", HERE / "phase2.py")
base = _load("context_arena_adapter_base", HERE / "adapters" / "base.py")


def _tasks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: task must be an object")
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error("--budget must be positive")

    adapters = [base.NoContextAdapter()]
    task_rows = _tasks(args.tasks)
    results = [
        result
        for task in task_rows
        for result in phase2.prepare_all(adapters, task, args.budget)
    ]
    by_task: dict[str, list[Any]] = {}
    for result in results:
        by_task.setdefault(result.task_id, []).append(result)

    artifact = {
        "schema": "entroly.context-arena.phase2.preflight.v1",
        "protocol": "Context Arena v1",
        "token_budget": args.budget,
        "tasks": len(task_rows),
        "adapters": [adapter.name for adapter in adapters],
        "claimable": all(
            phase2.comparison_is_claimable(rows) for rows in by_task.values()
        ) if by_task else False,
        "results": [result.to_dict() for result in results],
        "limitations": [
            "Preflight only: no model was called and no task-success claim is produced.",
            "NO-CONTEXT is the only built-in adapter until external adapters are explicitly configured.",
            "Provider-observed token usage is required for economic claims.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: artifact[k] for k in ("schema", "tasks", "adapters", "claimable")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
