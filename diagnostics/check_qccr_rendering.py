from __future__ import annotations

import ast
import json

from benchmarks.answer_correctness_bridge import _context_for, build_probes
from benchmarks.graph_lane_quality import (
    REPO,
    Task,
    _pool_for,
    _read,
    _tokens,
    _tracked_python_files,
)

LIMIT = 5
BUDGET = 2000
SEED = 20260807


def signature_lines(callee: str, rel: str) -> tuple[str, ...]:
    text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    tree = ast.parse(text)
    target = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == callee:
            target = node
            break
        if isinstance(node, ast.ClassDef) and node.name == callee:
            target = next(
                (
                    sub
                    for sub in node.body
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and sub.name == "__init__"
                ),
                None,
            )
            break
    if target is None:
        raise RuntimeError((callee, rel))
    body_line = target.body[0].lineno if target.body else target.end_lineno
    return tuple(
        line.strip()
        for line in lines[target.lineno - 1 : max(target.lineno, body_line - 1)]
        if line.strip()
    )


def main() -> None:
    payload = json.loads(
        (REPO / "benchmarks/results/graph_lane_tasks.json").read_text(encoding="utf-8")
    )
    probes = build_probes([Task(**row) for row in payload["tasks"]], LIMIT)
    corpus = sorted(
        {
            str(path.relative_to(REPO)).replace("\\", "/")
            for path in _tracked_python_files()
        }
    )
    for idx, probe in enumerate(probes, 1):
        pool = _pool_for(probe.task, corpus, 48, SEED)
        texts = {rel: _read(rel) for rel in pool}
        context = _context_for("qccr", probe, pool, texts, BUDGET)
        sig = signature_lines(probe.task.callee, probe.task.callee_file)
        print(
            "POSTFIX_PROBE",
            idx,
            f"callee={probe.task.callee!r}",
            f"file_delivered={f'### {probe.task.callee_file}' in context}",
            f"full_signature_present={all(line in context for line in sig)}",
            f"gold_presence={tuple((param, param in context) for param in probe.params)!r}",
            f"context_tokens={_tokens(context)}",
        )


if __name__ == "__main__":
    main()
