"""Determinism-tax harness on ContextBench (see CONTEXTBENCH_DETERMINISM_TAX_PREREGISTRATION.md).

Measures file- and line-level retrieval quality of Entroly's deterministic
selector against baselines on human-annotated gold contexts. The metric core
(interval-overlap recall/precision/F1) is pure and unit-tested; the repo-checkout
run path is gated on the dataset + local git checkouts + (for neural arms)
compute budget.

Line-interval gold spans mean file+line metrics need no AST parser; block-level
(tree-sitter) is deferred per the preregistration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


# ── Metric core (pure, deterministic, unit-tested) ─────────────────────────────

# A retrieval result is a mapping {file_path -> set of covered 1-indexed lines}.
Spans = dict[str, set[int]]


def parse_gold(gold_context_json: str) -> Spans:
    """Parse ContextBench `gold_context` JSON into per-file covered line sets.

    Each entry is {"file", "start_line", "end_line", "content"}; the line range
    is inclusive. Malformed or non-positive ranges are skipped fail-closed.
    """
    spans: Spans = {}
    try:
        entries = json.loads(gold_context_json)
    except (ValueError, TypeError):
        return spans
    for e in entries or []:
        path = str(e.get("file") or "")
        try:
            start = int(e.get("start_line"))
            end = int(e.get("end_line"))
        except (TypeError, ValueError):
            continue
        if not path or start < 1 or end < start:
            continue
        spans.setdefault(path, set()).update(range(start, end + 1))
    return spans


def _overlap_lines(pred: Spans, gold: Spans) -> int:
    return sum(len(lines & gold.get(f, set())) for f, lines in pred.items())


def _total_lines(spans: Spans) -> int:
    return sum(len(lines) for lines in spans.values())


@dataclass
class Score:
    recall: float
    precision: float
    f1: float
    overlap: int
    pred_total: int
    gold_total: int


def _prf(overlap: int, pred_total: int, gold_total: int) -> Score:
    recall = overlap / gold_total if gold_total else 0.0
    precision = overlap / pred_total if pred_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return Score(recall, precision, f1, overlap, pred_total, gold_total)


def line_score(pred: Spans, gold: Spans) -> Score:
    """Line-level recall/precision/F1 by interval overlap."""
    return _prf(_overlap_lines(pred, gold), _total_lines(pred), _total_lines(gold))


def file_score(pred: Spans, gold: Spans) -> Score:
    """File-level recall/precision/F1 over the SET of files touched."""
    pf, gf = set(pred), set(gold)
    inter = len(pf & gf)
    return _prf(inter, len(pf), len(gf))


def evidence_drop(pred: Spans, gold: Spans) -> float:
    """Fraction of predicted lines that overlap no gold span (retrieved-but-useless)."""
    total = _total_lines(pred)
    if not total:
        return 0.0
    useful = _overlap_lines(pred, gold)
    return (total - useful) / total


def macro_average(scores: list[Score]) -> dict[str, float]:
    """Macro-average over tasks so large repos do not dominate."""
    n = len(scores) or 1
    return {
        "recall": sum(s.recall for s in scores) / n,
        "precision": sum(s.precision for s in scores) / n,
        "f1": sum(s.f1 for s in scores) / n,
        "tasks": len(scores),
    }


def composite_q(f1_line: float, pass_at_1: float, efficiency: float, ev_drop: float,
                *, alpha: float = 1.0, beta: float = 0.5, lam: float = 0.5) -> float:
    """Preregistered SECONDARY composite (not the decision gate)."""
    return f1_line + alpha * pass_at_1 + beta * efficiency - lam * ev_drop


def determinism_tax(f1_best_nondeterministic: float, f1_deterministic: float) -> float:
    """Primary decision metric, in percentage points."""
    return 100.0 * (f1_best_nondeterministic - f1_deterministic)


DECISION_TABLE = (
    (3.0, "Strong general-purpose thesis"),
    (7.0, "Strong high-trust product"),
    (15.0, "Compliance / security niche"),
    (float("inf"), "Reject broad RCP positioning (unless hybridization closes the gap)"),
)


def decide(tax_pp: float) -> str:
    for threshold, verdict in DECISION_TABLE:
        if tax_pp <= threshold:
            return verdict
    return DECISION_TABLE[-1][1]


# ── Run scaffold (gated on dataset + checkouts) ────────────────────────────────

@dataclass
class Task:
    instance_id: str
    repo: str
    repo_url: str
    base_commit: str
    language: str
    problem_statement: str
    gold: Spans = field(default_factory=dict)


def load_tasks(limit: int | None = 25, language: str | None = "python") -> list[Task]:
    """Load ContextBench tasks (streaming). Requires network to HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset("Contextbench/ContextBench", "default", split="train", streaming=True)
    tasks: list[Task] = []
    for row in ds:
        if language and str(row.get("language")) != language:
            continue
        tasks.append(
            Task(
                instance_id=str(row["instance_id"]),
                repo=str(row["repo"]),
                repo_url=str(row["repo_url"]),
                base_commit=str(row["base_commit"]),
                language=str(row["language"]),
                problem_statement=str(row["problem_statement"]),
                gold=parse_gold(str(row["gold_context"])),
            )
        )
        if limit and len(tasks) >= limit:
            break
    tasks.sort(key=lambda t: t.instance_id)  # deterministic order, never by result
    return tasks


def entroly_select(repo_dir: str, query: str, budget: int) -> Spans:
    """Ingest a checked-out repo, run the deterministic selector, map to line spans.

    Gated on a local checkout of repo@base_commit. Selected content is located
    back to source line ranges by exact-substring match; unmatched content is
    NOT credited (fail-closed — it inflates evidence-drop instead).
    """
    raise NotImplementedError(
        "repo-checkout run path is gated on dataset + git checkout budget; "
        "the metric core above is what is validated pre-run"
    )


if __name__ == "__main__":  # pragma: no cover - thin CLI
    import sys

    if "--peek" in sys.argv:
        tasks = load_tasks(limit=3)
        for t in tasks:
            gold_lines = _total_lines(t.gold)
            print(f"{t.instance_id}  {t.repo}@{t.base_commit[:8]}  "
                  f"gold: {len(t.gold)} files / {gold_lines} lines")
    else:
        print("see CONTEXTBENCH_DETERMINISM_TAX_PREREGISTRATION.md; "
              "run with --peek to sample tasks")
