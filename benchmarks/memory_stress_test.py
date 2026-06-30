"""Agent Memory Stress Test for Entroly Memory OS.

Run:

    python benchmarks/memory_stress_test.py
    python benchmarks/memory_stress_test.py --json

This benchmark is intentionally deterministic and offline. It does not claim to
measure LLM answer quality. It measures whether the local memory-control layer
keeps high-value evidence, suppresses stale/unsafe/noisy memory, respects token
budgets, and survives persistence.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from entroly.memory import MemoryOS


@dataclass(slots=True)
class ScenarioResult:
    name: str
    passed: bool
    score: float
    details: dict[str, object]


@dataclass(slots=True)
class BenchmarkReport:
    name: str
    passed: bool
    score: float
    scenarios: list[ScenarioResult]
    thresholds: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "score": self.score,
            "thresholds": self.thresholds,
            "scenarios": [asdict(s) for s in self.scenarios],
        }


def scenario_recall_precision() -> ScenarioResult:
    mem = MemoryOS(default_budget=80)
    target = mem.remember(
        "Auth timeout regression was fixed in auth/session.py by increasing refresh slack.",
        agent_id="coder",
        importance=0.95,
        source="incident/auth-timeout",
        tags=["auth", "timeout", "critical"],
    )
    mem.remember("Marketing headline changed on homepage.", agent_id="coder", importance=0.2, source="docs/marketing")
    mem.remember("Billing export CSV delimiter changed.", agent_id="coder", importance=0.3, source="billing/export")

    ctx = mem.recall("login auth timeout regression", agent_id="coder", budget=80)
    top = ctx.selected[0].id if ctx.selected else None
    passed = top == target and ctx.used_tokens <= ctx.budget
    return ScenarioResult(
        "recall_precision",
        passed,
        1.0 if passed else 0.0,
        {"top": top, "target": target, "selected": [m.id for m in ctx.selected], "used_tokens": ctx.used_tokens},
    )


def scenario_stale_suppression() -> ScenarioResult:
    mem = MemoryOS(death_threshold=0.2)
    stale = mem.remember("Temporary debugging guess: auth timeout is probably DNS.", agent_id="coder", importance=0.1)
    stable = mem.remember("Stable invariant: auth/session.py owns token refresh slack.", agent_id="coder", importance=0.2, tier="semantic")
    mem.tick(20)
    forgotten = mem.forget()
    ctx = mem.recall("auth timeout token refresh slack", agent_id="coder", budget=120)
    selected = {m.id for m in ctx.selected}
    passed = stale not in selected and stable in selected and forgotten >= 1
    return ScenarioResult(
        "stale_suppression",
        passed,
        1.0 if passed else 0.0,
        {"forgotten": forgotten, "stale_selected": stale in selected, "semantic_selected": stable in selected},
    )


def scenario_secret_blocking() -> ScenarioResult:
    mem = MemoryOS()
    blocked = False
    try:
        mem.remember("Do not store this key sk-abcdefghijklmnopqrstuvwxyz123456", agent_id="coder")
    except ValueError:
        blocked = True
    redacting = MemoryOS(safety_policy="redact")
    redacting.remember("Contact dev@example.com about auth", agent_id="coder")
    ctx = redacting.recall("contact auth", agent_id="coder")
    redacted = bool(ctx.selected and "EMAIL_REDACTED" in ctx.selected[0].content)
    passed = blocked and redacted
    return ScenarioResult(
        "safety_block_redact",
        passed,
        1.0 if passed else 0.0,
        {"secret_blocked": blocked, "email_redacted": redacted},
    )


def scenario_budget_omission() -> ScenarioResult:
    mem = MemoryOS(default_budget=3)
    mem.remember("auth", agent_id="coder", importance=0.9)
    mem.remember("auth timeout regression investigation details are long and should not fit", agent_id="coder", importance=0.8)
    ctx = mem.recall("auth timeout", agent_id="coder", budget=3)
    passed = ctx.used_tokens <= 3 and any(m.reason == "over_budget" for m in ctx.omitted)
    return ScenarioResult(
        "budget_omission",
        passed,
        1.0 if passed else 0.0,
        {"used_tokens": ctx.used_tokens, "budget": ctx.budget, "omitted_reasons": [m.reason for m in ctx.omitted]},
    )


def scenario_persistence() -> ScenarioResult:
    mem = MemoryOS(default_budget=99, max_entries=10, max_tokens=1000)
    mem.remember("Persisted auth lesson", agent_id="coder", importance=0.8, source="incident/auth")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "memory.json"
        mem.save(path)
        restored = MemoryOS.load(path)
        ctx = restored.recall("persisted auth", agent_id="coder", budget=99)
    passed = bool(ctx.selected and ctx.selected[0].source == "incident/auth" and restored.stats()["default_budget"] == 99)
    return ScenarioResult(
        "persistence_roundtrip",
        passed,
        1.0 if passed else 0.0,
        {"selected": [m.source for m in ctx.selected], "default_budget": restored.stats()["default_budget"]},
    )


def scenario_capacity_eviction() -> ScenarioResult:
    mem = MemoryOS(max_entries=2, max_tokens=10_000)
    weak = mem.remember("weak scratch note", agent_id="coder", importance=0.1)
    semantic = mem.remember("semantic memory survives capacity pressure", agent_id="coder", importance=0.1, tier="semantic")
    strong = mem.remember("critical auth memory survives capacity pressure", agent_id="coder", importance=0.95)
    ctx = mem.recall("semantic critical weak auth", agent_id="coder", budget=120)
    selected = {m.id for m in ctx.selected}
    passed = mem.stats()["total_entries"] == 2 and weak not in selected and semantic in selected and strong in selected
    return ScenarioResult(
        "capacity_eviction",
        passed,
        1.0 if passed else 0.0,
        {"weak_selected": weak in selected, "semantic_selected": semantic in selected, "strong_selected": strong in selected, "stats": mem.stats()},
    )


def run_benchmark() -> BenchmarkReport:
    scenarios = [
        scenario_recall_precision(),
        scenario_stale_suppression(),
        scenario_secret_blocking(),
        scenario_budget_omission(),
        scenario_persistence(),
        scenario_capacity_eviction(),
    ]
    score = sum(s.score for s in scenarios) / max(1, len(scenarios))
    thresholds = {"pass_score": 1.0}
    return BenchmarkReport(
        name="Agent Memory Stress Test",
        passed=score >= thresholds["pass_score"] and all(s.passed for s in scenarios),
        score=round(score, 4),
        scenarios=scenarios,
        thresholds=thresholds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Entroly MemoryOS stress benchmark")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args()
    report = run_benchmark()
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(f"{report.name}: {'PASS' if report.passed else 'FAIL'} score={report.score:.2f}")
        for s in report.scenarios:
            print(f"  {'PASS' if s.passed else 'FAIL'} {s.name}: score={s.score:.2f}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
