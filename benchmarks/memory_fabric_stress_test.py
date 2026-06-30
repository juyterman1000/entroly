"""Memory Fabric Stress Test.

Run:

    python benchmarks/memory_fabric_stress_test.py
    python benchmarks/memory_fabric_stress_test.py --json

This benchmark is deterministic and offline. It checks that the public
orchestration layer behaves like one coherent memory system even when optional
engines are absent.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from entroly.memory_fabric import MemoryFabric


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

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "score": self.score,
            "scenarios": [asdict(s) for s in self.scenarios],
        }


def scenario_public_orchestrator_recall() -> ScenarioResult:
    fabric = MemoryFabric(enable_long_term=False, enable_native=False)
    target = fabric.remember(
        "Auth timeout was fixed in auth/session.py by increasing refresh slack.",
        agent_id="coder",
        importance=0.95,
        source="incident/auth-timeout",
        tags=["auth", "timeout"],
    )
    fabric.remember("Marketing copy changed.", agent_id="coder", importance=0.1, source="marketing")
    result = fabric.recall("login auth timeout refresh", agent_id="coder", budget=1200)
    selected = [m.id for m in result.context.selected]
    passed = target in selected and result.context.used_tokens <= result.context.budget
    return ScenarioResult(
        "public_orchestrator_recall",
        passed,
        1.0 if passed else 0.0,
        {"target": target, "selected": selected, "used_tokens": result.context.used_tokens},
    )


def scenario_capability_contract() -> ScenarioResult:
    fabric = MemoryFabric(enable_long_term=False, enable_native=False)
    layers = {layer.name: layer.status for layer in fabric.capabilities()}
    required = {
        "memory_os",
        "hippocampus_bridge",
        "rust_memory_manager",
        "schipc",
        "compliance_gate",
        "pollination",
        "federation",
        "receipts_witness",
    }
    passed = required.issubset(layers) and layers["memory_os"] == "active"
    return ScenarioResult(
        "capability_contract",
        passed,
        1.0 if passed else 0.0,
        {"layers": layers, "missing": sorted(required - set(layers))},
    )


def scenario_safety_contract() -> ScenarioResult:
    fabric = MemoryFabric(enable_long_term=False, enable_native=False)
    blocked = False
    try:
        fabric.remember("Bad memory sk-abcdefghijklmnopqrstuvwxyz123456", agent_id="coder")
    except ValueError:
        blocked = True
    passed = blocked
    return ScenarioResult("safety_contract", passed, 1.0 if passed else 0.0, {"blocked": blocked})


def scenario_receipt_contract() -> ScenarioResult:
    fabric = MemoryFabric(enable_long_term=False, enable_native=False)
    fabric.remember("Receipt evidence for auth timeout", agent_id="coder", importance=0.8)
    result = fabric.recall("auth timeout", agent_id="coder", budget=100)
    receipt = result.receipt()
    passed = (
        "memory_os" in receipt
        and "layers" in receipt
        and receipt["memory_os"]["used_tokens"] <= receipt["memory_os"]["budget"]
        and any(layer["name"] == "receipts_witness" for layer in receipt["layers"])
    )
    return ScenarioResult(
        "receipt_contract",
        passed,
        1.0 if passed else 0.0,
        {"receipt_keys": sorted(receipt.keys()), "layer_count": len(receipt.get("layers", []))},
    )


def scenario_persistence_contract(tmp_path=None) -> ScenarioResult:
    import tempfile
    from pathlib import Path

    fabric = MemoryFabric(enable_long_term=False, enable_native=False)
    fabric.remember("Persistent fabric memory", agent_id="coder", importance=0.8)
    if tmp_path is None:
        td = tempfile.TemporaryDirectory()
        path = Path(td.name) / "fabric.json"
    else:
        td = None
        path = Path(tmp_path) / "fabric.json"
    try:
        fabric.save(path)
        restored = MemoryFabric.load(path, enable_long_term=False, enable_native=False)
        result = restored.recall("persistent fabric", agent_id="coder", budget=100)
        passed = bool(result.context.selected)
        details = {"path_exists": path.exists(), "selected_count": len(result.context.selected)}
    finally:
        if td is not None:
            td.cleanup()
    return ScenarioResult("persistence_contract", passed, 1.0 if passed else 0.0, details)


def run_benchmark() -> BenchmarkReport:
    scenarios = [
        scenario_public_orchestrator_recall(),
        scenario_capability_contract(),
        scenario_safety_contract(),
        scenario_receipt_contract(),
        scenario_persistence_contract(),
    ]
    score = sum(s.score for s in scenarios) / max(1, len(scenarios))
    passed = score == 1.0 and all(s.passed for s in scenarios)
    return BenchmarkReport(
        name="Memory Fabric Stress Test",
        passed=passed,
        score=round(score, 4),
        scenarios=scenarios,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Entroly Memory Fabric stress benchmark")
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
