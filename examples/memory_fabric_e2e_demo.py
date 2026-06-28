"""End-to-end demo for Entroly Memory Fabric.

Run:

    python examples/memory_fabric_e2e_demo.py
    python examples/memory_fabric_e2e_demo.py --json

This demo is deterministic and offline. It shows the product-level path:

    MemoryFabric -> MemoryOS -> safety -> recall -> receipt -> capability map -> persistence

Optional engines such as Hippocampus or native Rust MemoryManager are detected
and reported, but the demo passes without them.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from entroly.memory_fabric import MemoryFabric


def run_demo() -> dict[str, object]:
    fabric = MemoryFabric(
        default_budget=1200,
        max_entries=100,
        max_tokens=20_000,
        safety_policy="block",
        enable_long_term=True,
        enable_native=True,
    )

    auth_id = fabric.remember(
        "Auth timeout recurrence was fixed in auth/session.py by increasing refresh slack.",
        agent_id="coder",
        importance=0.95,
        tier="working",
        source="incident/auth-timeout",
        tags=["auth", "timeout", "critical"],
    )
    semantic_id = fabric.remember(
        "Semantic invariant: auth/session.py owns token refresh slack and retry timing.",
        agent_id="coder",
        importance=0.8,
        tier="semantic",
        source="architecture/auth",
        tags=["auth", "semantic"],
    )
    fabric.remember(
        "Billing export delimiter changed from comma to pipe.",
        agent_id="coder",
        importance=0.2,
        source="billing/export",
        tags=["billing"],
    )

    unsafe_blocked = False
    try:
        fabric.remember("Never store this key sk-abcdefghijklmnopqrstuvwxyz123456", agent_id="coder")
    except ValueError:
        unsafe_blocked = True

    recall = fabric.recall("why is login timeout happening again", agent_id="coder", budget=1200)
    receipt = recall.receipt()
    selected_ids = [m.id for m in recall.context.selected]
    layer_status = {layer.name: layer.status for layer in fabric.capabilities()}

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "fabric.json"
        fabric.save(path)
        restored = MemoryFabric.load(path, enable_long_term=False, enable_native=False)
        restored_recall = restored.recall("auth refresh slack", agent_id="coder", budget=1200)

    passed = (
        unsafe_blocked
        and auth_id in selected_ids
        and semantic_id in selected_ids
        and recall.context.used_tokens <= recall.context.budget
        and "memory_os" in layer_status
        and "schipc" in layer_status
        and "pollination" in layer_status
        and "receipts_witness" in layer_status
        and bool(restored_recall.context.selected)
    )

    return {
        "scenario": "memory_fabric_auth_recurrence",
        "passed": passed,
        "unsafe_blocked": unsafe_blocked,
        "expected_memory_ids": [auth_id, semantic_id],
        "selected_ids": selected_ids,
        "used_tokens": recall.context.used_tokens,
        "budget": recall.context.budget,
        "prompt_evidence": recall.as_text(),
        "receipt": receipt,
        "layer_status": layer_status,
        "restored_selected_ids": [m.id for m in restored_recall.context.selected],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Entroly Memory Fabric e2e demo")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    result = run_demo()
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print("Entroly Memory Fabric demo")
        print(f"  passed: {result['passed']}")
        print(f"  unsafe memory blocked: {result['unsafe_blocked']}")
        print(f"  selected ids: {', '.join(result['selected_ids'])}")
        print(f"  used tokens: {result['used_tokens']}/{result['budget']}")
        print("  layers:")
        for name, status in result["layer_status"].items():
            print(f"    - {name}: {status}")
        print("\nPrompt-ready evidence:\n")
        print(result["prompt_evidence"])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
