"""End-to-end MemoryOS demo.

Run:

    python examples/memory_os_e2e_demo.py
    python examples/memory_os_e2e_demo.py --json

The demo is deterministic and offline. It shows the product path users should
understand immediately:

    remember evidence -> block unsafe memory -> recall under budget -> receipt -> persist
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from entroly import MemoryOS


def run_demo() -> dict[str, object]:
    mem = MemoryOS(default_budget=1200, max_entries=100, max_tokens=20_000, safety_policy="block")

    auth_id = mem.remember(
        "Login timeout was fixed in auth/session.py by increasing refresh slack before token renewal.",
        agent_id="coder",
        importance=0.95,
        tier="working",
        source="incident/auth-timeout",
        tags=["auth", "timeout", "critical"],
    )
    mem.remember(
        "The billing CSV delimiter changed from comma to pipe.",
        agent_id="coder",
        importance=0.3,
        source="billing/export",
        tags=["billing"],
    )
    semantic_id = mem.remember(
        "Semantic invariant: auth/session.py owns token refresh slack and retry timing.",
        agent_id="coder",
        importance=0.7,
        tier="semantic",
        source="architecture/auth",
        tags=["auth", "semantic"],
    )

    unsafe_blocked = False
    try:
        mem.remember("Never store this API key sk-abcdefghijklmnopqrstuvwxyz123456", agent_id="coder")
    except ValueError:
        unsafe_blocked = True

    ctx = mem.recall("why is login timeout happening again", agent_id="coder", budget=1200)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "memory.json"
        mem.save(path)
        restored = MemoryOS.load(path)
        restored_ctx = restored.recall("auth token refresh slack", agent_id="coder", budget=1200)

    receipt = ctx.receipt()
    return {
        "scenario": "auth_timeout_recurrence",
        "unsafe_blocked": unsafe_blocked,
        "expected_memories": [auth_id, semantic_id],
        "selected_ids": [m.id for m in ctx.selected],
        "omitted": [m.reason for m in ctx.omitted],
        "prompt_evidence": ctx.as_text(),
        "receipt": receipt,
        "persistence_roundtrip_selected": [m.id for m in restored_ctx.selected],
        "passed": unsafe_blocked
        and auth_id in {m.id for m in ctx.selected}
        and semantic_id in {m.id for m in ctx.selected}
        and bool(restored_ctx.selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Entroly MemoryOS end-to-end demo")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    result = run_demo()
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print("Entroly MemoryOS demo")
        print(f"  unsafe memory blocked: {result['unsafe_blocked']}")
        print(f"  selected ids: {', '.join(result['selected_ids'])}")
        print(f"  used tokens: {result['receipt']['used_tokens']}/{result['receipt']['budget']}")
        print("\nPrompt-ready evidence:\n")
        print(result["prompt_evidence"])
        print(f"\nDemo: {'PASS' if result['passed'] else 'FAIL'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
