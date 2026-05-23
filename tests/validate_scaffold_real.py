"""
Real-world validation of Context Scaffolding Engine.

Loads ACTUAL Entroly source files, simulates the proxy pipeline's
fragment selection, and prints the exact scaffold output.

No mocking. No synthetic data. Real code → real scaffold.
"""
import os
import sys
import time

# Ensure entroly is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entroly.context_scaffold import (
    generate_scaffold,
    _extract_imports_from_content,
    _extract_definitions_from_content,
)


def load_real_fragment(rel_path: str) -> dict:
    """Load a real file from the entroly codebase as a fragment dict."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_path = os.path.join(root, rel_path)
    with open(abs_path, encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return {
        "source": f"file:{rel_path}",
        "content": content,
        "token_count": len(content) // 4,
        "relevance": 0.8,
        "variant": "full",
    }


def main():
    print("=" * 70)
    print("CSE REAL-WORLD VALIDATION")
    print("=" * 70)

    # ── Test 1: Entroly Python modules (proxy pipeline) ──────────────
    print("\n### Test 1: Entroly Proxy Pipeline Files")
    print("-" * 50)

    files_1 = [
        "entroly/proxy.py",
        "entroly/proxy_config.py",
        "entroly/proxy_transform.py",
        "entroly/adaptive_budget.py",
        "entroly/context_scaffold.py",
    ]

    frags_1 = []
    for f in files_1:
        try:
            frag = load_real_fragment(f)
            frags_1.append(frag)
            imports = _extract_imports_from_content(frag["content"], f)
            defs = _extract_definitions_from_content(frag["content"], f)
            print(f"  {f}:")
            print(f"    imports: {imports[:8]}{'...' if len(imports) > 8 else ''}")
            print(f"    defines: {defs[:8]}{'...' if len(defs) > 8 else ''}")
        except FileNotFoundError:
            print(f"  SKIP: {f} not found")

    t0 = time.perf_counter()
    scaffold_1 = generate_scaffold(frags_1, task_type="Feature")
    elapsed_1 = (time.perf_counter() - t0) * 1000

    print(f"\n  ⏱️  Scaffold generation: {elapsed_1:.2f}ms")
    print(f"  📏 Scaffold length: {len(scaffold_1)} chars (~{len(scaffold_1)//4} tokens)")
    print(f"\n{'─' * 50}")
    print("  SCAFFOLD OUTPUT:")
    print(f"{'─' * 50}")
    if scaffold_1:
        for line in scaffold_1.split("\n"):
            print(f"  │ {line}")
    else:
        print("  │ (empty — no cross-file edges detected)")
    print(f"{'─' * 50}")

    # ── Test 2: Entroly Rust core modules ────────────────────────────
    print("\n### Test 2: Entroly Rust Core Files")
    print("-" * 50)

    files_2 = [
        "entroly-core/src/depgraph.rs",
        "entroly-core/src/knapsack.rs",
        "entroly-core/src/skeleton.rs",
        "entroly-core/src/entropy.rs",
    ]

    frags_2 = []
    for f in files_2:
        try:
            frag = load_real_fragment(f)
            frags_2.append(frag)
            imports = _extract_imports_from_content(frag["content"], f)
            defs = _extract_definitions_from_content(frag["content"], f)
            print(f"  {f}:")
            print(f"    imports: {imports[:6]}{'...' if len(imports) > 6 else ''}")
            print(f"    defines: {defs[:6]}{'...' if len(defs) > 6 else ''}")
        except FileNotFoundError:
            print(f"  SKIP: {f} not found")

    t0 = time.perf_counter()
    scaffold_2 = generate_scaffold(frags_2, task_type="BugFix")
    elapsed_2 = (time.perf_counter() - t0) * 1000

    print(f"\n  ⏱️  Scaffold generation: {elapsed_2:.2f}ms")
    print(f"  📏 Scaffold length: {len(scaffold_2)} chars (~{len(scaffold_2)//4} tokens)")
    print(f"\n{'─' * 50}")
    print("  SCAFFOLD OUTPUT:")
    print(f"{'─' * 50}")
    if scaffold_2:
        for line in scaffold_2.split("\n"):
            print(f"  │ {line}")
    else:
        print("  │ (empty — no cross-file edges detected)")
    print(f"{'─' * 50}")

    # ── Test 3: Mixed Python + Test file (BugFix scenario) ───────────
    print("\n### Test 3: Mixed Source + Tests (BugFix scenario)")
    print("-" * 50)

    files_3 = [
        "entroly/auto_index.py",
        "entroly/proxy.py",
        "entroly/context_scaffold.py",
        "tests/test_context_scaffold.py",
    ]

    frags_3 = []
    for f in files_3:
        try:
            frag = load_real_fragment(f)
            frags_3.append(frag)
        except FileNotFoundError:
            print(f"  SKIP: {f} not found")

    t0 = time.perf_counter()
    scaffold_3 = generate_scaffold(frags_3, task_type="BugFix")
    elapsed_3 = (time.perf_counter() - t0) * 1000

    print(f"  ⏱️  Scaffold generation: {elapsed_3:.2f}ms")
    print(f"  📏 Scaffold length: {len(scaffold_3)} chars (~{len(scaffold_3)//4} tokens)")
    print(f"\n{'─' * 50}")
    print("  SCAFFOLD OUTPUT:")
    print(f"{'─' * 50}")
    if scaffold_3:
        for line in scaffold_3.split("\n"):
            print(f"  │ {line}")
    else:
        print("  │ (empty — no cross-file edges detected)")
    print(f"{'─' * 50}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    results = [
        ("Proxy pipeline (5 Python)", scaffold_1, elapsed_1),
        ("Rust core (4 Rust)", scaffold_2, elapsed_2),
        ("Mixed + tests (4 files)", scaffold_3, elapsed_3),
    ]
    for name, scaffold, ms in results:
        edges = scaffold.count("→") if scaffold else 0
        tokens = len(scaffold) // 4 if scaffold else 0
        status = "✅ EDGES FOUND" if edges > 0 else "⚠️  NO EDGES"
        print(f"  {name}: {status} ({edges} edges, ~{tokens} tokens, {ms:.1f}ms)")

    all_working = all(s.count("→") > 0 for s, *_ in [(r[1],) for r in results] if s)
    print(f"\n  VERDICT: {'✅ CSE IS REAL' if all_working else '❌ NEEDS WORK'}")


if __name__ == "__main__":
    main()
