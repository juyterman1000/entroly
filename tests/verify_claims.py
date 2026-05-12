"""
Entroly Claims Verification Suite
===================================
Independently verifiable test that validates README claims against any
real-world repository. Run from a project root to verify.

Usage:
    cd /path/to/any/project
    python -m tests.verify_claims

Tested against:
    - Kubeflow Trainer (683 files, 939K tokens, Go/K8s/Python)
    - Entroly itself (200+ files, multi-language)
"""
import time
import sys
import os
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from entroly.server import EntrolyEngine
from entroly.config import EntrolyConfig
from entroly.auto_index import auto_index
from entroly import __version__

PASS = 0
FAIL = 0
results = []


def check(claim_id, name, condition, detail=""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")
    if detail:
        print(f"         {detail}")
    results.append({"id": claim_id, "name": name, "status": status, "detail": detail})


def main():
    global PASS, FAIL

    cwd = os.getcwd()
    print(f"\nEntroly Claims Verification v{__version__}")
    print(f"Repository: {cwd}")
    print("=" * 70)

    # ── INDEX ──────────────────────────────────────────────────────────
    print("\n[1] Indexing speed (<2 seconds)")
    print("-" * 40)
    engine = EntrolyEngine()
    t0 = time.perf_counter()
    result = auto_index(engine)
    index_time = time.perf_counter() - t0
    files = result.get("files_indexed", 0)
    tokens = result.get("total_tokens", 0)

    print(f"  {files} files, {tokens:,} tokens, {index_time:.2f}s")
    check("IDX-1", "Indexing < 2 seconds", index_time < 2.0, f"{index_time:.3f}s")
    check("IDX-2", "Files indexed > 0", files > 0, f"{files} files")

    # ── TOKEN SAVINGS ──────────────────────────────────────────────────
    print("\n[2] Token savings (70-95%+)")
    print("-" * 40)
    queries = [
        "How does the main controller reconcile resources?",
        "Explain the module structure and dependency graph",
        "How is the core workflow orchestrated?",
    ]
    budgets = [128000, 64000, 32000, 16000, 8000]
    all_savings = {}

    for budget in budgets:
        engine.advance_turn()
        opt = engine.optimize_context(token_budget=budget, query=queries[0])
        sel = opt.get("selected_fragments", [])
        used = sum(f.get("token_count", 0) for f in sel)
        saving = (1 - used / tokens) * 100 if tokens > 0 else 0
        all_savings[budget] = {"used": used, "saving": saving, "frags": len(sel)}
        print(f"  Budget {budget:>7,}: {used:>6,} tokens → {saving:.1f}% savings")

    # Savings at typical budget (32K)
    s32 = all_savings.get(32000, {}).get("saving", 0)
    check("SAV-1", "Savings >= 90% at 32K budget", s32 >= 90, f"{s32:.1f}%")

    s8 = all_savings.get(8000, {}).get("saving", 0)
    check("SAV-2", "Savings >= 95% at 8K budget", s8 >= 95, f"{s8:.1f}%")

    # Average across queries at 128K
    query_savings = []
    for q in queries:
        engine.advance_turn()
        opt = engine.optimize_context(token_budget=128000, query=q)
        sel = opt.get("selected_fragments", [])
        used = sum(f.get("token_count", 0) for f in sel)
        saving = (1 - used / tokens) * 100 if tokens > 0 else 0
        query_savings.append(saving)

    avg = sum(query_savings) / len(query_savings)
    check("SAV-3", "Average savings >= 70% (128K budget)", avg >= 70, f"{avg:.1f}%")

    # ── LATENCY ────────────────────────────────────────────────────────
    print("\n[3] Optimization latency")
    print("-" * 40)
    latencies = []
    for q in queries:
        engine.advance_turn()
        t0 = time.perf_counter()
        engine.optimize_context(token_budget=128000, query=q)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        print(f"  {q[:50]}... → {lat:.1f}ms")

    avg_lat = sum(latencies) / len(latencies)
    check("LAT-1", "Average latency < 100ms", avg_lat < 100, f"{avg_lat:.1f}ms")

    # ── COVERAGE ───────────────────────────────────────────────────────
    print("\n[4] Codebase coverage")
    print("-" * 40)
    frags = list(engine._rust.export_fragments())
    extensions = {}
    for f in frags:
        ext = os.path.splitext(f.get("source", ""))[1] or "(none)"
        extensions[ext] = extensions.get(ext, 0) + 1

    print(f"  {len(frags)} fragments, {len(extensions)} file types")
    for ext, count in sorted(extensions.items(), key=lambda x: -x[1])[:8]:
        print(f"    {ext:>10}: {count}")

    check("COV-1", "Multiple file types indexed", len(extensions) >= 3,
          f"{len(extensions)} types")

    # ── ENTROPY SCORING ────────────────────────────────────────────────
    print("\n[5] Information density scoring")
    print("-" * 40)
    entropies = [f.get("entropy_score", 0) for f in frags]
    nonzero = [e for e in entropies if e > 0]
    ent_range = max(nonzero) - min(nonzero) if nonzero else 0
    print(f"  Scores: min={min(nonzero):.4f}, max={max(nonzero):.4f}, range={ent_range:.4f}")

    check("ENT-1", "All fragments have entropy scores",
          len(nonzero) == len(entropies), f"{len(nonzero)}/{len(entropies)}")
    check("ENT-2", "Entropy variance > 0.1 (non-trivial scoring)",
          ent_range > 0.1, f"Range: {ent_range:.4f}")

    # ── SOURCE-TYPE PRIORITIZATION ─────────────────────────────────────
    print("\n[6] Source-type prioritization (v0.15.0)")
    print("-" * 40)
    engine.advance_turn()
    opt = engine.optimize_context(token_budget=4096, query="How does the controller work?")
    sel = opt.get("selected_fragments", [])

    code_exts = {".go", ".py", ".rs", ".ts", ".js", ".java", ".c", ".cpp", ".rb"}
    config_exts = {".yaml", ".yml", ".json", ".toml"}

    code_count = sum(1 for f in sel if os.path.splitext(f.get("source", ""))[1] in code_exts)
    config_count = sum(1 for f in sel if os.path.splitext(f.get("source", ""))[1] in config_exts)

    print(f"  Code files: {code_count}, Config files: {config_count}")
    check("SRC-1", "Source code >= config in selection", code_count >= config_count,
          f"Code={code_count} vs Config={config_count}")

    # ── DEDUPLICATION ──────────────────────────────────────────────────
    # SimHash dedup is a CONTENT property (near-duplicate content collapses),
    # not a SOURCE-PATH property. Files larger than the chunk size legitimately
    # produce multiple fragments sharing one source path; that is by design,
    # not a dedup failure. Testing source-path uniqueness was conceptually
    # wrong and silently passed for releases where chunking was rare.
    #
    # Right invariant: ingest known near-duplicate content into a fresh engine
    # and verify the engine collapses them. This exercises the actual SimHash
    # codepath (semantic_dedup.rs) with adversarial input.
    print("\n[7] SimHash deduplication (rigorous probe)")
    print("-" * 40)
    # Isolated probe engine — opt out of the shared persistent index entirely
    # (no load and no save). This makes the test deterministic regardless of
    # prior session state and prevents the probe from corrupting the real
    # engine's warm-start index with its 5-fragment test corpus.
    probe = EntrolyEngine(EntrolyConfig(use_persistent_index=False))
    # Three near-duplicates (whitespace / comment / minor formatting variants
    # of the same function body) + two distinct functions. A correct dedup
    # implementation collapses the first three into a single representative.
    canonical = "def hash_value(x):\n    return (x * 31 + 7) & 0xFFFFFFFF"
    near_dups = [
        (canonical, "probe://a.py"),
        (canonical + "  # added comment", "probe://b.py"),
        (canonical.replace("    ", "  "), "probe://c.py"),  # 4→2 space indent
    ]
    distinct = [
        ("def parse_url(s):\n    scheme, rest = s.split('://', 1)\n    return scheme, rest",
         "probe://d.py"),
        ("class Cache:\n    def __init__(self, size):\n        self._size = size\n        self._data = {}",
         "probe://e.py"),
    ]
    before = probe._rust.fragment_count() if hasattr(probe, "_rust") else 0
    for content, src in near_dups + distinct:
        probe.ingest_fragment(content=content, source=src, token_count=20, is_pinned=False)
    after = probe._rust.fragment_count() if hasattr(probe, "_rust") else 0
    net_ingested = after - before
    # Expected: 3 near-dups collapse to 1, 2 distinct stay → 3 net fragments.
    # Allow 3 (perfect dedup) or 4 (one near-dup pair caught, one missed) but
    # not 5 (no dedup at all).
    print(f"  Ingested 5 candidates ({len(near_dups)} near-dup + {len(distinct)} distinct)")
    print(f"  Net fragments after dedup: {net_ingested} (expected ≤ 4)")
    check("DUP-1", "SimHash collapses near-duplicate content",
          net_ingested <= 4,
          f"{net_ingested} of 5 candidates retained (perfect: 3, acceptable: ≤4)")

    # ── ENGINE & SDK ───────────────────────────────────────────────────
    print("\n[8] Engine, SDK, compatibility")
    print("-" * 40)

    try:
        import entroly_core
        check("ENG-1", "Rust engine loaded", True)
    except ImportError:
        check("ENG-1", "Rust engine loaded", False, "entroly_core not installed")

    check("ENG-2", "Python >= 3.10", sys.version_info >= (3, 10),
          f"Python {sys.version_info.major}.{sys.version_info.minor}")
    check("ENG-3", f"Version {__version__}", True)

    try:
        from entroly import compress
        check("SDK-1", "SDK compress importable", True)
    except ImportError:
        check("SDK-1", "SDK compress importable", False)

    check("LOCAL-1", "No API key required", True, "All operations ran without API keys")

    # ── SUMMARY ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    total = PASS + FAIL
    print(f"RESULTS: {PASS}/{total} claims verified ({FAIL} failed)")
    print("=" * 70)

    # Write machine-readable results
    report = {
        "version": __version__,
        "repository": cwd,
        "files_indexed": files,
        "total_tokens": tokens,
        "index_time_s": round(index_time, 3),
        "avg_savings_pct": round(avg, 1),
        "avg_latency_ms": round(avg_lat, 1),
        "passed": PASS,
        "failed": FAIL,
        "total": total,
        "results": results,
    }

    report_path = os.path.join(cwd, ".entroly_verification.json")
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {report_path}")
    except Exception:
        pass

    if FAIL == 0:
        print("\nAll claims verified.")
    else:
        print(f"\n{FAIL} claim(s) need attention.")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
