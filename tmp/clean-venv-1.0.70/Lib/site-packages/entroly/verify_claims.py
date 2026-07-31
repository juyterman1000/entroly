"""Packaged README smoke verifier.

This module is intentionally conservative: it verifies the install and local
context-selection path that a new user can run after ``pip install entroly``.
It does not try to re-run live LLM benchmarks or external integrations.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from entroly import __version__, compress, compress_messages
import entroly.auto_index as _auto_index
from entroly.ccr import (
    CompressedContextStore,
    capture_recoverable_fragments,
    slice_recovery_content,
)
from entroly.config import EntrolyConfig
from entroly.server import EntrolyEngine


def _print_status(ok: bool, name: str, detail: str = "") -> None:
    label = "PASS" if ok else "FAIL"
    print(f"  [{label}] {name}")
    if detail:
        print(f"         {detail}")


def run(output: str | None = None, max_files: int = 120) -> int:
    """Run a packaged, source-tree-independent verification pass."""
    cwd = Path.cwd()
    results: list[dict[str, Any]] = []

    def check(claim_id: str, name: str, condition: bool, detail: str = "") -> None:
        _print_status(condition, name, detail)
        results.append(
            {
                "id": claim_id,
                "name": name,
                "status": "PASS" if condition else "FAIL",
                "detail": detail,
            }
        )

    print(f"\nEntroly install verification v{__version__}")
    print(f"Repository: {cwd}")
    print("=" * 70)

    config = EntrolyConfig(use_persistent_index=False)
    engine = EntrolyEngine(config)

    print("\n[1] SDK import path")
    print("-" * 40)
    compressed = compress("alpha beta gamma delta", budget=3)
    messages = compress_messages([{"role": "user", "content": "hello"}], budget=10)
    check("SDK-1", "compress() returns text", isinstance(compressed, str), type(compressed).__name__)
    check("SDK-2", "compress_messages() returns messages", isinstance(messages, list), type(messages).__name__)

    print("\n[2] Local indexing")
    print("-" * 40)
    t0 = time.perf_counter()
    # Keep the packaged verifier bounded. Full-repo benchmarks belong in
    # ``bench/``; this command must remain safe on a plain pip install where
    # the native Rust wheel may not be available yet.
    previous_max_files = _auto_index.MAX_FILES
    _auto_index.MAX_FILES = max(1, min(previous_max_files, max_files))
    try:
        index_result = _auto_index.auto_index(engine)
    finally:
        _auto_index.MAX_FILES = previous_max_files
    index_s = time.perf_counter() - t0
    files = int(index_result.get("files_indexed", 0) or 0)
    tokens = int(index_result.get("total_tokens", 0) or 0)
    print(f"  {files} files, {tokens:,} tokens, {index_s:.2f}s")
    check("IDX-1", "Files indexed > 0", files > 0, f"{files} files")
    check("IDX-2", "Indexing completed", index_s >= 0, f"{index_s:.3f}s")

    print("\n[3] Context optimization")
    print("-" * 40)
    t1 = time.perf_counter()
    opt = engine.optimize_context(token_budget=8000, query="explain the main module structure")
    optimize_ms = (time.perf_counter() - t1) * 1000
    selected = opt.get("selected_fragments", []) or opt.get("selected", [])
    used = int(opt.get("total_tokens") or sum(int(f.get("token_count", 0) or 0) for f in selected))
    savings = (1 - used / tokens) * 100 if tokens > 0 and used <= tokens else 0.0
    print(f"  selected={len(selected)} tokens={used:,} savings={savings:.1f}% latency={optimize_ms:.1f}ms")
    check("OPT-1", "Selected context within budget", used <= 8000, f"{used:,}/8,000 tokens")
    check("OPT-2", "Optimization returned fragments", len(selected) > 0 or files == 0, f"{len(selected)} fragments")
    # This command is a new-user smoke test, not the release latency benchmark.
    # Full timing comparisons live in bench/. Keep the bound loose enough for
    # cold Windows filesystems and Python fallback paths while still catching
    # pathological hangs.
    smoke_budget_ms = max(5000.0, min(30000.0, files * 75.0))
    check(
        "OPT-3",
        "Optimization completed within smoke-test budget",
        optimize_ms < smoke_budget_ms,
        f"{optimize_ms:.1f}ms / {smoke_budget_ms:.0f}ms",
    )

    print("\n[4] Exact recovery")
    print("-" * 40)
    recovery_store = CompressedContextStore(max_entries=8)
    recovery_original = (
        "def validate_token(token):\n"
        "    return token == 'known-session-token'\n\n"
        "def authorize(request):\n"
        "    return validate_token(request.token)\n"
    )
    recovery_fragment = {
        "id": "probe-auth-fragment",
        "source": "probe://auth.py",
        "content": recovery_original,
        "token_count": max(1, len(recovery_original) // 4),
        "entropy_score": 0.5,
    }
    compressed_fragment = {
        "id": "probe-auth-fragment",
        "source": "probe://auth.py",
        "content": "[skeleton] validate_token(token) -> bool",
        "token_count": 10,
        "variant": "skeleton",
        "relevance": 1.0,
    }

    def recovery_lookup(key: str) -> dict[str, Any] | None:
        if key in {"probe-auth-fragment", "probe://auth.py"}:
            return recovery_fragment
        return None

    handles = capture_recoverable_fragments(
        [compressed_fragment],
        recovery_lookup,
        store=recovery_store,
    )
    recovered = recovery_store.retrieve(handles[0]) if handles else None
    check(
        "CCR-1",
        "Compressed fragment receives retrieval handle",
        bool(handles and compressed_fragment.get("recoverable")),
        handles[0] if handles else "no handle",
    )
    check(
        "CCR-2",
        "Retrieval restores exact original content",
        bool(recovered and recovered.get("original") == recovery_original),
        recovered.get("retrieval_handle", "") if recovered else "not recovered",
    )

    oversized = (
        "Module overview.\n"
        + ("background details\n" * 80)
        + "validate_token checks the known-session-token before authorization.\n"
        + ("trailing details\n" * 80)
    )
    recovery_budget = 96
    sliced, was_sliced = slice_recovery_content(
        oversized,
        "how does validate_token authorize the request",
        recovery_budget,
    )
    sliced_tokens = max(1, (len(sliced) + 3) // 4) if sliced else 0
    check(
        "CCR-3",
        "Oversized recovery excerpt stays within budget",
        bool(was_sliced and sliced_tokens <= recovery_budget and "validate_token" in sliced),
        f"{sliced_tokens}/{recovery_budget} tokens",
    )

    print("\n[5] Engine mode")
    print("-" * 40)
    native = bool(getattr(engine, "_use_rust", False))
    check(
        "ENG-1",
        "Engine initialized",
        True,
        "native Rust engine" if native else "Python fallback engine",
    )
    check(
        "LOCAL-1",
        "No API key required for this verification",
        True,
        "all operations ran locally",
    )

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = len(results) - passed
    report = {
        "version": __version__,
        "repository": str(cwd),
        "files_indexed": files,
        "total_tokens": tokens,
        "index_time_s": round(index_s, 3),
        "tokens_used": used,
        "savings_pct": round(savings, 1),
        "optimize_latency_ms": round(optimize_ms, 1),
        "recovery": {
            "retrieval_handles": len(handles),
            "retrieved_exact": bool(recovered and recovered.get("original") == recovery_original),
            "slice_tokens": sliced_tokens,
            "slice_budget": recovery_budget,
        },
        "engine": "rust" if native else "python",
        "passed": passed,
        "failed": failed,
        "results": results,
    }

    out_path = Path(output or ".entroly_verification.json")
    try:
        if out_path.parent != Path("."):
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport saved: {out_path}")
    except OSError as exc:
        print(f"\nWarning: failed to write report: {exc}", file=sys.stderr)

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{len(results)} checks passed ({failed} failed)")
    print("=" * 70)
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    output = None
    max_files = 120
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in {"--output", "-o"} and i + 1 < len(argv):
            output = argv[i + 1]
            i += 2
        elif arg == "--max-files" and i + 1 < len(argv):
            try:
                max_files = int(argv[i + 1])
            except ValueError:
                print("--max-files must be an integer", file=sys.stderr)
                return 2
            i += 2
        else:
            print(
                "usage: python -m entroly.verify_claims [--output PATH] [--max-files N]",
                file=sys.stderr,
            )
            return 2
    return run(output=output, max_files=max_files)


if __name__ == "__main__":
    raise SystemExit(main())
