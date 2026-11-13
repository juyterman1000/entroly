#!/usr/bin/env python3
"""Model-free RAW / COMPRESS / CLOSED_LOOP evidence-retention trials.

A source-preservation oracle on 12 real Entroly symbols, NOT agent task
completion, model-token billing, or proof of invoice savings. Fixed tasks,
budgets, and permutation seeds are declared before any measurements.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ENTROLY_AIR_GAP", "1")
os.environ.setdefault("ENTROLY_DISABLE_UPDATE_CHECK", "1")
os.environ.setdefault("ENTROLY_DISABLE_TELEMETRY", "1")

from benchmarks.agentic_arms import Arm, Fragment, build_closed_loop, build_for_arm  # noqa: E402
from entroly.native_status import native_status  # noqa: E402
from entroly.sdk import optimize  # noqa: E402

# Real code; no synthetic known-answer toy corpus.
CASES = (
    ("entroly/net_savings.py", "estimate_net_saving", "How does estimate_net_saving charge new and cached added context?"),
    ("entroly/usage_ledger.py", "price_usage", "Where does price_usage calculate provider usage cost?"),
    ("entroly/proof_bundle.py", "verify_proof_bundle", "How does verify_proof_bundle validate a signed proof?"),
    ("entroly/proxy_config.py", "resolve_context_injection", "When does resolve_context_injection select context injection?"),
    ("entroly/compression_proxy.py", "compress_proxy_payload", "Where does compress_proxy_payload handle tool text?"),
    ("entroly/context_commit.py", "verify_context_commit", "How does verify_context_commit check replay integrity?"),
    ("entroly/proxy_transform.py", "extract_user_message", "How does extract_user_message read the latest user text?"),
    ("entroly/receipt_attestation.py", "AttestationKey", "How does AttestationKey work with Ed25519?"),
    ("entroly/receipt_merkle.py", "merkle_root", "How does merkle_root compute receipt hashes?"),
    ("entroly/epistemic_router.py", "EpistemicRouter", "Where does EpistemicRouter select the next flow?"),
    ("entroly/usage_ledger.py", "parse_provider_usage", "Where does parse_provider_usage normalize Gemini usage?"),
    ("entroly/proxy_traffic_value.py", "build_traffic_value_snapshot", "How is build_traffic_value_snapshot computed?"),
)
BUDGETS = (600, 1600)
SEEDS = (17, 42)
MAX_FRAGMENT_CHARS = 2700


def source_corpus() -> list[Fragment]:
    corpus = []
    for path, symbol, _ in CASES:
        text = (ROOT / path).read_text(encoding="utf-8")
        matching = [node for node in ast.parse(text).body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and node.name == symbol]
        if len(matching) != 1:
            raise RuntimeError(f"missing/ambiguous source symbol: {path}::{symbol}")
        content = ast.get_source_segment(text, matching[0])
        if not content:
            raise RuntimeError(f"cannot extract source: {path}::{symbol}")
        corpus.append(Fragment(source=f"{path}::{symbol}", content=content[:MAX_FRAGMENT_CHARS]))
    return corpus


def run() -> dict:
    fragments = source_corpus()
    index = {f.source: f for f in fragments}
    trials = []
    for seed in SEEDS:
        ordered = list(fragments)
        random.Random(seed).shuffle(ordered)
        baseline = build_for_arm(Arm.RAW, ordered, query="", budget=1)
        for budget in BUDGETS:
            for path, symbol, query in CASES:
                target = f"{path}::{symbol}"
                started = time.perf_counter_ns()
                try:
                    selected = build_for_arm(Arm.COMPRESS, ordered, query=query,
                                             budget=budget, optimize_fn=optimize)
                except Exception as exc:
                    trials.append({"seed": seed, "budget": budget, "target": target,
                                   "state": "SELECTION_UNAVAILABLE",
                                   "reason": f"{type(exc).__name__}: {exc}"[:300]})
                    continue
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                sources = {f.source for f in selected.included}
                if not sources.issubset(index):
                    raise AssertionError(f"Foreign sources in selection: {sources-set(index)}")
                exact = any(f.source == target and f.content == index[target].content
                            for f in selected.included)
                final = build_closed_loop(selected,
                                          recovered=[index[target]] if not exact else [])
                after = any(f.source == target and f.content == index[target].content
                            for f in final.included)
                if not after:
                    raise AssertionError(f"Expected exact-source recovery for {target}")
                trials.append({
                    "seed": seed, "budget": budget, "target": target,
                    "state": "MODEL_FREE_MEASURED",
                    "raw_estimated_tokens": baseline.estimated_tokens,
                    "selected_estimated_tokens": selected.estimated_tokens,
                    "after_recovery_estimated_tokens": final.estimated_tokens,
                    "source_hit": target in sources, "first_pass_exact": exact,
                    "exact_after_recovery": after, "recovery_needed": not exact,
                    "selection_ms": round(elapsed_ms, 3),
                    "selected_sources": sorted(sources),
                })
    measured = [x for x in trials if x["state"] == "MODEL_FREE_MEASURED"]
    latency = sorted(x["selection_ms"] for x in measured)
    reduction = [100*(1-x["selected_estimated_tokens"]/x["raw_estimated_tokens"])
                 for x in measured]
    native = native_status()
    return {
        "schema": "entroly.intensive-dogfood.model-free.v1",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "sha": os.environ.get("GITHUB_SHA", "locally-unset"),
        "python": sys.version.split()[0],
        "native_available": bool(native.available),
        "native_version": native.version,
        "protocol": {"cases": len(CASES), "budgets": list(BUDGETS), "seeds": list(SEEDS),
                     "max_fragment_chars": MAX_FRAGMENT_CHARS},
        "limitations": ["Not a live LLM or coding-agent test",
                        "Estimated tokens are NOT provider-reported usage",
                        "Selection latency excludes provider network and generation",
                        "No invoice-backed cost evidence"],
        "aggregate": {
            "planned": len(CASES)*len(BUDGETS)*len(SEEDS),
            "completed": len(measured),
            "selection_unavailable": len(trials)-len(measured),
            "source_hit_rate": (sum(x["source_hit"] for x in measured)/len(measured)
                                if measured else None),
            "first_pass_exact_rate": (sum(x["first_pass_exact"] for x in measured)/len(measured)
                                       if measured else None),
            "exact_recovery_rate": (sum(x["exact_after_recovery"] for x in measured)/len(measured)
                                     if measured else None),
            "median_estimated_context_reduction_pct": (round(statistics.median(reduction), 2)
                                                       if reduction else None),
            "median_selection_ms": (round(statistics.median(latency), 3)
                                    if latency else None),
            "p95_selection_ms": (latency[int(.95*(len(latency)-1))] if latency else None),
        },
        "trials": trials,
    }


def main() -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("--out", required=True, type=Path)
    output = cli.parse_args().out
    report = run()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], sort_keys=True))
    return 0 if report["aggregate"]["completed"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
