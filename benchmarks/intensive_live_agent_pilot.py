#!/usr/bin/env python3
"""REAL Ollama coding-model A/B/C pilot. Not the 400-task preregistered study.

Every result requires a real local model generation, provider-reported Ollama
tokens, and independent pytest correctness. CLOSED_LOOP shares COMPRESS's
first result and retries only on test failure with statically imported omitted
source. No oracle expected answer or hidden task label is used for retrieval.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import statistics
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.agentic_arms import Arm, Fragment, build_closed_loop, build_for_arm  # noqa: E402
from benchmarks.agentic_tasks_run import (  # noqa: E402
    PROMPT, Task, build_dependency_tasks, call_model, classify_failure, extract_source,
    mcnemar_exact,
)
from benchmarks.engine_isolation import assert_engine_isolated, isolated_engine_dir  # noqa: E402
from benchmarks.intensive_oracle_sandbox import run_sandboxed_oracle  # noqa: E402
from entroly.native_status import native_status  # noqa: E402


def model_identity(url: str, model: str) -> dict[str, Any]:
    """Reject remote model endpoints, and require the named model's digest."""
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
            or parsed.port is None):
        raise ValueError("only a local Ollama endpoint is allowed")
    with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=15) as response:
        available = json.load(response).get("models", [])
    found = [x for x in available if x.get("name") == model or x.get("model") == model]
    if len(found) != 1 or not found[0].get("digest"):
        raise RuntimeError("requested Ollama model is not locally available with a digest")
    return {"name": model, "digest": found[0]["digest"], "size_bytes": found[0].get("size")}


def import_recovery(*, target_path: str, target_source: str, omitted: list[Fragment]) -> list[Fragment]:
    """Recover target and statically imported modules using visible input ONLY.

    This is deterministic import closure, not autonomous model inference.
    task.dependency_file, test answers and hidden_shape are never consulted.
    """
    try:
        tree = ast.parse(target_source)
    except SyntaxError:
        return []
    paths = {target_path}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            paths.add(node.module.replace(".", "/") + ".py")
        elif isinstance(node, ast.Import):
            paths.update(x.name.replace(".", "/") + ".py" for x in node.names)
    return [fragment for fragment in omitted if fragment.source in paths]


def answer(task: Task, context: str, *, model: str, base_url: str, seed: int,
           timeout: float) -> dict[str, Any]:
    prompt = PROMPT.format(query=task.query, context=context, target=task.target_file,
                           test=task.test_source)
    result = call_model(base_url=base_url, model=model, prompt=prompt, seed=seed,
                        timeout=timeout, max_output_tokens=384, context_window=4096)
    for field in ("input_tokens", "output_tokens"):
        count = result.get(field)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise RuntimeError(f"model omitted measured positive {field}; not imputing zero")
    submitted_patch = extract_source(result["text"])
    passed, oracle = run_sandboxed_oracle(task, submitted_patch)
    return {"passed": bool(passed), "outcome": classify_failure(passed, oracle),
            "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
            "latency_s": result["latency_s"],
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "answer_sha256": hashlib.sha256(result["text"].encode()).hexdigest(),
            "model_response": result["text"], "submitted_patch": submitted_patch,
            "oracle_tail": oracle.strip()[-300:]}


def one_task(task: Task, *, optimize_fn: Any, model: str, base_url: str,
             seed: int, budget: int, timeout: float, raw_first: bool) -> dict[str, Any]:
    fragments = task.fragments()
    raw = build_for_arm(Arm.RAW, fragments, query=task.query, budget=budget)
    compressed = build_for_arm(Arm.COMPRESS, fragments, query=task.query,
                               budget=budget, optimize_fn=optimize_fn)
    first = {}
    for name in (("raw", "compress") if raw_first else ("compress", "raw")):
        built = raw if name == "raw" else compressed
        first[name] = answer(task, built.text, model=model, base_url=base_url,
                             seed=seed, timeout=timeout)
        first[name]["sources"] = [f.source for f in built.included]
        first[name]["omitted_sources"] = [f.source for f in built.omitted]
    comp = first["compress"]
    recovered = []
    retry = None
    if not comp["passed"]:
        recovered = import_recovery(target_path=task.target_file,
                                    target_source=task.broken_source, omitted=compressed.omitted)
        if recovered:
            second_context = build_closed_loop(compressed, recovered=recovered)
            retry = answer(task, second_context.text, model=model, base_url=base_url,
                           seed=seed, timeout=timeout)
            retry["sources"] = [f.source for f in second_context.included]
    closed = {
        "passed": retry["passed"] if retry else comp["passed"],
        "retry_performed": bool(retry),
        "recovered_sources": [f.source for f in recovered],
        "first_call_shared_with_compress": True,
        "input_tokens": comp["input_tokens"] + (retry["input_tokens"] if retry else 0),
        "output_tokens": comp["output_tokens"] + (retry["output_tokens"] if retry else 0),
        "latency_s": comp["latency_s"] + (retry["latency_s"] if retry else 0),
        "second_attempt": retry,
    }
    return {"task_id": task.task_id, "seed": seed, "raw": first["raw"],
            "compress": first["compress"], "closed_loop": closed}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for arm in ("raw", "compress", "closed_loop"):
        entries = [row[arm] for row in rows]
        summary[arm] = {
            "trials": len(entries),
            "passed": sum(int(x["passed"]) for x in entries),
            "reported_input_tokens": sum(x["input_tokens"] for x in entries),
            "reported_output_tokens": sum(x["output_tokens"] for x in entries),
            "latency_s_total": round(sum(x["latency_s"] for x in entries), 3),
        }
    raw_only = sum(x["raw"]["passed"] and not x["compress"]["passed"] for x in rows)
    comp_only = sum(x["compress"]["passed"] and not x["raw"]["passed"] for x in rows)
    baseline = summary["raw"]["reported_input_tokens"]
    return {
        "arms": summary,
        "raw_only_success": raw_only, "compress_only_success": comp_only,
        "mcnemar_exact_p": mcnemar_exact(raw_only, comp_only),
        "closed_loop_real_retry_count": sum(x["closed_loop"]["retry_performed"] for x in rows),
        "input_token_reduction_compress_vs_raw_pct": (
            round(100 * (1 - summary["compress"]["reported_input_tokens"] / baseline), 2)
            if baseline else None),
        "input_token_reduction_closed_loop_vs_raw_pct": (
            round(100 * (1 - summary["closed_loop"]["reported_input_tokens"] / baseline), 2)
            if baseline else None),
        "median_raw_latency_s": statistics.median(x["raw"]["latency_s"] for x in rows),
        "median_compress_latency_s": statistics.median(x["compress"]["latency_s"] for x in rows),
    }


def run(*, model: str, base_url: str, budget: int, seeds: list[int],
        limit: int, timeout: float) -> dict[str, Any]:
    if budget < 1 or limit < 1 or not seeds:
        raise ValueError("budget, task count and seeds must be nonempty/positive")
    identity = model_identity(base_url, model)
    tasks = build_dependency_tasks(6)[:limit]
    rows = []
    with isolated_engine_dir():
        assert_engine_isolated()
        from entroly import optimize
        status = native_status()
        if not status.ok:
            raise RuntimeError(f"native engine unavailable: {status}")
        for seed in seeds:
            for i, task in enumerate(tasks):
                print(f"REAL_AGENT task={task.task_id} seed={seed}", flush=True)
                row = one_task(task, model=model, base_url=base_url, seed=seed, budget=budget,
                               timeout=timeout, optimize_fn=optimize,
                               raw_first=(i + seed) % 2 == 0)
                rows.append(row)
                print(f"  RAW={row['raw']['passed']} COMPRESS={row['compress']['passed']} "
                      f"CLOSED_LOOP={row['closed_loop']['passed']}", flush=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    expected = os.environ.get("ENTROLY_DOGFOOD_EXPECTED_SHA")
    if expected and commit != expected:
        raise RuntimeError(f"source mismatch: expected {expected}, got {commit}")
    return {
        "schema": "entroly.live-agent-pilot.v1",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": commit, "head_provenance_verified": bool(expected and commit == expected),
        "model": identity,
        "protocol": {"task_source": "4 synthetic dependency-bearing coding tasks",
                     "task_ids": [x.task_id for x in tasks], "seed_values": seeds,
                     "budget": budget, "temperature": 0, "max_output_tokens": 384,
                     "ollama_context_window": 4096,
                     "closed_loop": "reuse COMPRESS first call, then static import recovery on test failure"},
        "limitations": [
            "CPU Ollama pilot, not a frontier LLM; no provider invoice",
            "Synthetic tasks, not 400 mined real-repository test-oracle tasks",
            "Too few independent tasks for any noninferiority quality claim",
            "Model-reported token counts, no measured dollar billing or compute energy cost",
            "Recovery uses static imports, not model autonomous missing-evidence detection",
            "Generated patches execute only inside networkless, nonroot, read-only Docker oracle containers",
        ],
        "summary": summarize(rows), "trials": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5-coder:0.5b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--out", required=True, type=Path)
    options = parser.parse_args()
    result = run(model=options.model, base_url=options.base_url, budget=options.budget,
                 seeds=options.seeds, limit=options.limit, timeout=options.timeout)
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
