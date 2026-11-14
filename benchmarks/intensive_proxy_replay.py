#!/usr/bin/env python3
"""Deterministic in-process 25-turn proxy transform replay.

Synthetic tool outputs; no model, HTTP provider call, provider billing, or real
task-success oracle. Report serialized byte differences and local transform time.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from entroly.compression_proxy import ToolResultPolicy, compress_proxy_payload

SOURCE = "\n".join(
    f"def reconcile_{i}(account, amount):\n    current = account.balance\n"
    f"    if current < amount: raise InsufficientFunds(amount)\n"
    f"    return current - amount\n" for i in range(180)
)
LOG = "\n".join(f"2026-11-13T12:00:{i%60:02d} INFO request-{i}: completed"
                for i in range(800)) + "\n2026-11-13T12:59:59 FATAL E_CONNRESET retries=3\n"
RECORDS = json.dumps([{"id": i, "status": "overloaded_error" if i == 488 else "ok",
                       "latency_ms": i%19, "message": "x"*70} for i in range(600)])
POLICY = ToolResultPolicy.from_names("")


def append_tool(body: dict, provider: str, i: int, name: str, output: str) -> None:
    if provider == "openai":
        ref = f"call_{i}"
        body["messages"].append({"role": "assistant", "content": None, "tool_calls": [
            {"id": ref, "type": "function", "function": {"name": name, "arguments": "{}"}}]})
        body["messages"].append({"role": "tool", "tool_call_id": ref, "content": output})
    else:
        ref = f"tool_{i}"
        body["messages"].append({"role": "assistant", "content": [
            {"type": "tool_use", "id": ref, "name": name, "input": {}}]})
        body["messages"].append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": ref, "content": output}]})


def outputs(body: dict, provider: str) -> list[str]:
    if provider == "openai":
        return [m.get("content") for m in body["messages"] if m.get("role") == "tool"]
    return [block["content"] for m in body["messages"]
            for block in (m.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "tool_result"]


def run() -> dict:
    rows = []
    for provider in ("openai", "anthropic"):
        body = {"model": "gpt-4o" if provider == "openai" else "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Fix reconcile invoice problem"}]}
        for i in range(25):
            name, content = ("Read", SOURCE) if i in (0, 17) else (
                ("Bash", LOG) if i % 4 in (1, 2) else
                ("Bash", RECORDS) if i % 4 == 3 else
                ("Bash", f"INFO turn {i} "*100))
            append_tool(body, provider, i, name, content)
            t0 = time.perf_counter_ns()
            optimized = compress_proxy_payload(body, provider=provider, policy=POLICY)
            elapsed_ms = (time.perf_counter_ns() - t0)/1e6
            original, sent = outputs(body, provider), outputs(optimized.body, provider)
            if len(original) != len(sent) or sent[0] != SOURCE:
                raise AssertionError(f"first source view lost: {provider} turn {i}")
            if i >= 17 and sent[17] != SOURCE:
                # Content-addressed duplicate pointer is acceptable only when
                # the earlier full file is still available in this request.
                if not (sent[17].startswith("[entroly: byte-identical to the earlier Read result")
                        and "not repeated" in sent[17]):
                    raise AssertionError(f"duplicate source lacks provenance: {provider} turn {i}")
            a = len(json.dumps(body, sort_keys=True).encode())
            b = len(json.dumps(optimized.body, sort_keys=True).encode())
            rows.append({"provider": provider, "turn": i+1, "raw_bytes": a,
                         "forwarded_bytes": b, "byte_reduction_pct": round(100*(1-b/a), 2),
                         "transform_ms": round(elapsed_ms, 3),
                         "first_source_available": sent[0] == SOURCE,
                         "error_marker_preserved": (
                             "E_CONNRESET" in json.dumps(optimized.body) if
                             "E_CONNRESET" in json.dumps(body) else None)})
    aggregate = {}
    for provider in ("openai", "anthropic"):
        group = [r for r in rows if r["provider"] == provider]
        times = sorted(r["transform_ms"] for r in group)
        aggregate[provider] = {
            "turns": len(group),
            "median_transform_ms": round(statistics.median(times), 3),
            "p95_transform_ms": times[int(.95*(len(times)-1))],
            "last_turn_byte_reduction_pct": group[-1]["byte_reduction_pct"],
            "first_file_view_always_exact": all(r["first_source_available"] for r in group),
            "fatal_error_marker_preserved": all(r["error_marker_preserved"] is not False for r in group)}
    return {
        "schema": "entroly.intensive-dogfood.proxy-transform.v1",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": ["Synthetic tool outputs, not real LLM task success",
                        "In-process only; not end-to-end provider latency",
                        "Bytes are not provider tokens or dollar savings"],
        "aggregate": aggregate, "turns": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    path = parser.parse_args().out
    data = run()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
