#!/usr/bin/env python3
"""Entroly against an external context runtime, measured on identical files.

Unlike the earlier attempt in `head_to_head.py` -- which never obtained a valid
number because that system's compressor refused to engage on this platform --
this comparison ran successfully end to end.

The external binary is located through `ENTROLY_EXTERNAL_CTX_BIN`, or found on
PATH under a name the operator supplies. Its product name is deliberately
absent from this source, per the repository's external-name policy; the
convention matches `compression_frontier.py`, which imports a third party under
a neutral alias rather than naming it.

Axes
----
  cold read    tokens emitted the first time a file is read, per mode
  warm re-read tokens emitted on a second read of the same file
  evidence     fraction of derived required evidence surviving
  latency      wall clock per call

Entroly arms
------------
  compress()   the public query-agnostic API, which routes through the codec
               registry
  index        signatures only, no bodies -- the arm from `addressability.py`,
               included because it is the closest analogue to the external
               system's signature mode and the fairest structural comparison

The external system exposes an explicit mode per call; Entroly picks one
strategy internally. That asymmetry is the finding, not a defect in the
harness: a caller can ask that system for a resolution and cannot ask Entroly.

Required evidence is derived from each file (long digit runs, identifiers,
error tokens) rather than hand-written, so neither side is scored against a
list drawn up with its behaviour in mind.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
_EVIDENCE = re.compile(
    r"\b\d{3,}\b|\b[A-Za-z_][A-Za-z0-9_]{7,}\b|\b(?:ERROR|FAILED|Exception|Traceback)\b"
)
_MODES = ("full", "map", "signatures")


@dataclass
class Row:
    sample: str
    system: str
    mode: str
    in_tokens: int
    out_tokens: int
    ratio: float
    evidence_total: int
    evidence_kept: int
    latency_ms: float

    @property
    def retention(self) -> float:
        return self.evidence_kept / self.evidence_total if self.evidence_total else 1.0


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _required(text: str, cap: int = 300) -> list[str]:
    seen: list[str] = []
    for match in _EVIDENCE.findall(text):
        if match not in seen:
            seen.append(match)
        if len(seen) >= cap:
            break
    return seen


def _external_binary() -> str | None:
    explicit = os.environ.get("ENTROLY_EXTERNAL_CTX_BIN")
    if explicit and Path(explicit).exists():
        return explicit
    name = os.environ.get("ENTROLY_EXTERNAL_CTX_NAME", "")
    return shutil.which(name) if name else None


def _run_external(binary: str, rel: str, mode: str) -> tuple[str, float]:
    started = time.perf_counter()
    try:
        # Explicit utf-8 with replacement: the default Windows codepage raised
        # UnicodeDecodeError on box-drawing characters in the external tool's
        # output, and a failed decode returned "" -- which scored as 100%
        # compression and dragged the `full` median to a meaningless 50%.
        proc = subprocess.run([binary, "read", rel, "-m", mode], cwd=REPO,
                              capture_output=True, text=True, timeout=180,
                              encoding="utf-8", errors="replace")
        out = proc.stdout or ""
    except (OSError, subprocess.SubprocessError):
        out = ""
    return out, (time.perf_counter() - started) * 1000


def run(files: list[str], binary: str) -> dict[str, Any]:
    from benchmarks.addressability import index_of
    from entroly import compress

    rows: list[Row] = []
    warm: list[dict[str, Any]] = []

    for rel in files:
        try:
            text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        required = _required(text)
        in_tokens = _tokens(text)

        def add(system: str, mode: str, out: str, ms: float) -> None:
            rows.append(Row(
                sample=rel, system=system, mode=mode,
                in_tokens=in_tokens, out_tokens=_tokens(out),
                ratio=1.0 - _tokens(out) / max(in_tokens, 1),
                evidence_total=len(required),
                evidence_kept=sum(1 for item in required if item in out),
                latency_ms=ms,
            ))

        started = time.perf_counter()
        entroly_out = compress(text)
        add("entroly", "compress", entroly_out, (time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        index_out = index_of(text, rel)
        add("entroly", "index", index_out, (time.perf_counter() - started) * 1000)

        for mode in _MODES:
            out, ms = _run_external(binary, rel, mode)
            add("external", mode, out, ms)

        # Second read of the same file, to measure the caching claim.
        out, ms = _run_external(binary, rel, "full")
        warm.append({"sample": rel, "cold_tokens": in_tokens,
                     "warm_tokens": _tokens(out), "latency_ms": ms})

    return {
        "files": len(files),
        "rows": [r.__dict__ | {"retention": r.retention} for r in rows],
        "warm_reread": warm,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=10)
    # The external system caches reads ACROSS process invocations, so a file
    # read by an earlier run is no longer cold. Point this at an untouched set
    # to measure a genuine first read.
    ap.add_argument("--pattern", default="entroly/*.py")
    ap.add_argument("--out", type=Path,
                    default=REPO / "benchmarks" / "results" / "external_ctx_comparison.json")
    args = ap.parse_args()

    binary = _external_binary()
    if not binary:
        print("external binary not configured; set ENTROLY_EXTERNAL_CTX_BIN "
              "or ENTROLY_EXTERNAL_CTX_NAME")
        return 1

    listed = subprocess.run(["git", "ls-files", args.pattern], cwd=REPO,
                            capture_output=True, text=True, check=False).stdout.splitlines()
    picked: list[str] = []
    for rel in listed:
        try:
            size = (REPO / rel).stat().st_size
        except OSError:
            continue
        if 6_000 <= size <= 60_000:
            picked.append(rel)
        if len(picked) >= args.limit:
            break

    payload = run(picked, binary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = [Row(**{k: v for k, v in r.items() if k != "retention"})
            for r in payload["rows"]]
    print(f"\n  {len(picked)} source files, both systems on identical input\n")
    print(f"  {'system / mode':<24}{'median ratio':>14}{'median evidence':>17}{'median ms':>12}")
    for system, mode in (("entroly", "compress"), ("entroly", "index"),
                         ("external", "full"), ("external", "map"),
                         ("external", "signatures")):
        sub = [r for r in rows if r.system == system and r.mode == mode]
        if not sub:
            continue
        print(f"  {system + ' / ' + mode:<24}"
              f"{statistics.median(r.ratio for r in sub):>13.1%}"
              f"{statistics.median(r.retention for r in sub):>17.1%}"
              f"{statistics.median(r.latency_ms for r in sub):>12.1f}")

    warm = payload["warm_reread"]
    if warm:
        cold = statistics.median(w["cold_tokens"] for w in warm)
        hot = statistics.median(w["warm_tokens"] for w in warm)
        print(f"\n  warm re-read (external): {cold:,.0f} -> {hot:,.0f} tokens "
              f"({1 - hot / max(cold, 1):.2%} of a cold full read)")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
