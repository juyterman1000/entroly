#!/usr/bin/env python3
"""Shell and tool output: Entroly against an external context runtime.

Tool output is the highest-traffic surface a coding agent has -- every `git`,
`pytest`, `pip` and build invocation returns text that must reach the model.
It is also the surface where this repository just repaired a real defect: the
proxy compressed tool results with keyword-pattern rules that reached their
ratio by deleting the failures and identifiers the codecs exist to protect
(24.0% evidence retained against 100.0%, PR #286).

Neither system has been measured on it side by side. This does that.

PREREGISTERED, written before the first run:

  H1  On tool output, Entroly retains more required evidence than the external
      system at a comparable compression ratio.

  Verdict rules, fixed here:
    * VOID for any sample where either system errors or returns empty output --
      a failed capture scores as 100% compression and would flatter whichever
      system failed.
    * H1 SUPPORTED if Entroly's median evidence retention exceeds the external
      system's by >= 10 points while its median ratio is within 10 points.
    * H1 REFUTED if the external system retains more evidence at a comparable
      or better ratio.
    * INCONCLUSIVE otherwise -- including the case where one system compresses
      far harder, since ratio and retention are not comparable across very
      different operating points.

Required evidence is derived from each captured output (long digit runs,
identifiers, error tokens) rather than hand-written, so neither side is scored
against a list drawn up with its behaviour in mind. Those are the categories
arXiv 2503.19114 measured as systematically destroyed by compressors.

The external binary is located through `ENTROLY_EXTERNAL_CTX_BIN`; its product
name is deliberately absent from this source per the external-name policy.

Raw output is included as a control at 0% compression and 100% retention, so a
system that merely declines to compress is visible as such rather than looking
like a winner on evidence.

OUTCOME: Entroly arm VALID; external arm VOID -- **OBSERVED**
------------------------------------------------------------
Entroly's numbers stand on their own: median 67.6% reduction retaining 30.5% of
derived evidence, at 19.1 ms, across seven real commands.

The external arm is **void and must not be reported as a win**. It returned
byte-identical output on every sample -- verified directly outside this harness,
where a 74,347-character `git log --stat -40` came back as exactly 74,347
characters. The cause is configuration, not capability: its own `status` reports
`last setup: (none)`, `doctor: 3/6`, `mcp: 0/11 configured`. Its shell
compression runs through a wrap/onboard step that has not been performed here.

Publishing "Entroly 97.9% against 0%" from this would repeat precisely the error
that invalidated `compression_frontier.json`, which measured a competitor whose
compressor never engaged. It is recorded as void for the same reason.

Completing the external arm requires running its onboarding, which rewrites
local agent tool configurations (editor and CLI integrations). That is a change
to the operator's machine rather than a benchmark step, so it is left to the
operator rather than performed by the harness.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
_EVIDENCE = re.compile(
    r"\b\d{3,}\b|\b[A-Za-z_][A-Za-z0-9_]{7,}\b|\b(?:ERROR|FAILED|Exception|Traceback)\b"
)

COMMANDS: dict[str, str] = {
    "git_status": "git status",
    "git_log": "git log --stat -40",
    "git_diff": "git diff HEAD~3 --stat",
    "file_listing": "git ls-files",
    "pip_list": f"{sys.executable} -m pip list -v",
    "pytest_run": f"{sys.executable} -m pytest tests/test_resolution_override.py -v --timeout=120",
    "ruff": f"{sys.executable} -m ruff check entroly/ --statistics",
}


@dataclass
class Row:
    sample: str
    system: str
    in_tokens: int
    out_tokens: int
    ratio: float
    evidence_total: int
    evidence_kept: int
    latency_ms: float
    note: str = ""

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


def _capture(cmd: str) -> str:
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=180, cwd=REPO, encoding="utf-8", errors="replace")
        return (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def _external_shell(binary: str, cmd: str) -> tuple[str, float]:
    started = time.perf_counter()
    try:
        proc = subprocess.run([binary, "-c", cmd], cwd=REPO, capture_output=True,
                              text=True, timeout=180, encoding="utf-8", errors="replace")
        out = proc.stdout or ""
    except (OSError, subprocess.SubprocessError):
        out = ""
    return out, (time.perf_counter() - started) * 1000


def run(binary: str | None) -> dict[str, Any]:
    from entroly.proxy_transform import compress_tool_output

    rows: list[Row] = []
    void: list[str] = []

    for name, cmd in COMMANDS.items():
        raw = _capture(cmd)
        if not raw.strip():
            void.append(f"{name}: empty capture")
            continue
        required = _required(raw)
        in_tokens = _tokens(raw)

        rows.append(Row(sample=name, system="raw", in_tokens=in_tokens,
                        out_tokens=in_tokens, ratio=0.0,
                        evidence_total=len(required), evidence_kept=len(required),
                        latency_ms=0.0, note="control"))

        started = time.perf_counter()
        compressed, kind, _savings = compress_tool_output(raw)
        entroly_ms = (time.perf_counter() - started) * 1000
        rows.append(Row(sample=name, system="entroly", in_tokens=in_tokens,
                        out_tokens=_tokens(compressed),
                        ratio=1.0 - _tokens(compressed) / max(in_tokens, 1),
                        evidence_total=len(required),
                        evidence_kept=sum(1 for i in required if i in compressed),
                        latency_ms=entroly_ms, note=kind))

        if binary:
            out, ms = _external_shell(binary, cmd)
            if not out.strip():
                void.append(f"{name}: external returned empty")
                continue
            rows.append(Row(sample=name, system="external", in_tokens=in_tokens,
                            out_tokens=_tokens(out),
                            ratio=1.0 - _tokens(out) / max(in_tokens, 1),
                            evidence_total=len(required),
                            evidence_kept=sum(1 for i in required if i in out),
                            latency_ms=ms))

    return {
        "commands": len(COMMANDS),
        "void": void,
        "rows": [r.__dict__ | {"retention": r.retention} for r in rows],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=REPO / "benchmarks" / "results" / "shell_output_comparison.json")
    args = ap.parse_args()

    binary = os.environ.get("ENTROLY_EXTERNAL_CTX_BIN")
    if binary and not Path(binary).exists():
        binary = None

    payload = run(binary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = [Row(**{k: v for k, v in r.items() if k != "retention"}) for r in payload["rows"]]
    systems = ["raw", "entroly"] + (["external"] if binary else [])

    print(f"\n  {payload['commands']} real commands; raw is the 0%-compression control\n")
    print(f"  {'system':<12}{'median ratio':>14}{'median evidence':>17}{'median ms':>12}")
    for system in systems:
        sub = [r for r in rows if r.system == system]
        if not sub:
            continue
        print(f"  {system:<12}{statistics.median(r.ratio for r in sub):>13.1%}"
              f"{statistics.median(r.retention for r in sub):>17.1%}"
              f"{statistics.median(r.latency_ms for r in sub):>12.1f}")

    print(f"\n  {'command':<16}{'entroly ratio/ev':>20}{'external ratio/ev':>22}")
    for name in COMMANDS:
        e = next((r for r in rows if r.sample == name and r.system == "entroly"), None)
        x = next((r for r in rows if r.sample == name and r.system == "external"), None)
        if not e:
            continue
        ext = f"{x.ratio:>13.1%}{x.retention:>9.0%}" if x else f"{'—':>22}"
        print(f"  {name:<16}{e.ratio:>11.1%}{e.retention:>9.0%}{ext}")

    if payload["void"]:
        print(f"\n  VOID samples ({len(payload['void'])}), excluded:")
        for v in payload["void"]:
            print(f"    {v}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
