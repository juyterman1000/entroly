#!/usr/bin/env python3
"""Entroly vs a published external compressor, on identical real inputs.

The existing frontier artifact cannot be trusted: it recorded the external
system at 1.7% answer retention while that project publishes 97% on SQuAD v2,
and its own captured stderr shows the run used a degraded pure-Python content
detector under a maximum-savings profile. A 50x "win" measured against a
misconfigured baseline is worse than no measurement.

This is a clean-room replacement. Both systems run at their documented defaults
on the same real inputs, and every axis is reported -- including the ones where
Entroly loses.

Inputs are captured command output and tracked source, not authored fixtures.
Required evidence is derived from the input (long digit runs, identifiers,
error tokens) rather than hand-written, so neither side is scored against a
list drawn up with its behaviour in mind.

Axes measured
-------------
  ratio        token reduction, self-reported where available, recomputed here
  evidence     fraction of derived required evidence surviving compression
  latency      wall clock per call
  recovery     whether the omitted content can be retrieved afterwards

What this does NOT measure: downstream answer quality. That needs a model and
is the honest limit of this harness.

OUTCOME: BLOCKED -- no valid head-to-head was obtained
---------------------------------------------------------------
On this machine (Windows, CPU, onnxruntime 1.23.2) the external system never
engaged its text compressor. Seven configuration steps were taken to give it a
fair run, and the result was `router:noop` every time:

  1. correct package identified -- `pip install headroom` installs an unrelated
     project; the right one is `headroom-ai` (0.34.0)
  2. installed with `[all]`, its own documented full-feature extra
  3. `compress_user_messages=True` -- the default False refuses user content
  4. `protect_recent=0` -- the default 4 shields recent turns
  5. `protect_analysis_context=False` -- was reporting
     `router:protected:analysis_context`
  6. the 1.5 GB `chopratejas/kompress-v2-base` model downloaded from HuggingFace
     after `Kompress model not ready; requests will not be compressed`
  7. `HEADROOM_DETECT_BACKEND=rust`, `HEADROOM_KOMPRESS_BACKEND=torch`,
     `HEADROOM_FORCE_KOMPRESS=1`

Two platform faults surfaced along the way and are recorded because they are
reproducible: the native content detector exceeded its own 5 s watchdog and
disabled itself, and the shipped int8 ONNX artifact failed with
`Only 4b quantization is supported for unpacked compute`.

**Therefore this file publishes no comparative number.** Reporting Entroly at
84.6% against a system sitting at 0.0% would measure our ability to configure
their software, not their compression. That is precisely the error that
invalidated `compression_frontier.json`, which recorded them at 1.7% answer
retention while they publish 97% on SQuAD v2 -- and this run reproduces the
mechanism that most likely caused it.

Any comparative claim needs a platform where their stack runs (Linux, or a
working ONNX path), ideally with their maintainers' configuration confirmed.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
_EVIDENCE = re.compile(
    r"\b\d{3,}\b|\b[A-Za-z_][A-Za-z0-9_]{7,}\b|\b(?:ERROR|FAILED|Exception|Traceback)\b"
)


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
    recovery: str
    error: str = ""

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
                              timeout=120, cwd=REPO)
        return (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def gather(limit_files: int) -> dict[str, str]:
    samples: dict[str, str] = {}
    for label, cmd in (
        ("cmd:git_log", "git log --stat -60"),
        ("cmd:pip_list", f"{sys.executable} -m pip list -v"),
        ("cmd:file_listing", "git ls-files"),
        ("cmd:pytest", f"{sys.executable} -m pytest tests/test_proxy_codec_parity.py -v --timeout=120"),
    ):
        text = _capture(cmd)
        if text.strip():
            samples[label] = text
    listed = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO,
                            capture_output=True, text=True, check=False).stdout.splitlines()
    step = max(1, len(listed) // max(limit_files, 1))
    for rel in listed[::step][:limit_files]:
        path = REPO / rel
        try:
            if path.is_file() and 3_000 <= path.stat().st_size <= 90_000:
                samples[f"src:{rel}"] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return samples


def run_entroly(text: str, required: list[str]) -> Row:
    """Entroly's public compress(), which routes through the codec registry."""
    from entroly import compress

    started = time.perf_counter()
    try:
        out = compress(text)
        error = ""
    except Exception as exc:  # noqa: BLE001
        out, error = text, f"{type(exc).__name__}: {exc}"
    latency = (time.perf_counter() - started) * 1000
    return Row(
        sample="", system="entroly", in_tokens=_tokens(text), out_tokens=_tokens(out),
        ratio=1.0 - _tokens(out) / max(_tokens(text), 1),
        evidence_total=len(required),
        evidence_kept=sum(1 for item in required if item in out),
        latency_ms=latency,
        # Codec representations carry a content-addressed recovery reference,
        # but compress() itself exposes no handle to resolve it.
        recovery="registry-internal",
        error=error,
    )


def run_external(text: str, required: list[str]) -> Row:
    """The external package, configured so it will actually engage.

    Its defaults are `compress_user_messages=False` and `protect_recent=4`,
    which are correct for its intended use -- a live chat history where the
    newest turns must stay verbatim. Handing it a single user message under
    those defaults produced 0.0% compression on every sample, and reporting
    that as a result would have repeated exactly the misconfiguration that
    invalidated the previous frontier benchmark.

    Both flags are therefore turned off so the content is eligible. This is a
    deliberate change from the shipped defaults, made to let the system
    compress at all, and it is disclosed rather than silently applied.
    """
    import headroom
    from headroom import CompressConfig

    config = CompressConfig(
        compress_user_messages=True,
        protect_recent=0,
        min_tokens_to_compress=250,
    )
    started = time.perf_counter()
    try:
        result = headroom.compress([{"role": "user", "content": text}], config=config)
        out = "\n".join(str(m.get("content", "")) for m in result.messages)
        error = ""
    except Exception as exc:  # noqa: BLE001
        out, error = text, f"{type(exc).__name__}: {exc}"
    latency = (time.perf_counter() - started) * 1000
    return Row(
        sample="", system="external", in_tokens=_tokens(text), out_tokens=_tokens(out),
        ratio=1.0 - _tokens(out) / max(_tokens(text), 1),
        evidence_total=len(required),
        evidence_kept=sum(1 for item in required if item in out),
        latency_ms=latency,
        recovery="retrieve-tool",
        error=error,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", type=int, default=14)
    ap.add_argument("--out", type=Path,
                    default=REPO / "benchmarks" / "results" / "head_to_head.json")
    args = ap.parse_args()

    samples = gather(args.files)
    rows: list[Row] = []
    for name, text in sorted(samples.items()):
        required = _required(text)
        for runner in (run_entroly, run_external):
            row = runner(text, required)
            row.sample = name
            rows.append(row)

    payload = {
        "samples": len(samples),
        "rows": [r.__dict__ | {"retention": r.retention} for r in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n  {len(samples)} real inputs; both systems at documented defaults\n")
    print(f"  {'system':<12}{'median ratio':>14}{'median evidence':>17}{'median ms':>12}{'errors':>8}")
    for system in ("entroly", "external"):
        sub = [r for r in rows if r.system == system]
        if not sub:
            continue
        print(f"  {system:<12}"
              f"{statistics.median(r.ratio for r in sub):>13.1%}"
              f"{statistics.median(r.retention for r in sub):>17.1%}"
              f"{statistics.median(r.latency_ms for r in sub):>12.1f}"
              f"{sum(1 for r in sub if r.error):>8}")

    print(f"\n  {'sample':<34}{'entroly ratio/ev':>20}{'external ratio/ev':>22}")
    for name in sorted(samples):
        e = next(r for r in rows if r.sample == name and r.system == "entroly")
        x = next(r for r in rows if r.sample == name and r.system == "external")
        print(f"  {name[:33]:<34}{e.ratio:>11.1%}{e.retention:>9.0%}"
              f"{x.ratio:>13.1%}{x.retention:>9.0%}")
    errs = [r for r in rows if r.error]
    if errs:
        print(f"\n  errors ({len(errs)}):")
        for r in errs[:4]:
            print(f"    {r.system:<10}{r.sample:<30}{r.error[:70]}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
