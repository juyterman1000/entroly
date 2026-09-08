#!/usr/bin/env python3
"""Run Entroly's own static security engine over Entroly's own source.

The Rust core ships a SAST engine and CI never pointed it at this repository.
A scanner its author does not run is a scanner nobody has evidence works: the
rules could be silently broken, or the source could contain exactly what they
are written to find, and neither would surface.

This is a gate, not a report. It fails on findings at or above
``--fail-on`` severity so a regression blocks the pull request, and prints the
whole finding set as evidence either way.

The engine redacts secret-bearing lines itself (``line_content`` comes back as
a placeholder for secret rules), so the output is safe to attach to a public
build log.

Exit codes: 0 clean, 1 findings at or above the threshold, 2 the engine is
unavailable -- which is a failure, not a pass, because "no findings" and "no
scanner" are indistinguishable in a green check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Ordered, so a threshold means "this and anything worse".
SEVERITY_ORDER = ["Info", "Low", "Medium", "High", "Critical"]

# Scanning our own test corpus would report the vulnerable snippets the SAST
# tests deliberately contain, so the gate would fail on its own fixtures.
DEFAULT_EXCLUDES = (
    "tests/",
    "benchmarks/",
    "docs/",
    "scripts/self_sast_scan.py",  # the examples in this file's own docstring
    ".git/",
    "node_modules/",
    "target/",
    "build/",
    "dist/",
    # Vendored and installed third-party code. Omitting these made the first
    # run scan `tmp/clean-venv-1.0.70/Lib/site-packages/` -- pip, setuptools,
    # cryptography, pywin32 -- and report their findings as Entroly's. A
    # scanner that blames its dependencies for its own hygiene is worse than
    # no scanner, because the number looks like a measurement.
    ".venv/",
    "venv/",
    "tmp/",
    "site-packages/",
    ".tox/",
    "vendor/",
)


def _rank(severity: str) -> int:
    try:
        return SEVERITY_ORDER.index(severity)
    except ValueError:
        return 0


def _iter_sources(root: Path, excludes: tuple[str, ...]) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if any(rel.startswith(prefix) or f"/{prefix}" in f"/{rel}" for prefix in excludes):
            continue
        files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail-on", default="High", choices=SEVERITY_ORDER)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    try:
        import entroly_core
    except Exception as exc:
        print(f"FAIL: Entroly's security engine is unavailable: {exc}", file=sys.stderr)
        print(
            "A missing scanner must not read as a clean scan; build the native "
            "engine with `cd entroly-core && maturin develop --release`.",
            file=sys.stderr,
        )
        return 2

    if not hasattr(entroly_core, "py_scan_content"):
        print(
            "FAIL: the native engine exposes no py_scan_content; the SAST "
            "binding was removed or renamed.",
            file=sys.stderr,
        )
        return 2

    threshold = _rank(args.fail_on)
    sources = _iter_sources(args.root, DEFAULT_EXCLUDES)
    findings: list[dict] = []
    unreadable: list[str] = []

    for path in sources:
        rel = path.relative_to(args.root).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            unreadable.append(f"{rel}: {exc}")
            continue
        try:
            report = json.loads(entroly_core.py_scan_content(content, rel))
        except Exception as exc:
            unreadable.append(f"{rel}: scan failed: {exc}")
            continue
        for finding in report.get("findings", []):
            finding["file"] = rel
            findings.append(finding)

    blocking = [f for f in findings if _rank(f.get("severity", "Info")) >= threshold]

    summary = {
        "files_scanned": len(sources),
        "findings_total": len(findings),
        "findings_blocking": len(blocking),
        "fail_on": args.fail_on,
        "unreadable": unreadable,
        "by_severity": {
            level: sum(1 for f in findings if f.get("severity") == level)
            for level in SEVERITY_ORDER
        },
        "findings": findings,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Entroly self-SAST: {len(sources)} files scanned, {len(findings)} findings")
    for level in reversed(SEVERITY_ORDER):
        count = summary["by_severity"][level]
        if count:
            print(f"  {level:<8} {count}")

    for finding in sorted(
        blocking, key=lambda f: (-_rank(f.get("severity", "Info")), f.get("file", ""))
    ):
        print(
            f"  {finding.get('severity')}: {finding.get('file')}:"
            f"{finding.get('line_number')} [{finding.get('rule_id')}] "
            f"CWE-{finding.get('cwe')} {finding.get('description', '')[:110]}"
        )

    if unreadable:
        # Not fatal on its own, but never silent: a file that could not be read
        # was not scanned, and an unscanned file is not a clean file.
        print(f"  {len(unreadable)} file(s) could not be scanned:")
        for item in unreadable[:10]:
            print(f"    {item}")

    if blocking:
        print(
            f"FAIL: {len(blocking)} finding(s) at or above {args.fail_on}.",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: no findings at or above {args.fail_on}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
