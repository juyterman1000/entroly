"""
`entroly verify-code` — CLI surface for the hallucination verifiers
====================================================================

Distinct from `entroly verify`, which runs a verification pass on the
vault's belief artifacts. This module verifies *generated source code*
against a SymbolVerifier built from the local repository — the symbol
manifest + n-gram surprisal model that powers BIPT-style hallucination
detection at the identifier level.

Usage::

    # Verify code from a file
    entroly verify-code path/to/generated.py

    # Verify from stdin (e.g. piped from an LLM)
    cat generated.py | entroly verify-code -

    # JSON output for programmatic use
    entroly verify-code generated.py --json

    # Lower the strictness (more permissive)
    entroly verify-code generated.py --lambda 8.0

    # Force rebuild the manifest cache
    entroly verify-code generated.py --rebuild
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from .cache import load_or_build_verifier
from .symbol_resolution import DEFAULT_LAMBDA


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Returns process exit code.

    Exit codes:
        0 — code passed verification (H < threshold)
        1 — code FAILED verification (H >= threshold)  → for CI gating
        2 — usage error
    """
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args is None:
        return 2
    if args.get("help"):
        return 0

    # Read source
    source = _read_source(args["path"])
    if source is None:
        print(f"error: could not read {args['path']!r}", file=sys.stderr)
        return 2

    # Load (or build) verifier with caching
    t0 = time.time()
    verifier, meta = load_or_build_verifier(
        repo_root=args["repo"],
        lambda_calibration=args["lambda"],
        force_rebuild=args["rebuild"],
    )
    load_ms = (time.time() - t0) * 1000

    # Verify
    t1 = time.time()
    result = verifier.verify(source)
    verify_ms = (time.time() - t1) * 1000

    if args["json"]:
        out = {
            "verdict": "pass" if result.passed(args["threshold"]) else "hallucinated",
            "h_score": result.h_score,
            "threshold": args["threshold"],
            "n_resolved": result.n_resolved,
            "n_unresolved": result.n_unresolved,
            "n_suspicious": result.n_suspicious,
            "manifest_size": result.manifest_size,
            "load_ms": round(load_ms, 1),
            "verify_ms": round(verify_ms, 1),
            "judgments": [
                {
                    "name": j.ref.name,
                    "kind": j.ref.kind,
                    "line": j.ref.line,
                    "resolved": j.resolved,
                    "provenance": j.provenance,
                    "surprisal": round(j.surprisal, 4),
                    "p_hallucinated": round(j.p_hallucinated, 4),
                    "context": j.ref.context,
                }
                for j in result.judgments
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print(result.explain(max_items=args["max_items"]))
        print()
        print(f"load:   {load_ms:6.1f} ms ({'cache hit' if load_ms < 500 else 'cold build'})")
        print(f"verify: {verify_ms:6.1f} ms")

    return 0 if result.passed(args["threshold"]) else 1


def _parse_args(argv: list[str]) -> dict[str, Any] | None:
    args: dict[str, Any] = {
        "path": None,
        "repo": ".",
        "lambda": DEFAULT_LAMBDA,
        "threshold": 0.5,
        "json": False,
        "rebuild": False,
        "max_items": 30,
        "help": False,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help"):
            _print_help()
            args["help"] = True
            return args
        elif a == "--repo":
            i += 1
            args["repo"] = argv[i]
        elif a == "--lambda":
            i += 1
            args["lambda"] = float(argv[i])
        elif a == "--threshold":
            i += 1
            args["threshold"] = float(argv[i])
        elif a == "--json":
            args["json"] = True
        elif a == "--rebuild":
            args["rebuild"] = True
        elif a == "--max-items":
            i += 1
            args["max_items"] = int(argv[i])
        elif a.startswith("--"):
            print(f"unknown flag: {a}", file=sys.stderr)
            return None
        else:
            args["path"] = a
        i += 1

    if args["path"] is None:
        print("error: missing path argument", file=sys.stderr)
        _print_help()
        return None
    return args


def _read_source(path: str) -> str | None:
    if path == "-":
        return sys.stdin.read()
    p = Path(path)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _print_help() -> None:
    print(__doc__)
    print()
    print("Options:")
    print("  --repo PATH        Root of the codebase to ground against (default: cwd)")
    print("  --lambda FLOAT     Surprisal threshold offset (default: 6.5 nats/char)")
    print("  --threshold FLOAT  H(code) decision threshold (default: 0.5)")
    print("  --json             Emit JSON instead of human-readable report")
    print("  --rebuild          Force rebuild of the manifest/n-gram cache")
    print("  --max-items N      Max symbols to show in the report (default: 30)")
    print("  -h, --help         This help")


if __name__ == "__main__":
    sys.exit(main())
