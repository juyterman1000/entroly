#!/usr/bin/env python3
"""Compatibility entry point for Entroly public trust verification.

The canonical verifier rejects positive guarantees while allowing an explicit
statement that Entroly does *not* promise a guaranteed bill reduction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import verify_context_assurance_public as _impl
except ImportError:  # Direct execution: python scripts/verify_public_trust.py
    import verify_context_assurance_public as _impl

PROMINENT_PUBLIC_FILES = _impl.PROMINENT_PUBLIC_FILES
_collect_prism_r_public_failures = _impl._collect_prism_r_public_failures
_collect_stale_public_claim_failures = _impl._collect_stale_public_claim_failures
collect_online_failures = _impl.collect_online_failures
collect_published_version_failures = _impl.collect_published_version_failures

_FALSE_POSITIVE_GUARANTEE = (
    "README/PyPI contains forbidden promise 'guaranteed bill reduction.'"
)
_EXPLICIT_NON_GUARANTEE = (
    "does not promise a universal compression percentage or guaranteed bill reduction"
)


def collect_offline_failures() -> list[str]:
    """Return canonical failures without misclassifying an explicit disclaimer."""

    failures = _impl.collect_offline_failures()
    root = Path(__file__).resolve().parents[1]
    public_copy = "\n".join(
        (root / path).read_text(encoding="utf-8").casefold()
        for path in ("README.md", "PYPI_README.md")
    )
    if _EXPLICIT_NON_GUARANTEE in public_copy:
        failures = [
            failure
            for failure in failures
            if failure != _FALSE_POSITIVE_GUARANTEE
        ]
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--require-published-version", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    failures = collect_offline_failures()
    if args.online:
        failures.extend(
            collect_online_failures(retries=args.retries, timeout=args.timeout)
        )
    if args.require_published_version:
        failures.extend(collect_published_version_failures(timeout=args.timeout))

    if failures:
        print("Public trust verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    scopes = ["offline Context Assurance contracts"]
    if args.online:
        scopes.append("online destinations")
    if args.require_published_version:
        scopes.append("published-version parity")
    print("Public trust verification passed: " + ", ".join(scopes))
    return 0


__all__ = [
    "PROMINENT_PUBLIC_FILES",
    "_collect_prism_r_public_failures",
    "_collect_stale_public_claim_failures",
    "collect_offline_failures",
    "collect_online_failures",
    "collect_published_version_failures",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
