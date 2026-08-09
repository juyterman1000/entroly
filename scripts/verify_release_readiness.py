#!/usr/bin/env python3
"""Fail a release when tag, tree, version, or native capabilities are stale."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from entroly import __version__  # noqa: E402
from entroly.native_status import RELEASE_NATIVE_SYMBOLS, native_status  # noqa: E402


def _git(*args: str) -> tuple[int, str]:
    process = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return process.returncode, process.stdout.strip()


def release_readiness(*, development: bool = False) -> list[str]:
    failures: list[str] = []
    tag_code, tag = _git("describe", "--tags", "--exact-match", "HEAD")
    expected = {f"entroly-v{__version__}", f"v{__version__}"}
    if tag_code != 0 or tag not in expected:
        failures.append(
            f"HEAD is not the exact release tag for {__version__} "
            f"(expected one of {sorted(expected)!r}, got {tag or 'untagged'})"
        )

    _, dirty = _git("status", "--porcelain")
    if dirty:
        failures.append("worktree is dirty; release artifacts would not identify one commit")

    native = native_status(RELEASE_NATIVE_SYMBOLS)
    if not native.available:
        failures.append(f"native wheel unavailable: {native.error or 'not installed'}")
    else:
        if native.version != __version__:
            failures.append(
                f"native wheel version {native.version or 'unknown'} != Python {__version__}"
            )
        if native.missing_symbols:
            failures.append(
                "native wheel is capability-stale despite its version: missing "
                + ", ".join(native.missing_symbols)
            )
        if native.version_ok is False:
            failures.append(f"native wheel {native.version} is below the declared minimum")

    if development:
        return failures
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development",
        action="store_true",
        help="report blockers but return success for an intentionally untagged checkout",
    )
    args = parser.parse_args()
    failures = release_readiness(development=args.development)
    if failures:
        label = "DEVELOPMENT BLOCKERS" if args.development else "RELEASE READINESS FAILED"
        print(f"{label} ({len(failures)})")
        for failure in failures:
            print(f"  - {failure}")
        return 0 if args.development else 1
    print(f"release readiness OK - {__version__} tag and native capabilities align")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
