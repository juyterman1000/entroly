#!/usr/bin/env python3
"""Fail when prohibited external product names enter the current tree.

Names are represented only by SHA-256 digests. This keeps the policy itself
brand-neutral while detecting plain, hyphenated, underscored, URL, package,
identifier, filename, and directory forms after alphanumeric normalization.
"""

from __future__ import annotations

import hashlib
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROHIBITED = {
    8: {"26e93d81a9553eabd165301cad992369094b6a1759a62e94a998a94aa5315902"},
    7: {"d8a564a233ed75c8d55c193f8a56b5937cb2b5dec3b3566fa0537f7fa434dca7"},
}
# The policy governs what this project *authors and ships*. Scanning the whole
# working tree also swept vendored dependencies, generated state and local
# machine config, which produced violations nobody can act on -- a downloaded
# third-party README, a belief file the vault wrote about a source file, an old
# release installed under `tmp/`, and the developer's own settings.
#
# Those are also why the check took 3m47s and therefore always died inside the
# 60s subprocess timeout in `tests/test_docs_code_sync.py`, reporting
# `TimeoutExpired` instead of the violation list. A gate that cannot distinguish
# "policy broken" from "machine slow" fails opaquely under CI load.
SKIP_PARTS = {
    ".git",
    ".venv",
    "node_modules",
    "target",
    "__pycache__",
    # generated or cached state, rewritten by tooling
    ".entroly",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    "htmlcov",
    # build outputs -- derived from source, so clean when source is clean
    "build",
    "dist",
    # scratch and vendored third-party trees we neither author nor ship
    "tmp",
    "site-packages",
    # local developer configuration, never published
    ".claude",
}


def normalized(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


@lru_cache(maxsize=1 << 20)
def _window_hit(window: str) -> bool:
    """Whether one fixed-length window is a prohibited name.

    Memoised because the same short windows recur constantly across a source
    tree -- ordinary words, indentation runs, repeated identifiers. Digesting
    every window of every line unmemoised was the dominant cost, and it is pure,
    so caching cannot change a verdict.
    """
    digests = PROHIBITED.get(len(window))
    if not digests:
        return False
    return hashlib.sha256(window.encode()).hexdigest() in digests


@lru_cache(maxsize=1 << 18)
def matches_prohibited(value: str) -> bool:
    value = normalized(value)
    for length in PROHIBITED:
        if len(value) < length:
            continue
        # Distinct windows only: a line repeating a token digests it once.
        for window in {value[i : i + length] for i in range(len(value) - length + 1)}:
            if _window_hit(window):
                return True
    return False


def _scannable_files() -> list[Path]:
    """Files under ROOT, pruning skipped directories instead of descending them.

    `ROOT.rglob("*")` walked 49,273 files and then discarded all but 1,339,
    because it still descends `.git`, `node_modules`, `target` and vendored
    trees before the filter runs. Pruning at the directory level skips those
    subtrees entirely.
    """
    found: list[Path] = []
    stack = [ROOT]
    while stack:
        directory = stack.pop()
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name in SKIP_PARTS:
                continue
            if entry.is_dir():
                stack.append(entry)
            elif entry.is_file():
                found.append(entry)
    return sorted(found)


def violations() -> list[str]:
    found: list[str] = []
    for path in _scannable_files():
        relative = path.relative_to(ROOT)
        if matches_prohibited(relative.as_posix()):
            found.append(f"{relative}:path")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if matches_prohibited(line):
                found.append(f"{relative}:{line_number}")
    return found


def main() -> int:
    found = violations()
    if found:
        print("prohibited external product name found in current tree:", file=sys.stderr)
        for location in found:
            print(f"- {location}", file=sys.stderr)
        return 1
    print("external-name policy check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
