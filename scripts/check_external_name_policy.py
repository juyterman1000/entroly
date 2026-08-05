#!/usr/bin/env python3
"""Fail when prohibited external product names enter the current tree.

Names are represented only by SHA-256 digests. This keeps the policy itself
brand-neutral while detecting plain, hyphenated, underscored, URL, package, and
identifier forms after alphanumeric normalization.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROHIBITED = {
    8: {"26e93d81a9553eabd165301cad992369094b6a1759a62e94a998a94aa5315902"},
    7: {"d8a564a233ed75c8d55c193f8a56b5937cb2b5dec3b3566fa0537f7fa434dca7"},
}
SKIP_PARTS = {".git", ".venv", "node_modules", "target", "__pycache__"}


def normalized(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def violations() -> list[str]:
    found: list[str] = []
    for path in sorted(candidate for candidate in ROOT.rglob("*") if candidate.is_file()):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            value = normalized(line)
            for length, digests in PROHIBITED.items():
                if len(value) < length:
                    continue
                matched = False
                for index in range(len(value) - length + 1):
                    digest = hashlib.sha256(
                        value[index : index + length].encode()
                    ).hexdigest()
                    if digest in digests:
                        found.append(f"{relative}:{line_number}")
                        matched = True
                        break
                if matched:
                    break
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
