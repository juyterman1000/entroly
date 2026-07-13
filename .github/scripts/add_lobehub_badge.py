#!/usr/bin/env python3
"""Add the exact LobeHub ownership badge to the primary README."""

from pathlib import Path


README = Path("README.md")
BADGE = (
    '  <a href="https://lobehub.com/mcp/juyterman1000-entroly">'
    '<img src="https://lobehub.com/badge/mcp/juyterman1000-entroly" '
    'alt="Entroly on LobeHub"></a>\n'
)
ANCHOR = '  <img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="PyPI">\n'


def main() -> int:
    text = README.read_text(encoding="utf-8")
    if BADGE.strip() in text:
        print("LobeHub badge already present")
        return 0
    if text.count(ANCHOR) != 1:
        raise SystemExit("expected exactly one README badge anchor")
    README.write_text(text.replace(ANCHOR, ANCHOR + BADGE, 1), encoding="utf-8")
    print("Added LobeHub ownership badge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
