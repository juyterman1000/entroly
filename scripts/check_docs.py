#!/usr/bin/env python3
"""Validate local links in Entroly's public Markdown and HTML documentation.

The checker is deliberately offline: external destinations are covered by the
separate public-trust verifier, while this script prevents broken repository
paths from reaching users in the first place.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOTS = (
    ROOT,
    ROOT / ".github",
    ROOT / "cookbook",
    ROOT / "docs",
    ROOT / "examples",
)
SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "artifacts",
    "node_modules",
    "target",
    "vendor",
}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_LINK = re.compile(r"(?:href|src)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class BrokenLink:
    source: Path
    line: int
    target: str
    resolved: Path


def public_files() -> list[Path]:
    """Return deterministic, de-duplicated public documentation paths."""
    files: set[Path] = set()
    for base in PUBLIC_ROOTS:
        if not base.exists():
            continue
        if base == ROOT:
            candidates = [*base.glob("*.md"), *base.glob("*.html")]
        else:
            candidates = [*base.rglob("*.md"), *base.rglob("*.html")]
        files.update(
            path
            for path in candidates
            if not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)
        )
    return sorted(files, key=lambda path: path.as_posix().casefold())


def _strip_code_fences(text: str) -> list[str]:
    """Preserve line numbers while hiding links in fenced examples."""
    visible: list[str] = []
    fence_marker: str | None = None
    for line in text.splitlines():
        match = FENCE.match(line)
        if match:
            marker = match.group(1)
            if fence_marker is None:
                fence_marker = marker
            elif marker == fence_marker:
                fence_marker = None
            visible.append("")
        elif fence_marker is None:
            visible.append(line)
        else:
            visible.append("")
    return visible


def _clean_target(raw: str) -> str:
    target = raw.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    elif " " in target:
        # Markdown permits an optional quoted title after the destination.
        target = target.split(maxsplit=1)[0]
    return target.strip()


def resolve_local_link(source: Path, raw_target: str) -> Path | None:
    """Resolve a documentation target, or return None for non-local links."""
    target = _clean_target(raw_target)
    if not target or target.startswith(("#", "//")):
        return None
    if any(token in target for token in ("{{", "}}", "${", "<%")):
        return None

    split = urlsplit(target)
    if split.scheme or split.netloc:
        return None
    path_text = unquote(split.path).replace("\\", "/")
    if not path_text:
        return None

    if path_text.startswith("/entroly/"):
        candidate = ROOT / path_text.removeprefix("/entroly/")
    elif path_text.startswith("/"):
        candidate = ROOT / path_text.removeprefix("/")
    else:
        candidate = source.parent / path_text
    return candidate.resolve()


def check_file(path: Path) -> list[BrokenLink]:
    """Return broken local links for one Markdown or HTML document."""
    text = path.read_text(encoding="utf-8")
    lines = (
        _strip_code_fences(text)
        if path.suffix.casefold() == ".md"
        else text.splitlines()
    )
    pattern = MARKDOWN_LINK if path.suffix.casefold() == ".md" else HTML_LINK
    broken: list[BrokenLink] = []
    for line_number, line in enumerate(lines, start=1):
        for raw_target in pattern.findall(line):
            resolved = resolve_local_link(path, raw_target)
            if resolved is not None and not resolved.exists():
                broken.append(BrokenLink(path, line_number, raw_target, resolved))
    return broken


def check_paths(paths: list[Path]) -> list[BrokenLink]:
    return [broken for path in paths for broken in check_file(path)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", type=Path, help="Optional files to validate"
    )
    args = parser.parse_args(argv)
    paths = [path.resolve() for path in args.paths] if args.paths else public_files()
    missing_inputs = [path for path in paths if not path.is_file()]
    if missing_inputs:
        for path in missing_inputs:
            print(f"input is not a file: {path}", file=sys.stderr)
        return 2

    broken = check_paths(paths)
    if broken:
        for item in broken:
            source = (
                item.source.relative_to(ROOT)
                if item.source.is_relative_to(ROOT)
                else item.source
            )
            print(
                f"{source}:{item.line}: broken local link {item.target!r} -> {item.resolved}"
            )
        print(f"FAILED: {len(broken)} broken local link(s) across {len(paths)} files")
        return 1

    print(f"OK: {len(paths)} public documentation files have valid local links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
