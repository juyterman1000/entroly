"""Bounded command-line surface for repository intelligence."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from . import build_repository_index
from .graph import analyze_change_impact, localize_tests
from .models import RepositoryLimits, normalize_relative

CLI_SCHEMA_VERSION = "entroly.repository-cli.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m entroly.repository_intelligence",
        description=(
            "Build a bounded local symbol graph, inspect change impact, and "
            "rank relevant tests. Output is deterministic JSON."
        ),
    )
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--max-files", type=int, default=20_000)
    parser.add_argument("--max-total-mb", type=int, default=256)
    parser.add_argument("--max-file-mb", type=int, default=2)
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("summary", help="report bounded index counts")

    impact = subcommands.add_parser("impact", help="find reverse change impact")
    impact.add_argument("--changed", action="append", required=True)
    impact.add_argument("--max-depth", type=int, default=4)
    impact.add_argument("--limit", type=int, default=5_000)

    tests = subcommands.add_parser("tests", help="rank tests for changed files")
    tests.add_argument("--changed", action="append", required=True)
    tests.add_argument("--limit", type=int, default=20)
    return parser


def _limits(args: argparse.Namespace) -> RepositoryLimits:
    return RepositoryLimits(
        max_files=args.max_files,
        max_total_bytes=args.max_total_mb * 1024 * 1024,
        max_file_bytes=args.max_file_mb * 1024 * 1024,
    )


def _changed(args: argparse.Namespace, known: set[str]) -> tuple[list[str], list[str]]:
    requested = sorted({normalize_relative(value) for value in args.changed})
    unknown = [path for path in requested if path not in known]
    return [path for path in requested if path in known], unknown


def _summary(index) -> dict[str, object]:
    languages = Counter(record.language for record in index.files.values())
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "command": "summary",
        "root": index.root,
        "files": len(index.files),
        "symbols": len(index.symbols),
        "call_edges": len(index.call_edges),
        "file_edges": sum(len(values) for values in index.file_dependencies.values()),
        "tests": len(index.test_paths),
        "languages": dict(sorted(languages.items())),
        "diagnostics": list(index.diagnostics),
    }


def run(argv: Sequence[str] | None = None) -> tuple[int, dict[str, object]]:
    args = _parser().parse_args(argv)
    try:
        root = Path(args.root).expanduser().resolve(strict=True)
        index = build_repository_index(root, limits=_limits(args))
    except (OSError, ValueError) as exc:
        return 2, {
            "schema_version": CLI_SCHEMA_VERSION,
            "error": "invalid_repository",
            "detail": str(exc),
        }

    if args.command == "summary":
        return 0, _summary(index)

    changed, unknown = _changed(args, set(index.files))
    if unknown:
        return 2, {
            "schema_version": CLI_SCHEMA_VERSION,
            "error": "unknown_changed_paths",
            "unknown": unknown,
            "diagnostics": list(index.diagnostics),
        }

    if args.command == "impact":
        report = analyze_change_impact(
            index,
            changed,
            max_depth=args.max_depth,
            max_impacted_paths=args.limit,
        )
        return 0, {
            "schema_version": CLI_SCHEMA_VERSION,
            "command": "impact",
            "report": report.to_dict(),
            "diagnostics": list(index.diagnostics),
        }

    candidates = localize_tests(index, changed, limit=args.limit)
    return 0, {
        "schema_version": CLI_SCHEMA_VERSION,
        "command": "tests",
        "changed_paths": changed,
        "candidates": [candidate.to_dict() for candidate in candidates],
        "diagnostics": list(index.diagnostics),
    }


def main(argv: Sequence[str] | None = None) -> int:
    code, payload = run(argv)
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return code
