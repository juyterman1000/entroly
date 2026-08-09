"""Bounded command-line surface for repository intelligence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .models import RepositoryLimits
from .service import RepositoryIntelligenceError, RepositoryIntelligenceService

CLI_SCHEMA_VERSION = "entroly.repository-cli.v2"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m entroly.repository_intelligence",
        description=(
            "Build a bounded local symbol graph, inspect change impact, and "
            "emit receipt-backed task context. Output is deterministic JSON."
        ),
    )
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--max-files", type=int, default=20_000)
    parser.add_argument("--max-total-mb", type=int, default=256)
    parser.add_argument("--max-file-mb", type=int, default=2)
    parser.add_argument(
        "--cache-dir",
        help="opt-in content-addressed parse cache directory",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("summary", help="report bounded index counts")

    impact = subcommands.add_parser("impact", help="find reverse change impact")
    impact.add_argument("--changed", action="append", required=True)
    impact.add_argument("--max-depth", type=int, default=4)
    impact.add_argument("--limit", type=int, default=5_000)

    tests = subcommands.add_parser("tests", help="rank tests for changed files")
    tests.add_argument("--changed", action="append", required=True)
    tests.add_argument("--limit", type=int, default=20)

    context = subcommands.add_parser(
        "context",
        help="build a verified, budgeted partial graph for a task",
    )
    context.add_argument("--query", required=True)
    context.add_argument("--token-budget", type=int, default=2_000)
    context.add_argument("--max-hops", type=int, default=2)
    context.add_argument("--max-fragments", type=int, default=24)
    context.add_argument("--include-history", action="store_true")
    context.add_argument("--max-history-commits", type=int, default=20)

    repository_map = subcommands.add_parser(
        "map",
        help="rank a verified whole-repository map under a token budget",
    )
    repository_map.add_argument("--query", default="")
    repository_map.add_argument("--token-budget", type=int, default=2_000)
    repository_map.add_argument("--max-entries", type=int, default=100)

    graph = subcommands.add_parser(
        "graph",
        help="trace freshness-checked static calls for an unambiguous symbol",
    )
    graph.add_argument("--symbol", required=True)
    graph.add_argument(
        "--direction",
        choices=("callers", "callees", "both"),
        default="both",
    )
    graph.add_argument("--max-depth", type=int, default=3)
    graph.add_argument("--limit", type=int, default=200)

    program = subcommands.add_parser(
        "program",
        help="build verified control and reaching-definition flow for a symbol",
    )
    program.add_argument("--symbol", required=True)
    program.add_argument("--limit", type=int, default=1_000)

    runtime = subcommands.add_parser(
        "runtime",
        help="bind bounded trace events to verified source locations",
    )
    runtime.add_argument("--events-json", required=True)
    runtime.add_argument("--producer", default="external-trace")
    runtime.add_argument("--max-events", type=int, default=100_000)

    semantic = subcommands.add_parser(
        "semantic",
        help="verify external LSP/compiler relationship locations",
    )
    semantic.add_argument("--relationships-json", required=True)
    semantic.add_argument("--provider", required=True)
    semantic.add_argument("--max-relationships", type=int, default=100_000)
    return parser


def _limits(args: argparse.Namespace) -> RepositoryLimits:
    return RepositoryLimits(
        max_files=args.max_files,
        max_total_bytes=args.max_total_mb * 1024 * 1024,
        max_file_bytes=args.max_file_mb * 1024 * 1024,
    )


def run(argv: Sequence[str] | None = None) -> tuple[int, dict[str, object]]:
    args = _parser().parse_args(argv)
    try:
        root = Path(args.root).expanduser().resolve(strict=True)
        service = RepositoryIntelligenceService(
            root,
            limits=_limits(args),
            cache_dir=args.cache_dir,
        )
        if args.command == "summary":
            payload = service.summary()
            payload["command"] = "summary"
            payload["schema_version"] = CLI_SCHEMA_VERSION
            return 0, payload
        if args.command == "impact":
            payload = service.impact(
                args.changed,
                max_depth=args.max_depth,
                limit=args.limit,
            )
            payload["command"] = "impact"
            payload["schema_version"] = CLI_SCHEMA_VERSION
            return 0, payload
        if args.command == "context":
            payload = service.context(
                args.query,
                token_budget=args.token_budget,
                max_hops=args.max_hops,
                max_fragments=args.max_fragments,
                include_history=args.include_history,
                max_history_commits=args.max_history_commits,
            )
            payload["command"] = "context"
            return 0, payload
        if args.command == "graph":
            payload = service.symbol_graph(
                args.symbol,
                direction=args.direction,
                max_depth=args.max_depth,
                limit=args.limit,
            )
            payload["command"] = "graph"
            return 0, payload
        if args.command == "map":
            payload = service.repository_map(
                args.query,
                token_budget=args.token_budget,
                max_entries=args.max_entries,
            )
            payload["command"] = "map"
            return 0, payload
        if args.command == "program":
            payload = service.program_graph(args.symbol, limit=args.limit)
            payload["command"] = "program"
            return 0, payload
        if args.command == "runtime":
            events_path = Path(args.events_json).expanduser().resolve(strict=True)
            if events_path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("events JSON must be at most 16 MiB")
            events = json.loads(events_path.read_text(encoding="utf-8"))
            if not isinstance(events, list):
                raise ValueError("events JSON must contain an array")
            payload = service.runtime_overlay(
                events,
                producer=args.producer,
                max_events=args.max_events,
            )
            payload["command"] = "runtime"
            return 0, payload
        if args.command == "semantic":
            relationships_path = Path(args.relationships_json).expanduser().resolve(strict=True)
            if relationships_path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("relationships JSON must be at most 16 MiB")
            relationships = json.loads(relationships_path.read_text(encoding="utf-8"))
            if not isinstance(relationships, list):
                raise ValueError("relationships JSON must contain an array")
            payload = service.semantic_overlay(
                relationships,
                provider=args.provider,
                max_relationships=args.max_relationships,
            )
            payload["command"] = "semantic"
            return 0, payload
        payload = service.tests(args.changed, limit=args.limit)
        payload["command"] = "tests"
        payload["schema_version"] = CLI_SCHEMA_VERSION
        return 0, payload
    except RepositoryIntelligenceError as exc:
        payload = exc.to_dict()
        payload["schema_version"] = CLI_SCHEMA_VERSION
        return 2, payload
    except (OSError, ValueError) as exc:
        return 2, {
            "schema_version": CLI_SCHEMA_VERSION,
            "error": "invalid_repository",
            "detail": str(exc),
        }


def main(argv: Sequence[str] | None = None) -> int:
    code, payload = run(argv)
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return code
