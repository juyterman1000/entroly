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

    graph_query = subcommands.add_parser(
        "query",
        help="query verified file/symbol neighbors, paths, relatedness, or impact",
    )
    graph_query.add_argument("--query", required=True)
    graph_query.add_argument(
        "--operation",
        choices=("explain", "neighbors", "path", "related", "impact"),
        default="neighbors",
    )
    graph_query.add_argument("--target")
    graph_query.add_argument(
        "--direction",
        choices=("incoming", "outgoing", "both"),
        default="both",
    )
    graph_query.add_argument("--max-depth", type=int, default=4)
    graph_query.add_argument("--limit", type=int, default=100)
    graph_query.add_argument("--max-visited", type=int, default=10_000)

    program = subcommands.add_parser(
        "program",
        help="build verified control and reaching-definition flow for a symbol",
    )
    program.add_argument("--symbol", required=True)
    program.add_argument("--limit", type=int, default=1_000)

    health = subcommands.add_parser(
        "health",
        help="audit verified complexity, cycles, coupling, and navigability",
    )
    health.add_argument("--max-findings", type=int, default=500)
    health.add_argument("--max-symbols", type=int, default=2_000)

    architecture = subcommands.add_parser(
        "architecture",
        help="build verified layers, communities, cycles, routes, and hotspots",
    )
    architecture.add_argument("--max-components", type=int, default=5_000)
    architecture.add_argument("--max-communities", type=int, default=1_000)
    architecture.add_argument("--max-cycles", type=int, default=1_000)
    architecture.add_argument("--max-dependency-edges", type=int, default=100_000)
    architecture.add_argument("--max-hotspots", type=int, default=100)
    architecture.add_argument("--max-routes", type=int, default=100)

    architecture_diff = subcommands.add_parser(
        "architecture-diff",
        help="compare two committed architecture JSON snapshots",
    )
    architecture_diff.add_argument("--before-json", required=True)
    architecture_diff.add_argument("--after-json", required=True)
    architecture_diff.add_argument("--limit", type=int, default=5_000)

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

    rename_preview = subcommands.add_parser(
        "rename-preview",
        help="preview an exact two-phase rename without writing files",
    )
    rename_preview.add_argument("--symbol", required=True)
    rename_preview.add_argument("--new-name", required=True)
    rename_preview.add_argument("--semantic-json")
    rename_preview.add_argument("--provider", default="none")
    rename_preview.add_argument("--max-changes", type=int, default=10_000)

    rename_apply = subcommands.add_parser(
        "rename-apply",
        help="apply a committed rename plan after explicit risk acknowledgement",
    )
    rename_apply.add_argument("--plan-json", required=True)
    rename_apply.add_argument("--expected-plan-sha", required=True)
    rename_apply.add_argument("--acknowledge-incomplete", action="store_true")

    lsp_preview = subcommands.add_parser(
        "lsp-rename-preview",
        help="run an explicitly configured LSP and build a no-write rename plan",
    )
    lsp_preview.add_argument("--symbol", required=True)
    lsp_preview.add_argument("--new-name", required=True)
    lsp_preview.add_argument("--language-id", required=True)
    lsp_preview.add_argument("--command-json", required=True)
    lsp_preview.add_argument("--timeout-seconds", type=float, default=15.0)
    lsp_preview.add_argument("--max-relationships", type=int, default=10_000)
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
        if args.command == "query":
            payload = service.graph_query(
                args.query,
                operation=args.operation,
                target_query=args.target,
                direction=args.direction,
                max_depth=args.max_depth,
                limit=args.limit,
                max_visited=args.max_visited,
            )
            payload["command"] = "query"
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
        if args.command == "health":
            payload = service.code_health(
                max_findings=args.max_findings,
                max_symbols=args.max_symbols,
            )
            payload["command"] = "health"
            return 0, payload
        if args.command == "architecture":
            payload = service.architecture(
                max_components=args.max_components,
                max_communities=args.max_communities,
                max_cycles=args.max_cycles,
                max_dependency_edges=args.max_dependency_edges,
                max_hotspots=args.max_hotspots,
                max_routes=args.max_routes,
            )
            payload["command"] = "architecture"
            return 0, payload
        if args.command == "architecture-diff":
            inputs: list[dict[str, object]] = []
            for raw_path in (args.before_json, args.after_json):
                input_path = Path(raw_path).expanduser().resolve(strict=True)
                if input_path.stat().st_size > 64 * 1024 * 1024:
                    raise ValueError("architecture JSON must be at most 64 MiB")
                raw_payload = json.loads(input_path.read_text(encoding="utf-8"))
                if not isinstance(raw_payload, dict):
                    raise ValueError("architecture JSON must contain an object")
                inputs.append(raw_payload)
            payload = service.architecture_diff(
                inputs[0], inputs[1], limit=args.limit
            )
            payload["command"] = "architecture-diff"
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
        if args.command == "rename-preview":
            relationships: list[dict[str, object]] = []
            if args.semantic_json:
                semantic_path = Path(args.semantic_json).expanduser().resolve(strict=True)
                if semantic_path.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError("semantic JSON must be at most 16 MiB")
                raw_relationships = json.loads(semantic_path.read_text(encoding="utf-8"))
                if not isinstance(raw_relationships, list) or any(
                    not isinstance(item, dict) for item in raw_relationships
                ):
                    raise ValueError("semantic JSON must contain an array of objects")
                relationships = raw_relationships
            payload = service.rename_preview(
                args.symbol,
                args.new_name,
                semantic_relationships=relationships,
                provider=args.provider,
                max_changes=args.max_changes,
            )
            payload["command"] = "rename-preview"
            return 0, payload
        if args.command == "rename-apply":
            plan_path = Path(args.plan_json).expanduser().resolve(strict=True)
            if plan_path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("plan JSON must be at most 16 MiB")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if not isinstance(plan, dict):
                raise ValueError("plan JSON must contain an object")
            payload = service.rename_apply(
                plan,
                expected_plan_sha256=args.expected_plan_sha,
                acknowledge_incomplete=args.acknowledge_incomplete,
            )
            payload["command"] = "rename-apply"
            return 0, payload
        if args.command == "lsp-rename-preview":
            command_path = Path(args.command_json).expanduser().resolve(strict=True)
            if command_path.stat().st_size > 64 * 1024:
                raise ValueError("LSP command JSON must be at most 64 KiB")
            command = json.loads(command_path.read_text(encoding="utf-8"))
            if (
                not isinstance(command, list)
                or not 1 <= len(command) <= 32
                or any(
                    not isinstance(item, str) or not item or len(item) > 4096
                    for item in command
                )
            ):
                raise ValueError("LSP command JSON must contain 1 to 32 strings")
            payload = service.lsp_rename_preview(
                args.symbol,
                args.new_name,
                command=command,
                language_id=args.language_id,
                timeout_seconds=args.timeout_seconds,
                max_relationships=args.max_relationships,
            )
            payload["command"] = "lsp-rename-preview"
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
