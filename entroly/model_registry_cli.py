"""Command-line inspection for Entroly model intelligence."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .model_discovery import discover_ollama_models
from .model_registry import (
    ModelRegistryError,
    ModelSpec,
    UnknownModelError,
    default_registry,
    resolve_model,
)


def _spec_to_dict(spec: ModelSpec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "provider": spec.provider,
        "context_window": spec.context_window,
        "max_output_tokens": spec.max_output_tokens,
        "supports_tools": spec.supports_tools,
        "supports_vision": spec.supports_vision,
        "supports_reasoning": spec.supports_reasoning,
        "reasoning_levels": list(spec.reasoning_levels),
        "source": spec.source,
        "verified_at": spec.verified_at,
        "confidence": spec.confidence,
        "trust": spec.trust,
        "status": spec.status,
        "local": spec.local,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="entroly-models",
        description="Inspect Entroly's provenance-aware model intelligence registry.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser("resolve", help="Resolve one model")
    resolve_parser.add_argument("model")
    resolve_parser.add_argument("--provider")
    resolve_parser.add_argument("--strict", action="store_true")
    resolve_parser.add_argument("--json", action="store_true")

    list_parser = subparsers.add_parser("list", help="List known models")
    list_parser.add_argument("--provider")
    list_parser.add_argument("--json", action="store_true")

    subparsers.add_parser("fingerprint", help="Print active registry fingerprint")

    discover_parser = subparsers.add_parser(
        "discover-ollama", help="Discover local Ollama models"
    )
    discover_parser.add_argument("--base-url")
    discover_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "resolve":
            resolution = resolve_model(
                args.model,
                provider=args.provider,
                unknown_policy="error" if args.strict else None,
            )
            if args.json:
                print(json.dumps(resolution.to_dict(), indent=2, sort_keys=True))
            else:
                marker = "verified" if resolution.verified else "unverified"
                canonical = resolution.canonical_id or "<unknown>"
                print(
                    f"{args.model} -> {canonical} | "
                    f"{resolution.effective_context_window:,} tokens | {marker}"
                )
                for warning in resolution.warnings:
                    print(f"warning: {warning}", file=sys.stderr)
            return 0

        if args.command == "list":
            models = [
                spec
                for spec in default_registry().models
                if args.provider is None or spec.provider == args.provider.lower()
            ]
            if args.json:
                print(json.dumps([_spec_to_dict(spec) for spec in models], indent=2))
            else:
                for spec in models:
                    window = (
                        f"{spec.context_window:,}"
                        if spec.context_window is not None
                        else "unknown"
                    )
                    print(
                        f"{spec.provider:18} {spec.id:44} "
                        f"{window:>10}  confidence={spec.confidence:.2f}"
                    )
            return 0

        if args.command == "fingerprint":
            print(default_registry().fingerprint)
            return 0

        if args.command == "discover-ollama":
            models = discover_ollama_models(base_url=args.base_url)
            if args.json:
                print(json.dumps([_spec_to_dict(spec) for spec in models], indent=2))
            else:
                for spec in models:
                    window = (
                        f"{spec.context_window:,}"
                        if spec.context_window is not None
                        else "unknown"
                    )
                    print(f"{spec.id:44} {window:>10}")
            return 0
    except (ModelRegistryError, UnknownModelError, OSError, json.JSONDecodeError) as exc:
        print(f"entroly-models: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
