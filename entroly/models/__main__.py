from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Sequence

from .registry import RegistryTrust, get_model_registry


def _capability_payload(capability) -> dict[str, Any] | None:
    if capability is None:
        return None
    payload = asdict(capability)
    payload["trust"] = capability.trust.value
    payload["aliases"] = list(capability.aliases)
    payload["reasoning_levels"] = list(capability.reasoning_levels)
    return payload


def _resolution_payload(result, *, output_tokens: int | None, safety_fraction: float) -> dict[str, Any]:
    return {
        "requested_model": result.requested_model,
        "model_id": result.model_id,
        "context_window": result.context_window,
        "effective_input_budget": result.effective_input_budget(
            requested_output_tokens=output_tokens,
            safety_fraction=safety_fraction,
        ),
        "exact": result.exact,
        "trust": result.trust.value,
        "warning": result.warning,
        "registry_digest": result.registry_digest,
        "base_registry_digest": result.base_registry_digest,
        "capability": _capability_payload(result.capability),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m entroly.models",
        description="Inspect Entroly's provenance-aware model intelligence registry.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve", help="Resolve one model name or alias")
    resolve.add_argument("model")
    resolve.add_argument("--output-tokens", type=int)
    resolve.add_argument("--safety-fraction", type=float, default=0.05)

    listing = subparsers.add_parser("list", help="List registry capabilities")
    listing.add_argument("--provider")
    listing.add_argument("--trust", choices=[item.value for item in RegistryTrust])

    subparsers.add_parser("diagnostics", help="Show registry provenance and discovery status")

    discover = subparsers.add_parser(
        "discover",
        help="Discover Ollama, LM Studio, or authenticated OpenRouter metadata",
    )
    discover.add_argument(
        "providers",
        nargs="?",
        default="ollama,lmstudio",
        help="Comma-separated providers: ollama,lmstudio,openrouter",
    )
    discover.add_argument("--inspect-ollama-context", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "discover":
        providers = {item.strip().lower() for item in args.providers.split(",") if item.strip()}
        local = providers & {"ollama", "lmstudio"}
        remote = providers & {"openrouter"}
        os.environ["ENTROLY_DISCOVER_LOCAL_MODELS"] = ",".join(sorted(local))
        os.environ["ENTROLY_DISCOVER_REMOTE_MODELS"] = ",".join(sorted(remote))
        if args.inspect_ollama_context:
            os.environ["ENTROLY_OLLAMA_INSPECT_CONTEXT"] = "1"
        get_model_registry.cache_clear()

    registry = get_model_registry()

    if args.command == "resolve":
        payload = _resolution_payload(
            registry.resolve(args.model),
            output_tokens=args.output_tokens,
            safety_fraction=args.safety_fraction,
        )
    elif args.command == "list":
        capabilities = registry.all()
        if args.provider:
            capabilities = tuple(
                item for item in capabilities if item.provider == args.provider.lower()
            )
        if args.trust:
            capabilities = tuple(
                item for item in capabilities if item.trust.value == args.trust
            )
        payload = {
            "registry_digest": registry.registry_digest,
            "base_registry_digest": registry.base_registry_digest,
            "models": [_capability_payload(item) for item in capabilities],
        }
    else:
        payload = registry.diagnostics()
        if args.command == "discover":
            payload["discovered_models"] = [
                _capability_payload(item)
                for item in registry.all()
                if item.trust is RegistryTrust.DISCOVERED
            ]

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
