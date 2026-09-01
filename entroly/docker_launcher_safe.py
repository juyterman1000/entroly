"""Truthful cross-platform launcher for the public ``entroly`` command.

The legacy launcher remains responsible for version-pinned Docker MCP execution.
Proxy mode is deliberately native and loopback-only: Docker bridge publication
would otherwise require every client to understand an additional access header,
which is neither zero-friction nor supported by all OpenAI-compatible clients.

Safe model routing is exposed only through an explicit opt-in CLI surface. Plain
``entroly proxy`` keeps its existing behavior; ``entroly proxy --routing ...``
uses the hardened native proxy and translates reviewed flags into the strict
routing-safety environment contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import _docker_launcher as _legacy


def _proxy_requested(argv: list[str]) -> bool:
    return "--proxy" in argv or os.environ.get("ENTROLY_PROXY") == "1"


def _routing_proxy_requested(argv: list[str]) -> bool:
    return bool(argv and argv[0] == "proxy" and "--routing" in argv)


def _proxy_arguments(argv: list[str]) -> list[str]:
    remaining = list(argv)
    if remaining and remaining[0] in {"serve", "proxy"}:
        remaining.pop(0)
    return [value for value in remaining if value != "--proxy"]


def _routing_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="entroly proxy",
        description=(
            "Start the hardened local proxy. Routing starts in observe mode and "
            "requires explicit provider, origin, model, pricing, and credential controls."
        ),
    )
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--routing", choices=("observe", "execute"), default=None)
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        choices=("openai", "anthropic", "gemini"),
        help="Provider transport authorized for this proxy process (exactly one).",
    )
    parser.add_argument(
        "--allow-model",
        action="append",
        default=[],
        metavar="MODEL",
        help="Exact source or target model ID; repeat for every authorized model.",
    )
    parser.add_argument(
        "--upstream-origin",
        default=None,
        metavar="HTTPS_ORIGIN",
        help=(
            "Explicit HTTPS upstream origin for observe mode. Execute mode currently "
            "permits only the provider's official API origin."
        ),
    )
    parser.add_argument(
        "--pricing-catalog",
        default=None,
        metavar="ABSOLUTE_JSON_PATH",
        help="Auditable pricing catalog containing every allowlisted model.",
    )
    parser.add_argument(
        "--ack-authorized-api",
        action="store_true",
        help="Confirm that the API credential is authorized for this account or organization.",
    )
    parser.add_argument(
        "--no-routing-headers",
        action="store_true",
        help="Do not add bounded routing receipt headers to provider responses.",
    )
    return parser


def _canonical_models(provider: str, values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = raw.strip()
        if not value:
            raise ValueError("--allow-model cannot be empty")
        if ":" in value:
            explicit_provider, model = value.split(":", 1)
            if explicit_provider.strip().casefold() != provider or not model.strip():
                raise ValueError(
                    "--allow-model provider prefixes must match the selected --provider"
                )
            canonical = f"{provider}:{model.strip()}"
        else:
            canonical = f"{provider}:{value}"
        if canonical not in seen:
            result.append(canonical)
            seen.add(canonical)
    return result


def _apply_proxy_cli_overrides(argv: list[str]) -> None:
    """Translate the documented native proxy CLI surface into validated env config."""
    try:
        args = _routing_parser().parse_args(_proxy_arguments(argv))
    except SystemExit as exc:
        if exc.code == 0:
            raise
        raise ValueError("unsupported or incomplete proxy argument") from exc
    if args.port is not None:
        os.environ["ENTROLY_PROXY_PORT"] = str(args.port)
    if args.host is not None:
        os.environ["ENTROLY_PROXY_HOST"] = str(args.host)

    if args.routing is None:
        return

    providers = list(dict.fromkeys(args.provider))
    if len(providers) != 1:
        raise ValueError("safe routing requires exactly one --provider per proxy process")
    provider = providers[0]

    from .proxy_routing_safety import official_origin

    origin = args.upstream_origin or official_origin(provider)
    models = _canonical_models(provider, args.allow_model)
    if args.routing == "execute":
        if len(models) < 2:
            raise ValueError(
                "routing execute requires at least two --allow-model values "
                "covering the original and permitted target models"
            )
        if not args.pricing_catalog:
            raise ValueError("routing execute requires --pricing-catalog")
        if not Path(args.pricing_catalog).expanduser().is_absolute():
            raise ValueError("--pricing-catalog must be an absolute path")
        if not args.ack_authorized_api:
            raise ValueError("routing execute requires --ack-authorized-api")

    os.environ["ENTROLY_ROUTING_AUTHORITY"] = "1"
    os.environ["ENTROLY_ROUTING_AUTHORITY_MODE"] = args.routing
    os.environ["ENTROLY_ROUTING_AUTHORITY_ALLOWED_PROVIDERS"] = provider
    os.environ["ENTROLY_ROUTING_AUTHORITY_ALLOWED_ORIGINS"] = f"{provider}={origin}"
    os.environ["ENTROLY_ROUTING_AUTHORITY_ALLOWED_MODELS"] = ",".join(models)
    os.environ["ENTROLY_ROUTING_AUTHORITY_REQUIRE_PRICING"] = "1"
    os.environ["ENTROLY_ROUTING_AUTHORITY_HEADERS"] = (
        "0" if args.no_routing_headers else "1"
    )
    os.environ["ENTROLY_RAVS_ROUTER"] = "1"
    os.environ.setdefault("ENTROLY_ESCALATION_MODE", "observe")
    if args.pricing_catalog:
        os.environ["ENTROLY_PRICING_CATALOG"] = str(
            Path(args.pricing_catalog).expanduser()
        )
    if args.ack_authorized_api:
        os.environ["ENTROLY_ROUTING_AUTHORITY_ACK"] = "authorized-official-api"


def _run_native_proxy(argv: list[str]) -> None:
    _apply_proxy_cli_overrides(argv)
    os.environ.setdefault("ENTROLY_PROXY_HOST", "127.0.0.1")

    from .container_proxy import main as proxy_main

    try:
        proxy_main()
    except (RuntimeError, ValueError) as exc:
        print(f"[entroly] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _routing_status(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="entroly routing status")
    parser.add_argument("--port", type=int, default=9377)
    args = parser.parse_args(argv)
    url = f"http://127.0.0.1:{args.port}/routing-authority"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"unable to read routing authority at {url}; verify the proxy is running"
        ) from exc
    print(json.dumps(payload, indent=2, sort_keys=True))


def _routing_check(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="entroly routing check")
    parser.add_argument(
        "--host",
        default=os.environ.get("ENTROLY_PROXY_HOST", "127.0.0.1"),
    )
    args = parser.parse_args(argv)

    from .proxy_config import ProxyConfig
    from .proxy_routing_official_guard import validate_official_routing_boundary
    from .proxy_routing_safety import validate_routing_environment

    proxy_config = ProxyConfig.from_env()
    proxy_config.host = args.host
    config = validate_routing_environment(proxy_config=proxy_config, host=args.host)
    config = validate_official_routing_boundary(config)
    print(json.dumps(config.public_summary(), indent=2, sort_keys=True))


def _run_routing_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="entroly routing")
    parser.add_argument("command", choices=("status", "check"))
    parsed, remaining = parser.parse_known_args(argv)
    if parsed.command == "status":
        _routing_status(remaining)
    else:
        _routing_check(remaining)


def _wrapper_flag(argv: list[str], flag: str) -> bool:
    boundary = argv.index("--") if "--" in argv else len(argv)
    return flag in argv[:boundary]


def _prepare_copilot_subscription(argv: list[str]) -> bool:
    """Prepare and pre-start the hardened proxy for explicit Copilot subscription mode."""
    from .copilot_subscription import (
        is_subscription_wrap,
        prepare_subscription_wrap,
        start_subscription_proxy,
    )

    if not is_subscription_wrap(argv):
        return False

    dry_run = _wrapper_flag(argv, "--dry-run")
    plan = prepare_subscription_wrap(argv)
    sys.argv[1:] = list(plan.cleaned_argv)
    summary = plan.public_summary()
    print(
        "[entroly] Copilot subscription route: "
        f"{summary['wire_api']} -> {summary['upstream_origin']} "
        f"via dedicated localhost:{summary['proxy_port']}"
        + (" [dry-run]" if dry_run else ""),
        file=sys.stderr,
    )
    if not dry_run:
        log_path = start_subscription_proxy(plan)
        print(
            f"[entroly] Hardened Copilot subscription proxy ready; log: {log_path}",
            file=sys.stderr,
        )
    return True


def launch() -> None:
    """Launch local commands, native proxy mode, or version-pinned Docker MCP."""
    argv = sys.argv[1:]
    try:
        if _prepare_copilot_subscription(argv):
            _legacy.launch()
            return
        if argv and argv[0] == "routing":
            _run_routing_command(argv[1:])
            return
        if (argv and argv[0] == "serve" and _proxy_requested(argv)) or (
            _routing_proxy_requested(argv)
        ):
            _run_native_proxy(argv)
            return
        _legacy.launch()
    except ValueError as exc:
        print(f"[entroly] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


__all__ = [
    "_apply_proxy_cli_overrides",
    "_prepare_copilot_subscription",
    "_routing_proxy_requested",
    "launch",
]
