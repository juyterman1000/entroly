"""Privacy-safe, offline inspection of installed Entroly capabilities.

The report deliberately distinguishes code availability from active health and
provider connectivity.  It performs no network requests and exposes no local
filesystem paths or raw import exceptions.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from typing import Any

from . import __version__
from .native_status import QCCR_SYMBOLS, native_status

_SCHEMA_VERSION = "entroly.runtime-capabilities.v1"


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def runtime_capabilities() -> dict[str, Any]:
    """Return a stable, non-mutating description of installed capabilities."""
    native = native_status(QCCR_SYMBOLS)
    native_healthy = bool(native.ok)
    native_available = bool(native.available)

    return {
        "schema_version": _SCHEMA_VERSION,
        "package": {
            "name": "entroly",
            "version": __version__,
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system().lower(),
        },
        "engine": {
            "active": "native" if native_healthy else "pure-python",
            "pure_python_available": True,
            "native": {
                "installed": native_available,
                "healthy": native_healthy,
                "version": native.version if native_healthy else None,
                "missing_symbol_count": len(native.missing_symbols),
            },
            "shared_native_wasm_source": True,
        },
        "dependencies": {
            "mcp": {
                "installed": _module_available("mcp"),
                "required_for": ["mcp-stdio", "mcp-http"],
            },
            "http_proxy": {
                "installed": all(
                    _module_available(name) for name in ("httpx", "starlette", "uvicorn")
                ),
                "required_for": ["provider-proxy", "dashboard"],
            },
        },
        "assurance": {
            "identity_when_in_budget": True,
            "secure_recovery_store": True,
            "exact_source_receipts": True,
            "unsafe_compression_bypass": True,
        },
        "provider_protocols": {
            "openai_chat_completions": {"implemented": True, "connectivity_verified": False},
            "openai_responses": {"implemented": True, "connectivity_verified": False},
            "anthropic_messages": {"implemented": True, "connectivity_verified": False},
            "gemini_generate_content": {"implemented": True, "connectivity_verified": False},
        },
        # Some protections are installed everywhere but can only be *active* in
        # the mode that sees the data they protect. Reporting them as a flat
        # "implemented: true" is how a user ends up believing an inactive
        # guard is running, so the mode is part of the capability.
        "session_protection": {
            "implemented": True,
            # The distinction is automatic vs available, not proxy vs nothing.
            # `SessionRescueController` is pure policy -- it imports nothing
            # HTTP-aware -- so any caller that can hand over its conversation
            # can drive it via `entroly.rescue_session`. What the proxy alone
            # provides is doing it *for* you: rescue rewrites the outbound
            # provider request, and the proxy is the only surface Entroly owns
            # that sees one. MCP tools are invoked with their own arguments and
            # never receive the host's transcript, so an MCP host that wants
            # rescue has to pass the conversation in.
            "automatic_modes": ["proxy"],
            "callable_from": ["sdk", "cli", "mcp-host", "provider-sdk-wrapper"],
            "entry_point": "entroly.rescue_session",
            "reason_not_automatic_elsewhere": (
                "session rescue rewrites the outbound provider request; only "
                "the proxy sees one, so every other surface must pass its "
                "conversation to entroly.rescue_session"
            ),
            "enable_with": "entroly proxy",
            "disable_env": "ENTROLY_SESSION_RESCUE",
            # Both are properties of the rescue itself, not of the mode, and are
            # what separate it from summarizing compaction.
            "omissions_recoverable": True,
            "prefix_cache_stable": True,
        },
        "operations": {
            "doctor": True,
            "proxy": True,
            "dashboard": True,
            "wrap": True,
            "unwrap": True,
            "status": True,
        },
        "claims": {
            "provider_connectivity_verified": False,
            "benchmark_leadership_implied": False,
            "production_readiness_implied": False,
        },
    }


def render_capabilities_text(report: dict[str, Any]) -> str:
    """Render a concise human-readable view without adding new claims."""
    engine = report["engine"]
    dependencies = report["dependencies"]
    session = report.get("session_protection") or {}
    lines = [
        f"Entroly {report['package']['version']} runtime capabilities",
        f"  Engine: {engine['active']}",
        f"  MCP dependency: {'installed' if dependencies['mcp']['installed'] else 'missing'}",
        (
            "  HTTP proxy dependencies: "
            + ("installed" if dependencies["http_proxy"]["installed"] else "missing")
        ),
    ]
    if session:
        # Two facts, because reporting only the first reads as "you don't have
        # this": where it happens for you, and how to invoke it where it does
        # not. The user's real question is "am I protected right now", and the
        # answer depends on how they are running Entroly.
        lines.append(
            "  Runaway-session rescue: automatic in "
            + ", ".join(session["automatic_modes"])
            + " (`"
            + session["enable_with"]
            + "`); callable anywhere via `"
            + session["entry_point"]
            + "`"
        )
    lines += [
        "  Provider connectivity: not checked (offline report)",
        "  Benchmark leadership: not implied",
    ]
    return "\n".join(lines)


__all__ = ["render_capabilities_text", "runtime_capabilities"]
