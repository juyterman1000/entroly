"""Deterministic runtime capability reporting for installation and support tooling.

The report is intentionally conservative: it describes what this installed
runtime can load locally.  It does not claim provider connectivity, benchmark
quality, or semantic conformance merely because an adapter module exists.
"""
from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .native_status import MIN_ENTROLY_CORE_VERSION, QCCR_SYMBOLS, native_status
from .proxy_config import PROVIDER_CAPABILITIES

SCHEMA_VERSION = "entroly.runtime-capabilities.v1"

_OPTIONAL_MODULES: tuple[tuple[str, str], ...] = (
    ("mcp", "mcp"),
    ("proxy_http", "httpx"),
    ("proxy_asgi", "starlette"),
    ("proxy_server", "uvicorn"),
    ("tokenizer", "tiktoken"),
)


def _module_available(module_name: str) -> bool:
    """Return module discoverability without importing optional dependencies."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _secure_recovery_available() -> tuple[bool, str | None]:
    try:
        from .compression_retrieval_store_secure import CompressionRetrievalStore

        module = getattr(CompressionRetrievalStore, "__module__", "")
        return module.endswith("compression_retrieval_store_secure"), module or None
    except Exception as exc:  # diagnostics must never make the CLI unusable
        return False, type(exc).__name__


def build_runtime_capabilities() -> dict[str, Any]:
    """Build a JSON-serializable, side-effect-free runtime capability report."""
    native = native_status(QCCR_SYMBOLS)
    optional = {name: _module_available(module) for name, module in _OPTIONAL_MODULES}
    secure_recovery, recovery_detail = _secure_recovery_available()

    providers: list[dict[str, Any]] = []
    for name in sorted(PROVIDER_CAPABILITIES):
        capability = PROVIDER_CAPABILITIES[name]
        providers.append(
            {
                "name": name,
                "native_protocol_adapter": True,
                "streaming_route_detection": bool(capability.streaming_path_markers)
                or name in {"openai", "anthropic"},
                "tools_declared": bool(capability.supports_tools),
                "vision_declared": bool(capability.supports_vision),
                "connectivity_verified": False,
            }
        )

    proxy_dependencies = {
        key: optional[key]
        for key in ("proxy_http", "proxy_asgi", "proxy_server")
    }
    proxy_available = all(proxy_dependencies.values())

    return {
        "schema_version": SCHEMA_VERSION,
        "package": {
            "name": "entroly",
            "version": __version__,
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": sys.platform,
            "machine": platform.machine() or "unknown",
        },
        "engine": {
            "active_mode": "native" if native.ok else "python",
            "native": {
                "available": native.available,
                "healthy": native.ok,
                "version": native.version,
                "minimum_version": MIN_ENTROLY_CORE_VERSION,
                "version_ok": native.version_ok,
                "missing_symbols": list(native.missing_symbols),
                "module_file": Path(native.path).name if native.path else None,
                "unavailable_reason": "import_failed" if native.error else None,
            },
            "pure_python_fallback": True,
        },
        "surfaces": {
            "python_sdk": {"available": True},
            "assured_compression": {"available": True, "default": False},
            "context_receipts": {"available": True},
            "secure_recovery": {
                "available": secure_recovery,
                "default": secure_recovery,
                "implementation": recovery_detail,
            },
            "proxy": {
                "available": proxy_available,
                "dependencies": proxy_dependencies,
                "connectivity_verified": False,
            },
            "mcp": {
                "available": optional["mcp"],
                "connectivity_verified": False,
            },
        },
        "providers": providers,
        "optional_dependencies": optional,
        "claims": {
            "local_runtime_inspection_only": True,
            "provider_connectivity_verified": False,
            "benchmark_leadership_implied": False,
        },
    }


def capability_exit_code(report: dict[str, Any]) -> int:
    """Return non-zero only when the installed base runtime is unusable."""
    package_ok = bool(report.get("surfaces", {}).get("python_sdk", {}).get("available"))
    recovery_ok = bool(report.get("surfaces", {}).get("secure_recovery", {}).get("available"))
    return 0 if package_ok and recovery_ok else 1
