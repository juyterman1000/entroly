"""Machine-readable, privacy-safe Entroly installation diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from .runtime_capabilities import runtime_capabilities
from .runtime_status import runtime_status

SCHEMA_VERSION = "entroly.runtime-doctor.v1"


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _configuration_checks(data_dir: Path) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    candidates = {
        "runtime_config": data_dir / "config.json",
        "tuning_config": data_dir / "tuning_config.json",
        "model_registry": data_dir / "model_registry.json",
        "routing_policy": data_dir / "routing_policy.json",
    }
    for label, path in candidates.items():
        if not path.exists():
            checks.append(_check(label, "pass", "not_configured"))
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            checks.append(_check(label, "error", "invalid_json"))
            continue
        if not isinstance(payload, dict):
            checks.append(_check(label, "error", "expected_object"))
            continue
        checks.append(_check(label, "pass", "valid_json_object"))
    return checks


def _writable_state_check(data_dir: Path) -> dict[str, str]:
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        descriptor, probe_name = tempfile.mkstemp(prefix=".doctor-", dir=data_dir)
        try:
            os.write(descriptor, b"ok")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
            Path(probe_name).unlink(missing_ok=True)
    except OSError:
        return _check("state_directory", "error", "not_writable")
    return _check("state_directory", "pass", "writable")


def _secure_recovery_check() -> dict[str, str]:
    try:
        import entroly

        store = entroly.CompressionRetrievalStore
        module = str(getattr(store, "__module__", ""))
    except (AttributeError, ImportError):
        return _check("secure_recovery_default", "error", "unavailable")
    if module != "entroly.compression_retrieval_store_secure":
        return _check("secure_recovery_default", "error", "legacy_binding")
    return _check("secure_recovery_default", "pass", "hardened_binding")


def runtime_doctor(
    *,
    data_dir: Path,
    port: int = 9377,
    status_factory: Callable[..., dict[str, Any]] = runtime_status,
    capability_factory: Callable[[], dict[str, Any]] = runtime_capabilities,
) -> dict[str, Any]:
    """Return a stable local diagnostic report with no raw paths or errors."""
    capabilities = capability_factory()
    local_status = status_factory(port=port)

    checks = [
        _check("package", "pass", "importable"),
        _writable_state_check(data_dir),
        _secure_recovery_check(),
    ]
    checks.extend(_configuration_checks(data_dir))

    native = capabilities.get("engine", {}).get("native", {})
    if native.get("available"):
        checks.append(_check("native_engine", "pass", "available"))
    else:
        checks.append(_check("native_engine", "warning", "pure_python_fallback"))

    if local_status.get("healthy"):
        checks.append(_check("local_proxy", "pass", "ready"))
    else:
        checks.append(_check("local_proxy", "warning", "not_running"))

    errors = sum(check["status"] == "error" for check in checks)
    warnings = sum(check["status"] == "warning" for check in checks)
    passed = sum(check["status"] == "pass" for check in checks)

    return {
        "schema_version": SCHEMA_VERSION,
        "healthy": errors == 0,
        "summary": {"passed": passed, "warnings": warnings, "errors": errors},
        "checks": checks,
        "capabilities_schema": capabilities.get("schema_version"),
        "status_schema": local_status.get("schema_version"),
        "claims": {
            "provider_connectivity_verified": False,
            "production_readiness_implied": False,
        },
    }


__all__ = ["SCHEMA_VERSION", "runtime_doctor"]
