"""Native engine capability diagnostics."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib import metadata
from types import ModuleType


MIN_ENTROLY_CORE_VERSION = "1.0.25"
QCCR_SYMBOLS = (
    "py_qccr_expand_query",
    "py_qccr_rank_files",
    "py_qccr_select",
)


@dataclass(frozen=True)
class NativeStatus:
    available: bool
    module: ModuleType | None
    version: str | None
    path: str | None
    missing_symbols: tuple[str, ...]
    version_ok: bool | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.available
            and not self.missing_symbols
            and self.version_ok is not False
        )


def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    parts: list[int] = []
    for chunk in value.replace("-", ".").split("."):
        if not chunk.isdigit():
            break
        parts.append(int(chunk))
    return tuple(parts)


def _version_at_least(value: str | None, minimum: str) -> bool | None:
    parsed = _version_tuple(value)
    if not parsed:
        return None
    return parsed >= _version_tuple(minimum)


def native_status(required_symbols: tuple[str, ...] = ()) -> NativeStatus:
    """Inspect the loaded native engine without raising import-time failures."""
    try:
        module = importlib.import_module("entroly_core")
    except Exception as exc:
        return NativeStatus(
            available=False,
            module=None,
            version=None,
            path=None,
            missing_symbols=required_symbols,
            version_ok=None,
            error=str(exc),
        )

    try:
        version = metadata.version("entroly-core")
    except metadata.PackageNotFoundError:
        version = getattr(module, "__version__", None)

    missing = tuple(name for name in required_symbols if not hasattr(module, name))
    return NativeStatus(
        available=True,
        module=module,
        version=version,
        path=getattr(module, "__file__", None),
        missing_symbols=missing,
        version_ok=_version_at_least(version, MIN_ENTROLY_CORE_VERSION),
    )


def native_status_message(
    status: NativeStatus,
    *,
    feature: str = "this feature",
) -> str:
    if not status.available:
        return (
            f"{feature} requires the Entroly Rust engine. "
            "Install it with: python -m pip install --no-cache-dir -U "
            f"\"entroly-core>={MIN_ENTROLY_CORE_VERSION},<2\""
        )

    details: list[str] = []
    if status.version:
        details.append(f"loaded version {status.version}")
    if status.path:
        details.append(f"from {status.path}")
    if status.version_ok is False:
        details.append(f"requires >= {MIN_ENTROLY_CORE_VERSION}")
    if status.missing_symbols:
        details.append(f"missing symbols: {', '.join(status.missing_symbols)}")
    suffix = f" ({'; '.join(details)})" if details else ""
    return (
        f"{feature} requires a newer Entroly Rust engine{suffix}. "
        "Fix: python -m pip install --no-cache-dir -U "
        f"\"entroly-core>={MIN_ENTROLY_CORE_VERSION},<2\""
    )