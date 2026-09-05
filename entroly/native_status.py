"""Native engine capability diagnostics."""
from __future__ import annotations

import functools
import importlib
import logging
from dataclasses import dataclass
from importlib import metadata
from types import ModuleType


MIN_ENTROLY_CORE_VERSION = "1.0.81"
QCCR_SYMBOLS = (
    "py_qccr_expand_query",
    "py_qccr_rank_files",
    "py_qccr_select",
)
# The Work Graph binding is a distinct capability from QCCR: a core can satisfy
# MIN_ENTROLY_CORE_VERSION and still predate `WorkGraph`, which is exactly the
# state of the published entroly-core 1.0.78 -- version_ok is True and the
# symbol is absent. Checking the version alone would let `pip install entroly`
# resolve to a core that cannot run the Work Graph while reporting a healthy
# engine, the same silent capability loss QCCR gating exists to prevent.
WORK_GRAPH_SYMBOLS = (
    "WorkGraph",
    # Canonical node/edge identity. A core carrying WorkGraph but not these
    # cannot address artifacts the way the graph does, which is how a second
    # free-form id scheme grew in the Python layer. Treat identity as part of
    # the capability, not an optional extra.
    "work_graph_node_id",
    "work_graph_edge_id",
    # The production continuity surface is the complete context/execution/
    # trust chain. A core with the original WorkGraph class but without these
    # canonical contracts would otherwise pass the capability probe and fail
    # only after a handoff tried to record an execution.
    "routing_decision_build_json",
    "model_execution_outcome_build_json",
    "verification_record_build_json",
    "work_continuation_proof_build_json",
    "context_receipt_build_json",
    "context_receipt_verify_json",
    "recovery_handle_build_json",
    "recovery_handle_verify_json",
    "recovery_handle_verify_bytes",
)

RELEASE_NATIVE_SYMBOLS = QCCR_SYMBOLS + (
    "extract_skeleton",
    "py_compress_block",
)

# What `usable_core()` consumers dereference off the module it hands back.
# `checkpoint.py` takes `ContextFragment` with no guard, so a core missing it
# raises AttributeError at import time instead of using the pure-Python
# fallback sitting in the very next branch. `engine.py` takes the other three.
#
# Feature-specific sets stay out on purpose. QCCR_SYMBOLS and
# WORK_GRAPH_SYMBOLS are checked by the features that need them, via
# `native_status(...)`; a core without QCCR should lose QCCR, not all native
# acceleration. This tuple is the opposite case -- the symbols whose absence
# leaves the process unable to use the core at all.
CORE_SYMBOLS = (
    "ContextFragment",
    "EntrolyEngine",
    "py_analyze_query",
    "py_refine_heuristic",
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
    value = value.split("+", 1)[0]
    parts: list[int] = []
    for chunk in value.replace("-", ".").split("."):
        if not chunk.isdigit():
            break
        parts.append(int(chunk))
    return tuple(parts)


def _is_prerelease(value: str | None) -> bool:
    if not value:
        return False
    return "-" in value.split("+", 1)[0]


def _version_at_least(value: str | None, minimum: str) -> bool | None:
    parsed = _version_tuple(value)
    if not parsed:
        return None
    minimum_parsed = _version_tuple(minimum)
    if parsed != minimum_parsed:
        return parsed > minimum_parsed
    if _is_prerelease(value) and not _is_prerelease(minimum):
        return False
    return True


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
            f"Install entroly-core>={MIN_ENTROLY_CORE_VERSION},<2."
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
    if status.version_ok is False:
        return (
            f"{feature} requires a newer Entroly Rust engine{suffix}. "
            f"Install entroly-core>={MIN_ENTROLY_CORE_VERSION},<2."
        )
    if status.missing_symbols:
        return (
            f"{feature} found the Rust engine but required symbols are missing{suffix}. "
            f"Reinstall entroly-core>={MIN_ENTROLY_CORE_VERSION},<2."
        )
    return (
        f"{feature} found an incompatible Entroly Rust engine{suffix}. "
        f"Install entroly-core>={MIN_ENTROLY_CORE_VERSION},<2."
    )


@functools.lru_cache(maxsize=1)
def usable_core() -> ModuleType | None:
    """The native engine module, but only when it is safe to use.

    Single source of truth for "may this process call into entroly_core?".
    Sixteen modules import the native engine; before this existed, each decided
    on its own with a bare ``try: import entroly_core``, and only two consulted
    the declared minimum version. A core below that minimum was therefore
    refused by the engine and used by everything else *in the same process*.

    That is not a theoretical split. With a core one release behind the
    package, ``server.py`` correctly fell back to Python and passed
    ``recency_score`` to a ``ContextFragment`` that ``checkpoint.py`` had
    imported from the stale Rust core, which has no such field:

        TypeError: ContextFragment.__new__() got an unexpected keyword
                   argument 'recency_score'

    A mixed process is worse than either pure mode, so the decision has to be
    made once. Returns None when the core is absent, incomplete, or below the
    minimum, and callers keep their pure-Python fallback.

    Cached: the answer cannot change within a process, and this sits on import
    paths that run per request.

    "Incomplete" means missing any of ``CORE_SYMBOLS``. This used to call
    ``native_status()`` with no required symbols, so ``missing_symbols`` was
    always empty and the incomplete case could not be detected -- the promise
    above was documented but not implemented. A core new enough to pass the
    version gate but missing ``ContextFragment`` was handed to ``checkpoint.py``,
    which dereferences it without a guard and raised ``AttributeError`` at
    import time rather than falling back. That is not hypothetical: the comment
    on ``WORK_GRAPH_SYMBOLS`` records a published core that satisfied the
    minimum version while lacking a symbol.
    """
    status = native_status(CORE_SYMBOLS)
    logger = logging.getLogger("entroly")
    if status.available and status.version_ok is False:
        logger.warning(
            "entroly_core %s is below the %s this release requires; using the "
            "pure-Python engine. Rebuild with `cd entroly-core && maturin "
            "develop --release` to restore native acceleration.",
            status.version,
            MIN_ENTROLY_CORE_VERSION,
        )
    elif status.available and status.missing_symbols:
        logger.warning(
            "entroly_core %s is missing required symbols (%s); using the "
            "pure-Python engine. Rebuild with `cd entroly-core && maturin "
            "develop --release` to restore native acceleration.",
            status.version,
            ", ".join(status.missing_symbols),
        )
    return status.module if status.ok else None
