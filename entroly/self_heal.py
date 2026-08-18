"""Automatic repair of a degraded Entroly install.

Entroly's value depends on query-conditioned selection (QCCR), which is gated on
the native engine: `EntrolyEngine.optimize_context` only takes the QCCR path
`if self._use_rust`, and its candidate set comes from
`self._rust.export_fragments()`, which has no pure-Python equivalent. Without the
native module the optimizer still returns fragments and still fills the token
budget -- it just never reads the query.

Measured on this repository with the engine absent, three unrelated questions --
including the nonsense control "banana bicycle weather forecast tuna sandwich" --
returned a byte-identical 23 fragments / 7,588 tokens and the same "76.29%
saved", because that figure is `(baseline - selected_tokens) / baseline` and
nothing about the query can move it. With the engine present the same three
queries returned 12/5,386, 11/3,180 and 12/4,827.

Before this module, eight separate places told the user to install the engine
(`cli.py`, `dashboard.py`, `parser_compatibility.py`, `hardening.py`, `qccr.py`).
Everywhere told; nowhere fixed. This module fixes it instead, and is the single
remediation path those sites should call.

**This performs an outbound network install.** That is a deliberate product
decision and a departure from Entroly's otherwise strictly opt-in outbound
behaviour (`cli._check_for_update` requires `ENTROLY_ENABLE_UPDATE_CHECK=1`).
Set `ENTROLY_NO_SELF_HEAL=1` to disable it -- appropriate for locked or audited
environments, reproducible CI, and air-gapped builds. When repair is disabled or
impossible, callers must fall back to reporting the figure *labelled* as
unearned rather than silently presenting it as a measured saving.

Never call this from an import path. Installing a package while a module is
being imported risks re-entrant imports, partially initialised state, and
multiprocessing deadlock, so `entroly/__init__.py` and `entroly/sdk.py` must
not. SDK users call `entroly.repair()` explicitly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path

from .native_status import MIN_ENTROLY_CORE_VERSION, QCCR_SYMBOLS, native_status

# Set to "1" to disable automatic repair entirely.
ENV_DISABLE = "ENTROLY_NO_SELF_HEAL"
# Set by the re-exec below so a repaired process never tries to repair again.
ENV_GUARD = "ENTROLY_SELF_HEAL_DONE"

_REQUIREMENT = f"entroly-core>={MIN_ENTROLY_CORE_VERSION},<2"

# A wheel is ~6.4 MB. Generous enough for a slow link, short enough that a
# hung index server or a captive-portal proxy cannot wedge the CLI.
INSTALL_TIMEOUT_SECONDS = 180

# Repair is attempted at most once per process even without the re-exec, so a
# failing install cannot turn one command into an install loop.
_attempted_in_this_process = False


@dataclass
class RepairOutcome:
    """What repair did, and whether the caller must re-exec to benefit."""

    attempted: bool = False
    healed: bool = False
    needs_reexec: bool = False
    blocked_reason: str | None = None
    steps: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.blocked_reason is not None


def native_engine_ready() -> bool:
    """True when the native engine is present, complete, and new enough."""
    return native_status(QCCR_SYMBOLS).ok


def disabled() -> bool:
    return os.environ.get(ENV_DISABLE, "0") == "1"


def already_healed() -> bool:
    return os.environ.get(ENV_GUARD, "0") == "1"


def _externally_managed() -> bool:
    """True when PEP 668 forbids installing into this interpreter.

    Debian/Ubuntu system Pythons and Homebrew Python ship an
    ``EXTERNALLY-MANAGED`` marker; pip refuses to install there without
    ``--break-system-packages``, which this must never pass. A virtualenv is
    always safe, and its own stdlib directory carries no marker.
    """
    if sys.prefix != sys.base_prefix:  # inside a venv
        return False
    stdlib = sysconfig.get_path("stdlib")
    return bool(stdlib) and Path(stdlib).with_name("EXTERNALLY-MANAGED").exists()


def _installer_command() -> list[str] | None:
    """The command that can install into *this* interpreter, or None.

    `uv` is checked first only when Entroly is already running inside a
    uv-managed environment; otherwise `pip` targeted at `sys.executable` is the
    correct tool, because installing into some other interpreter would leave the
    running one just as broken.
    """
    if _externally_managed():
        return None

    if os.environ.get("UV_PROJECT_ENVIRONMENT") or os.environ.get("VIRTUAL_ENV"):
        uv = shutil.which("uv")
        if uv:
            return [uv, "pip", "install", "--python", sys.executable, "-U", _REQUIREMENT]

    try:
        import pip  # noqa: F401
    except Exception:
        return None
    return [sys.executable, "-m", "pip", "install", "--no-input", "-U", _REQUIREMENT]


def install_native_engine() -> tuple[bool, str]:
    """Install the native engine. Returns (ok, human-readable detail).

    Never raises: every failure mode here (offline, proxy, PEP 668, read-only
    filesystem, resolver conflict) is expected in real deployments and must
    degrade to a labelled report rather than an exception.
    """
    command = _installer_command()
    if command is None:
        if _externally_managed():
            return False, (
                "this Python is externally managed (PEP 668), so Entroly will not "
                "install into it"
            )
        return False, "no usable installer (pip is unavailable in this interpreter)"

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"install exceeded {INSTALL_TIMEOUT_SECONDS}s"
    except Exception as exc:  # pragma: no cover - platform specific
        return False, f"install could not start: {exc}"

    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit code {completed.returncode}"
        return False, detail[:200]
    return True, f"installed {_REQUIREMENT}"


def repair_native(*, force: bool = False) -> RepairOutcome:
    """Install the native engine when it is missing.

    `force` bypasses the once-per-process guard for an explicit `entroly doctor`
    style invocation; automatic callers should leave it False.
    """
    global _attempted_in_this_process

    outcome = RepairOutcome()
    if native_engine_ready():
        outcome.healed = True
        return outcome

    if disabled():
        outcome.blocked_reason = f"repair disabled by {ENV_DISABLE}=1"
        return outcome
    if already_healed() and not force:
        outcome.blocked_reason = "already repaired once in this process tree"
        return outcome
    if _attempted_in_this_process and not force:
        outcome.blocked_reason = "repair already attempted in this process"
        return outcome

    _attempted_in_this_process = True
    outcome.attempted = True

    ok, detail = install_native_engine()
    outcome.steps.append(("install native engine", ok, detail))
    if not ok:
        outcome.blocked_reason = detail
        return outcome

    # The engine object in this process was already built with _use_rust=False,
    # and native_status.usable_core() is lru_cached, so importing the freshly
    # installed module here would still leave a half-native process -- the exact
    # mixed state usable_core() exists to prevent. Re-exec instead.
    outcome.healed = True
    outcome.needs_reexec = True
    return outcome


def reexec_after_repair() -> int:
    """Re-run this command in a fresh interpreter. Returns its exit code.

    `os.execv` is avoided deliberately: on Windows it returns control to the
    parent shell before the replacement finishes, which reorders output and
    breaks exit-code propagation for callers such as the `entroly-mcp` npm
    bridge, which spawns the CLI with `spawnSync` and forwards its status.
    """
    env = dict(os.environ)
    env[ENV_GUARD] = "1"
    command = [sys.executable, "-m", "entroly", *sys.argv[1:]]
    try:
        return subprocess.run(command, env=env, check=False).returncode
    except Exception:  # pragma: no cover - platform specific
        return 1


def repair() -> RepairOutcome:
    """Public entry point for SDK users.

    Import paths must never repair implicitly, so library users who want the
    native engine restored call this explicitly:

        import entroly
        entroly.repair()
    """
    return repair_native(force=True)
