"""The Python fallback must say why it engaged, not guess.

Four sites hardcoded "entroly_core not installed" in a branch reached for
several different reasons: the core genuinely absent, present but below
``MIN_ENTROLY_CORE_VERSION``, present but missing a required symbol, or an
import that raised. Only the first makes that sentence true.

The product already knows which it is. ``native_status()`` returns
``available``, ``version``, ``version_ok``, ``missing_symbols`` and ``error``,
``usable_core()`` logs an accurate warning from exactly those fields, and
``entroly doctor`` prints "Installed Rust engine is stale or incomplete" with
the loaded version and a working fix. Observed on a machine with
``entroly_core`` 1.0.84 installed against a required 1.0.85, the two lines are
emitted back to back:

    entroly_core 1.0.84 is below the 1.0.85 this release requires; using the
        pure-Python engine.                                  <- usable_core()
    Using Python fallback engine (entroly_core not installed) <- engine.py

A user reading the second reinstalls a package they already have. The advice is
also wrong: the fix for a stale core is an upgrade or a rebuild, not an install.

`entroly/mcp_sdk.py` exists because the same class of bug bit the MCP SDK
guard, and states the rule this file enforces: failing closed is correct,
failing closed with a false reason is not.
"""
from __future__ import annotations

import logging

import pytest

from entroly import native_status as ns


def _status(**kw):
    """Build a NativeStatus with explicit fields."""
    defaults = dict(
        available=False,
        module=None,
        version=None,
        path=None,
        missing_symbols=(),
        version_ok=None,
        error=None,
    )
    defaults.update(kw)
    return ns.NativeStatus(**defaults)


# ── The reason has to distinguish the cases ──────────────────────────


def test_a_genuinely_absent_core_is_reported_as_not_installed():
    reason = ns.fallback_reason(_status(available=False, error="No module named 'entroly_core'"))

    assert "not installed" in reason.lower()


def test_a_stale_core_is_not_reported_as_not_installed():
    """The case that sent users to reinstall what they already had."""
    reason = ns.fallback_reason(
        _status(available=True, version="1.0.84", version_ok=False)
    )

    assert "not installed" not in reason.lower(), (
        f"an installed-but-stale core was reported as absent: {reason!r}"
    )


def test_a_stale_core_names_the_loaded_and_required_versions():
    reason = ns.fallback_reason(
        _status(available=True, version="1.0.84", version_ok=False)
    )

    assert "1.0.84" in reason
    assert ns.MIN_ENTROLY_CORE_VERSION in reason


def test_an_incomplete_core_names_the_missing_symbols():
    reason = ns.fallback_reason(
        _status(
            available=True,
            version="1.0.85",
            version_ok=True,
            missing_symbols=("ContextFragment",),
        )
    )

    assert "not installed" not in reason.lower()
    assert "ContextFragment" in reason


def test_an_import_error_is_surfaced_rather_than_relabelled():
    reason = ns.fallback_reason(
        _status(available=False, error="DLL load failed while importing entroly_core")
    )

    assert "DLL load failed" in reason


# ── The live path, which is what a user actually reads ───────────────


def test_the_live_reason_agrees_with_the_live_status():
    """Whatever this machine's state is, the two must not contradict."""
    status = ns.native_status(ns.CORE_SYMBOLS)
    reason = ns.fallback_reason(status)

    if status.available:
        assert "not installed" not in reason.lower(), (
            f"core is importable from {status.path} but the reason claims it is "
            f"absent: {reason!r}"
        )
    else:
        assert "not installed" in reason.lower()


def test_the_engine_fallback_log_does_not_claim_absence_when_present(caplog):
    """The user-visible line from `engine.py`.

    Asserted against the live status rather than a fixture, so it holds whether
    the machine running the suite has a usable core, a stale one, or none.
    """
    from entroly.engine import EntrolyEngine

    status = ns.native_status(ns.CORE_SYMBOLS)
    if status.ok:
        pytest.skip("native core is usable here; the fallback branch is not taken")

    # engine.py logs to the package logger "entroly", not "entroly.engine".
    with caplog.at_level(logging.INFO, logger="entroly"):
        EntrolyEngine()

    fallback_lines = [
        r.getMessage() for r in caplog.records if "fallback" in r.getMessage().lower()
    ]
    assert fallback_lines, "no fallback line was logged at all"
    if status.available:
        for line in fallback_lines:
            assert "not installed" not in line.lower(), (
                f"engine logged absence for an importable core: {line!r}"
            )
