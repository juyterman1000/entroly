"""The engine must refuse a native core the package declares incompatible.

`qccr.py` gates on `native_status().ok`, which includes `version_ok`.
`server.py` did not -- it imported `entroly_core` and trusted whatever it got.
So the same shared library, in the same process, was refused by one component
and used by the other, and the one using it performed every selection.

That is not theoretical. Measured on this repository with a stale core:

  * the same query over the same three fragments returned three fragments on
    entroly_core 1.0.73 and one on 1.0.74 -- the core version changes what the
    model is shown;
  * tests/test_simulate_small_project.py reported 0.0% savings against the
    stale core and passes against a matched one, so skeleton demotion silently
    stopped happening.

`entroly doctor` already reported the core as stale. The defect was that
nothing acted on it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Run in a subprocess: the gate executes at module import, so it cannot be
# exercised by monkeypatching a module this process has already imported.
_PROBE = textwrap.dedent(
    """
    import sys, types

    # Stand in for a native core whose reported version is below the floor.
    fake_core = types.ModuleType("entroly_core")
    fake_core.EntrolyEngine = object
    fake_core.py_analyze_query = lambda q: {}
    fake_core.py_refine_heuristic = lambda *a, **k: None
    sys.modules["entroly_core"] = fake_core

    import entroly.native_status as ns

    _real = ns.native_status

    def _stale(required_symbols=()):
        status = _real(required_symbols)
        return ns.NativeStatus(
            available=True,
            module=fake_core,
            version="0.0.1",
            path="<test>",
            missing_symbols=(),
            version_ok=False,
            error=None,
        )

    ns.native_status = _stale

    import entroly.server as server
    print("RUST_AVAILABLE=" + str(server._RUST_AVAILABLE))
    """
)


def _probe_with_stale_core() -> str:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        pytest.fail(f"probe failed:\n{completed.stdout}\n{completed.stderr}")
    return completed.stdout + completed.stderr


def test_engine_refuses_a_below_minimum_core() -> None:
    """A core reporting version_ok=False must not be used for selection."""
    output = _probe_with_stale_core()
    assert "RUST_AVAILABLE=False" in output, (
        "server.py accepted a native core the package declares below the "
        f"minimum supported version:\n{output}"
    )


def test_refusal_is_announced_not_silent() -> None:
    """Degrading must be visible; a silent downgrade is the original defect."""
    output = _probe_with_stale_core()
    assert "below the minimum" in output, (
        f"the fallback happened without telling anyone:\n{output}"
    )
    assert "maturin develop" in output, (
        "the warning should say how to fix it, not just that it happened"
    )


def test_server_and_qccr_agree_on_the_installed_core() -> None:
    """Two components must not reach opposite verdicts on the same library."""
    import entroly.qccr as qccr
    import entroly.server as server

    assert server._RUST_AVAILABLE == qccr._HAS_RUST, (
        "server.py and qccr.py disagree about whether the installed "
        f"entroly_core is usable: server={server._RUST_AVAILABLE}, "
        f"qccr={qccr._HAS_RUST}"
    )


def test_a_matched_core_is_still_accepted() -> None:
    """The gate must not cost native acceleration when the core is correct."""
    from entroly.native_status import native_status

    import entroly.server as server

    status = native_status()
    if not status.available:
        pytest.skip("native engine not installed in this environment")
    if status.version_ok is False:
        pytest.skip(
            f"installed entroly_core {status.version} is below the minimum; "
            "rebuild with `cd entroly-core && maturin develop --release`"
        )
    assert server._RUST_AVAILABLE is True, (
        "a compatible native core was rejected; the gate is too strict"
    )
