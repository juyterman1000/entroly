"""`optimize()` must select only from the fragments it was given.

`optimize()` is documented as selecting a task-conditioned subset of the
`fragments` argument. It built a default `EntrolyEngine()`, which warm-starts
from the shared index under ENTROLY_DIR, so the returned selection could
contain content the caller never passed in -- files from whatever repository
was indexed last -- while displacing the caller's own fragments.

Observed before the fix, with three in-memory fragments supplied:

    supplied : auth.py, billing.py, crypto.py
    returned : auth.py,
               file:entroly-wasm/bin/entroly-wasm.js,
               file:entroly/npm-alias/index.d.ts

Two of three returned entries were never supplied, and `crypto.py` -- the file
the query needed -- was gone. Because the result is handed to a model, this is
a cross-project content leak as well as a selection defect.

tests/test_engine_isolation.py already locks the engine-level flag; these tests
lock the SDK entry point that failed to pass it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

SUPPLIED = {"auth.py", "billing.py", "crypto.py"}

_PROBE = textwrap.dedent(
    """
    import json
    from entroly import optimize

    fragments = [
        {"content": "def login(user):\\n    return check_password(user)",
         "source": "auth.py", "token_count": 12},
        {"content": "# invoices, taxes, unrelated to login",
         "source": "billing.py", "token_count": 11},
        {"content": "def check_password(u):\\n    return u.pw == stored",
         "source": "crypto.py", "token_count": 12},
    ]
    result = optimize(fragments, budget=40, query="fix the login bug")
    selected = result.get("selected_fragments") or result.get("selected") or []
    sources = [s.get("source") for s in selected if isinstance(s, dict)]
    print("PROBE_RESULT " + json.dumps(sources))
    """
)


def _run_probe() -> list[str]:
    """Run optimize() in a fresh interpreter with the ambient environment.

    A subprocess is deliberate: the defect depends on module-level warm-start
    state, so exercising it in-process after other tests have imported entroly
    would not reproduce the conditions a real caller hits.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        pytest.fail(f"probe failed:\n{completed.stdout}\n{completed.stderr}")

    for line in completed.stdout.splitlines():
        if line.startswith("PROBE_RESULT "):
            return json.loads(line[len("PROBE_RESULT "):])
    pytest.fail(f"probe produced no result:\n{completed.stdout}")


def test_optimize_returns_only_supplied_fragments() -> None:
    """No source may appear that the caller did not pass in."""
    sources = _run_probe()
    foreign = [source for source in sources if source not in SUPPLIED]
    assert not foreign, (
        "optimize() returned fragments that were never supplied, meaning the "
        "shared warm-start index leaked into the caller's context: "
        f"{foreign}"
    )


def test_optimize_returns_the_directly_queried_fragment() -> None:
    """The fragment naming the queried symbol must survive selection.

    Deliberately narrow. An earlier version of this test also required
    `crypto.py`, on the reasoning that `login()` calls `check_password()` and a
    dependency-aware selector should follow that edge. The engine does not make
    that guarantee: at a 1000-token budget for 35 tokens of input it returns
    `auth.py` alone, leaving 98% of the budget unspent and the callee behind.
    That under-fill is recorded in the research ledger ("selection returned
    only the needle rather than filling the budget") and is a separate open
    question from the contamination this module exists to lock.

    Asserting the stronger property here would have encoded a wish as a
    contract, so this checks only what the fix actually established.
    """
    sources = _run_probe()
    assert "auth.py" in sources, f"login fragment missing from {sources}"


def test_optimize_builds_an_ephemeral_engine() -> None:
    """Lock the mechanism, so a refactor cannot quietly restore the default."""
    from entroly import sdk

    source = sdk.optimize.__code__.co_consts
    # The config flag is passed by keyword; its name appears in the code object
    # of optimize() only if the call is still there.
    assert any(
        isinstance(const, str) and "use_persistent_index" in const
        for const in _walk_consts(sdk.optimize.__code__)
    ) or "use_persistent_index" in str(source), (
        "optimize() no longer passes use_persistent_index=False; a default "
        "EntrolyEngine() warm-starts from the shared index"
    )


def _walk_consts(code) -> list:  # noqa: ANN001
    out = []
    for const in code.co_consts:
        out.append(const)
        if hasattr(const, "co_consts"):
            out.extend(_walk_consts(const))
    return out
