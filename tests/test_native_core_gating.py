"""One process must not mix a refused native core with the Python fallback.

Sixteen modules import ``entroly_core``. Each used to decide on its own with a
bare ``try: import entroly_core``, and only two consulted
``MIN_ENTROLY_CORE_VERSION``. A core below the minimum was therefore refused by
the engine and used by everything else at the same time, which is worse than
either pure mode: the halves disagree about the shape of shared types.

Observed with a 1.0.74 core against a 1.0.75 package -- ``server.py`` fell back
to Python and passed ``recency_score`` to a ``ContextFragment`` that
``checkpoint.py`` had taken from the stale Rust core::

    TypeError: ContextFragment.__new__() got an unexpected keyword argument
               'recency_score'

``native_status.usable_core()`` is now the single gate. These tests keep the
decision in one place and stop the ungated set from growing.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "entroly"

def _imports_core_at_module_scope(source: str) -> bool:
    """True when a module binds entroly_core at import time.

    Module scope is the hazard: a top-level import binds a shared type for the
    life of the process, so two modules can hold different definitions of the
    same class -- which is how ContextFragment broke. Function-local imports
    re-evaluate per call and are usually guarded by an availability flag, so
    they degrade rather than corrupt type identity.

    Parsed rather than matched. The offending import sat inside a ``try:``
    block, so it is indented and invisible to an anchored regex, while still
    executing at module scope. Walking the tree and skipping function and class
    bodies gets this right regardless of nesting.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # a different scope, evaluated later
        if isinstance(node, ast.ImportFrom) and node.module == "entroly_core":
            return True
        if isinstance(node, ast.Import) and any(
            alias.name == "entroly_core" or alias.name.startswith("entroly_core.")
            for alias in node.names
        ):
            return True
        for field in ("body", "orelse", "finalbody", "handlers"):
            stack.extend(getattr(node, field, []) or [])
    return False

#: Modules that still import the native core without consulting the shared
#: gate. Each is a place a stale core can leak into a process that has already
#: refused it. This list may only shrink -- route the module through
#: ``native_status.usable_core()`` and delete its entry.
UNGATED_NATIVE_IMPORTERS = frozenset({
    "adaptive_pruner.py",
    "belief_compiler.py",
    "query_refiner.py",
    "witness.py",
})


def _modules_with_bare_native_import() -> set[str]:
    found: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if _imports_core_at_module_scope(text):
            found.add(path.relative_to(PACKAGE).as_posix())
    return found


def test_gated_modules_do_not_regress_to_a_bare_import() -> None:
    """server.py and checkpoint.py must keep using the shared gate."""
    bare = _modules_with_bare_native_import()
    for module in ("server.py", "checkpoint.py"):
        assert module not in bare, (
            f"entroly/{module} imports entroly_core directly again; it must go "
            "through native_status.usable_core() or a stale core can be used "
            "by one half of the process and refused by the other"
        )


def test_ungated_native_importers_do_not_grow() -> None:
    """New code must consult the gate; the exemption list may only shrink."""
    bare = _modules_with_bare_native_import()
    new = sorted(bare - UNGATED_NATIVE_IMPORTERS)
    assert not new, (
        f"new module(s) importing entroly_core without the shared gate: {new}. "
        "Use `from .native_status import usable_core` and keep a pure-Python "
        "fallback."
    )


def test_exemption_list_stays_honest() -> None:
    """A module that no longer bare-imports must leave the list."""
    bare = _modules_with_bare_native_import()
    stale = sorted(UNGATED_NATIVE_IMPORTERS - bare)
    assert not stale, (
        f"these no longer import entroly_core directly: {stale}. Remove them "
        "from UNGATED_NATIVE_IMPORTERS so the list keeps meaning something."
    )


def test_usable_core_refuses_a_below_minimum_core() -> None:
    """The gate must decline a core the package declares too old."""
    probe = (
        "import entroly.native_status as ns\n"
        "ns.usable_core.cache_clear()\n"
        "real = ns.native_status\n"
        "ns.native_status = lambda *a, **k: ns.NativeStatus(\n"
        "    available=True, module=object(), version='0.0.1', path='<t>',\n"
        "    missing_symbols=(), version_ok=False, error=None)\n"
        "print('CORE=' + repr(ns.usable_core()))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        pytest.fail(f"probe failed:\n{result.stdout}\n{result.stderr}")
    assert "CORE=None" in result.stdout, (
        f"usable_core() accepted a below-minimum core:\n{result.stdout}"
    )


def test_usable_core_refuses_a_core_missing_the_symbols_its_callers_use() -> None:
    """The gate documents "incomplete" -- it has to actually detect it.

    ``usable_core`` called ``native_status()`` with no required symbols, so
    ``missing_symbols`` was always empty and the incomplete branch could never
    fire, while the docstring promised it did. A core new enough to pass the
    version gate but missing ``ContextFragment`` was handed back, and
    ``checkpoint.py`` -- which dereferences it with no guard -- raised
    ``AttributeError`` at import time instead of using the pure-Python fallback
    in the next branch.

    Partially featured cores are not hypothetical: the comment on
    ``WORK_GRAPH_SYMBOLS`` records published 1.0.78 passing the version check
    while lacking a symbol.
    """
    probe = (
        "import sys, types\n"
        "import entroly_core as real\n"
        "fake = types.ModuleType('entroly_core')\n"
        "fake.__file__ = getattr(real, '__file__', '<t>')\n"
        "for n in dir(real):\n"
        "    if n != 'ContextFragment':\n"
        "        try: setattr(fake, n, getattr(real, n))\n"
        "        except Exception: pass\n"
        "sys.modules['entroly_core'] = fake\n"
        "import entroly.native_status as ns\n"
        "ns.usable_core.cache_clear()\n"
        "print('CORE=' + repr(ns.usable_core()))\n"
        "import entroly.checkpoint as cp\n"
        "print('FRAGMENT_MODULE=' + cp.ContextFragment.__module__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        pytest.fail(
            "an incomplete core crashed instead of falling back:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    assert "CORE=None" in result.stdout, (
        f"usable_core() accepted a core missing ContextFragment:\n{result.stdout}"
    )
    assert "FRAGMENT_MODULE=entroly" in result.stdout, (
        "checkpoint did not fall back to the Python ContextFragment:\n"
        f"{result.stdout}"
    )


def test_core_symbols_are_present_in_the_engine_this_release_builds() -> None:
    """Guard the other direction: the gate must not disable a healthy core.

    ``CORE_SYMBOLS`` is a hard requirement, so a name that is misspelled or
    renamed in the Rust core would silently drop the whole process to
    pure-Python -- a large, quiet performance regression rather than a failure.
    """
    import entroly.native_status as ns

    core = pytest.importorskip("entroly_core")
    missing = [s for s in ns.CORE_SYMBOLS if not hasattr(core, s)]
    assert not missing, (
        f"CORE_SYMBOLS names symbols the built core does not export: {missing}. "
        "Every process would fall back to pure-Python."
    )
    assert ns.usable_core() is not None, (
        "a healthy core was refused by the capability gate"
    )


def test_engine_and_checkpoint_agree_on_the_core() -> None:
    """The two halves of the crash must reach the same verdict."""
    import entroly.checkpoint as checkpoint
    import entroly.qccr as qccr
    import entroly.server as server

    assert server._RUST_AVAILABLE == qccr._HAS_RUST, (
        f"server={server._RUST_AVAILABLE} qccr={qccr._HAS_RUST}"
    )
    # When the engine is on the Python path, the shared fragment type must be
    # the Python one, or fragments built here cannot be consumed there.
    if not server._RUST_AVAILABLE:
        assert checkpoint.ContextFragment.__module__.startswith("entroly"), (
            "engine fell back to Python but ContextFragment came from the "
            "native core; this is the mixed process that raised TypeError"
        )
