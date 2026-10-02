"""Pytest bootstrap: guarantee the suite tests *this* working tree.

Without this file the repository root never reaches ``sys.path``. pytest's
default ``prepend`` import mode inserts the first directory that is not a
package -- ``tests/`` has no ``__init__.py``, so it inserts ``tests/``, not the
root. ``entroly`` then resolves through whatever is installed, and an editable
install writes a ``.pth`` into site-packages that is placed on ``sys.path`` at
interpreter startup.

Measured on this machine before the fix: ``_editable_impl_entroly.pth``
contained ``C:\\Users\\abhis\\entroly``, so ``pytest tests/`` imported
``entroly`` from a *different checkout* -- a different version (1.0.84 against a
1.0.85 tree) in a different directory. The suite passed, and none of it
exercised the code under review. A green run meant nothing, and edits to
``entroly/`` here were invisible to it.

Two things are therefore done below, and both are load-bearing:

1. Put the repository root first on ``sys.path`` so the local package wins.
2. Assert that ``entroly`` actually resolved inside this repository, and fail
   loudly if not. Step 1 alone is a fix that can silently regress; step 2 is
   what makes the regression impossible to miss, because the failure mode is a
   *passing* suite.

Set ``ENTROLY_ALLOW_EXTERNAL_PACKAGE=1`` to bypass the assertion when you are
deliberately testing an installed artifact (the release matrix in CLAUDE.md
force-installs a wheel). The override is explicit so it shows up in the command
that used it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

# Must precede any `import entroly`, including the one below.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
elif sys.path[0] != str(_REPO_ROOT):
    sys.path.remove(str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT))


def _is_within(child: Path, parent: Path) -> bool:
    """Containment test that tolerates Windows case and short-path forms."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        # Fall back to a normalised comparison: `C:\\Users\\ABHIS\\...` and
        # `C:\\Users\\abhis\\...` are the same directory on Windows, and
        # `relative_to` is purely lexical.
        return os.path.normcase(str(child)).startswith(
            os.path.normcase(str(parent)) + os.sep
        )


def _verify_package_origin() -> None:
    if os.environ.get("ENTROLY_ALLOW_EXTERNAL_PACKAGE") == "1":
        return

    import importlib.util

    spec = importlib.util.find_spec("entroly")
    origin = getattr(spec, "origin", None) if spec else None
    if not origin:
        # No installed or local package at all: let the individual test's own
        # ImportError say so, rather than masking it with a path complaint.
        return

    resolved = Path(origin).resolve()
    if _is_within(resolved, _REPO_ROOT):
        return

    raise RuntimeError(
        "pytest would import `entroly` from outside this repository, so the "
        "suite would not test the working tree.\n"
        f"  repository root : {_REPO_ROOT}\n"
        f"  entroly resolves: {resolved}\n"
        "This is usually an editable install pointing at another checkout "
        "(look for a `_editable_impl_entroly.pth` or `__editable__*entroly*` "
        "file in site-packages, and check `pip show entroly`).\n"
        "Fix with `pip install -e .` from this directory, or set "
        "ENTROLY_ALLOW_EXTERNAL_PACKAGE=1 if you are deliberately testing an "
        "installed artifact."
    )


_verify_package_origin()
