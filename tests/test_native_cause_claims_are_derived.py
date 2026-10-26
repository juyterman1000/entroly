"""No module may hardcode *why* the native engine is unavailable.

A module that falls back to pure Python knows that it fell back. It does not
know why. The reason lives in `native_status()`, which reports `available`,
`version`, `version_ok`, `missing_symbols` and `error` from the actual import,
and four modules nonetheless asserted a cause in a string literal:

    "entroly_core not installed"

That sentence is true only when the core is genuinely absent. For a core that
is installed but a release behind -- the case a developer and a partially
upgraded user both hit -- it is false, and the fix it implies is wrong: a stale
core needs an upgrade or a rebuild, not an install. Observed on one run, the
truth and the denial were logged back to back:

    entroly_core 1.0.84 is below the 1.0.85 this release requires  <- usable_core()
    Using Python fallback engine (entroly_core not installed)       <- engine.py

This gate exists because fixing the four instances does not stop a fifth. The
pattern is easy to reintroduce: `except ImportError` is a natural place to
write down why, and the honest answer is only available from `native_status`.

Scope is deliberately narrow, and the narrowing is the point. Stating *what*
happened -- "ArchetypeOptimizer: using pure-Python fallback" -- claims nothing
untrue and is left alone. So is the word "unavailable", which asserts no cause
and is correct whether the core is missing, stale or incomplete; `cli.py` and
`engine.py` already use it well. Only a claim of *absence* is rejected, because
absence is one specific cause among several and the one a module cannot know
without asking. Modules guarding genuinely optional third-party packages
(`playwright`, `hippocampus`, PyYAML) are untouched: for those, absence really
is the expected cause.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "entroly"

#: A claim that the engine is *absent*, which is one specific cause among
#: several. Excludes "unavailable" deliberately -- see the module docstring.
_HARDCODED_CAUSE = re.compile(
    r"""(entroly[_-]core|rust\s+engine|native\s+engine)
        [^"'\n]{0,40}?
        (not\s+installed|isn't\s+installed|not\s+present|not\s+found)""",
    re.IGNORECASE | re.VERBOSE,
)

#: `native_status.py` defines the vocabulary and documents the defect, so it is
#: the one file allowed to contain these phrases.
_ALLOWED = {"native_status.py"}


def _docstring_lines(text: str) -> set[int]:
    """Line numbers occupied by docstrings, via `ast`.

    An earlier version counted triple-quote fences per line. On a real module
    that tracker desynchronised -- by `engine.py` line 1020 it believed it was
    inside a docstring and silently skipped every line after, so reintroducing
    the exact bug this gate exists to catch produced a pass. A gate that stops
    working quietly is worse than no gate, which is the same defect class the
    gate itself guards against. `ast` cannot desynchronise.

    A file that does not parse yields no exemptions, so its code is still
    scanned rather than silently skipped.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:  # pragma: no cover - a broken file still gets scanned
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            end = first.end_lineno or first.lineno
            lines.update(range(first.lineno, end + 1))
    return lines


def _violations(root: Path | None = None) -> list[tuple[Path, int, str]]:
    """Hardcoded absence claims under ``root``, skipping prose.

    ``root`` is a parameter rather than a patched global so the self-test below
    can aim the same code at a fixture directory.
    """
    base = PACKAGE if root is None else root
    found: list[tuple[Path, int, str]] = []
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" in path.parts or path.name in _ALLOWED:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover - env dependent
            continue
        prose = _docstring_lines(text)
        for lineno, line in enumerate(text.splitlines(), 1):
            # Comments and docstrings explain the rule; they ship no claim.
            if lineno in prose or line.lstrip().startswith("#"):
                continue
            if _HARDCODED_CAUSE.search(line):
                found.append((path, lineno, line.strip()))
    return found


def test_no_module_hardcodes_why_the_native_engine_is_unavailable():
    violations = _violations()

    assert not violations, "\n".join(
        [
            "A module asserted a cause it had not checked. Derive it instead:",
            "",
            "    from .native_status import CORE_SYMBOLS, native_status",
            "    from .native_status import fallback_reason        # for a log line",
            "    from .native_status import native_status_message  # for a user error",
            "",
            *(
                f"  {path.relative_to(PACKAGE.parent)}:{lineno}: {line}"
                for path, lineno, line in violations
            ),
        ]
    )


def test_the_gate_detects_a_reintroduced_claim(tmp_path):
    """A gate that cannot fail is not a gate.

    The test above passes on a clean tree, which proves nothing by itself --
    the same mistake that let an earlier regression test in this repo pass with
    its fix reverted. This writes the exact string the four fixed sites used and
    asserts the pattern catches it.
    """
    offender = tmp_path / "pkg" / "regressed.py"
    offender.parent.mkdir(parents=True)
    offender.write_text(
        'logger.info("Using Python fallback engine (entroly_core not installed)")\n',
        encoding="utf-8",
    )

    violations = _violations(offender.parent)

    assert violations, "the gate did not catch a reintroduced hardcoded cause"
    assert violations[0][0].name == "regressed.py"


def test_the_gate_ignores_the_same_claim_in_a_docstring(tmp_path):
    """Prose that explains the rule must not trip it."""
    doc = tmp_path / "pkg" / "documented.py"
    doc.parent.mkdir(parents=True)
    doc.write_text(
        '"""This used to say entroly_core not installed, which was false."""\n'
        "value = 1\n",
        encoding="utf-8",
    )

    assert not _violations(doc.parent)


@pytest.mark.parametrize(
    "line",
    [
        'logger.info("ArchetypeOptimizer: using pure-Python fallback")',
        'logger.info("AdaptivePruner: Python fallback active")',
        'raise ImportError("entroly_core unavailable or below minimum version")',
        '"The native engine is unavailable, so selection did not read the query: "',
        'raise ImportError("browser support is not installed; run pip install")',
        'logger.warning("hippocampus-sharp-memory not installed - disabled")',
    ],
)
def test_the_gate_does_not_fire_on_legitimate_messages(line):
    """What-not-why, the cause-agnostic word, and genuinely optional packages.

    Without these the gate could be tightened into rejecting every fallback log
    line, which would push authors toward saying nothing at all -- or toward the
    false-but-specific wording this exists to stop.
    """
    assert not _HARDCODED_CAUSE.search(line), f"false positive on: {line}"
