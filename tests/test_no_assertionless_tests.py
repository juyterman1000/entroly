"""Guard: a test that cannot fail is not a test.

A test function with no assertion passes no matter how broken the code is. That
is how bugs leak past a green suite — the `entroly doctor` false-green shipped
under a test that asserted `"7/8 checks passed"` as a *substring*, which stayed
true when the summary changed.

This guard freezes the existing debt and blocks new additions. It does not
rewrite the legacy suites; it stops them growing while they are paid down.

To pay debt down: add real assertions, then lower that file's number in
LEGACY_ASSERTIONLESS. The guard fails if a count goes *up*, and also if a count
is stale-high, so the baseline can never silently drift.
"""

from __future__ import annotations

import ast
import pathlib

TESTS_DIR = pathlib.Path(__file__).parent

# Known debt at the time this guard was installed. Only ever edit these numbers
# DOWNWARD. A new entry here must be justified in review.
LEGACY_ASSERTIONLESS: dict[str, int] = {
    "test_deep_functional.py": 32,
    "test_intensive_functional.py": 22,
    "test_functional.py": 18,
    "test_comprehensive_eval.py": 15,
    "test_ios.py": 3,
    "test_pagerank_integration.py": 3,
    "test_forge_live.py": 2,
    "test_zero_token_invariants.py": 2,
}

# Tests that legitimately assert "this must not raise" carry it in the name.
_NO_RAISE_MARKERS = ("never_raises", "does_not_raise", "no_exception", "is_importable")


def _asserts_something(fn: ast.AST) -> bool:
    nodes = list(ast.walk(fn))
    if any(isinstance(n, (ast.Assert, ast.Raise)) for n in nodes):
        return True
    for n in nodes:
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr in {"raises", "warns", "fail", "exit"}:
                return True
            if isinstance(f, ast.Name) and f.id in {"fail"}:
                return True
    return False


def _scan() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        bad = [
            fn.name
            for fn in ast.walk(tree)
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            and fn.name.startswith("test_")
            and not any(m in fn.name for m in _NO_RAISE_MARKERS)
            and not _asserts_something(fn)
        ]
        if bad:
            found[path.name] = sorted(bad)
    return found


def test_no_new_assertionless_tests_are_added():
    found = _scan()
    new_files = sorted(set(found) - set(LEGACY_ASSERTIONLESS))
    assert not new_files, (
        "these files add tests with no assertion — a test that cannot fail is "
        f"not a test: { {f: found[f] for f in new_files} }"
    )

    grown = {
        f: (len(found[f]), LEGACY_ASSERTIONLESS[f])
        for f in found
        if len(found[f]) > LEGACY_ASSERTIONLESS[f]
    }
    assert not grown, (
        f"assertion-less test debt grew (actual, allowed): {grown}. "
        "Add assertions instead of raising the baseline."
    )


def test_baseline_is_not_stale():
    # If debt was paid down, the baseline must be lowered in the same change,
    # otherwise the guard silently permits regressions back up to the old number.
    found = _scan()
    stale = {
        f: (len(found.get(f, [])), allowed)
        for f, allowed in LEGACY_ASSERTIONLESS.items()
        if len(found.get(f, [])) < allowed
    }
    assert not stale, (
        f"debt was reduced but the baseline was not lowered (actual, allowed): "
        f"{stale}. Update LEGACY_ASSERTIONLESS to lock the improvement in."
    )
