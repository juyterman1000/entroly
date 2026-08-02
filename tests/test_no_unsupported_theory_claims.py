"""Guard: unsupported theoretical guarantees must not reappear.

Entroly's selection code sits close to some real theory -- submodular
maximisation, knapsack approximation, rate-distortion -- and that adjacency
makes it easy to write a guarantee the implementation does not earn.

Two concrete instances were in the repo when this test was written:

* ``CLAUDE.md`` described ``knapsack.rs`` / ``knapsack_sds.rs`` as a "0/1 DP
  token budget solver with (1-1/e) guarantee". Both files' own headers refute
  it: ``knapsack.rs`` says the objective is modular, so density-greedy gives
  Dantzig-style 1/2 and "NOT (1-1/e)"; ``knapsack_sds.rs`` says the subtractive
  redundancy penalty can violate monotonicity and "no tight worst-case ratio is
  claimed here".
* ``HCCEngine`` in ``entroly/context_bridge.py`` claimed "(1 - 1/e) = 63.2%
  optimal (submodular monotone)" for an objective that is separable across
  fragments -- a fragment's value depends only on its own assigned level, never
  on what else was selected -- i.e. modular, so the bound cannot apply.

The (1-1/e) bound (Nemhauser-Wolsey-Fisher 1978; tight per Feige 1998) needs a
*monotone submodular* objective and an *exact* marginal-gain oracle. Under a
knapsack rather than a cardinality constraint it additionally needs Sviridenko's
2004 partial enumeration to reach (1-1/e-eps). Naming a ratio is a claim about
all of those assumptions at once.

This test bans phrasings that *assert* a guarantee. It deliberately does not ban
discussing the bound: every file cited above still explains why the ratio does
not apply, and that prose is the useful part. Only unhedged assertions fail.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCANNED_SUFFIXES = {".py", ".rs", ".md", ".ts", ".js", ".toml", ".txt", ".rst"}

# Each pattern asserts a guarantee. Phrasings that merely mention a bound while
# explaining its inapplicability are not matched -- see the module docstring.
BANNED = [
    (r"\(?\s*1\s*[-−]\s*1\s*/\s*e\s*\)?[^.\n]{0,40}\bguarantee",
     "states a (1-1/e) guarantee; requires monotone submodular + exact oracle "
     "(NWF 1978), and Sviridenko 2004 partial enumeration under a knapsack"),
    (r"63\.2\s*%\s*optimal",
     "states the (1-1/e) ratio as an achieved optimality percentage"),
    (r"\b(?:provably|mathematically|formally)\s+(?:optimal|guaranteed)",
     "asserts a formal guarantee"),
    (r"\bguaranteed\s+optimal\b",
     "asserts optimality"),
    (r"\bzero\s+(?:quality|information)\s+loss\b",
     "asserts lossless compression"),
    (r"\bexact\s+semantic\s+preservation\b",
     "asserts semantics are preserved exactly"),
    (r"\bguarantees?\s+the\s+same\s+answers?\b",
     "asserts identical model answers"),
]

COMPILED = [(re.compile(p, re.IGNORECASE), why) for p, why in BANNED]

# Files that quote these phrasings in order to forbid or analyse them. The
# regexes match assertions by shape and cannot tell "X is guaranteed" from
# "never claim X is guaranteed", so prose *about* the claims is listed here.
ALLOWLISTED = {
    # A positioning spec whose own text is a list of claims not to make:
    # 'Do not claim: "zero quality loss" without a bounded test domain'.
    "docs/ENTROLY_WIN_MASTER_PROMPT.md",
}


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    if out.returncode != 0:  # pragma: no cover - only when git is unavailable
        pytest.skip(f"git ls-files failed: {out.stderr[:200]}")
    files = []
    for line in out.stdout.splitlines():
        p = ROOT / line.strip()
        if p.suffix.lower() in SCANNED_SUFFIXES and p.is_file():
            files.append(p)
    return files


def test_no_unsupported_theory_claims():
    files = _tracked_files()
    assert files, "scanned nothing -- the guard would pass vacuously"

    # This test is itself full of the banned phrasings, by necessity.
    this_file = Path(__file__).resolve()

    violations: list[str] = []
    for path in files:
        if path.resolve() == this_file:
            continue
        if path.relative_to(ROOT).as_posix() in ALLOWLISTED:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover
            continue
        for pattern, why in COMPILED:
            for m in pattern.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                rel = path.relative_to(ROOT).as_posix()
                violations.append(f"  {rel}:{line_no}: {m.group(0).strip()!r} -- {why}")

    assert not violations, (
        "Unsupported theoretical guarantee(s) found:\n"
        + "\n".join(violations)
        + "\n\nState the implemented objective and the algorithm instead, and say "
          "which assumption fails. See this file's docstring for the two claims "
          "that motivated the guard."
    )
