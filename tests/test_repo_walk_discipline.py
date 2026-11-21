"""No test may enumerate the whole repository with `rglob`.

`Path.rglob` descends into a directory in order to list it, so excluding
`node_modules` by filtering the *results* excludes nothing. Two surface tests
did exactly that and both broke on Windows, where a vendored pnpm tree exceeds
the 260-character path limit: one aborted pytest collection before any test ran,
the other failed after 8123 files. Use `_repo_walk.discover_repo_files`, which
prunes during the walk and reports what it could not enumerate.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS = Path(__file__).resolve().parent

#: Module-level names that denote the repository root in this suite. A
#: subdirectory-scoped call reads `(ROOT / "entroly").rglob(...)` and is fine;
#: only an unscoped walk from the root is the problem.
_ROOT_WALK = re.compile(r"\b(?:ROOT|REPO|REPO_ROOT)\.r?glob\(")

_ALLOWED = {"_repo_walk.py", "test_repo_walk_discipline.py"}


def test_no_test_walks_the_repository_root_unpruned() -> None:
    offenders: list[str] = []
    for path in sorted(TESTS.glob("*.py")):
        if path.name in _ALLOWED:
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if _ROOT_WALK.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, (
        "use _repo_walk.discover_repo_files instead of walking the repo root:\n  "
        + "\n  ".join(offenders)
    )
