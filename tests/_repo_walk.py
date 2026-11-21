"""Repository-wide file discovery that prunes excluded directories mid-walk.

`Path.rglob` has to descend into a directory to enumerate it, so filtering its
*results* cannot exclude anything. A surface test that lists `ROOT.rglob("*.md")`
and then skips `node_modules` afterwards still walks `node_modules`: on Windows a
vendored pnpm tree exceeds the 260-character path limit and the walk raises
`FileNotFoundError` partway through. That aborted collection of
`test_shipped_mcp_serve_manifests.py` before any test ran, and failed
`test_release_surface_consistency.py` after 8123 files.

So prune during traversal, the way `entroly/repository_intelligence/parsers.py`
already does, and hand enumeration failures back to the caller instead of
dropping them. A surface test that silently skipped part of the tree would pass
for the wrong reason, which is worse than failing.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Directories that hold vendored or generated files, never shipped surfaces.
PRUNED_DIRS = frozenset({
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
})


def discover_repo_files(
    root: Path,
    suffix: str,
    *,
    pruned: frozenset[str] = PRUNED_DIRS,
) -> tuple[list[Path], list[str]]:
    """Return every ``suffix`` file under ``root``, plus enumeration failures.

    Never raises and never descends into ``pruned``, so importing a module that
    calls this cannot abort pytest collection. The second element is empty when
    the whole tree was readable; callers must assert on it, because a partial
    walk cannot support a claim about every shipped surface.
    """
    failures: list[str] = []

    def _record(error: OSError) -> None:
        target = error.filename or str(root)
        try:
            target = Path(target).relative_to(root).as_posix()
        except (TypeError, ValueError):
            target = "<outside-root>"
        failures.append(f"{target} ({type(error).__name__})")

    found: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False, onerror=_record):
        dirnames[:] = sorted(name for name in dirnames if name not in pruned)
        for filename in sorted(filenames):
            if filename.endswith(suffix):
                found.append(Path(directory) / filename)
    return sorted(found), failures
