"""Say when a local check leaves Entroly state inside the user's project.

`entroly verify-claims` and `entroly simulate` are presented as local checks,
but they leave `.entroly/` (the project vault, filled by the background
belief autoseed) and `.entroly_verification.json` in the working tree, where
`git status` shows them as untracked files. This module reports that, and
makes `.entroly/` ignore itself so its contents stay out of commits.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


_SELF_IGNORE = (
    "# Written by Entroly. This directory holds local state (index, vault,\n"
    "# receipts) and is safe to delete. Delete this file to commit it.\n"
    "*\n"
)


def mark_local_state_dir(directory: str | Path) -> bool:
    """Make a project's ``.entroly`` directory ignore itself in git.

    `entroly go` fills ``<project>/.entroly/vault/`` with one generated summary
    per source file (172 files on pallets/click), and the first ``git add -A``
    after onboarding committed all of them. The same convention as
    ``.pytest_cache`` and ``.ruff_cache`` keeps them out: a ``.gitignore``
    containing ``*`` inside the directory. The user's own ``.gitignore`` is
    never edited, and deleting this one file opts back in.

    Only a directory named ``.entroly`` is marked, an existing ``.gitignore`` is
    never replaced, and failure is silent: this must not break the write it
    accompanies. Returns True when the file was created.
    """
    path = Path(directory)
    if path.name != ".entroly":
        return False
    target = path / ".gitignore"
    try:
        if target.exists():
            return False
        path.mkdir(parents=True, exist_ok=True)
        # Exclusive create: two processes starting at once cannot interleave.
        with open(target, "x", encoding="utf-8") as handle:
            handle.write(_SELF_IGNORE)
        return True
    except OSError:
        return False


def _git_ignored(root: Path, names: list[str]) -> set[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "--", *names],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if result.returncode not in (0, 1):  # 128: not a repository
        return set()
    return {line.strip().rstrip("/") for line in result.stdout.splitlines() if line.strip()}


def _vault_lands_in_project(root: Path) -> bool:
    """The autoseed thread writes <project>/.entroly unless redirected."""
    if os.environ.get("ENTROLY_VAULT") or os.environ.get("ENTROLY_DIR"):
        return False
    try:
        from entroly.belief_autoseed import autoseed_enabled
    except Exception:
        return False
    return autoseed_enabled() and os.access(root, os.W_OK)


def project_footprint_notice(
    project_dir: str | Path, names: tuple[str, ...] = (".entroly",)
) -> str | None:
    """Return a one-line notice for Entroly files left here and not git-ignored.

    The ``.entroly`` directory Entroly fills (now, or from the background
    autoseed a moment later) is marked self-ignoring first, so the notice
    reports what git will actually see rather than racing the autoseed.
    """
    root = Path(project_dir)
    if ".entroly" in names and (
        (root / ".entroly").is_dir() or _vault_lands_in_project(root)
    ):
        mark_local_state_dir(root / ".entroly")
    present = [
        name
        for name in names
        if (root / name).exists() or (name == ".entroly" and _vault_lands_in_project(root))
    ]
    if not present:
        return None
    # Ask about a path inside each directory: a directory that ignores its own
    # contents (see `mark_local_state_dir`) is not itself matched by any
    # pattern, so `git check-ignore .entroly` says "not ignored" while
    # `git status` shows nothing.
    probes = {
        (f"{name}/.entroly-probe" if name == ".entroly" or (root / name).is_dir() else name): name
        for name in present
    }
    ignored_probes = _git_ignored(root, list(probes))
    ignored = {probes[probe] for probe in ignored_probes if probe in probes}
    shown = [name for name in present if name not in ignored]
    if not shown:
        return None
    listed = ", ".join(
        name if (root / name).is_file() else f"{name}/" for name in shown
    )
    return (
        f"Entroly writes local state into this project: {listed} ({root}). "
        "Add it to .gitignore to keep it out of commits."
    )
