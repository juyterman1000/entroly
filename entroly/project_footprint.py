"""Say when a local check leaves Entroly state inside the user's project.

`entroly verify-claims` and `entroly simulate` are presented as local checks,
but they leave `.entroly/` (the project vault, filled by the background
belief autoseed) and `.entroly_verification.json` in the working tree, where
`git status` shows them as untracked files. This module only reports that;
it moves nothing.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


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
    """Return a one-line notice for Entroly files left here and not git-ignored."""
    root = Path(project_dir)
    present = [
        name
        for name in names
        if (root / name).exists() or (name == ".entroly" and _vault_lands_in_project(root))
    ]
    if not present:
        return None
    ignored = _git_ignored(root, present)
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
