"""A workflow path filter that matches nothing is a gate that never runs.

`public-trust.yml` filtered on a root-level `llms-full.txt`. No such file has
ever existed -- the real one is `docs/llms-full.txt` -- so that entry matched on
no commit in its history, and edits to the actual file never triggered the
workflow through it. `agent-integration-contracts.yml` filtered on
`integrations/hermes/**`, a path `.gitignore` excludes, so no file under it can
ever reach a commit; the Hermes integration lives in `entroly/integrations/`,
which a neighbouring entry already covered.

Both workflows still ran via their other paths, which is exactly why neither was
noticed: a dead filter produces no error, no warning and no failing run. It
simply covers nothing.

This is the defect #412 fixed for `visibility-integrity.yml`. That fix pinned
one workflow; this asserts the property across all of them, because the failure
is silent by construction and a per-workflow check only ever finds the instance
someone already suspected.

Matching is done against **git-tracked files**, not the working tree, because
that is what GitHub matches against. A first version of this test used
`glob.glob` over the filesystem and was vacuous on Windows: for a missing
directory, `glob("nope/**", recursive=True)` returns `['nope/']` there while
returning `[]` on Linux, so every `dir/**` pattern looked alive regardless. It
passed locally and failed in CI -- the same class of silent non-coverage it
exists to catch. Using the index also makes an untracked or ignored local
directory unable to mask a dead filter.
"""
from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is required to parse workflows")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"


@lru_cache(maxsize=1)
def _tracked_paths() -> tuple[str, ...]:
    """Every path in the git index, repo-relative with forward slashes."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(f"git ls-files failed: {result.stderr.strip()[:200]}")
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a GitHub path-filter glob into a regex.

    `**` crosses directory separators; `*` and `?` do not. Built by scanning
    rather than with `fnmatch`, whose `*` matches `/` and would call every
    `dir/**` pattern alive.
    """
    out: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**", index):
            out.append(".*")
            index += 2
        elif pattern[index] == "*":
            out.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            out.append("[^/]")
            index += 1
        else:
            out.append(re.escape(pattern[index]))
            index += 1
    return re.compile("^" + "".join(out) + "$")


def _matches_anything(pattern: str) -> bool:
    normalized = pattern.lstrip("!").strip()
    if not normalized:
        return False
    if not any(char in normalized for char in "*?["):
        # A literal entry may name a file or a directory prefix.
        return any(
            tracked == normalized or tracked.startswith(normalized.rstrip("/") + "/")
            for tracked in _tracked_paths()
        )
    matcher = _pattern_to_regex(normalized)
    return any(matcher.match(tracked) for tracked in _tracked_paths())


def _path_filters() -> list[tuple[str, str, str]]:
    """Every (workflow, trigger.key, pattern) path filter in the repository."""
    found: list[tuple[str, str, str]] = []
    for workflow in sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml")):
        document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            continue
        # `on:` parses as the boolean True under the YAML 1.1 rules PyYAML uses.
        triggers = document.get(True) or document.get("on")
        if not isinstance(triggers, dict):
            continue
        for trigger, config in triggers.items():
            if not isinstance(config, dict):
                continue
            for key in ("paths", "paths-ignore"):
                for pattern in config.get(key) or []:
                    found.append((workflow.name, f"{trigger}.{key}", str(pattern)))
    return found


def test_the_sweep_actually_sees_the_repository():
    """Guard the guard: an empty sweep makes every assertion below vacuous."""
    assert len(_tracked_paths()) > 500, (
        f"only {len(_tracked_paths())} tracked files found; the index is not "
        f"being read and this file would pass without checking anything"
    )
    filters = _path_filters()
    assert len(filters) > 100, (
        f"only {len(filters)} path filters parsed; the sweep is not seeing the "
        f"workflows and this file would pass without checking anything"
    )


def test_a_known_live_and_a_known_dead_pattern_are_told_apart():
    """The matcher itself must discriminate, or the sweep proves nothing."""
    assert _matches_anything("docs/llms-full.txt"), "a tracked file read as dead"
    assert _matches_anything("entroly/integrations/**"), "a tracked tree read as dead"
    assert not _matches_anything("llms-full.txt"), (
        "a root-level llms-full.txt does not exist but read as live"
    )
    assert not _matches_anything("integrations/hermes/**"), (
        "a gitignored directory read as live -- this is the Windows glob "
        "behaviour that made the first version of this test vacuous"
    )


def test_no_workflow_path_filter_matches_nothing():
    """Every filter must match something in the index today."""
    dead = [
        f"{workflow} {location}: {pattern!r}"
        for workflow, location, pattern in _path_filters()
        if not _matches_anything(pattern)
    ]
    assert not dead, (
        "these path filters match no tracked file, so the workflows they gate "
        "never trigger on the files they name:\n  " + "\n  ".join(dead) + "\n"
        "Either point the entry at a real path or delete it. A filter that "
        "matches nothing fails silently and reads as coverage."
    )
