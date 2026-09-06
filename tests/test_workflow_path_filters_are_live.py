"""A workflow path filter that matches nothing is a gate that never runs.

`public-trust.yml` filtered on a root-level `llms-full.txt`. No such file has
ever existed -- the real one is `docs/llms-full.txt` -- so that entry matched on
no commit in its history, and edits to the actual file never triggered the
workflow through it. The workflow still ran via its other paths, which is
exactly why nobody noticed: a dead entry produces no error, no warning and no
failing run. It simply covers nothing.

This is the same defect #412 fixed for `visibility-integrity.yml`, where two of
the most claim-dense files were absent from the filter that was supposed to
check them. That fix pinned one workflow. This asserts the property across
every workflow in the repository, because the failure is silent by
construction and a per-workflow check only ever finds the instance someone
already suspected.

Measured when written: 195 path entries across all workflows, 1 dead.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is required to parse workflows")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

# Globs are matched, not existence-checked, so a pattern that is *meant* to
# cover files that do not exist yet still has to match something today.
_GLOB_CHARS = "*?["


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


def test_the_repository_actually_has_path_filters_to_check():
    """Guard the guard: an empty sweep would make every assertion below vacuous."""
    filters = _path_filters()
    assert len(filters) > 100, (
        f"only {len(filters)} path filters parsed; the sweep is not seeing the "
        f"workflows and this file would pass without checking anything"
    )


def test_no_workflow_path_filter_matches_nothing():
    """Every filter must resolve to something in the tree today."""
    dead: list[str] = []
    for workflow, location, pattern in _path_filters():
        if any(char in pattern for char in _GLOB_CHARS):
            alive = bool(glob.glob(str(REPO_ROOT / pattern), recursive=True))
            reason = "glob matches no file"
        else:
            alive = (REPO_ROOT / pattern).exists()
            reason = "file does not exist"
        if not alive:
            dead.append(f"{workflow} {location}: {pattern!r} — {reason}")

    assert not dead, (
        "these path filters cover nothing, so the workflows they gate never "
        "trigger on the files they name:\n  " + "\n  ".join(dead) + "\n"
        "Either point the entry at the real path or delete it. A filter that "
        "matches nothing fails silently and reads as coverage."
    )
