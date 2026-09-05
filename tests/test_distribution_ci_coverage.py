"""Every file the distribution checker validates must also trigger it.

Issue #279 Phase 5 asks that the distribution and external-name validators run
in normal CI. `visibility-integrity.yml` runs them behind a `paths:` filter, so
a file the checker *reads* but the filter does not *list* is silently exempt: a
pull request touching only that file never runs the check that guards it.

Measured before this gate, two of the twenty required discovery files were
uncovered -- `README.md` and `docs/BENCHMARKS.md`. Those are the two most
claim-dense files in the repository: install commands and benchmark claims,
which are exactly what Phase 1's exit criteria ("install commands match the
current released package surfaces", "no unlinked benchmark ... claim") rest on.

Asserting the property rather than the two filenames, so adding a file to
`REQUIRED_DISCOVERY_FILES` without extending the trigger fails here instead of
quietly losing coverage.
"""
from __future__ import annotations

import fnmatch
import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github/workflows/visibility-integrity.yml"
CHECKER = REPO_ROOT / "scripts/check_distribution_surface.py"

# Read by `check_distribution_surface.py` outside REQUIRED_DISCOVERY_FILES: the
# documentation map must point `benchmarks` at this path, so its contents are
# part of what the check certifies.
EXTRA_VALIDATED = ("docs/BENCHMARKS.md",)


def _load_checker():
    spec = importlib.util.spec_from_file_location("_cds", CHECKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cds"] = module
    spec.loader.exec_module(module)
    return module


def _trigger_path_blocks() -> list[list[str]]:
    """Every `paths:` list in the workflow (pull_request and push)."""
    text = WORKFLOW.read_text(encoding="utf-8")
    blocks = re.findall(r"paths:\n((?:\s+- '[^']+'\n)+)", text)
    return [re.findall(r"- '([^']+)'", block) for block in blocks]


def _covered(target: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/**"):
            if target.startswith(pattern[:-3] + "/"):
                return True
        elif fnmatch.fnmatch(target, pattern):
            return True
    return False


def test_every_validated_file_triggers_the_workflow():
    if not WORKFLOW.exists() or not CHECKER.exists():
        pytest.skip("distribution workflow or checker is absent")

    checker = _load_checker()
    required = [str(p).replace("\\", "/") for p in checker.REQUIRED_DISCOVERY_FILES]
    required.extend(EXTRA_VALIDATED)

    blocks = _trigger_path_blocks()
    assert blocks, f"no `paths:` filter parsed from {WORKFLOW.name}"

    for index, patterns in enumerate(blocks):
        uncovered = [f for f in required if not _covered(f, patterns)]
        assert not uncovered, (
            f"{WORKFLOW.name} path block #{index + 1} does not trigger on "
            f"{uncovered}, but check_distribution_surface.py validates them. "
            "A pull request touching only those files would skip the check "
            "that guards their install commands and benchmark claims."
        )


def test_the_workflow_still_runs_both_validators():
    """The trigger is only useful if the job actually invokes the checks."""
    if not WORKFLOW.exists():
        pytest.skip("distribution workflow is absent")

    text = WORKFLOW.read_text(encoding="utf-8")
    for script in ("check_distribution_surface.py", "check_external_name_policy.py"):
        assert script in text, (
            f"{WORKFLOW.name} no longer runs {script}; issue #279 Phase 5 "
            "requires both validators in normal CI"
        )


def test_pull_request_and_push_filters_agree():
    """A file guarded on PRs but not on main lets drift land unchecked."""
    if not WORKFLOW.exists():
        pytest.skip("distribution workflow is absent")

    blocks = _trigger_path_blocks()
    if len(blocks) < 2:
        pytest.skip("only one paths filter present")

    first, rest = set(blocks[0]), [set(b) for b in blocks[1:]]
    for index, other in enumerate(rest, start=2):
        assert first == other, (
            f"paths block #1 and #{index} differ: "
            f"only in #1={sorted(first - other)}, only in #{index}={sorted(other - first)}"
        )
