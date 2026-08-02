"""`entroly simulate` must not tell a first-time user it did nothing.

A project smaller than the default 4,096-token budget has nothing to drop, so
every query honestly reports "0.0% fewer; 0 tokens saved". That is
arithmetically correct and, as the output a new user is most likely to see
first, actively harmful -- it reads as "this tool does nothing" on precisely
the run that decides whether they keep going.

The fix is not to manufacture a demo by squeezing the budget until something
gets cut. An earlier attempt did that and made it worse: the narrowed budget
fell below a single fragment plus per-request overhead, so selection returned
nothing and the output degraded from "0% saved" to "0 fragments, top: none".

Instead the report flags the case, and the CLI explains what happened and where
the value actually appears.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_project(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "auth.py").write_text(
        "def login(user, pw):\n"
        "    token = verify_password(user, pw)\n"
        "    return issue_session_token(token)\n",
        encoding="utf-8",
    )
    (src / "billing.py").write_text(
        "def charge_card(customer, amount):\n"
        "    return StripeGateway().charge(customer.card, amount)\n",
        encoding="utf-8",
    )
    return tmp_path


def _simulate(cwd: Path, *extra: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "entroly", "simulate", "--json", *extra],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
        env={**__import__("os").environ, "ENTROLY_DISABLE_UPDATE_CHECK": "1"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    start = result.stdout.find("{")
    assert start >= 0, f"no JSON in output: {result.stdout[:500]}"
    return json.loads(result.stdout[start:])


def test_small_project_is_flagged_rather_than_reported_as_zero(tiny_project: Path):
    report = _simulate(tiny_project)
    assert report["files_indexed"] == 2
    assert report["budget_narrowed_to_demonstrate"] is True, (
        "a project smaller than the budget must be flagged, so the CLI can "
        "explain the 0% instead of printing it bare"
    )
    # The budget is reported as-is; we do not secretly shrink it to fake a win.
    assert report["budget"] >= report["repo_tokens_indexed"]


def test_small_project_still_selects_everything(tiny_project: Path):
    """0% saved must mean 'nothing needed dropping', not 'nothing was selected'."""
    report = _simulate(tiny_project)
    for row in report["queries"]:
        assert row["selected_fragments"] > 0, (
            "selection returned nothing -- 'top: none' is a worse first "
            f"impression than 0% saved. row={row!r}"
        )


def test_forcing_a_smaller_budget_demonstrates_selection(tiny_project: Path):
    """The command the CLI suggests must actually work."""
    report = _simulate(tiny_project, "--budget", "30")
    assert report["budget_narrowed_to_demonstrate"] is False
    assert report["average_reduction_pct"] > 0, (
        f"--budget 30 should force real selection, got {report!r}"
    )
    for row in report["queries"]:
        assert row["selected_fragments"] > 0
