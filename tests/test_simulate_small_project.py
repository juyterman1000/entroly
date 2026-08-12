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


def _native_engine_available() -> bool:
    """True when the installed native engine passes Entroly's shared gate.

    The resolution ladder -- full / skeleton / reference -- lives in the native
    engine only. Behaviour that depends on it has to be asserted per surface,
    or the pure-Python CI job fails for a feature it does not ship.
    """
    from entroly.native_status import usable_core

    return usable_core() is not None


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


def test_a_realistic_small_project_reports_its_savings(tmp_path: Path):
    """The "already fits" message must never suppress real savings.

    A 6-module, ~2,600-token project fits inside the default 4,096-token budget
    and still reduces context 40-98%, because the resolution ladder demotes
    lower-ranked files to skeletons even when nothing has to be dropped.

    An earlier version of the small-project message gated on "does the repo fit
    in the budget" rather than "did we actually save anything", and printed
    "there is nothing to save yet" directly beneath "1,104 tokens saved".
    """
    app = tmp_path / "app"
    app.mkdir()
    modules = {
        "auth": ("login", "verify_password", "issue_session_token", "refresh_session"),
        "billing": ("charge_card", "refund_payment", "create_invoice", "void_invoice"),
        "users": ("create_user", "get_user", "update_profile", "delete_user"),
        "orders": ("place_order", "cancel_order", "track_shipment", "apply_coupon"),
    }
    for name, fns in modules.items():
        lines = [f'"""{name} module."""', "import logging", "", "logger = logging.getLogger(__name__)", ""]
        for fn in fns:
            lines += [
                f"def {fn}(request, context=None):",
                f'    """Handle {fn.replace("_", " ")} for {name}."""',
                f'    logger.info("{fn} called with %s", request)',
                "    validated = validate_input(request)",
                "    if not validated:",
                '        raise ValueError("invalid request")',
                f"    result = process_{name}(validated, context)",
                "    return finalize(result)",
                "",
            ]
        (app / f"{name}.py").write_text("\n".join(lines), encoding="utf-8")

    report = _simulate(tmp_path)
    assert report["repo_tokens_indexed"] < report["budget"], (
        "fixture must fit inside the budget, or it does not test this path"
    )

    # The saving comes from the resolution ladder, which only the native engine
    # implements: the pure-Python fallback has no skeleton/multi-resolution
    # path, so a project that fits its budget genuinely has nothing to drop and
    # honestly reports 0%. Asserting a native-only number on both surfaces made
    # the pure-Python CI job red for a capability that does not exist there --
    # a real gap, recorded rather than papered over.
    #
    # This is a user-visible difference: a default `pip install entroly` with no
    # Rust wheel sees 0% on exactly the small-project run that decides whether a
    # first-time user keeps going.
    if _native_engine_available():
        assert report["average_reduction_pct"] > 0, (
            "on the native engine a project that fits should still save via "
            f"skeleton demotion; got {report['average_reduction_pct']}%"
        )
        assert report["total_tokens_saved"] > 0
    else:
        assert report["average_reduction_pct"] == 0, (
            "the pure-Python fallback has no resolution ladder, so it should "
            "report 0% rather than a number it cannot have earned; got "
            f"{report['average_reduction_pct']}%"
        )
        # Whatever it reports, it must not have silently dropped context.
        for row in report["queries"]:
            assert row["selected_fragments"] > 0, (
                f"0% saved must mean 'nothing needed dropping', not "
                f"'nothing was selected'. row={row!r}"
            )


def test_results_are_ordered_by_relevance_not_by_density(tmp_path: Path, monkeypatch):
    """The top result must be the most relevant, not the shortest.

    The greedy knapsack sorts candidates by value-per-token, which is correct
    for filling a budget -- but that ordering leaked out as the order callers
    display. Measured on this fixture for "who verifies the password":

        auth.py     24 tokens  relevance 0.58640  density 0.024308
        billing.py  23 tokens  relevance 0.57694  density 0.024960

    auth.py is the better answer and lost the top slot purely for being one
    token longer, so `entroly simulate` reported "top: billing.py" for an
    authentication question.
    """
    from entroly.auto_index import auto_index
    from entroly.server import EntrolyEngine

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
    monkeypatch.chdir(tmp_path)

    engine = EntrolyEngine()
    auto_index(engine, project_dir=str(tmp_path), force=True)

    for query, expected in [
        ("who verifies the password", "auth.py"),
        ("How are credit cards charged?", "billing.py"),
        ("stripe payment gateway", "billing.py"),
        ("login session token", "auth.py"),
    ]:
        result = engine.optimize_context(token_budget=4096, query=query)
        selected = result.get("selected_fragments") or []
        assert selected, f"nothing selected for {query!r}"
        top = selected[0]["source"].rsplit("/", 1)[-1]
        assert top == expected, (
            f"{query!r} ranked {top} above {expected}. Ordering must follow "
            f"relevance; density is for packing a budget, not for display. "
            f"got={[f['source'].rsplit('/', 1)[-1] for f in selected]}"
        )
